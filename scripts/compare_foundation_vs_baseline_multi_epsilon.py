#!/usr/bin/env python3
"""Multi-epsilon comparison: Foundation Run 3 (Phase 2) vs Stage 2 baseline.

For each subset (val / test / robustness), at each model's selected threshold,
compute the within-epsilon rate at the chosen stop point using TWO different
prediction sources:

  - foundation's own throughput prediction at the stop point
  - XGBoost's prediction at the stop point (stored in the dataset)

This tests whether the Run 3 result generalizes beyond eps=10.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

if "TMPDIR" not in os.environ:
    for c in ("/dev/shm", "/tmp", "/var/tmp", "/usr/tmp"):
        if os.path.isdir(c) and os.access(c, os.W_OK):
            os.environ["TMPDIR"] = c
            break

import torch
from torch import nn

from fmfstlt.models.two_stage import (
    FMNetTwoStage,
    FMNetTwoStageConfig,
    expm1_mbps,
)

try:
    from tqdm.auto import tqdm
except ImportError:

    class tqdm:  # type: ignore
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())

        def set_postfix_str(self, *a, **k):
            pass

        def close(self):
            pass


# Inline baseline Stage 2 Transformer (mirrors train_stage2_transformer.py:Stage2Transformer)
class Stage2TransformerBaseline(nn.Module):
    def __init__(
        self,
        *,
        max_sequence_buckets: int,
        feature_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_sequence_buckets, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, x_full: torch.Tensor, attention_mask: torch.Tensor, history_lengths: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.input_projection(x_full) + self.position_embedding[:, : x_full.shape[1], :]
        key_padding_mask = ~attention_mask.bool()
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        last_indices = torch.clamp(history_lengths.long() - 1, min=0)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_indices]
        pooled = self.norm(pooled)
        return self.head(pooled).squeeze(-1)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    artifacts = root / "artifacts_exact_public"
    p = argparse.ArgumentParser()
    p.add_argument(
        "--foundation-checkpoint",
        type=Path,
        default=artifacts / "foundation_twostage_run3_cosine_ema/phase2_checkpoint.pt",
    )
    p.add_argument("--foundation-threshold", type=float, default=0.45)
    p.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=artifacts
        / "stage2_transformer_eps_10_local_gpu_bs1024_acc4/stage2_transformer_model.pt",
    )
    p.add_argument("--baseline-threshold", type=float, default=0.25)
    p.add_argument(
        "--input-root", type=Path, default=artifacts / "stage2_transformer_dataset_eps_10"
    )
    p.add_argument(
        "--output-root", type=Path, default=artifacts / "foundation_vs_baseline_multi_epsilon"
    )
    p.add_argument("--subsets", nargs="+", default=["val", "test", "robustness"])
    p.add_argument("--epsilons", nargs="+", type=float, default=[5, 10, 15, 20, 25, 30, 35])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--max-shards", type=int, default=None)
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda but CUDA not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_foundation(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = FMNetTwoStageConfig.from_dict(ckpt["config"])
    model = FMNetTwoStage(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def load_baseline(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = Stage2TransformerBaseline(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, cfg


def collect_per_test(paths, foundation, baseline, device, batch_size):
    """For each test in the subset, gather:
    - foundation stop_prob per decision [N, 20]
    - foundation predicted throughput Mbps per decision [N, 20]
    - baseline stop_prob per decision [N, 20]
    - decision_valid_mask, decision_end_bucket, decision_elapsed_ms
    - y_true_mbps per test [N], xgb_y_pred per decision [N, 20]
    """
    fields = {
        "foundation_stop_prob": [],
        "foundation_pred_mbps": [],
        "baseline_stop_prob": [],
        "decision_valid_mask": [],
        "decision_end_bucket": [],
        "decision_elapsed_ms": [],
        "y_true_mbps": [],
        "xgb_y_pred": [],
    }
    with torch.no_grad():
        for path in tqdm(paths, desc="infer", unit="shard"):
            with np.load(path, allow_pickle=False) as d:
                x_full = d["x_full"].astype(np.float32, copy=False)
                decision_valid_mask = d["decision_valid_mask"].astype(bool, copy=False)
                decision_end_bucket = d["decision_end_bucket"].astype(np.int64, copy=False)
                decision_elapsed_ms = d["decision_elapsed_ms"].astype(np.int32, copy=False)
                decision_observed = d["decision_observed_buckets_seen"].astype(np.int32, copy=False)
                xgb_y_pred = d["y_pred_mbps"].astype(np.float32, copy=False)
                y_true = d["y_true_mbps"].astype(np.float32, copy=False)

            N, T, _ = x_full.shape
            D = decision_end_bucket.shape[1]
            f_stop = np.zeros((N, D), dtype=np.float32)
            f_pred = np.zeros((N, D), dtype=np.float32)
            b_stop = np.zeros((N, D), dtype=np.float32)

            # Foundation: one forward per batch-of-tests, all decisions at once
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                x = torch.from_numpy(x_full[start:end]).to(device, non_blocking=True)
                db = torch.from_numpy(decision_end_bucket[start:end]).to(device, non_blocking=True)
                de = torch.from_numpy(decision_elapsed_ms[start:end]).to(device, non_blocking=True)
                obs = torch.from_numpy(decision_observed[start:end]).to(device, non_blocking=True)
                fout = foundation.forward_full(x, db, de, obs, detach_stage1=True)
                f_stop[start:end] = torch.sigmoid(fout["stop_logit"]).cpu().numpy()
                f_pred[start:end] = expm1_mbps(fout["stage1_mu"]).cpu().numpy()

            # Baseline: per (test, decision) pair, attention masks differ
            test_idx, dec_idx = np.nonzero(decision_valid_mask)
            n_pairs = len(test_idx)
            positions = np.arange(T, dtype=np.int32)
            for start in range(0, n_pairs, batch_size):
                bt = test_idx[start : start + batch_size]
                bd = dec_idx[start : start + batch_size]
                eb = decision_end_bucket[bt, bd]
                hist_len = eb.astype(np.int32) + 1
                attn_mask = (positions[None, :] < hist_len[:, None]).astype(np.uint8)

                x = torch.from_numpy(x_full[bt]).to(device, non_blocking=True)
                attn = torch.from_numpy(attn_mask).to(device, non_blocking=True)
                hl = torch.from_numpy(hist_len).to(device, non_blocking=True, dtype=torch.long)
                logits = baseline(x, attn, hl)
                probs = torch.sigmoid(logits).cpu().numpy()
                for k in range(len(bt)):
                    b_stop[bt[k], bd[k]] = probs[k]

            fields["foundation_stop_prob"].append(f_stop)
            fields["foundation_pred_mbps"].append(f_pred)
            fields["baseline_stop_prob"].append(b_stop)
            fields["decision_valid_mask"].append(decision_valid_mask)
            fields["decision_end_bucket"].append(decision_end_bucket)
            fields["decision_elapsed_ms"].append(decision_elapsed_ms)
            fields["y_true_mbps"].append(y_true)
            fields["xgb_y_pred"].append(xgb_y_pred)

    return {k: np.concatenate(v, axis=0) for k, v in fields.items()}


def select_stop_idx(stop_prob: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    N, D = stop_prob.shape
    fired = (stop_prob >= threshold) & valid
    chosen = np.full(N, -1, dtype=np.int64)
    for i in range(N):
        f = np.flatnonzero(fired[i])
        if f.size > 0:
            chosen[i] = f[0]
        else:
            v = np.flatnonzero(valid[i])
            chosen[i] = v[-1] if v.size > 0 else -1
    return chosen


def within_eps_at_stop(
    predictions: np.ndarray, y_true: np.ndarray, chosen_idx: np.ndarray, epsilons: list[float]
) -> dict[str, float]:
    N = predictions.shape[0]
    rows = np.arange(N)
    valid_mask = chosen_idx >= 0
    chosen_safe = np.clip(chosen_idx, 0, predictions.shape[1] - 1)
    sel_pred = predictions[rows, chosen_safe]
    rel = np.abs(sel_pred - y_true) / np.maximum(np.abs(y_true), 1e-6)
    rel = rel[valid_mask]
    return {
        f"within_{int(eps)}pct": float((rel <= eps / 100.0).mean()) if rel.size > 0 else 0.0
        for eps in epsilons
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    foundation, _ = load_foundation(args.foundation_checkpoint, device)
    baseline, _ = load_baseline(args.baseline_checkpoint, device)
    print(
        f"[loaded] foundation thr={args.foundation_threshold} | "
        f"baseline thr={args.baseline_threshold} on {device}"
    )

    summary: dict = {
        "foundation_threshold": args.foundation_threshold,
        "baseline_threshold": args.baseline_threshold,
        "epsilons": args.epsilons,
        "subsets": {},
    }

    for subset in args.subsets:
        subset_dir = args.input_root / subset
        paths = sorted(subset_dir.glob("*.npz"))
        if args.max_shards is not None:
            paths = paths[: args.max_shards]
        if not paths:
            print(f"[skip] {subset}: no shards")
            continue
        print(f"\n=== subset={subset} ({len(paths)} shards) ===")

        data = collect_per_test(paths, foundation, baseline, device, args.batch_size)

        f_chosen = select_stop_idx(
            data["foundation_stop_prob"], data["decision_valid_mask"], args.foundation_threshold
        )
        b_chosen = select_stop_idx(
            data["baseline_stop_prob"], data["decision_valid_mask"], args.baseline_threshold
        )

        y_true = data["y_true_mbps"]

        f_with_f_pred = within_eps_at_stop(
            data["foundation_pred_mbps"], y_true, f_chosen, args.epsilons
        )
        f_with_x_pred = within_eps_at_stop(data["xgb_y_pred"], y_true, f_chosen, args.epsilons)
        b_with_x_pred = within_eps_at_stop(data["xgb_y_pred"], y_true, b_chosen, args.epsilons)
        b_with_f_pred = within_eps_at_stop(
            data["foundation_pred_mbps"], y_true, b_chosen, args.epsilons
        )

        rows = np.arange(len(y_true))
        f_idx_safe = np.clip(f_chosen, 0, data["decision_elapsed_ms"].shape[1] - 1)
        b_idx_safe = np.clip(b_chosen, 0, data["decision_elapsed_ms"].shape[1] - 1)
        last_valid_idx = np.array(
            [
                (
                    np.flatnonzero(data["decision_valid_mask"][i])[-1]
                    if data["decision_valid_mask"][i].any()
                    else 0
                )
                for i in range(len(y_true))
            ]
        )
        full_elapsed = data["decision_elapsed_ms"][rows, last_valid_idx]
        f_savings = float((full_elapsed - data["decision_elapsed_ms"][rows, f_idx_safe]).mean())
        b_savings = float((full_elapsed - data["decision_elapsed_ms"][rows, b_idx_safe]).mean())

        summary["subsets"][subset] = {
            "tests": int(len(y_true)),
            "foundation_mean_savings_ms": f_savings,
            "baseline_mean_savings_ms": b_savings,
            "foundation_stops_with_foundation_pred": f_with_f_pred,
            "foundation_stops_with_xgboost_pred": f_with_x_pred,
            "baseline_stops_with_xgboost_pred": b_with_x_pred,
            "baseline_stops_with_foundation_pred": b_with_f_pred,
        }

        print(
            f"  tests={len(y_true)}  foundation savings={f_savings:.0f}ms  "
            f"baseline savings={b_savings:.0f}ms  (Δ={f_savings - b_savings:+.0f})"
        )
        print()
        eps_hdr = "  epsilon" + " ".join(f"{int(e):>7}" for e in args.epsilons)
        print(eps_hdr)
        print("  " + "-" * (len(eps_hdr) - 2))
        for label, table in [
            ("F stops + F pred", f_with_f_pred),
            ("F stops + X pred", f_with_x_pred),
            ("B stops + X pred", b_with_x_pred),
            ("B stops + F pred", b_with_f_pred),
        ]:
            row = " ".join(f"{table[f'within_{int(e)}pct']:>7.4f}" for e in args.epsilons)
            print(f"  {label:<18}{row}")

    out_path = args.output_root / "multi_epsilon_comparison.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
