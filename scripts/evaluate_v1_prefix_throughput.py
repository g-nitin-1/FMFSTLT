#!/usr/bin/env python3
"""Evaluate foundation v1 (full-trace throughput regressor) in PREFIX mode.

Diagnostic question: v1 was trained on full traces with bucket_mask reflecting
real observation. Does it produce accurate per-decision throughput when we
feed it a prefix-truncated mask?

For each test and each valid Stage 2 decision d (end-bucket 5d+4):
  - prefix_mask[i] = original bucket_mask[i] AND (i <= 5d+4)
  - run v1.encoder + head on (x_full, prefix_mask)
  - prediction in Mbps = expm1(model_output) [model trained with log1p target]

Reports per-decision MAE/RMSE/within-10pct vs y_true_mbps, alongside the
already-computed XGBoost prefix predictions stored in the Stage 2 dataset.
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from foundation_model import TraceFoundationConfig, ThroughputRegressionModel

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())
        def set_postfix(self, *a, **k): pass
        def close(self): pass


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    artifacts = root / "artifacts_exact_public"
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=artifacts / "foundation_throughput_regressor_v1" / "foundation_throughput_model.pt")
    p.add_argument("--input-root", type=Path,
                   default=artifacts / "stage2_transformer_dataset_eps_10")
    p.add_argument("--output-root", type=Path,
                   default=artifacts / "foundation_v1_prefix_throughput_eval")
    p.add_argument("--subsets", nargs="+", default=["val", "test", "robustness"])
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


def load_v1(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = TraceFoundationConfig(
        feature_dim=ckpt["model_config"]["feature_dim"],
        max_sequence_buckets=ckpt["model_config"]["max_sequence_buckets"],
        patch_size=ckpt["model_config"]["patch_size"],
        d_model=ckpt["model_config"]["d_model"],
        patch_hidden_dim=ckpt["model_config"]["patch_hidden_dim"],
        num_heads=ckpt["model_config"]["num_heads"],
        num_layers=ckpt["model_config"]["num_layers"],
        ff_dim=ckpt["model_config"]["ff_dim"],
        dropout=ckpt["model_config"]["dropout"],
    )
    model = ThroughputRegressionModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    target_transform = ckpt.get("target_transform", "log1p")
    return model, cfg, target_transform


def inverse_transform(pred: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "log1p":
        return torch.expm1(pred).clamp_min(0.0)
    return pred


def throughput_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.abs(y_pred - y_true)
    sq = (y_pred - y_true) ** 2
    rel = err / np.maximum(np.abs(y_true), 1e-6)
    return {
        "count": int(y_true.shape[0]),
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(sq.mean())),
        "mre": float(rel.mean()),
        "within_10pct": float((rel <= 0.10).mean()),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    model, cfg, target_transform = load_v1(args.checkpoint, device)
    print(f"[init] v1 throughput regressor on {device} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")
    print(f"[init] target_transform = {target_transform}")

    summary: dict = {
        "checkpoint": str(args.checkpoint),
        "input_root": str(args.input_root),
        "subsets": {},
    }

    for subset in args.subsets:
        subset_dir = args.input_root / subset
        paths = sorted(subset_dir.glob("*.npz"))
        if args.max_shards is not None:
            paths = paths[: args.max_shards]
        if not paths:
            print(f"[skip] {subset}: no shards found")
            continue
        print(f"\n=== subset={subset} ({len(paths)} shards) ===")

        # accumulators
        all_v1_pred_decisions = []   # [N, 20] foundation v1 predictions per decision
        all_xgb_pred_decisions = []  # [N, 20] XGBoost predictions per decision (from shard)
        all_y_true = []              # [N]
        all_valid = []               # [N, 20]
        all_end_bucket = []          # [N, 20]

        for path in tqdm(paths, desc=f"v1-prefix {subset}", unit="shard"):
            with np.load(path, allow_pickle=False) as d:
                x_full = d["x_full"].astype(np.float32, copy=False)
                bucket_mask = d["bucket_mask"].astype(np.uint8, copy=False)
                decision_valid_mask = d["decision_valid_mask"].astype(bool, copy=False)
                decision_end_bucket = d["decision_end_bucket"].astype(np.int32, copy=False)
                y_pred_xgb = d["y_pred_mbps"].astype(np.float32, copy=False)
                y_true = d["y_true_mbps"].astype(np.float32, copy=False)

            N, T, _ = x_full.shape
            D = decision_end_bucket.shape[1]
            v1_pred = np.zeros((N, D), dtype=np.float32)

            # We process (test, decision) pairs in batches.
            # For each pair: prefix_mask[i] = bucket_mask[i] AND (i <= end_bucket).
            # Pack all valid pairs across the shard, then batch through v1.
            test_idx, dec_idx = np.nonzero(decision_valid_mask)
            n_pairs = len(test_idx)
            if n_pairs == 0:
                all_v1_pred_decisions.append(v1_pred)
                all_xgb_pred_decisions.append(y_pred_xgb)
                all_y_true.append(y_true)
                all_valid.append(decision_valid_mask)
                all_end_bucket.append(decision_end_bucket)
                continue

            positions = np.arange(T, dtype=np.int32)
            with torch.no_grad():
                for start in range(0, n_pairs, args.batch_size):
                    bt = test_idx[start: start + args.batch_size]
                    bd = dec_idx[start: start + args.batch_size]
                    eb = decision_end_bucket[bt, bd]
                    # prefix_mask for each (test, decision): visible if observed AND <= end_bucket
                    bm = bucket_mask[bt]                                        # [b, T]
                    in_prefix = (positions[None, :] <= eb[:, None]).astype(np.uint8)
                    prefix_mask = (bm & in_prefix).astype(np.float32)
                    x_b = x_full[bt]                                            # [b, T, F]

                    x_t = torch.from_numpy(x_b).to(device, non_blocking=True)
                    pm_t = torch.from_numpy(prefix_mask).to(device, non_blocking=True)
                    out = model(x_t, pm_t)
                    pred_mbps = inverse_transform(out, target_transform).cpu().numpy()
                    for k in range(len(bt)):
                        v1_pred[bt[k], bd[k]] = pred_mbps[k]

            all_v1_pred_decisions.append(v1_pred)
            all_xgb_pred_decisions.append(y_pred_xgb)
            all_y_true.append(y_true)
            all_valid.append(decision_valid_mask)
            all_end_bucket.append(decision_end_bucket)

        v1_pred = np.concatenate(all_v1_pred_decisions, axis=0)
        xgb_pred = np.concatenate(all_xgb_pred_decisions, axis=0)
        y_true = np.concatenate(all_y_true, axis=0)
        valid = np.concatenate(all_valid, axis=0)
        end_bucket = np.concatenate(all_end_bucket, axis=0)

        # Overall (all valid (test, decision) pairs flattened)
        mask = valid
        target = np.broadcast_to(y_true[:, None], v1_pred.shape)
        overall_v1 = throughput_metrics(target[mask], v1_pred[mask])
        overall_xgb = throughput_metrics(target[mask], xgb_pred[mask])

        # Per-decision-index breakdown
        per_decision = []
        for d in range(v1_pred.shape[1]):
            col_mask = valid[:, d]
            if not col_mask.any():
                per_decision.append({"decision_index": d, "v1": {}, "xgboost": {}})
                continue
            v_v1 = throughput_metrics(target[col_mask, d], v1_pred[col_mask, d])
            v_xgb = throughput_metrics(target[col_mask, d], xgb_pred[col_mask, d])
            per_decision.append({"decision_index": d, "v1": v_v1, "xgboost": v_xgb})

        # Final-decision-only (decision corresponding to bucket 99)
        # In Stage 2 ε=10 dataset, decision d=19 is end_bucket=99 for full-length tests
        is_final = (end_bucket == 99) & valid
        if is_final.any():
            final_v1 = throughput_metrics(target[is_final], v1_pred[is_final])
            final_xgb = throughput_metrics(target[is_final], xgb_pred[is_final])
        else:
            final_v1 = {}
            final_xgb = {}

        summary["subsets"][subset] = {
            "overall": {"v1": overall_v1, "xgboost": overall_xgb},
            "final_position_only": {"v1": final_v1, "xgboost": final_xgb},
            "per_decision_index": per_decision,
        }

        # Print summary
        print(f"\n  {subset} OVERALL prefix throughput (all valid decisions):")
        print(f"    v1:      MAE {overall_v1['mae']:8.3f}  RMSE {overall_v1['rmse']:8.3f}  "
              f"within10% {overall_v1['within_10pct']:.4f}  count {overall_v1['count']}")
        print(f"    xgboost: MAE {overall_xgb['mae']:8.3f}  RMSE {overall_xgb['rmse']:8.3f}  "
              f"within10% {overall_xgb['within_10pct']:.4f}  count {overall_xgb['count']}")
        print(f"\n  {subset} FINAL POSITION (decision at bucket 99):")
        if final_v1:
            print(f"    v1:      MAE {final_v1['mae']:8.3f}  RMSE {final_v1['rmse']:8.3f}  count {final_v1['count']}")
            print(f"    xgboost: MAE {final_xgb['mae']:8.3f}  RMSE {final_xgb['rmse']:8.3f}  count {final_xgb['count']}")

        print(f"\n  {subset} PER-DECISION MAE (v1 vs xgboost):")
        print(f"  {'d':>3} {'end_bucket':>10} {'count':>8} {'v1_MAE':>10} {'xgb_MAE':>10}")
        for entry in per_decision:
            d = entry["decision_index"]
            if entry["v1"]:
                print(f"  {d:>3} {5*d+4:>10} {entry['v1']['count']:>8} "
                      f"{entry['v1']['mae']:>10.3f} {entry['xgboost']['mae']:>10.3f}")

    out_path = args.output_root / "v1_prefix_throughput_eval.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
