#!/usr/bin/env python3
"""Offline threshold sweep for Stage 2 Transformer models.

For each trained epsilon model, runs inference on the val and test sets,
sweeps 99 evenly-spaced decision thresholds (0.01 to 0.99), and records
policy-level metrics at each threshold. Results are saved as JSON for
Pareto frontier plotting.

Usage (WSL, CUDA):
    python -m experimental_scripts.rescore_stage2_thresholds --device cuda

Outputs:
    artifacts_exact_public/stage2_threshold_sweep/threshold_sweep_eps_<N>.json
    artifacts_exact_public/stage2_threshold_sweep/threshold_sweep_all.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

if "TMPDIR" not in os.environ:
    for candidate in ("/dev/shm", "/tmp", "/var/tmp", "/usr/tmp"):
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            os.environ["TMPDIR"] = candidate
            break

import torch
from torch import nn

try:
    from tqdm.auto import tqdm
except ImportError:

    class tqdm:  # type: ignore[override]
        def __init__(self, iterable=None, **kwargs) -> None:
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())

        def set_postfix_str(self, *a, **k) -> None:
            pass

        def close(self) -> None:
            pass


EPSILON_VALUES = [5, 10, 15, 20, 25, 30, 35]
DEFAULT_MODEL_PATTERN = "stage2_transformer_eps_{eps}_local_gpu_bs1024_acc4"
DEFAULT_DATASET_PATTERN = "stage2_transformer_dataset_eps_{eps}"


# ---------------------------------------------------------------------------
# Model (must match train_stage2_transformer.py exactly)
# ---------------------------------------------------------------------------


class Stage2Transformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        max_sequence_buckets: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
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
        self,
        x_full: torch.Tensor,
        attention_mask: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(x_full) + self.position_embedding[:, : x_full.shape[1], :]
        key_padding_mask = ~attention_mask.bool()
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        last_indices = torch.clamp(history_lengths.long() - 1, min=0)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_indices]
        pooled = self.norm(pooled)
        return self.head(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    artifacts = root_dir / "artifacts_exact_public"
    p = argparse.ArgumentParser(
        description="Offline policy-level threshold sweep for Stage 2 models."
    )
    p.add_argument(
        "--artifacts-root",
        type=Path,
        default=artifacts,
        help="Root that contains both model dirs and dataset dirs.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=artifacts / "stage2_threshold_sweep",
        help="Directory for per-epsilon sweep JSON files.",
    )
    p.add_argument(
        "--epsilon-values",
        nargs="+",
        type=int,
        default=EPSILON_VALUES,
        help="Epsilon values to process.",
    )
    p.add_argument(
        "--model-pattern",
        default=DEFAULT_MODEL_PATTERN,
        help="Dir name pattern with {eps} placeholder for model dirs.",
    )
    p.add_argument(
        "--dataset-pattern",
        default=DEFAULT_DATASET_PATTERN,
        help="Dir name pattern with {eps} placeholder for dataset dirs.",
    )
    p.add_argument(
        "--subsets",
        nargs="+",
        default=["val", "test"],
        help="Dataset subsets to evaluate.",
    )
    p.add_argument(
        "--threshold-steps",
        type=int,
        default=99,
        help="Number of evenly-spaced thresholds from 0.01 to 0.99.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Inference batch size (decision points per forward pass).",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    p.add_argument(
        "--max-eval-shards",
        type=int,
        default=None,
        help="Optional shard limit for a quick probe.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: Path, device: torch.device) -> nn.Module:
    ckpt_path = model_dir / "stage2_transformer_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = Stage2Transformer(
        input_dim=cfg["feature_dim"],
        max_sequence_buckets=cfg["max_sequence_buckets"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg.get("ff_dim", 512),
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def safe_mean(lst: list[float]) -> float | None:
    return float(np.mean(lst)) if lst else None


def safe_median(lst: list[float]) -> float | None:
    return float(np.median(lst)) if lst else None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def collect_subset_predictions(
    model: nn.Module,
    dataset_dir: Path,
    subset: str,
    device: torch.device,
    batch_size: int,
    max_shards: int | None,
) -> dict[str, np.ndarray]:
    """Run inference on every valid decision point in a subset.

    Returns flat arrays, one entry per (test, decision) pair.
    Per-test fields (oracle_stop_*) are repeated for every decision of that test.
    """
    subset_dir = dataset_dir / subset
    paths = sorted(subset_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no shards found under {subset_dir}")
    if max_shards is not None:
        paths = paths[:max_shards]

    uuid_list: list[str] = []
    test_time_list: list[str] = []
    end_bucket_list: list[int] = []
    elapsed_ms_list: list[int] = []
    probs_list: list[float] = []
    inst_safe_list: list[int] = []
    rel_error_list: list[float] = []
    oracle_found_list: list[int] = []
    oracle_elapsed_list: list[int] = []

    with torch.no_grad():
        shard_bar = tqdm(paths, desc=f"  infer {subset}", unit="shard", dynamic_ncols=True)
        for path in shard_bar:
            with np.load(path, allow_pickle=False) as data:
                x_full = data["x_full"].astype(np.float32)
                decision_valid_mask = data["decision_valid_mask"].astype(bool)
                decision_end_bucket = data["decision_end_bucket"].astype(np.int16)
                decision_elapsed_ms = data["decision_elapsed_ms"].astype(np.int32)
                inst_safe = data["instantaneous_safe_window"].astype(np.uint8)
                rel_error = data["relative_error"].astype(np.float32)
                oracle_found = data["oracle_stop_found"].astype(np.uint8)
                oracle_elapsed = data["oracle_stop_elapsed_ms"].astype(np.int32)
                uuid = data["uuid"]
                test_time = data["test_time"]

            seq_len = x_full.shape[1]
            positions = np.arange(seq_len, dtype=np.int16)

            test_idx, dec_idx = np.nonzero(decision_valid_mask)
            n_decisions = len(test_idx)
            probs_flat = np.empty(n_decisions, dtype=np.float32)

            for start in range(0, n_decisions, batch_size):
                end = min(start + batch_size, n_decisions)
                bt = test_idx[start:end]
                bd = dec_idx[start:end]
                batch_end_bucket = decision_end_bucket[bt, bd]
                batch_history_lengths = batch_end_bucket.astype(np.int32) + 1
                batch_attn = (
                    positions.reshape(1, -1) < batch_history_lengths.reshape(-1, 1)
                ).astype(np.uint8)

                inputs = torch.from_numpy(x_full[bt]).to(device=device, dtype=torch.float32)
                attn = torch.from_numpy(batch_attn).to(device=device, dtype=torch.bool)
                hist = torch.from_numpy(batch_history_lengths).to(device=device, dtype=torch.long)

                logits = model(inputs, attn, hist)
                probs_flat[start:end] = torch.sigmoid(logits).cpu().numpy()

            uuid_list.extend(uuid[test_idx].tolist())
            test_time_list.extend(test_time[test_idx].tolist())
            end_bucket_list.extend(decision_end_bucket[test_idx, dec_idx].tolist())
            elapsed_ms_list.extend(decision_elapsed_ms[test_idx, dec_idx].tolist())
            probs_list.extend(probs_flat.tolist())
            inst_safe_list.extend(inst_safe[test_idx, dec_idx].tolist())
            rel_error_list.extend(rel_error[test_idx, dec_idx].tolist())
            oracle_found_list.extend(oracle_found[test_idx].tolist())
            oracle_elapsed_list.extend(oracle_elapsed[test_idx].tolist())

        shard_bar.close()

    return {
        "uuid": np.array(uuid_list),
        "test_time": np.array(test_time_list),
        "end_bucket": np.array(end_bucket_list, dtype=np.int16),
        "elapsed_ms": np.array(elapsed_ms_list, dtype=np.int32),
        "probabilities": np.array(probs_list, dtype=np.float32),
        "instantaneous_safe": np.array(inst_safe_list, dtype=np.uint8),
        "relative_error": np.array(rel_error_list, dtype=np.float32),
        "oracle_stop_found": np.array(oracle_found_list, dtype=np.uint8),
        "oracle_stop_elapsed_ms": np.array(oracle_elapsed_list, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Policy metric computation
# ---------------------------------------------------------------------------


def compute_policy_at_threshold(
    preds: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, object]:
    uuid = preds["uuid"]
    test_time = preds["test_time"]
    end_bucket = preds["end_bucket"]
    elapsed_ms = preds["elapsed_ms"]
    probs = preds["probabilities"]
    inst_safe = preds["instantaneous_safe"]
    rel_error = preds["relative_error"]
    oracle_found = preds["oracle_stop_found"]
    oracle_elapsed = preds["oracle_stop_elapsed_ms"]

    grouped: dict[tuple[str, str], list[int]] = {}
    for i, key in enumerate(zip(uuid.tolist(), test_time.tolist(), strict=True)):
        grouped.setdefault(key, []).append(i)  # type: ignore[arg-type]

    emitted = 0
    within_eps = 0
    stop_elapsed_vals: list[float] = []
    savings_vals: list[float] = []
    rel_err_vals: list[float] = []
    pct_transferred_vals: list[float] = []
    excess_vs_oracle_vals: list[float] = []

    for indices in grouped.values():
        indices.sort(key=lambda i: int(end_bucket[i]))
        last_idx = indices[-1]
        full_elapsed = float(elapsed_ms[last_idx])

        chosen_idx = last_idx
        fired = False
        for i in indices:
            if probs[i] >= threshold:
                chosen_idx = i
                fired = True
                break

        if fired:
            emitted += 1

        stop_el = float(elapsed_ms[chosen_idx])
        stop_elapsed_vals.append(stop_el)
        savings_vals.append(full_elapsed - stop_el)
        rel_err_vals.append(float(rel_error[chosen_idx]))
        pct_transferred_vals.append(stop_el / full_elapsed if full_elapsed > 0 else 1.0)

        if int(inst_safe[chosen_idx]) == 1:
            within_eps += 1

        if int(oracle_found[indices[0]]) == 1:
            excess_vs_oracle_vals.append(stop_el - float(oracle_elapsed[indices[0]]))

    n = len(grouped)
    return {
        "threshold": float(threshold),
        "tests": n,
        "emitted_stop_rate": emitted / n if n else 0.0,
        "within_epsilon_rate": within_eps / n if n else 0.0,
        "mean_stop_elapsed_ms": safe_mean(stop_elapsed_vals),
        "median_stop_elapsed_ms": safe_median(stop_elapsed_vals),
        "mean_savings_vs_full_ms": safe_mean(savings_vals),
        "median_savings_vs_full_ms": safe_median(savings_vals),
        "mean_pct_data_transferred": safe_mean(pct_transferred_vals),
        "median_pct_data_transferred": safe_median(pct_transferred_vals),
        "mean_relative_error_at_stop": safe_mean(rel_err_vals),
        "median_relative_error_at_stop": safe_median(rel_err_vals),
        "mean_excess_vs_oracle_ms": safe_mean(excess_vs_oracle_vals),
        "median_excess_vs_oracle_ms": safe_median(excess_vs_oracle_vals),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    thresholds = np.linspace(0.01, 0.99, args.threshold_steps).tolist()
    all_results: dict[str, object] = {}

    for eps in args.epsilon_values:
        model_dir = args.artifacts_root / args.model_pattern.format(eps=eps)
        dataset_dir = args.artifacts_root / args.dataset_pattern.format(eps=eps)

        if not model_dir.exists():
            print(f"[skip eps={eps}] model dir not found: {model_dir}")
            continue
        if not dataset_dir.exists():
            print(f"[skip eps={eps}] dataset dir not found: {dataset_dir}")
            continue

        print(f"\n{'=' * 50}")
        print(f"epsilon = {eps}")
        print(f"{'=' * 50}")

        model = load_model(model_dir, device)
        eps_result: dict[str, object] = {"epsilon": eps, "subsets": {}}

        for subset in args.subsets:
            print(f"\n  subset: {subset}")
            preds = collect_subset_predictions(
                model,
                dataset_dir,
                subset,
                device,
                args.batch_size,
                args.max_eval_shards,
            )
            n_tests = len(
                set(
                    zip(
                        preds["uuid"].tolist(),
                        preds["test_time"].tolist(),
                        strict=True,
                    )
                )
            )
            n_decisions = len(preds["probabilities"])
            print(f"  {n_tests} tests, {n_decisions} decision points")

            sweep = []
            thr_bar = tqdm(thresholds, desc=f"  sweep {subset}", unit="thr", dynamic_ncols=True)
            for t in thr_bar:
                sweep.append(compute_policy_at_threshold(preds, t))
            thr_bar.close()

            eps_result["subsets"][subset] = sweep  # type: ignore[index]

        all_results[str(eps)] = eps_result

        out_path = args.output_root / f"threshold_sweep_eps_{eps}.json"
        out_path.write_text(json.dumps(eps_result, indent=2) + "\n")
        print(f"\n  saved {out_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined_path = args.output_root / "threshold_sweep_all.json"
    combined_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nwrote combined sweep to {combined_path}")


if __name__ == "__main__":
    main()
