#!/usr/bin/env python3
"""Evaluate deployable stability policies from foundation prefix predictions."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path

import numpy as np

if "TMPDIR" not in os.environ:
    for candidate in ("/dev/shm", "/tmp", "/var/tmp", "/usr/tmp"):
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            os.environ["TMPDIR"] = candidate
            break

import torch

from foundation_model import TraceFoundationConfig
from foundation_model_v15 import CausalPatchFoundationV15
from train_foundation_v15_multitask import SPEED_TIERS
from train_stage2_transformer import compute_window_metrics, safe_divide, safe_mean, safe_median

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:  # type: ignore[override]
        def __init__(self, iterable=None, **kwargs) -> None:
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())

        def close(self) -> None:
            return None


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    artifacts = root_dir / "artifacts_exact_public"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deployable early-stop policies that stop when the foundation "
            "prefix throughput prediction becomes stable."
        )
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=artifacts / "stage2_transformer_dataset_eps_10")
    parser.add_argument("--output-root", type=Path, default=artifacts / "foundation_v15_stability_policy")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--subsets", nargs="+", default=["val", "test", "robustness"])
    parser.add_argument("--stability-kind", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--threshold-min", type=float, default=0.005)
    parser.add_argument("--threshold-max", type=float, default=0.50)
    parser.add_argument("--threshold-steps", type=int, default=100)
    parser.add_argument("--patience-values", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--min-decision-index", type=int, default=1)
    parser.add_argument("--min-within-epsilon-rate", type=float, default=0.66)
    parser.add_argument("--epsilon-fraction", type=float, default=0.10)
    parser.add_argument("--relative-denominator-floor", type=float, default=1e-6)
    parser.add_argument("--stability-denominator-floor", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-eval-shards", type=int, default=None)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_from_checkpoint(checkpoint: dict[str, object]) -> TraceFoundationConfig:
    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint missing model_config")
    allowed = {field.name for field in fields(TraceFoundationConfig)}
    return TraceFoundationConfig(**{key: raw_config[key] for key in allowed if key in raw_config})


def load_model(path: Path, device: torch.device) -> tuple[CausalPatchFoundationV15, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device)
    config = config_from_checkpoint(checkpoint)
    speed_tiers = checkpoint.get("speed_tiers", list(SPEED_TIERS))
    model = CausalPatchFoundationV15(config, num_speed_tiers=len(speed_tiers)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def mbps_from_log(mu: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.expm1(mu), min=0.0)


@torch.no_grad()
def collect_subset_predictions(
    *,
    model: CausalPatchFoundationV15,
    dataset_root: Path,
    subset: str,
    device: torch.device,
    batch_size: int,
    max_shards: int | None,
) -> dict[str, np.ndarray]:
    paths = sorted((dataset_root / subset).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no shards found for subset {subset}")
    if max_shards is not None:
        paths = paths[:max_shards]

    y_pred_batches: list[np.ndarray] = []
    y_true_batches: list[np.ndarray] = []
    valid_batches: list[np.ndarray] = []
    elapsed_batches: list[np.ndarray] = []
    labels_batches: list[np.ndarray] = []
    oracle_found_batches: list[np.ndarray] = []
    oracle_elapsed_batches: list[np.ndarray] = []

    progress = tqdm(paths, desc=f"infer {subset}", unit="shard")
    for path in progress:
        with np.load(path, allow_pickle=False) as data:
            x_full = data["x_full"].astype(np.float32, copy=False)
            bucket_mask = data["bucket_mask"].astype(bool, copy=False)
            decision_valid = data["decision_valid_mask"].astype(bool, copy=False)
            decision_elapsed_ms = data["decision_elapsed_ms"].astype(np.int32, copy=False)
            stop_label = data["stop_label"].astype(np.float32, copy=False)
            y_true = data["y_true_mbps"].astype(np.float32, copy=False)
            oracle_found = data["oracle_stop_found"].astype(np.uint8, copy=False)
            oracle_elapsed = data["oracle_stop_elapsed_ms"].astype(np.int32, copy=False)

        valid_tests = np.flatnonzero(decision_valid.any(axis=1))
        for start in range(0, valid_tests.shape[0], batch_size):
            idx = valid_tests[start : start + batch_size]
            x_tensor = torch.as_tensor(x_full[idx], dtype=torch.float32, device=device)
            mask_tensor = torch.as_tensor(bucket_mask[idx], dtype=torch.bool, device=device)
            decision_valid_tensor = torch.as_tensor(decision_valid[idx], dtype=torch.bool, device=device)
            outputs = model(x_tensor, mask_tensor, decision_valid_mask=decision_valid_tensor)
            y_pred = mbps_from_log(outputs["throughput_mu"]).cpu().numpy().astype(np.float32)

            y_pred_batches.append(y_pred)
            y_true_batches.append(y_true[idx])
            valid_batches.append(decision_valid[idx])
            elapsed_batches.append(decision_elapsed_ms[idx])
            labels_batches.append(stop_label[idx])
            oracle_found_batches.append(oracle_found[idx])
            oracle_elapsed_batches.append(oracle_elapsed[idx])
    progress.close()

    return {
        "y_pred_mbps": np.concatenate(y_pred_batches, axis=0),
        "y_true_mbps": np.concatenate(y_true_batches, axis=0),
        "decision_valid_mask": np.concatenate(valid_batches, axis=0).astype(bool, copy=False),
        "decision_elapsed_ms": np.concatenate(elapsed_batches, axis=0).astype(np.int32, copy=False),
        "stop_label": np.concatenate(labels_batches, axis=0).astype(np.float32, copy=False),
        "oracle_stop_found": np.concatenate(oracle_found_batches, axis=0).astype(np.uint8, copy=False),
        "oracle_stop_elapsed_ms": np.concatenate(oracle_elapsed_batches, axis=0).astype(np.int32, copy=False),
    }


def stability_mask(
    y_pred: np.ndarray,
    valid: np.ndarray,
    *,
    threshold: float,
    kind: str,
    denominator_floor: float,
    min_decision_index: int,
) -> np.ndarray:
    stable = np.zeros_like(valid, dtype=bool)
    previous = y_pred[:, :-1]
    current = y_pred[:, 1:]
    delta = np.abs(current - previous)
    if kind == "relative":
        denom = np.maximum(np.maximum(np.abs(previous), np.abs(current)), denominator_floor)
        score = delta / denom
    else:
        score = delta
    stable[:, 1:] = valid[:, 1:] & valid[:, :-1] & (score <= threshold)
    if min_decision_index > 0:
        stable[:, :min_decision_index] = False
    return stable


def choose_stop_indices(valid: np.ndarray, stable: np.ndarray, patience: int) -> tuple[np.ndarray, np.ndarray]:
    test_count, decision_count = valid.shape
    last_valid = np.maximum(valid.sum(axis=1).astype(np.int32) - 1, 0)
    stop_idx = last_valid.copy()
    emitted = np.zeros(test_count, dtype=bool)
    streak = np.zeros(test_count, dtype=np.int16)

    for decision_idx in range(decision_count):
        streak = np.where(stable[:, decision_idx], streak + 1, 0)
        can_stop = (streak >= patience) & ~emitted & valid[:, decision_idx]
        stop_idx[can_stop] = decision_idx
        emitted[can_stop] = True
    return stop_idx, emitted


def row_policy_predictions(valid: np.ndarray, stable: np.ndarray, patience: int) -> np.ndarray:
    predictions = np.zeros_like(valid, dtype=np.float32)
    streak = np.zeros(valid.shape[0], dtype=np.int16)
    for decision_idx in range(valid.shape[1]):
        streak = np.where(stable[:, decision_idx], streak + 1, 0)
        predictions[:, decision_idx] = ((streak >= patience) & valid[:, decision_idx]).astype(np.float32)
    return predictions


def evaluate_policy(
    data: dict[str, np.ndarray],
    *,
    threshold: float,
    patience: int,
    stability_kind: str,
    stability_denominator_floor: float,
    min_decision_index: int,
    epsilon_fraction: float,
    relative_denominator_floor: float,
) -> dict[str, object]:
    y_pred = data["y_pred_mbps"]
    y_true = data["y_true_mbps"]
    valid = data["decision_valid_mask"]
    elapsed = data["decision_elapsed_ms"]

    stable = stability_mask(
        y_pred,
        valid,
        threshold=threshold,
        kind=stability_kind,
        denominator_floor=stability_denominator_floor,
        min_decision_index=min_decision_index,
    )
    stop_idx, emitted = choose_stop_indices(valid, stable, patience)
    row_predictions = row_policy_predictions(valid, stable, patience)

    row = np.arange(y_true.shape[0])
    last_valid = np.maximum(valid.sum(axis=1).astype(np.int32) - 1, 0)
    stop_pred = y_pred[row, stop_idx]
    stop_elapsed = elapsed[row, stop_idx].astype(np.float64)
    full_elapsed = elapsed[row, last_valid].astype(np.float64)
    relative_error = (
        np.abs(stop_pred - y_true)
        / np.maximum(np.abs(y_true), relative_denominator_floor)
    ).astype(np.float64)
    abs_error = np.abs(stop_pred - y_true).astype(np.float64)
    within = relative_error <= epsilon_fraction

    oracle_found = data["oracle_stop_found"].astype(bool, copy=False)
    excess_vs_oracle = stop_elapsed[oracle_found] - data["oracle_stop_elapsed_ms"][oracle_found].astype(np.float64)

    valid_flat = valid.reshape(-1)
    labels_flat = data["stop_label"].reshape(-1)[valid_flat]
    predictions_flat = row_predictions.reshape(-1)[valid_flat]
    window_metrics = compute_window_metrics(labels_flat, predictions_flat, 0.5)
    savings = full_elapsed - stop_elapsed
    policy_metrics = {
        "tests": int(y_true.shape[0]),
        "threshold": float(threshold),
        "patience": int(patience),
        "emitted_stop_rate": float(emitted.mean()) if emitted.size else 0.0,
        "within_epsilon_rate": float(within.mean()) if within.size else 0.0,
        "mean_stop_elapsed_ms": safe_mean(stop_elapsed.tolist()),
        "median_stop_elapsed_ms": safe_median(stop_elapsed.tolist()),
        "mean_stop_relative_error": safe_mean(relative_error.tolist()),
        "mean_stop_abs_error_mbps": safe_mean(abs_error.tolist()),
        "mean_savings_vs_full_ms": safe_mean(savings.tolist()),
        "median_savings_vs_full_ms": safe_median(savings.tolist()),
        "tests_with_oracle_stop": int(oracle_found.sum()),
        "mean_excess_vs_oracle_ms": safe_mean(excess_vs_oracle.tolist()),
        "median_excess_vs_oracle_ms": safe_median(excess_vs_oracle.tolist()),
    }
    return {
        "threshold": float(threshold),
        "patience": int(patience),
        "window_metrics": window_metrics,
        "policy_metrics": policy_metrics,
    }


def best_summaries(sweep: list[dict[str, object]], min_within: float) -> dict[str, object]:
    best_product = max(
        sweep,
        key=lambda item: float(item["policy_metrics"]["within_epsilon_rate"])
        * float(item["policy_metrics"]["mean_savings_vs_full_ms"] or 0.0),
    )
    best_savings_at_min = max(
        sweep,
        key=lambda item: (
            float(item["policy_metrics"]["mean_savings_vs_full_ms"] or 0.0)
            if float(item["policy_metrics"]["within_epsilon_rate"]) >= min_within
            else -1.0 + float(item["policy_metrics"]["within_epsilon_rate"])
        ),
    )
    best_within = max(sweep, key=lambda item: float(item["policy_metrics"]["within_epsilon_rate"]))
    return {
        "best_within_times_savings": best_product,
        "best_savings_at_min_within": best_savings_at_min,
        "best_within_epsilon": best_within,
    }


def main() -> None:
    args = parse_args()
    if args.threshold_steps <= 0:
        raise SystemExit("--threshold-steps must be positive")
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.model_path, device)
    run_name = args.run_name or args.model_path.parent.name
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps, dtype=np.float64)

    subsets: dict[str, object] = {}
    for subset in args.subsets:
        data = collect_subset_predictions(
            model=model,
            dataset_root=args.dataset_root,
            subset=subset,
            device=device,
            batch_size=args.batch_size,
            max_shards=args.max_eval_shards,
        )
        sweep: list[dict[str, object]] = []
        progress = tqdm(args.patience_values, desc=f"sweep {subset}", unit="patience")
        for patience in progress:
            for threshold in thresholds.tolist():
                sweep.append(
                    evaluate_policy(
                        data,
                        threshold=float(threshold),
                        patience=int(patience),
                        stability_kind=args.stability_kind,
                        stability_denominator_floor=args.stability_denominator_floor,
                        min_decision_index=args.min_decision_index,
                        epsilon_fraction=args.epsilon_fraction,
                        relative_denominator_floor=args.relative_denominator_floor,
                    )
                )
        progress.close()

        pred = data["y_pred_mbps"]
        valid = data["decision_valid_mask"]
        deltas = np.abs(pred[:, 1:] - pred[:, :-1])
        valid_deltas = (valid[:, 1:] & valid[:, :-1])
        if args.stability_kind == "relative":
            denom = np.maximum(np.maximum(np.abs(pred[:, 1:]), np.abs(pred[:, :-1])), args.stability_denominator_floor)
            delta_score = deltas / denom
        else:
            delta_score = deltas
        valid_scores = delta_score[valid_deltas]

        subsets[subset] = {
            "tests": int(data["y_true_mbps"].shape[0]),
            "decision_rows": int(valid.sum()),
            "stability_score_summary": {
                "min": float(np.min(valid_scores)) if valid_scores.size else None,
                "p50": float(np.quantile(valid_scores, 0.5)) if valid_scores.size else None,
                "p90": float(np.quantile(valid_scores, 0.9)) if valid_scores.size else None,
                "max": float(np.max(valid_scores)) if valid_scores.size else None,
            },
            **best_summaries(sweep, args.min_within_epsilon_rate),
            "sweep": sweep,
        }

    output = {
        "run_name": run_name,
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "policy": "stop when consecutive prefix throughput predictions are stable",
        "stability_kind": args.stability_kind,
        "threshold_min": float(args.threshold_min),
        "threshold_max": float(args.threshold_max),
        "threshold_steps": int(args.threshold_steps),
        "patience_values": [int(item) for item in args.patience_values],
        "min_decision_index": int(args.min_decision_index),
        "min_within_epsilon_rate": float(args.min_within_epsilon_rate),
        "epsilon_fraction": float(args.epsilon_fraction),
        "checkpoint_summary": checkpoint.get("summary", {}),
        "subsets": subsets,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / f"{run_name}_stability_policy.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote stability policy sweep to {out_path}")


if __name__ == "__main__":
    main()
