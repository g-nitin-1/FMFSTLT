#!/usr/bin/env python3
"""Oracle diagnostic for direct policies from foundation prefix throughput error.

This script is not a deployable early-stop evaluator: it uses y_true_mbps to
compute prefix relative error at decision time. Keep it only for diagnostics.
"""

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
        description="Evaluate direct stop policies from foundation prefix throughput predictions."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=artifacts / "stage2_transformer_dataset_eps_10")
    parser.add_argument("--output-root", type=Path, default=artifacts / "foundation_v15_direct_error_policy")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--subsets", nargs="+", default=["val", "test", "robustness"])
    parser.add_argument("--error-threshold-min", type=float, default=0.02)
    parser.add_argument("--error-threshold-max", type=float, default=0.50)
    parser.add_argument("--error-threshold-steps", type=int, default=97)
    parser.add_argument("--patience-values", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--relative-denominator-floor", type=float, default=1e-6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-eval-shards", type=int, default=None)
    parser.add_argument(
        "--allow-oracle-policy",
        action="store_true",
        help="Required acknowledgment: this policy uses y_true_mbps and is not deployable.",
    )
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
def collect_subset_rows(
    *,
    model: CausalPatchFoundationV15,
    dataset_root: Path,
    subset: str,
    device: torch.device,
    batch_size: int,
    max_shards: int | None,
    relative_denominator_floor: float,
) -> dict[str, np.ndarray]:
    paths = sorted((dataset_root / subset).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no shards found for subset {subset}")
    if max_shards is not None:
        paths = paths[:max_shards]

    uuid_list: list[str] = []
    test_time_list: list[str] = []
    end_bucket_list: list[int] = []
    elapsed_ms_list: list[int] = []
    labels_list: list[float] = []
    y_true_list: list[float] = []
    y_pred_list: list[float] = []
    rel_error_list: list[float] = []
    oracle_found_list: list[int] = []
    oracle_elapsed_list: list[int] = []

    progress = tqdm(paths, desc=f"infer {subset}", unit="shard")
    for path in progress:
        with np.load(path, allow_pickle=False) as data:
            x_full = data["x_full"].astype(np.float32, copy=False)
            bucket_mask = data["bucket_mask"].astype(bool, copy=False)
            decision_valid = data["decision_valid_mask"].astype(bool, copy=False)
            decision_end_bucket = data["decision_end_bucket"].astype(np.int16, copy=False)
            decision_elapsed_ms = data["decision_elapsed_ms"].astype(np.int32, copy=False)
            stop_label = data["stop_label"].astype(np.float32, copy=False)
            y_true = data["y_true_mbps"].astype(np.float32, copy=False)
            oracle_found = data["oracle_stop_found"].astype(np.uint8, copy=False)
            oracle_elapsed = data["oracle_stop_elapsed_ms"].astype(np.int32, copy=False)
            uuid = data["uuid"]
            test_time = data["test_time"]

        valid_tests = np.flatnonzero(decision_valid.any(axis=1))
        for start in range(0, valid_tests.shape[0], batch_size):
            idx = valid_tests[start : start + batch_size]
            x_tensor = torch.as_tensor(x_full[idx], dtype=torch.float32, device=device)
            mask_tensor = torch.as_tensor(bucket_mask[idx], dtype=torch.bool, device=device)
            decision_valid_tensor = torch.as_tensor(decision_valid[idx], dtype=torch.bool, device=device)
            outputs = model(x_tensor, mask_tensor, decision_valid_mask=decision_valid_tensor)
            y_pred = mbps_from_log(outputs["throughput_mu"]).cpu().numpy().astype(np.float32)

            y_true_batch = y_true[idx].reshape(-1, 1)
            rel_error = (
                np.abs(y_pred - y_true_batch)
                / np.maximum(np.abs(y_true_batch), relative_denominator_floor)
            ).astype(np.float32)
            test_idx, decision_idx = np.nonzero(decision_valid[idx])
            original_tests = idx[test_idx]

            uuid_list.extend(uuid[original_tests].tolist())
            test_time_list.extend(test_time[original_tests].tolist())
            end_bucket_list.extend(decision_end_bucket[original_tests, decision_idx].tolist())
            elapsed_ms_list.extend(decision_elapsed_ms[original_tests, decision_idx].tolist())
            labels_list.extend(stop_label[original_tests, decision_idx].tolist())
            y_true_list.extend(y_true[original_tests].tolist())
            y_pred_list.extend(y_pred[test_idx, decision_idx].tolist())
            rel_error_list.extend(rel_error[test_idx, decision_idx].tolist())
            oracle_found_list.extend(oracle_found[original_tests].tolist())
            oracle_elapsed_list.extend(oracle_elapsed[original_tests].tolist())
    progress.close()

    return {
        "uuid": np.array(uuid_list),
        "test_time": np.array(test_time_list),
        "end_bucket": np.array(end_bucket_list, dtype=np.int16),
        "elapsed_ms": np.array(elapsed_ms_list, dtype=np.int32),
        "labels": np.array(labels_list, dtype=np.float32),
        "y_true_mbps": np.array(y_true_list, dtype=np.float32),
        "y_pred_mbps": np.array(y_pred_list, dtype=np.float32),
        "relative_error": np.array(rel_error_list, dtype=np.float32),
        "oracle_stop_found": np.array(oracle_found_list, dtype=np.uint8),
        "oracle_stop_elapsed_ms": np.array(oracle_elapsed_list, dtype=np.int32),
    }


def policy_probabilities(rows: dict[str, np.ndarray], error_threshold: float, patience: int) -> np.ndarray:
    probabilities = np.zeros(rows["relative_error"].shape[0], dtype=np.float32)
    grouped: dict[tuple[str, str], list[int]] = {}
    for idx, key in enumerate(zip(rows["uuid"].tolist(), rows["test_time"].tolist())):
        grouped.setdefault(key, []).append(idx)
    for row_indices in grouped.values():
        row_indices.sort(key=lambda idx: (int(rows["end_bucket"][idx]), int(rows["elapsed_ms"][idx])))
        streak = 0
        for idx in row_indices:
            if float(rows["relative_error"][idx]) <= error_threshold:
                streak += 1
            else:
                streak = 0
            if streak >= patience:
                probabilities[idx] = 1.0
    return probabilities


def compute_policy_metrics(
    rows: dict[str, np.ndarray],
    probabilities: np.ndarray,
    error_threshold: float,
) -> dict[str, float | int | None]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for idx, key in enumerate(zip(rows["uuid"].tolist(), rows["test_time"].tolist())):
        grouped.setdefault(key, []).append(idx)

    emitted_stops = 0
    within_epsilon = 0
    stop_elapsed_values: list[float] = []
    stop_relative_error_values: list[float] = []
    stop_abs_error_values: list[float] = []
    savings_values: list[float] = []
    excess_vs_oracle_values: list[float] = []
    oracle_tests = 0

    for row_indices in grouped.values():
        row_indices.sort(key=lambda idx: (int(rows["end_bucket"][idx]), int(rows["elapsed_ms"][idx])))
        chosen_idx = row_indices[-1]
        for idx in row_indices:
            if probabilities[idx] >= 0.5:
                chosen_idx = idx
                emitted_stops += 1
                break
        full_idx = row_indices[-1]
        stop_elapsed = float(rows["elapsed_ms"][chosen_idx])
        full_elapsed = float(rows["elapsed_ms"][full_idx])
        stop_rel_error = float(rows["relative_error"][chosen_idx])
        stop_elapsed_values.append(stop_elapsed)
        stop_relative_error_values.append(stop_rel_error)
        stop_abs_error_values.append(float(abs(rows["y_pred_mbps"][chosen_idx] - rows["y_true_mbps"][chosen_idx])))
        savings_values.append(full_elapsed - stop_elapsed)
        if stop_rel_error <= error_threshold:
            within_epsilon += 1
        if int(rows["oracle_stop_found"][row_indices[0]]) == 1:
            oracle_tests += 1
            excess_vs_oracle_values.append(stop_elapsed - float(rows["oracle_stop_elapsed_ms"][row_indices[0]]))

    total_tests = len(grouped)
    return {
        "tests": total_tests,
        "error_threshold": float(error_threshold),
        "emitted_stop_rate": safe_divide(emitted_stops, total_tests),
        "within_epsilon_rate": safe_divide(within_epsilon, total_tests),
        "mean_stop_elapsed_ms": safe_mean(stop_elapsed_values),
        "median_stop_elapsed_ms": safe_median(stop_elapsed_values),
        "mean_stop_relative_error": safe_mean(stop_relative_error_values),
        "mean_stop_abs_error_mbps": safe_mean(stop_abs_error_values),
        "mean_savings_vs_full_ms": safe_mean(savings_values),
        "median_savings_vs_full_ms": safe_median(savings_values),
        "tests_with_oracle_stop": oracle_tests,
        "mean_excess_vs_oracle_ms": safe_mean(excess_vs_oracle_values),
        "median_excess_vs_oracle_ms": safe_median(excess_vs_oracle_values),
    }


def main() -> None:
    args = parse_args()
    if not args.allow_oracle_policy:
        raise SystemExit(
            "This script is an oracle diagnostic because it uses y_true_mbps at decision time. "
            "Re-run with --allow-oracle-policy only if you explicitly want that diagnostic."
        )
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.model_path, device)
    run_name = args.run_name or args.model_path.parent.name
    thresholds = np.linspace(args.error_threshold_min, args.error_threshold_max, args.error_threshold_steps)

    subsets: dict[str, object] = {}
    for subset in args.subsets:
        rows = collect_subset_rows(
            model=model,
            dataset_root=args.dataset_root,
            subset=subset,
            device=device,
            batch_size=args.batch_size,
            max_shards=args.max_eval_shards,
            relative_denominator_floor=args.relative_denominator_floor,
        )
        sweep: list[dict[str, object]] = []
        for patience in args.patience_values:
            for error_threshold in thresholds.tolist():
                probabilities = policy_probabilities(rows, float(error_threshold), int(patience))
                sweep.append(
                    {
                        "patience": int(patience),
                        "error_threshold": float(error_threshold),
                        "window_metrics": compute_window_metrics(rows["labels"], probabilities, 0.5),
                        "policy_metrics": compute_policy_metrics(rows, probabilities, float(error_threshold)),
                    }
                )
        best_product = max(
            sweep,
            key=lambda item: float(item["policy_metrics"]["within_epsilon_rate"])
            * float(item["policy_metrics"]["mean_savings_vs_full_ms"] or 0.0),
        )
        best_savings_at_66 = max(
            sweep,
            key=lambda item: (
                float(item["policy_metrics"]["mean_savings_vs_full_ms"] or 0.0)
                if float(item["policy_metrics"]["within_epsilon_rate"]) >= 0.66
                else -1.0 + float(item["policy_metrics"]["within_epsilon_rate"])
            ),
        )
        subsets[subset] = {
            "rows": int(rows["relative_error"].shape[0]),
            "relative_error_summary": {
                "min": float(np.min(rows["relative_error"])) if rows["relative_error"].size else None,
                "p50": float(np.quantile(rows["relative_error"], 0.5)) if rows["relative_error"].size else None,
                "p90": float(np.quantile(rows["relative_error"], 0.9)) if rows["relative_error"].size else None,
                "max": float(np.max(rows["relative_error"])) if rows["relative_error"].size else None,
            },
            "best_within_times_savings": best_product,
            "best_savings_at_min_within_0_66": best_savings_at_66,
            "sweep": sweep,
        }

    output = {
        "run_name": run_name,
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "policy": "stop after patience consecutive decisions with model relative error <= error_threshold",
        "checkpoint_summary": checkpoint.get("summary", {}),
        "error_threshold_min": float(args.error_threshold_min),
        "error_threshold_max": float(args.error_threshold_max),
        "error_threshold_steps": int(args.error_threshold_steps),
        "patience_values": [int(item) for item in args.patience_values],
        "subsets": subsets,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / f"{run_name}_direct_error_policy.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote direct error policy sweep to {out_path}")


if __name__ == "__main__":
    main()
