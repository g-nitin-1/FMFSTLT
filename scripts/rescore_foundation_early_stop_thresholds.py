#!/usr/bin/env python3
"""Offline threshold sweep for foundation early-stop checkpoints."""

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

from foundation_model import EarlyStopFoundationModel, TraceFoundationConfig
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
        description="Evaluate a trained foundation early-stop checkpoint across decision thresholds."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to foundation_early_stop_model.pt.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=artifacts / "stage2_transformer_dataset_eps_10",
        help="Frozen epsilon=10 Stage 2 dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=artifacts / "foundation_early_stop_threshold_sweep",
        help="Directory for threshold sweep JSON.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name used for the output JSON file. Defaults to the model parent directory name.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["val", "test", "robustness"],
        help="Dataset subsets to evaluate.",
    )
    parser.add_argument("--threshold-min", type=float, default=0.01, help="Minimum threshold.")
    parser.add_argument("--threshold-max", type=float, default=0.99, help="Maximum threshold.")
    parser.add_argument("--threshold-steps", type=int, default=99, help="Number of threshold values.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Decision-point inference batch size.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Inference device.")
    parser.add_argument("--max-eval-shards", type=int, default=None, help="Optional shard limit for probes.")
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
    config_kwargs = {key: raw_config[key] for key in allowed if key in raw_config}
    return TraceFoundationConfig(**config_kwargs)


def load_model(model_path: Path, device: torch.device) -> tuple[EarlyStopFoundationModel, dict[str, object]]:
    checkpoint = torch.load(model_path, map_location=device)
    config = config_from_checkpoint(checkpoint)
    model = EarlyStopFoundationModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def collect_subset_predictions(
    *,
    model: EarlyStopFoundationModel,
    dataset_root: Path,
    subset: str,
    device: torch.device,
    batch_size: int,
    max_shards: int | None,
) -> dict[str, np.ndarray]:
    subset_dir = dataset_root / subset
    paths = sorted(subset_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no shards found under {subset_dir}")
    if max_shards is not None:
        paths = paths[:max_shards]

    uuid_list: list[str] = []
    test_time_list: list[str] = []
    end_bucket_list: list[int] = []
    elapsed_ms_list: list[int] = []
    probabilities_list: list[float] = []
    labels_list: list[float] = []
    instantaneous_safe_list: list[int] = []
    y_true_list: list[float] = []
    y_pred_list: list[float] = []
    relative_error_list: list[float] = []
    oracle_found_list: list[int] = []
    oracle_elapsed_list: list[int] = []

    with torch.no_grad():
        progress = tqdm(paths, desc=f"infer {subset}", unit="shard")
        for path in progress:
            with np.load(path, allow_pickle=False) as data:
                x_full = data["x_full"].astype(np.float32, copy=False)
                bucket_mask = data["bucket_mask"].astype(bool, copy=False)
                decision_valid_mask = data["decision_valid_mask"].astype(bool, copy=False)
                decision_end_bucket = data["decision_end_bucket"].astype(np.int16, copy=False)
                stop_label = data["stop_label"].astype(np.float32, copy=False)
                instantaneous_safe = data["instantaneous_safe_window"].astype(np.uint8, copy=False)
                y_true = data["y_true_mbps"].astype(np.float32, copy=False)
                y_pred = data["y_pred_mbps"].astype(np.float32, copy=False)
                relative_error = data["relative_error"].astype(np.float32, copy=False)
                oracle_found = data["oracle_stop_found"].astype(np.uint8, copy=False)
                oracle_elapsed = data["oracle_stop_elapsed_ms"].astype(np.int32, copy=False)
                uuid = data["uuid"]
                test_time = data["test_time"]

            test_idx, decision_idx = np.nonzero(decision_valid_mask)
            positions = np.arange(x_full.shape[1], dtype=np.int16)
            probabilities = np.empty(test_idx.shape[0], dtype=np.float32)

            for start in range(0, test_idx.shape[0], batch_size):
                end = min(start + batch_size, test_idx.shape[0])
                batch_tests = test_idx[start:end]
                batch_decisions = decision_idx[start:end]
                batch_end_bucket = decision_end_bucket[batch_tests, batch_decisions]
                history_lengths = batch_end_bucket.astype(np.int32) + 1
                prefix_bucket_mask = np.logical_and(
                    bucket_mask[batch_tests],
                    positions.reshape(1, -1) < history_lengths.reshape(-1, 1),
                )

                x_tensor = torch.as_tensor(x_full[batch_tests], dtype=torch.float32, device=device)
                mask_tensor = torch.as_tensor(prefix_bucket_mask, dtype=torch.bool, device=device)
                logits = model(x_tensor, mask_tensor)
                probabilities[start:end] = torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False)

            uuid_list.extend(uuid[test_idx].tolist())
            test_time_list.extend(test_time[test_idx].tolist())
            end_bucket_list.extend(decision_end_bucket[test_idx, decision_idx].tolist())
            elapsed_ms_list.extend(((decision_end_bucket[test_idx, decision_idx].astype(np.int32) + 1) * 100).tolist())
            probabilities_list.extend(probabilities.tolist())
            labels_list.extend(stop_label[test_idx, decision_idx].tolist())
            instantaneous_safe_list.extend(instantaneous_safe[test_idx, decision_idx].tolist())
            y_true_list.extend(y_true[test_idx].tolist())
            y_pred_list.extend(y_pred[test_idx, decision_idx].tolist())
            relative_error_list.extend(relative_error[test_idx, decision_idx].tolist())
            oracle_found_list.extend(oracle_found[test_idx].tolist())
            oracle_elapsed_list.extend(oracle_elapsed[test_idx].tolist())
        progress.close()

    return {
        "uuid": np.array(uuid_list),
        "test_time": np.array(test_time_list),
        "end_bucket": np.array(end_bucket_list, dtype=np.int16),
        "elapsed_ms": np.array(elapsed_ms_list, dtype=np.int32),
        "probabilities": np.array(probabilities_list, dtype=np.float32),
        "labels": np.array(labels_list, dtype=np.float32),
        "instantaneous_safe": np.array(instantaneous_safe_list, dtype=np.uint8),
        "y_true_mbps": np.array(y_true_list, dtype=np.float32),
        "y_pred_mbps": np.array(y_pred_list, dtype=np.float32),
        "relative_error": np.array(relative_error_list, dtype=np.float32),
        "oracle_stop_found": np.array(oracle_found_list, dtype=np.uint8),
        "oracle_stop_elapsed_ms": np.array(oracle_elapsed_list, dtype=np.int32),
    }


def compute_policy_metrics(preds: dict[str, np.ndarray], threshold: float) -> dict[str, float | int | None]:
    probabilities = preds["probabilities"]
    uuid = preds["uuid"]
    test_time = preds["test_time"]
    end_bucket = preds["end_bucket"]
    elapsed_ms = preds["elapsed_ms"]
    y_true_mbps = preds["y_true_mbps"]
    y_pred_mbps = preds["y_pred_mbps"]
    relative_error = preds["relative_error"]
    instantaneous_safe = preds["instantaneous_safe"]
    oracle_found = preds["oracle_stop_found"]
    oracle_elapsed = preds["oracle_stop_elapsed_ms"]

    grouped_rows: dict[tuple[str, str], list[int]] = {}
    for idx, key in enumerate(zip(uuid.tolist(), test_time.tolist())):
        grouped_rows.setdefault(key, []).append(idx)

    emitted_stops = 0
    within_epsilon = 0
    stop_elapsed_values: list[float] = []
    stop_relative_error_values: list[float] = []
    stop_abs_error_values: list[float] = []
    savings_values: list[float] = []
    excess_vs_oracle_values: list[float] = []
    oracle_tests = 0

    for row_indices in grouped_rows.values():
        row_indices.sort(key=lambda idx: (int(end_bucket[idx]), int(elapsed_ms[idx])))
        chosen_idx = row_indices[-1]
        for idx in row_indices:
            if probabilities[idx] >= threshold:
                chosen_idx = idx
                emitted_stops += 1
                break

        full_idx = row_indices[-1]
        stop_elapsed = float(elapsed_ms[chosen_idx])
        full_elapsed = float(elapsed_ms[full_idx])
        stop_elapsed_values.append(stop_elapsed)
        stop_relative_error_values.append(float(relative_error[chosen_idx]))
        stop_abs_error_values.append(float(abs(y_pred_mbps[chosen_idx] - y_true_mbps[chosen_idx])))
        savings_values.append(full_elapsed - stop_elapsed)

        if int(instantaneous_safe[chosen_idx]) == 1:
            within_epsilon += 1

        if int(oracle_found[row_indices[0]]) == 1:
            oracle_tests += 1
            excess_vs_oracle_values.append(stop_elapsed - float(oracle_elapsed[row_indices[0]]))

    total_tests = len(grouped_rows)
    return {
        "tests": total_tests,
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


def sweep_thresholds(preds: dict[str, np.ndarray], thresholds: list[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        row = {
            "threshold": float(threshold),
            "window_metrics": compute_window_metrics(preds["labels"], preds["probabilities"], float(threshold)),
            "policy_metrics": compute_policy_metrics(preds, float(threshold)),
        }
        rows.append(row)
    return rows


def pareto_frontier(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frontier: list[dict[str, object]] = []
    for row in rows:
        policy = row["policy_metrics"]
        within = float(policy["within_epsilon_rate"])
        savings = float(policy["mean_savings_vs_full_ms"])
        dominated = False
        for other in rows:
            other_policy = other["policy_metrics"]
            other_within = float(other_policy["within_epsilon_rate"])
            other_savings = float(other_policy["mean_savings_vs_full_ms"])
            if (
                other_within >= within
                and other_savings >= savings
                and (other_within > within or other_savings > savings)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(
        frontier,
        key=lambda item: (
            float(item["policy_metrics"]["within_epsilon_rate"]),
            float(item["policy_metrics"]["mean_savings_vs_full_ms"]),
        ),
        reverse=True,
    )


def best_by_metric(rows: list[dict[str, object]], metric_path: tuple[str, str]) -> dict[str, object]:
    section, metric = metric_path
    return max(rows, key=lambda row: float(row[section][metric]))


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.model_path, device)
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps, dtype=np.float64).tolist()
    run_name = args.run_name or args.model_path.parent.name

    result: dict[str, object] = {
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "checkpoint_summary": checkpoint.get("summary", {}),
        "thresholds": thresholds,
        "subsets": {},
    }

    for subset in args.subsets:
        preds = collect_subset_predictions(
            model=model,
            dataset_root=args.dataset_root,
            subset=subset,
            device=device,
            batch_size=args.batch_size,
            max_shards=args.max_eval_shards,
        )
        rows = sweep_thresholds(preds, thresholds)
        result["subsets"][subset] = {
            "decision_points": int(preds["probabilities"].shape[0]),
            "sweep": rows,
            "pareto_frontier": pareto_frontier(rows),
            "best_f1": best_by_metric(rows, ("window_metrics", "f1")),
            "best_within_epsilon": best_by_metric(rows, ("policy_metrics", "within_epsilon_rate")),
            "best_savings": best_by_metric(rows, ("policy_metrics", "mean_savings_vs_full_ms")),
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / f"{run_name}_threshold_sweep.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote foundation early-stop threshold sweep to {out_path}")


if __name__ == "__main__":
    main()
