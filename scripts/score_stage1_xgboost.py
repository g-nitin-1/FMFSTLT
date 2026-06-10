#!/usr/bin/env python3
"""Score Stage 1 window shards with a trained XGBoost regressor."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

try:
    from tqdm.auto import tqdm
except ImportError:

    class tqdm:  # type: ignore[override]
        def __init__(self, iterable=None, **kwargs) -> None:
            self.iterable = iterable

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def set_postfix_str(self, *args, **kwargs) -> None:
            return None

        def close(self) -> None:
            return None


DEFAULT_SUBSETS = ("train", "val", "test", "robustness")


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Score Stage 1 window shards and summarize error metrics."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage1_windows",
        help="Root directory containing Stage 1 window subsets.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=root_dir
        / "artifacts_exact_public"
        / "stage1_xgboost"
        / "stage1_xgboost_model.json",
        help="Path to a trained Stage 1 XGBoost model.",
    )
    parser.add_argument(
        "--training-summary-path",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage1_xgboost" / "training_summary.json",
        help="Training summary used to recover best_iteration.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage1_predictions",
        help="Directory for prediction shards and metric summaries.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(DEFAULT_SUBSETS),
        help="Stage 1 subsets to score.",
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help="Optional glob relative to each subset directory for a probe run.",
    )
    return parser.parse_args()


def list_subset_paths(input_root: Path, subset: str, input_glob: str | None) -> list[Path]:
    subset_dir = input_root / subset
    if input_glob:
        paths = sorted(subset_dir.glob(input_glob))
    else:
        paths = sorted(subset_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no Stage 1 shards found for subset {subset} under {subset_dir}")
    return paths


def init_metric_bucket() -> dict[str, float]:
    return {
        "count": 0,
        "sum_abs_error": 0.0,
        "sum_sq_error": 0.0,
        "sum_rel_error": 0.0,
    }


def update_metric_bucket(bucket: dict[str, float], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    abs_err = np.abs(y_pred - y_true)
    rel_err = abs_err / np.maximum(np.abs(y_true), 1e-6)
    bucket["count"] += int(y_true.shape[0])
    bucket["sum_abs_error"] += float(abs_err.sum())
    bucket["sum_sq_error"] += float(np.square(y_pred - y_true).sum())
    bucket["sum_rel_error"] += float(rel_err.sum())


def finalize_metric_bucket(bucket: dict[str, float]) -> dict[str, float]:
    count = int(bucket["count"])
    if count == 0:
        return {"count": 0, "rmse": 0.0, "mae": 0.0, "mean_relative_error": 0.0}
    return {
        "count": count,
        "rmse": float(np.sqrt(bucket["sum_sq_error"] / count)),
        "mae": float(bucket["sum_abs_error"] / count),
        "mean_relative_error": float(bucket["sum_rel_error"] / count),
    }


def main() -> None:
    args = parse_args()
    booster = xgb.Booster()
    booster.load_model(args.model_path)

    best_iteration = None
    if args.training_summary_path.exists():
        training_summary = json.loads(args.training_summary_path.read_text())
        best_iteration = training_summary.get("best_iteration")

    iteration_range = None
    if best_iteration is not None:
        iteration_range = (0, int(best_iteration) + 1)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    overall_metrics: dict[str, dict[str, float]] = {}
    by_elapsed: dict[str, dict[str, dict[str, float]]] = {}
    by_speed_tier: dict[str, dict[str, dict[str, float]]] = {}
    manifest_entries: list[dict[str, object]] = []

    for subset in args.subsets:
        paths = list_subset_paths(args.input_root, subset, args.input_glob)
        subset_dir = output_root / subset
        subset_dir.mkdir(parents=True, exist_ok=True)

        overall_bucket = init_metric_bucket()
        elapsed_buckets: dict[int, dict[str, float]] = defaultdict(init_metric_bucket)
        speed_tier_buckets: dict[str, dict[str, float]] = defaultdict(init_metric_bucket)

        path_bar = tqdm(paths, desc=f"score {subset}", unit="shard", dynamic_ncols=True)
        for path in path_bar:
            path_bar.set_postfix_str(path.stem)
            with np.load(path, allow_pickle=False) as data:
                x = data["x"].astype(np.float32, copy=False)
                y_true = data["y_true_mbps"].astype(np.float32, copy=False)
                if iteration_range is None:
                    y_pred = booster.inplace_predict(x)
                else:
                    y_pred = booster.inplace_predict(x, iteration_range=iteration_range)
                y_pred = np.asarray(y_pred, dtype=np.float32)

                out_path = subset_dir / path.name
                np.savez(
                    out_path,
                    y_pred_mbps=y_pred,
                    y_true_mbps=y_true,
                    uuid=data["uuid"],
                    date=data["date"],
                    speed_tier=data["speed_tier"],
                    test_time=data["test_time"],
                    end_bucket=data["end_bucket"],
                    elapsed_ms=data["elapsed_ms"],
                    observed_buckets_seen=data["observed_buckets_seen"],
                    last_observed_bucket=data["last_observed_bucket"],
                    source_split=data["source_split"],
                    source_shard=data["source_shard"],
                    model_path=np.array(str(args.model_path), dtype=np.str_),
                    best_iteration=np.array(
                        -1 if best_iteration is None else int(best_iteration), dtype=np.int32
                    ),
                )

                update_metric_bucket(overall_bucket, y_true, y_pred)

                elapsed_ms = data["elapsed_ms"].astype(np.int32, copy=False)
                for elapsed in np.unique(elapsed_ms):
                    mask = elapsed_ms == elapsed
                    update_metric_bucket(elapsed_buckets[int(elapsed)], y_true[mask], y_pred[mask])

                speed_tiers = data["speed_tier"].tolist()
                speed_tiers_unique = sorted(set(speed_tiers))
                speed_tiers_arr = np.array(speed_tiers, dtype=np.str_)
                for speed_tier in speed_tiers_unique:
                    mask = speed_tiers_arr == speed_tier
                    update_metric_bucket(speed_tier_buckets[speed_tier], y_true[mask], y_pred[mask])

                manifest_entries.append(
                    {
                        "subset": subset,
                        "source_npz": str(path),
                        "prediction_npz": str(out_path),
                        "examples": int(x.shape[0]),
                    }
                )
                print(f"scored {out_path}")
        path_bar.close()

        overall_metrics[subset] = finalize_metric_bucket(overall_bucket)
        by_elapsed[subset] = {
            str(elapsed): finalize_metric_bucket(bucket)
            for elapsed, bucket in sorted(elapsed_buckets.items())
        }
        by_speed_tier[subset] = {
            tier: finalize_metric_bucket(bucket)
            for tier, bucket in sorted(speed_tier_buckets.items())
        }

    summary = {
        "model_path": str(args.model_path),
        "best_iteration": best_iteration,
        "overall_metrics": overall_metrics,
        "by_elapsed_ms": by_elapsed,
        "by_speed_tier": by_speed_tier,
    }
    summary_path = output_root / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    manifest_path = output_root / "manifest_stage1_predictions.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2) + "\n")

    print(f"wrote metrics summary to {summary_path}")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
