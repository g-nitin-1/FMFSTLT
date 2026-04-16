#!/usr/bin/env python3
"""Materialize tensor-ready Stage 2 dataset shards from windows, predictions, and labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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
        description=(
            "Join Stage 1 windows, Stage 1 predictions, and epsilon-specific Stage 2 "
            "labels into tensor-ready Stage 2 dataset shards."
        )
    )
    parser.add_argument(
        "--window-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage1_windows",
        help="Root directory containing Stage 1 window subsets.",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage1_predictions",
        help="Root directory containing scored Stage 1 prediction subsets.",
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage2_labels",
        help="Root directory containing epsilon-specific Stage 2 label subsets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage2_transformer_dataset",
        help="Output directory for tensor-ready Stage 2 shards.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(DEFAULT_SUBSETS),
        help="Subsets to materialize.",
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help="Optional glob relative to each subset directory for a probe run.",
    )
    return parser.parse_args()


def list_subset_paths(root: Path, subset: str, input_glob: str | None) -> list[Path]:
    subset_dir = root / subset
    if input_glob:
        paths = sorted(subset_dir.glob(input_glob))
    else:
        paths = sorted(subset_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no NPZ shards found for subset {subset} under {subset_dir}")
    return paths


def ensure_equal(lhs: np.ndarray, rhs: np.ndarray, *, field_name: str, path: Path) -> None:
    if not np.array_equal(lhs, rhs):
        raise ValueError(f"mismatched {field_name} while joining {path}")


def load_tensor_windows(window_data, window_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    x = window_data["x"].astype(np.float32, copy=False)
    feature_names = window_data["feature_names"]
    feature_dim = int(feature_names.shape[0])
    window_size_buckets = int(window_data["window_size_buckets"].item())

    if x.ndim == 3:
        if x.shape[1] != window_size_buckets or x.shape[2] != feature_dim:
            raise ValueError(
                f"inconsistent tensor window shape in {window_path}: {tuple(x.shape)} "
                f"vs expected (*, {window_size_buckets}, {feature_dim})"
            )
        return x, feature_names, window_size_buckets, feature_dim

    if x.ndim == 2:
        expected_dim = window_size_buckets * feature_dim
        if x.shape[1] != expected_dim:
            raise ValueError(
                f"flat window dimension mismatch in {window_path}: {x.shape[1]} vs {expected_dim}"
            )
        return (
            x.reshape(x.shape[0], window_size_buckets, feature_dim).astype(np.float32, copy=False),
            feature_names,
            window_size_buckets,
            feature_dim,
        )

    raise ValueError(f"unsupported x rank in {window_path}: {x.ndim}")


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "window_root": str(args.window_root),
        "prediction_root": str(args.prediction_root),
        "label_root": str(args.label_root),
        "output_root": str(output_root),
        "subsets": {},
    }
    manifest_entries: list[dict[str, object]] = []

    recorded_window_size: int | None = None
    recorded_feature_dim: int | None = None
    recorded_feature_names: list[str] | None = None
    recorded_epsilon: float | None = None
    recorded_error_kind: str | None = None
    recorded_relative_epsilon_unit: str | None = None
    recorded_effective_relative_epsilon_fraction: float | None = None

    for subset in args.subsets:
        window_paths = list_subset_paths(args.window_root, subset, args.input_glob)
        subset_dir = output_root / subset
        subset_dir.mkdir(parents=True, exist_ok=True)

        subset_windows = 0
        subset_stop_positive_windows = 0
        subset_oracle_positive_windows = 0
        tests_seen: set[tuple[str, str]] = set()
        oracle_tests_seen: set[tuple[str, str]] = set()

        path_bar = tqdm(window_paths, desc=f"stage2 dataset {subset}", unit="shard", dynamic_ncols=True)
        for window_path in path_bar:
            path_bar.set_postfix_str(window_path.stem)
            prediction_path = args.prediction_root / subset / window_path.name
            label_path = args.label_root / subset / window_path.name

            if not prediction_path.exists():
                raise FileNotFoundError(f"missing prediction shard {prediction_path}")
            if not label_path.exists():
                raise FileNotFoundError(f"missing label shard {label_path}")

            with (
                np.load(window_path, allow_pickle=False) as window_data,
                np.load(prediction_path, allow_pickle=False) as prediction_data,
                np.load(label_path, allow_pickle=False) as label_data,
            ):
                x_tensor, feature_names, window_size_buckets, feature_dim = load_tensor_windows(
                    window_data, window_path
                )
                num_examples = int(x_tensor.shape[0])

                if prediction_data["y_pred_mbps"].shape[0] != num_examples:
                    raise ValueError(f"prediction row count mismatch for {window_path.name}")
                if label_data["stop_label"].shape[0] != num_examples:
                    raise ValueError(f"label row count mismatch for {window_path.name}")

                ensure_equal(window_data["uuid"], prediction_data["uuid"], field_name="uuid", path=window_path)
                ensure_equal(
                    window_data["test_time"],
                    prediction_data["test_time"],
                    field_name="test_time",
                    path=window_path,
                )
                ensure_equal(
                    window_data["end_bucket"],
                    prediction_data["end_bucket"],
                    field_name="end_bucket",
                    path=window_path,
                )
                ensure_equal(
                    window_data["elapsed_ms"],
                    prediction_data["elapsed_ms"],
                    field_name="elapsed_ms",
                    path=window_path,
                )
                ensure_equal(
                    window_data["y_true_mbps"],
                    prediction_data["y_true_mbps"],
                    field_name="y_true_mbps",
                    path=window_path,
                )

                source_prediction_name = Path(str(label_data["source_prediction_npz"].item())).name
                if source_prediction_name != prediction_path.name:
                    raise ValueError(
                        f"label shard {label_path} points at {source_prediction_name}, "
                        f"expected {prediction_path.name}"
                    )

                stop_label = label_data["stop_label"].astype(np.uint8, copy=False)
                oracle_stop_window = label_data["is_oracle_stop_window"].astype(np.uint8, copy=False)
                oracle_stop_found = label_data["oracle_stop_found"].astype(np.uint8, copy=False)

                out_path = subset_dir / window_path.name
                np.savez(
                    out_path,
                    x=x_tensor,
                    stop_label=stop_label,
                    is_oracle_stop_window=oracle_stop_window,
                    y_pred_mbps=prediction_data["y_pred_mbps"].astype(np.float32, copy=False),
                    y_true_mbps=prediction_data["y_true_mbps"].astype(np.float32, copy=False),
                    abs_error_mbps=label_data["abs_error_mbps"].astype(np.float32, copy=False),
                    relative_error=label_data["relative_error"].astype(np.float32, copy=False),
                    uuid=window_data["uuid"],
                    date=window_data["date"],
                    speed_tier=window_data["speed_tier"],
                    test_time=window_data["test_time"],
                    end_bucket=window_data["end_bucket"].astype(np.int16, copy=False),
                    elapsed_ms=window_data["elapsed_ms"].astype(np.int32, copy=False),
                    observed_buckets_seen=window_data["observed_buckets_seen"].astype(np.int16, copy=False),
                    last_observed_bucket=window_data["last_observed_bucket"].astype(np.int16, copy=False),
                    oracle_stop_found=oracle_stop_found,
                    oracle_stop_end_bucket=label_data["oracle_stop_end_bucket"].astype(np.int16, copy=False),
                    oracle_stop_elapsed_ms=label_data["oracle_stop_elapsed_ms"].astype(np.int32, copy=False),
                    oracle_stop_observed_buckets_seen=label_data["oracle_stop_observed_buckets_seen"].astype(
                        np.int16, copy=False
                    ),
                    oracle_stop_abs_error_mbps=label_data["oracle_stop_abs_error_mbps"].astype(
                        np.float32, copy=False
                    ),
                    oracle_stop_relative_error=label_data["oracle_stop_relative_error"].astype(
                        np.float32, copy=False
                    ),
                    feature_names=feature_names,
                    window_size_buckets=np.array(window_size_buckets, dtype=np.int16),
                    feature_dim=np.array(feature_dim, dtype=np.int16),
                    source_split=window_data["source_split"],
                    source_shard=window_data["source_shard"],
                    epsilon=label_data["epsilon"].astype(np.float32, copy=False),
                    error_kind=label_data["error_kind"],
                    relative_epsilon_unit=label_data["relative_epsilon_unit"],
                    effective_relative_epsilon_fraction=label_data["effective_relative_epsilon_fraction"].astype(
                        np.float32, copy=False
                    ),
                    source_window_npz=np.array(str(window_path), dtype=np.str_),
                    source_prediction_npz=np.array(str(prediction_path), dtype=np.str_),
                    source_label_npz=np.array(str(label_path), dtype=np.str_),
                )

                subset_windows += num_examples
                subset_stop_positive_windows += int(stop_label.sum())
                subset_oracle_positive_windows += int(oracle_stop_window.sum())

                uuids = window_data["uuid"].tolist()
                test_times = window_data["test_time"].tolist()
                tests_seen.update(zip(uuids, test_times))
                if np.any(oracle_stop_found):
                    oracle_tests_seen.update(
                        (uuids[idx], test_times[idx])
                        for idx in np.flatnonzero(oracle_stop_found)
                    )

                manifest_entries.append(
                    {
                        "subset": subset,
                        "source_window_npz": str(window_path),
                        "source_prediction_npz": str(prediction_path),
                        "source_label_npz": str(label_path),
                        "output_npz": str(out_path),
                        "examples": num_examples,
                    }
                )
                print(f"built {out_path}")

                if recorded_window_size is None:
                    recorded_window_size = window_size_buckets
                    recorded_feature_dim = feature_dim
                    recorded_feature_names = feature_names.tolist()
                    recorded_epsilon = float(label_data["epsilon"].item())
                    recorded_error_kind = str(label_data["error_kind"].item())
                    recorded_relative_epsilon_unit = str(label_data["relative_epsilon_unit"].item())
                    effective_value = float(label_data["effective_relative_epsilon_fraction"].item())
                    recorded_effective_relative_epsilon_fraction = (
                        None if np.isnan(effective_value) else effective_value
                    )
                else:
                    if window_size_buckets != recorded_window_size or feature_dim != recorded_feature_dim:
                        raise ValueError("Stage 2 dataset shape changed across shards")

        path_bar.close()

        summary["subsets"][subset] = {
            "windows": subset_windows,
            "tests": len(tests_seen),
            "tests_with_oracle_stop": len(oracle_tests_seen),
            "oracle_stop_rate": safe_rate(len(oracle_tests_seen), len(tests_seen)),
            "stop_positive_windows": subset_stop_positive_windows,
            "stop_positive_window_rate": safe_rate(subset_stop_positive_windows, subset_windows),
            "oracle_stop_windows": subset_oracle_positive_windows,
            "oracle_stop_window_rate": safe_rate(subset_oracle_positive_windows, subset_windows),
        }

    summary["window_size_buckets"] = recorded_window_size
    summary["feature_dim"] = recorded_feature_dim
    summary["feature_names"] = recorded_feature_names
    summary["epsilon"] = recorded_epsilon
    summary["error_kind"] = recorded_error_kind
    summary["relative_epsilon_unit"] = recorded_relative_epsilon_unit
    summary["effective_relative_epsilon_fraction"] = recorded_effective_relative_epsilon_fraction

    summary_path = output_root / "stage2_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    manifest_path = output_root / "manifest_stage2_transformer_dataset.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2) + "\n")

    print(f"wrote Stage 2 dataset summary to {summary_path}")
    print(f"wrote Stage 2 dataset manifest to {manifest_path}")


if __name__ == "__main__":
    main()
