#!/usr/bin/env python3
"""Build a paper-faithful Stage 2 Transformer dataset from normalized shards."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
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


WINDOW_DEFAULT = 20
DECISION_STRIDE_DEFAULT = 5
DEFAULT_SOURCE_SPLITS = ("train", "test", "robustness")


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Build a paper-faithful Stage 2 Transformer dataset with full-history "
            "inputs, 500 ms decision tables, Stage 1 predictions, and monotonic "
            "oracle labels."
        )
    )
    parser.add_argument(
        "--normalized-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "normalized_shards",
        help="Root directory containing normalized exact-public shard split subdirectories.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage1_uuid_split.npz",
        help="UUID-level train/validation split produced by make_stage1_uuid_split.py.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=root_dir
        / "artifacts_exact_public"
        / "stage1_xgboost_full_windows"
        / "stage1_xgboost_model.json",
        help="Path to the trained Stage 1 XGBoost regressor.",
    )
    parser.add_argument(
        "--training-summary-path",
        type=Path,
        default=root_dir
        / "artifacts_exact_public"
        / "stage1_xgboost_full_windows"
        / "training_summary.json",
        help="Training summary used to recover best_iteration for Stage 1 scoring.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage2_transformer_dataset",
        help="Output directory for Stage 2 Transformer dataset shards.",
    )
    parser.add_argument(
        "--source-splits",
        nargs="+",
        default=list(DEFAULT_SOURCE_SPLITS),
        help="Normalized source splits to process.",
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help="Optional glob evaluated relative to normalized root. Useful for probes.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="Accuracy tolerance used for the Stage 2 oracle labels.",
    )
    parser.add_argument(
        "--error-kind",
        choices=("relative", "absolute"),
        default="relative",
        help="Threshold errors by relative error or absolute Mbps error.",
    )
    parser.add_argument(
        "--relative-epsilon-unit",
        choices=("percent", "fraction"),
        default="percent",
        help=(
            "Unit for --epsilon when --error-kind=relative. "
            "Use 'percent' for paper-style values like 5, 10, ..., 35."
        ),
    )
    parser.add_argument(
        "--relative-denominator-floor",
        type=float,
        default=1e-6,
        help="Floor used in relative error computation to avoid division by zero.",
    )
    parser.add_argument(
        "--window-size-buckets",
        type=int,
        default=WINDOW_DEFAULT,
        help="Stage 1 lookback used when re-scoring decision points.",
    )
    parser.add_argument(
        "--decision-stride-buckets",
        type=int,
        default=DECISION_STRIDE_DEFAULT,
        help="Decision stride in 100 ms buckets. Paper default is 5 (500 ms).",
    )
    parser.add_argument(
        "--min-decision-bucket",
        type=int,
        default=DECISION_STRIDE_DEFAULT - 1,
        help="Smallest end bucket considered as a Stage 2 decision point.",
    )
    parser.add_argument(
        "--max-examples-per-file",
        type=int,
        default=20000,
        help="Maximum tests per output NPZ shard.",
    )
    return parser.parse_args()


def load_split_map(split_path: Path) -> dict[str, str]:
    with np.load(split_path, allow_pickle=False) as data:
        return dict(zip(data["uuid"].tolist(), data["subset"].tolist(), strict=True))


def iter_source_paths(
    normalized_root: Path,
    source_splits: Iterable[str],
    input_glob: str | None,
) -> list[Path]:
    if input_glob:
        paths = sorted(normalized_root.glob(input_glob))
        if not paths:
            raise SystemExit(f"no normalized shard paths matched {input_glob!r}")
        return paths

    paths: list[Path] = []
    for split in source_splits:
        split_dir = normalized_root / split
        split_paths = sorted(split_dir.glob("*.npz"))
        if not split_paths:
            raise SystemExit(f"no normalized shards found under {split_dir}")
        paths.extend(split_paths)
    return paths


def compute_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    relative_denominator_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    abs_error = np.abs(y_pred - y_true).astype(np.float32, copy=False)
    relative_error = (abs_error / np.maximum(np.abs(y_true), relative_denominator_floor)).astype(
        np.float32, copy=False
    )
    return abs_error, relative_error


def safe_mask_for_error_kind(
    *,
    abs_error: np.ndarray,
    relative_error: np.ndarray,
    epsilon: float,
    error_kind: str,
    relative_epsilon_unit: str,
) -> np.ndarray:
    if error_kind == "absolute":
        return abs_error <= epsilon
    if relative_epsilon_unit == "percent":
        return relative_error <= (epsilon / 100.0)
    return relative_error <= epsilon


def make_stage1_window(
    x_dense: np.ndarray, end_bucket: int, window_size_buckets: int
) -> np.ndarray:
    start = max(0, end_bucket - window_size_buckets + 1)
    window = x_dense[start : end_bucket + 1]
    if window.shape[0] < window_size_buckets:
        pad = np.repeat(window[-1:], window_size_buckets - window.shape[0], axis=0)
        window = np.concatenate([pad, window], axis=0)
    return window.astype(np.float32, copy=False)


def selected_decision_buckets(
    *,
    last_observed_bucket: int,
    min_decision_bucket: int,
    decision_stride_buckets: int,
) -> np.ndarray:
    if last_observed_bucket < min_decision_bucket:
        return np.empty((0,), dtype=np.int16)
    return np.arange(
        min_decision_bucket,
        last_observed_bucket + 1,
        decision_stride_buckets,
        dtype=np.int16,
    )


def load_best_iteration(training_summary_path: Path) -> int | None:
    if not training_summary_path.exists():
        return None
    training_summary = json.loads(training_summary_path.read_text())
    value = training_summary.get("best_iteration")
    if value is None:
        return None
    return int(value)


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


@dataclass
class Buffer:
    x_full: list[np.ndarray] = field(default_factory=list)
    bucket_mask: list[np.ndarray] = field(default_factory=list)
    decision_valid_mask: list[np.ndarray] = field(default_factory=list)
    decision_end_bucket: list[np.ndarray] = field(default_factory=list)
    decision_elapsed_ms: list[np.ndarray] = field(default_factory=list)
    decision_observed_buckets_seen: list[np.ndarray] = field(default_factory=list)
    y_pred_mbps: list[np.ndarray] = field(default_factory=list)
    abs_error_mbps: list[np.ndarray] = field(default_factory=list)
    relative_error: list[np.ndarray] = field(default_factory=list)
    instantaneous_safe_window: list[np.ndarray] = field(default_factory=list)
    stop_label: list[np.ndarray] = field(default_factory=list)
    continue_label: list[np.ndarray] = field(default_factory=list)
    is_oracle_stop_window: list[np.ndarray] = field(default_factory=list)
    oracle_stop_found: list[int] = field(default_factory=list)
    oracle_stop_end_bucket: list[int] = field(default_factory=list)
    oracle_stop_elapsed_ms: list[int] = field(default_factory=list)
    oracle_stop_observed_buckets_seen: list[int] = field(default_factory=list)
    oracle_stop_abs_error_mbps: list[float] = field(default_factory=list)
    oracle_stop_relative_error: list[float] = field(default_factory=list)
    y_true_mbps: list[float] = field(default_factory=list)
    uuid: list[str] = field(default_factory=list)
    date: list[str] = field(default_factory=list)
    speed_tier: list[str] = field(default_factory=list)
    test_time: list[str] = field(default_factory=list)
    source_split: list[str] = field(default_factory=list)
    source_shard: list[str] = field(default_factory=list)
    last_observed_bucket: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.uuid)

    def clear(self) -> None:
        for value in self.__dict__.values():
            value.clear()


def flush_buffer(
    buffer: Buffer,
    *,
    output_root: Path,
    subset: str,
    source_split: str,
    source_shard_name: str,
    counters: dict[tuple[str, str], int],
    feature_names: np.ndarray,
    max_sequence_buckets: int,
    max_decisions_per_test: int,
    window_size_buckets: int,
    decision_stride_buckets: int,
    epsilon: float,
    error_kind: str,
    relative_epsilon_unit: str,
) -> dict[str, object] | None:
    if len(buffer) == 0:
        return None

    key = (subset, source_shard_name)
    part_index = counters[key]
    counters[key] += 1

    subset_dir = output_root / subset
    subset_dir.mkdir(parents=True, exist_ok=True)
    out_path = subset_dir / f"{source_shard_name}__part{part_index:04d}.npz"

    effective_relative_epsilon_fraction = (
        epsilon / 100.0
        if error_kind == "relative" and relative_epsilon_unit == "percent"
        else (epsilon if error_kind == "relative" else np.nan)
    )

    np.savez(
        out_path,
        x_full=np.stack(buffer.x_full, axis=0).astype(np.float32, copy=False),
        bucket_mask=np.stack(buffer.bucket_mask, axis=0).astype(np.uint8, copy=False),
        decision_valid_mask=np.stack(buffer.decision_valid_mask, axis=0).astype(
            np.uint8, copy=False
        ),
        decision_end_bucket=np.stack(buffer.decision_end_bucket, axis=0).astype(
            np.int16, copy=False
        ),
        decision_elapsed_ms=np.stack(buffer.decision_elapsed_ms, axis=0).astype(
            np.int32, copy=False
        ),
        decision_observed_buckets_seen=np.stack(
            buffer.decision_observed_buckets_seen, axis=0
        ).astype(np.int16, copy=False),
        y_pred_mbps=np.stack(buffer.y_pred_mbps, axis=0).astype(np.float32, copy=False),
        abs_error_mbps=np.stack(buffer.abs_error_mbps, axis=0).astype(np.float32, copy=False),
        relative_error=np.stack(buffer.relative_error, axis=0).astype(np.float32, copy=False),
        instantaneous_safe_window=np.stack(buffer.instantaneous_safe_window, axis=0).astype(
            np.uint8, copy=False
        ),
        stop_label=np.stack(buffer.stop_label, axis=0).astype(np.uint8, copy=False),
        continue_label=np.stack(buffer.continue_label, axis=0).astype(np.uint8, copy=False),
        is_oracle_stop_window=np.stack(buffer.is_oracle_stop_window, axis=0).astype(
            np.uint8, copy=False
        ),
        oracle_stop_found=np.array(buffer.oracle_stop_found, dtype=np.uint8),
        oracle_stop_end_bucket=np.array(buffer.oracle_stop_end_bucket, dtype=np.int16),
        oracle_stop_elapsed_ms=np.array(buffer.oracle_stop_elapsed_ms, dtype=np.int32),
        oracle_stop_observed_buckets_seen=np.array(
            buffer.oracle_stop_observed_buckets_seen, dtype=np.int16
        ),
        oracle_stop_abs_error_mbps=np.array(buffer.oracle_stop_abs_error_mbps, dtype=np.float32),
        oracle_stop_relative_error=np.array(buffer.oracle_stop_relative_error, dtype=np.float32),
        y_true_mbps=np.array(buffer.y_true_mbps, dtype=np.float32),
        uuid=np.array(buffer.uuid, dtype=np.str_),
        date=np.array(buffer.date, dtype=np.str_),
        speed_tier=np.array(buffer.speed_tier, dtype=np.str_),
        test_time=np.array(buffer.test_time, dtype=np.str_),
        source_split=np.array(buffer.source_split, dtype=np.str_),
        source_shard=np.array(buffer.source_shard, dtype=np.str_),
        last_observed_bucket=np.array(buffer.last_observed_bucket, dtype=np.int16),
        feature_names=feature_names,
        max_sequence_buckets=np.array(max_sequence_buckets, dtype=np.int16),
        max_decisions_per_test=np.array(max_decisions_per_test, dtype=np.int16),
        window_size_buckets=np.array(window_size_buckets, dtype=np.int16),
        decision_stride_buckets=np.array(decision_stride_buckets, dtype=np.int16),
        decision_stride_ms=np.array(decision_stride_buckets * 100, dtype=np.int32),
        epsilon=np.array(epsilon, dtype=np.float32),
        error_kind=np.array(error_kind, dtype=np.str_),
        relative_epsilon_unit=np.array(relative_epsilon_unit, dtype=np.str_),
        effective_relative_epsilon_fraction=np.array(
            effective_relative_epsilon_fraction,
            dtype=np.float32,
        ),
        oracle_definition=np.array("permanent_safe_suffix", dtype=np.str_),
        stop_label_definition=np.array("monotonic_suffix_from_oracle", dtype=np.str_),
        stage2_input_definition=np.array("full_history_prefix_with_decision_mask", dtype=np.str_),
    )

    manifest_entry = {
        "subset": subset,
        "source_split": source_split,
        "source_shard": source_shard_name,
        "output_npz": str(out_path),
        "tests": len(buffer),
    }
    buffer.clear()
    return manifest_entry


def main() -> None:
    args = parse_args()
    if args.epsilon < 0:
        raise SystemExit("--epsilon must be non-negative")
    if args.relative_denominator_floor <= 0:
        raise SystemExit("--relative-denominator-floor must be positive")
    if args.window_size_buckets <= 0:
        raise SystemExit("--window-size-buckets must be positive")
    if args.decision_stride_buckets <= 0:
        raise SystemExit("--decision-stride-buckets must be positive")
    if args.min_decision_bucket < 0:
        raise SystemExit("--min-decision-bucket must be non-negative")
    if args.max_examples_per_file <= 0:
        raise SystemExit("--max-examples-per-file must be positive")

    split_map = load_split_map(args.split_path)
    source_paths = iter_source_paths(args.normalized_root, args.source_splits, args.input_glob)

    booster = xgb.Booster()
    booster.load_model(args.model_path)
    best_iteration = load_best_iteration(args.training_summary_path)
    iteration_range = None if best_iteration is None else (0, best_iteration + 1)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    buffers: dict[tuple[str, str, str], Buffer] = defaultdict(Buffer)
    counters: dict[tuple[str, str], int] = defaultdict(int)
    manifest_entries: list[dict[str, object]] = []

    overall_summary: dict[str, object] = {
        "normalized_root": str(args.normalized_root),
        "split_path": str(args.split_path),
        "model_path": str(args.model_path),
        "training_summary_path": str(args.training_summary_path),
        "output_root": str(output_root),
        "epsilon": args.epsilon,
        "error_kind": args.error_kind,
        "relative_epsilon_unit": args.relative_epsilon_unit,
        "effective_relative_epsilon_fraction": (
            args.epsilon / 100.0
            if args.error_kind == "relative" and args.relative_epsilon_unit == "percent"
            else (args.epsilon if args.error_kind == "relative" else None)
        ),
        "window_size_buckets": args.window_size_buckets,
        "decision_stride_buckets": args.decision_stride_buckets,
        "decision_stride_ms": args.decision_stride_buckets * 100,
        "min_decision_bucket": args.min_decision_bucket,
        "oracle_definition": "permanent_safe_suffix",
        "stop_label_definition": "monotonic_suffix_from_oracle",
        "stage2_input_definition": "full_history_prefix_with_decision_mask",
        "subsets": {},
    }

    subset_test_counts = defaultdict(int)
    subset_decision_counts = defaultdict(int)
    subset_positive_counts = defaultdict(int)
    subset_oracle_counts = defaultdict(int)
    subset_oracle_test_counts = defaultdict(int)
    subset_history_lengths: dict[str, list[int]] = defaultdict(list)

    max_sequence_buckets: int | None = None
    max_decisions_per_test: int | None = None
    recorded_feature_names: np.ndarray | None = None

    shard_bar = tqdm(source_paths, desc="stage2 dataset", unit="shard", dynamic_ncols=True)
    for shard_path in shard_bar:
        source_split = shard_path.parent.name
        source_shard_name = shard_path.stem
        shard_bar.set_postfix_str(f"{source_split}/{source_shard_name}")
        with np.load(shard_path, allow_pickle=False) as data:
            x_full = data["x"].astype(np.float32, copy=False)
            bucket_mask = data["bucket_mask"].astype(np.uint8, copy=False)
            y_true = data["y_true_mbps"].astype(np.float32, copy=False)
            uuids = data["uuid"].tolist()
            dates = data["date"].tolist()
            speed_tiers = data["speed_tier"].tolist()
            test_times = data["test_time"].tolist()
            feature_names = data["feature_names"]

            current_max_sequence_buckets = int(x_full.shape[1])
            current_max_decisions_per_test = int(
                max(
                    0,
                    ((current_max_sequence_buckets - 1) - args.min_decision_bucket)
                    // args.decision_stride_buckets
                    + 1,
                )
            )

            if max_sequence_buckets is None:
                max_sequence_buckets = current_max_sequence_buckets
                max_decisions_per_test = current_max_decisions_per_test
                recorded_feature_names = feature_names
            else:
                if current_max_sequence_buckets != max_sequence_buckets:
                    raise ValueError(
                        f"inconsistent sequence length in {shard_path}: "
                        f"{current_max_sequence_buckets} vs {max_sequence_buckets}"
                    )
                if current_max_decisions_per_test != max_decisions_per_test:
                    raise ValueError(
                        f"inconsistent decision count in {shard_path}: "
                        f"{current_max_decisions_per_test} vs {max_decisions_per_test}"
                    )
                if not np.array_equal(feature_names, recorded_feature_names):
                    raise ValueError(f"feature mismatch in {shard_path}")

            for idx in range(x_full.shape[0]):
                uuid = uuids[idx]
                if source_split == "train":
                    subset = split_map.get(uuid)
                    if subset is None:
                        raise KeyError(f"uuid {uuid} missing from {args.split_path}")
                else:
                    subset = source_split

                observed = np.flatnonzero(bucket_mask[idx])
                if observed.size == 0:
                    continue

                last_observed_bucket = int(observed[-1])
                decision_buckets = selected_decision_buckets(
                    last_observed_bucket=last_observed_bucket,
                    min_decision_bucket=args.min_decision_bucket,
                    decision_stride_buckets=args.decision_stride_buckets,
                )
                if decision_buckets.size == 0:
                    continue

                stage1_windows = np.stack(
                    [
                        make_stage1_window(
                            x_full[idx],
                            int(end_bucket),
                            args.window_size_buckets,
                        ).reshape(-1)
                        for end_bucket in decision_buckets.tolist()
                    ],
                    axis=0,
                ).astype(np.float32, copy=False)
                if iteration_range is None:
                    y_pred = booster.inplace_predict(stage1_windows)
                else:
                    y_pred = booster.inplace_predict(
                        stage1_windows, iteration_range=iteration_range
                    )
                y_pred = np.asarray(y_pred, dtype=np.float32)

                y_true_decisions = np.full(decision_buckets.shape, y_true[idx], dtype=np.float32)
                abs_error, relative_error = compute_errors(
                    y_true=y_true_decisions,
                    y_pred=y_pred,
                    relative_denominator_floor=args.relative_denominator_floor,
                )
                instantaneous_safe = safe_mask_for_error_kind(
                    abs_error=abs_error,
                    relative_error=relative_error,
                    epsilon=args.epsilon,
                    error_kind=args.error_kind,
                    relative_epsilon_unit=args.relative_epsilon_unit,
                )

                oracle_stop_found = bool(instantaneous_safe.size > 0 and instantaneous_safe[-1])
                oracle_position = None
                if oracle_stop_found:
                    unsafe_positions = np.flatnonzero(~instantaneous_safe)
                    oracle_position = (
                        0 if unsafe_positions.size == 0 else int(unsafe_positions[-1] + 1)
                    )

                decision_valid_mask = np.zeros((max_decisions_per_test,), dtype=np.uint8)
                decision_end_bucket_arr = np.full((max_decisions_per_test,), -1, dtype=np.int16)
                decision_elapsed_ms_arr = np.full((max_decisions_per_test,), -1, dtype=np.int32)
                decision_observed_buckets_seen_arr = np.full(
                    (max_decisions_per_test,), -1, dtype=np.int16
                )
                y_pred_arr = np.zeros((max_decisions_per_test,), dtype=np.float32)
                abs_error_arr = np.zeros((max_decisions_per_test,), dtype=np.float32)
                relative_error_arr = np.zeros((max_decisions_per_test,), dtype=np.float32)
                instantaneous_safe_arr = np.zeros((max_decisions_per_test,), dtype=np.uint8)
                stop_label_arr = np.zeros((max_decisions_per_test,), dtype=np.uint8)
                continue_label_arr = np.ones((max_decisions_per_test,), dtype=np.uint8)
                is_oracle_stop_window_arr = np.zeros((max_decisions_per_test,), dtype=np.uint8)

                valid_count = int(decision_buckets.shape[0])
                decision_valid_mask[:valid_count] = 1
                decision_end_bucket_arr[:valid_count] = decision_buckets
                decision_elapsed_ms_arr[:valid_count] = (
                    decision_buckets.astype(np.int32) + 1
                ) * 100
                decision_observed_buckets_seen_arr[:valid_count] = np.array(
                    [
                        int(bucket_mask[idx, : end_bucket + 1].sum())
                        for end_bucket in decision_buckets.tolist()
                    ],
                    dtype=np.int16,
                )
                y_pred_arr[:valid_count] = y_pred
                abs_error_arr[:valid_count] = abs_error
                relative_error_arr[:valid_count] = relative_error
                instantaneous_safe_arr[:valid_count] = instantaneous_safe.astype(np.uint8)

                oracle_stop_end_bucket = -1
                oracle_stop_elapsed_ms = -1
                oracle_stop_observed_buckets_seen = -1
                oracle_stop_abs_error_mbps = float("nan")
                oracle_stop_relative_error = float("nan")
                if oracle_position is not None:
                    stop_label_arr[oracle_position:valid_count] = 1
                    continue_label_arr[oracle_position:valid_count] = 0
                    is_oracle_stop_window_arr[oracle_position] = 1
                    oracle_stop_end_bucket = int(decision_buckets[oracle_position])
                    oracle_stop_elapsed_ms = int((oracle_stop_end_bucket + 1) * 100)
                    oracle_stop_observed_buckets_seen = int(
                        decision_observed_buckets_seen_arr[oracle_position]
                    )
                    oracle_stop_abs_error_mbps = float(abs_error[oracle_position])
                    oracle_stop_relative_error = float(relative_error[oracle_position])

                buffer_key = (subset, source_split, source_shard_name)
                buffer = buffers[buffer_key]
                buffer.x_full.append(x_full[idx])
                buffer.bucket_mask.append(bucket_mask[idx])
                buffer.decision_valid_mask.append(decision_valid_mask)
                buffer.decision_end_bucket.append(decision_end_bucket_arr)
                buffer.decision_elapsed_ms.append(decision_elapsed_ms_arr)
                buffer.decision_observed_buckets_seen.append(decision_observed_buckets_seen_arr)
                buffer.y_pred_mbps.append(y_pred_arr)
                buffer.abs_error_mbps.append(abs_error_arr)
                buffer.relative_error.append(relative_error_arr)
                buffer.instantaneous_safe_window.append(instantaneous_safe_arr)
                buffer.stop_label.append(stop_label_arr)
                buffer.continue_label.append(continue_label_arr)
                buffer.is_oracle_stop_window.append(is_oracle_stop_window_arr)
                buffer.oracle_stop_found.append(1 if oracle_stop_found else 0)
                buffer.oracle_stop_end_bucket.append(oracle_stop_end_bucket)
                buffer.oracle_stop_elapsed_ms.append(oracle_stop_elapsed_ms)
                buffer.oracle_stop_observed_buckets_seen.append(oracle_stop_observed_buckets_seen)
                buffer.oracle_stop_abs_error_mbps.append(oracle_stop_abs_error_mbps)
                buffer.oracle_stop_relative_error.append(oracle_stop_relative_error)
                buffer.y_true_mbps.append(float(y_true[idx]))
                buffer.uuid.append(uuid)
                buffer.date.append(dates[idx])
                buffer.speed_tier.append(speed_tiers[idx])
                buffer.test_time.append(test_times[idx])
                buffer.source_split.append(source_split)
                buffer.source_shard.append(source_shard_name)
                buffer.last_observed_bucket.append(last_observed_bucket)

                subset_test_counts[subset] += 1
                subset_decision_counts[subset] += valid_count
                subset_positive_counts[subset] += int(stop_label_arr[:valid_count].sum())
                subset_oracle_counts[subset] += int(is_oracle_stop_window_arr[:valid_count].sum())
                subset_oracle_test_counts[subset] += int(oracle_stop_found)
                subset_history_lengths[subset].append(last_observed_bucket + 1)

                if len(buffer) >= args.max_examples_per_file:
                    manifest_entry = flush_buffer(
                        buffer,
                        output_root=output_root,
                        subset=subset,
                        source_split=source_split,
                        source_shard_name=source_shard_name,
                        counters=counters,
                        feature_names=feature_names,
                        max_sequence_buckets=max_sequence_buckets,
                        max_decisions_per_test=max_decisions_per_test,
                        window_size_buckets=args.window_size_buckets,
                        decision_stride_buckets=args.decision_stride_buckets,
                        epsilon=args.epsilon,
                        error_kind=args.error_kind,
                        relative_epsilon_unit=args.relative_epsilon_unit,
                    )
                    if manifest_entry is not None:
                        manifest_entries.append(manifest_entry)
                        print(
                            f"built {manifest_entry['output_npz']} "
                            f"({manifest_entry['tests']} tests)"
                        )

    shard_bar.close()

    for (subset, source_split, source_shard_name), buffer in sorted(buffers.items()):
        if (
            recorded_feature_names is None
            or max_sequence_buckets is None
            or max_decisions_per_test is None
        ):
            raise RuntimeError(
                "dataset builder reached flush stage without recorded feature metadata"
            )
        manifest_entry = flush_buffer(
            buffer,
            output_root=output_root,
            subset=subset,
            source_split=source_split,
            source_shard_name=source_shard_name,
            counters=counters,
            feature_names=recorded_feature_names,
            max_sequence_buckets=max_sequence_buckets,
            max_decisions_per_test=max_decisions_per_test,
            window_size_buckets=args.window_size_buckets,
            decision_stride_buckets=args.decision_stride_buckets,
            epsilon=args.epsilon,
            error_kind=args.error_kind,
            relative_epsilon_unit=args.relative_epsilon_unit,
        )
        if manifest_entry is not None:
            manifest_entries.append(manifest_entry)
            print(f"built {manifest_entry['output_npz']} ({manifest_entry['tests']} tests)")

    if (
        recorded_feature_names is None
        or max_sequence_buckets is None
        or max_decisions_per_test is None
    ):
        raise RuntimeError("no examples were materialized")

    for subset in ("train", "val", "test", "robustness"):
        test_count = int(subset_test_counts.get(subset, 0))
        decision_count = int(subset_decision_counts.get(subset, 0))
        positive_count = int(subset_positive_counts.get(subset, 0))
        oracle_count = int(subset_oracle_counts.get(subset, 0))
        oracle_test_count = int(subset_oracle_test_counts.get(subset, 0))
        history_lengths = subset_history_lengths.get(subset, [])

        overall_summary["subsets"][subset] = {
            "tests": test_count,
            "decisions": decision_count,
            "decisions_per_test_mean": safe_rate(decision_count, test_count),
            "stop_positive_decisions": positive_count,
            "stop_positive_rate": safe_rate(positive_count, decision_count),
            "oracle_stop_decisions": oracle_count,
            "oracle_stop_rate_over_decisions": safe_rate(oracle_count, decision_count),
            "tests_with_oracle_stop": oracle_test_count,
            "oracle_stop_rate_over_tests": safe_rate(oracle_test_count, test_count),
            "mean_history_length_buckets": (
                float(np.mean(np.asarray(history_lengths, dtype=np.float64)))
                if history_lengths
                else None
            ),
        }

    overall_summary["feature_names"] = recorded_feature_names.tolist()
    overall_summary["feature_dim"] = int(recorded_feature_names.shape[0])
    overall_summary["max_sequence_buckets"] = max_sequence_buckets
    overall_summary["max_decisions_per_test"] = max_decisions_per_test
    overall_summary["best_iteration"] = best_iteration

    summary_path = output_root / "stage2_dataset_summary.json"
    summary_path.write_text(json.dumps(overall_summary, indent=2) + "\n")
    manifest_path = output_root / "manifest_stage2_transformer_dataset.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2) + "\n")

    print(f"wrote Stage 2 dataset summary to {summary_path}")
    print(f"wrote Stage 2 dataset manifest to {manifest_path}")


if __name__ == "__main__":
    main()
