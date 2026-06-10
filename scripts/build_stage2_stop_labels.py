#!/usr/bin/env python3
"""Build Stage 2 stop/continue labels from scored Stage 1 window predictions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:

    class tqdm:  # type: ignore[override]
        def __init__(self, iterable=None, total=None, **kwargs) -> None:
            self.iterable = iterable
            self.total = total

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
            "Build epsilon-specific Stage 2 stop/continue labels and oracle stop "
            "indices from scored Stage 1 predictions."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage1_predictions",
        help="Root directory containing scored Stage 1 prediction shards.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage2_labels",
        help="Directory for Stage 2 label shards and oracle stop summaries.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(DEFAULT_SUBSETS),
        help="Prediction subsets to process.",
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help="Optional glob relative to each subset directory for a probe run.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help=(
            "Accuracy tolerance used for the stop oracle. Combined with "
            "--error-kind to derive stop/continue labels."
        ),
    )
    parser.add_argument(
        "--error-kind",
        choices=("relative", "absolute"),
        default="relative",
        help=(
            "Threshold errors by relative error or absolute Mbps error. "
            "The natural default is relative error."
        ),
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
    return parser.parse_args()


def list_subset_paths(input_root: Path, subset: str, input_glob: str | None) -> list[Path]:
    subset_dir = input_root / subset
    if input_glob:
        paths = sorted(subset_dir.glob(input_glob))
    else:
        paths = sorted(subset_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(
            f"no Stage 1 prediction shards found for subset {subset} under {subset_dir}"
        )
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


def stop_mask_for_error_kind(
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


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


@dataclass
class OracleRecord:
    uuid: str
    test_time: str
    date: str
    speed_tier: str
    y_true_mbps: float
    last_observed_bucket: int
    num_windows: int = 0
    final_end_bucket: int = -1
    final_elapsed_ms: int = -1
    final_observed_buckets_seen: int = -1
    last_seen_end_bucket: int = -1
    last_window_safe: bool = False
    candidate_stop_found: bool = False
    candidate_stop_end_bucket: int = -1
    candidate_stop_elapsed_ms: int = -1
    candidate_stop_observed_buckets_seen: int = -1
    candidate_stop_abs_error_mbps: float = float("nan")
    candidate_stop_relative_error: float = float("nan")
    oracle_stop_found: bool = False
    oracle_stop_end_bucket: int = -1
    oracle_stop_elapsed_ms: int = -1
    oracle_stop_observed_buckets_seen: int = -1
    oracle_stop_abs_error_mbps: float = float("nan")
    oracle_stop_relative_error: float = float("nan")


def clear_candidate_stop(record: OracleRecord) -> None:
    record.candidate_stop_found = False
    record.candidate_stop_end_bucket = -1
    record.candidate_stop_elapsed_ms = -1
    record.candidate_stop_observed_buckets_seen = -1
    record.candidate_stop_abs_error_mbps = float("nan")
    record.candidate_stop_relative_error = float("nan")


def build_oracle_map(
    *,
    subset: str,
    paths: list[Path],
    epsilon: float,
    error_kind: str,
    relative_epsilon_unit: str,
    relative_denominator_floor: float,
) -> dict[tuple[str, str], OracleRecord]:
    oracle_map: dict[tuple[str, str], OracleRecord] = {}
    path_bar = tqdm(paths, desc=f"oracle {subset}", unit="shard", dynamic_ncols=True)
    for path in path_bar:
        path_bar.set_postfix_str(path.stem)
        with np.load(path, allow_pickle=False) as data:
            uuids = data["uuid"].tolist()
            test_times = data["test_time"].tolist()
            dates = data["date"].tolist()
            speed_tiers = data["speed_tier"].tolist()
            y_true = data["y_true_mbps"].astype(np.float32, copy=False)
            y_pred = data["y_pred_mbps"].astype(np.float32, copy=False)
            end_bucket = data["end_bucket"].astype(np.int16, copy=False)
            elapsed_ms = data["elapsed_ms"].astype(np.int32, copy=False)
            observed_buckets_seen = data["observed_buckets_seen"].astype(np.int16, copy=False)
            last_observed_bucket = data["last_observed_bucket"].astype(np.int16, copy=False)

            abs_error, relative_error = compute_errors(
                y_true=y_true,
                y_pred=y_pred,
                relative_denominator_floor=relative_denominator_floor,
            )
            stop_mask = stop_mask_for_error_kind(
                abs_error=abs_error,
                relative_error=relative_error,
                epsilon=epsilon,
                error_kind=error_kind,
                relative_epsilon_unit=relative_epsilon_unit,
            )

            for idx, (uuid, test_time) in enumerate(zip(uuids, test_times, strict=True)):
                key = (uuid, test_time)
                record = oracle_map.get(key)
                if record is None:
                    record = OracleRecord(
                        uuid=uuid,
                        test_time=test_time,
                        date=dates[idx],
                        speed_tier=speed_tiers[idx],
                        y_true_mbps=float(y_true[idx]),
                        last_observed_bucket=int(last_observed_bucket[idx]),
                    )
                    oracle_map[key] = record

                record.num_windows += 1

                current_end_bucket = int(end_bucket[idx])
                if current_end_bucket > record.final_end_bucket:
                    record.final_end_bucket = current_end_bucket
                    record.final_elapsed_ms = int(elapsed_ms[idx])
                    record.final_observed_buckets_seen = int(observed_buckets_seen[idx])

                if record.last_seen_end_bucket > current_end_bucket:
                    raise ValueError(
                        f"non-monotonic end_bucket order for test {(uuid, test_time)}: "
                        f"{current_end_bucket} after {record.last_seen_end_bucket}"
                    )
                record.last_seen_end_bucket = current_end_bucket

                if stop_mask[idx]:
                    record.last_window_safe = True
                    if not record.candidate_stop_found:
                        record.candidate_stop_found = True
                        record.candidate_stop_end_bucket = current_end_bucket
                        record.candidate_stop_elapsed_ms = int(elapsed_ms[idx])
                        record.candidate_stop_observed_buckets_seen = int(
                            observed_buckets_seen[idx]
                        )
                        record.candidate_stop_abs_error_mbps = float(abs_error[idx])
                        record.candidate_stop_relative_error = float(relative_error[idx])
                else:
                    record.last_window_safe = False
                    clear_candidate_stop(record)

    path_bar.close()

    for record in oracle_map.values():
        if record.last_window_safe and record.candidate_stop_found:
            record.oracle_stop_found = True
            record.oracle_stop_end_bucket = record.candidate_stop_end_bucket
            record.oracle_stop_elapsed_ms = record.candidate_stop_elapsed_ms
            record.oracle_stop_observed_buckets_seen = record.candidate_stop_observed_buckets_seen
            record.oracle_stop_abs_error_mbps = record.candidate_stop_abs_error_mbps
            record.oracle_stop_relative_error = record.candidate_stop_relative_error

    return oracle_map


def write_oracle_index_file(
    *,
    output_root: Path,
    subset: str,
    oracle_map: dict[tuple[str, str], OracleRecord],
    epsilon: float,
    error_kind: str,
    relative_epsilon_unit: str,
) -> Path:
    records = sorted(
        oracle_map.values(),
        key=lambda item: (item.date, item.test_time, item.uuid),
    )
    out_path = output_root / f"oracle_stop_indices_{subset}.npz"
    np.savez(
        out_path,
        uuid=np.array([record.uuid for record in records], dtype=np.str_),
        test_time=np.array([record.test_time for record in records], dtype=np.str_),
        date=np.array([record.date for record in records], dtype=np.str_),
        speed_tier=np.array([record.speed_tier for record in records], dtype=np.str_),
        y_true_mbps=np.array([record.y_true_mbps for record in records], dtype=np.float32),
        last_observed_bucket=np.array(
            [record.last_observed_bucket for record in records], dtype=np.int16
        ),
        num_windows=np.array([record.num_windows for record in records], dtype=np.int16),
        final_end_bucket=np.array([record.final_end_bucket for record in records], dtype=np.int16),
        final_elapsed_ms=np.array([record.final_elapsed_ms for record in records], dtype=np.int32),
        final_observed_buckets_seen=np.array(
            [record.final_observed_buckets_seen for record in records], dtype=np.int16
        ),
        oracle_stop_found=np.array(
            [record.oracle_stop_found for record in records], dtype=np.uint8
        ),
        oracle_stop_end_bucket=np.array(
            [record.oracle_stop_end_bucket for record in records], dtype=np.int16
        ),
        oracle_stop_elapsed_ms=np.array(
            [record.oracle_stop_elapsed_ms for record in records], dtype=np.int32
        ),
        oracle_stop_observed_buckets_seen=np.array(
            [record.oracle_stop_observed_buckets_seen for record in records], dtype=np.int16
        ),
        oracle_stop_abs_error_mbps=np.array(
            [record.oracle_stop_abs_error_mbps for record in records], dtype=np.float32
        ),
        oracle_stop_relative_error=np.array(
            [record.oracle_stop_relative_error for record in records], dtype=np.float32
        ),
        epsilon=np.array(epsilon, dtype=np.float32),
        error_kind=np.array(error_kind, dtype=np.str_),
        relative_epsilon_unit=np.array(relative_epsilon_unit, dtype=np.str_),
        effective_relative_epsilon_fraction=np.array(
            (
                epsilon / 100.0
                if error_kind == "relative" and relative_epsilon_unit == "percent"
                else (epsilon if error_kind == "relative" else np.nan)
            ),
            dtype=np.float32,
        ),
        oracle_definition=np.array("permanent_safe_suffix", dtype=np.str_),
        stop_label_definition=np.array("monotonic_suffix_from_oracle", dtype=np.str_),
    )
    return out_path


def summarize_oracle_map(
    oracle_map: dict[tuple[str, str], OracleRecord],
) -> dict[str, float | int | None]:
    records = list(oracle_map.values())
    found_records = [record for record in records if record.oracle_stop_found]
    return {
        "tests": len(records),
        "tests_with_oracle_stop": len(found_records),
        "oracle_stop_rate": (float(len(found_records) / len(records)) if records else 0.0),
        "mean_oracle_elapsed_ms": safe_mean(
            [record.oracle_stop_elapsed_ms for record in found_records]
        ),
        "median_oracle_elapsed_ms": safe_median(
            [record.oracle_stop_elapsed_ms for record in found_records]
        ),
        "mean_oracle_end_bucket": safe_mean(
            [record.oracle_stop_end_bucket for record in found_records]
        ),
        "median_oracle_end_bucket": safe_median(
            [record.oracle_stop_end_bucket for record in found_records]
        ),
        "mean_oracle_abs_error_mbps": safe_mean(
            [record.oracle_stop_abs_error_mbps for record in found_records]
        ),
        "mean_oracle_relative_error": safe_mean(
            [record.oracle_stop_relative_error for record in found_records]
        ),
    }


def write_label_shards(
    *,
    subset: str,
    paths: list[Path],
    output_root: Path,
    oracle_map: dict[tuple[str, str], OracleRecord],
    epsilon: float,
    error_kind: str,
    relative_epsilon_unit: str,
    relative_denominator_floor: float,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    subset_dir = output_root / subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    subset_windows = 0
    subset_positive_windows = 0
    subset_oracle_windows = 0
    manifest_entries: list[dict[str, object]] = []

    path_bar = tqdm(paths, desc=f"labels {subset}", unit="shard", dynamic_ncols=True)
    for path in path_bar:
        path_bar.set_postfix_str(path.stem)
        with np.load(path, allow_pickle=False) as data:
            uuids = data["uuid"].tolist()
            test_times = data["test_time"].tolist()
            y_true = data["y_true_mbps"].astype(np.float32, copy=False)
            y_pred = data["y_pred_mbps"].astype(np.float32, copy=False)
            end_bucket = data["end_bucket"].astype(np.int16, copy=False)

            abs_error, relative_error = compute_errors(
                y_true=y_true,
                y_pred=y_pred,
                relative_denominator_floor=relative_denominator_floor,
            )
            instantaneous_safe_mask = stop_mask_for_error_kind(
                abs_error=abs_error,
                relative_error=relative_error,
                epsilon=epsilon,
                error_kind=error_kind,
                relative_epsilon_unit=relative_epsilon_unit,
            )
            stop_label = np.zeros_like(instantaneous_safe_mask, dtype=np.uint8)
            continue_label = np.ones_like(instantaneous_safe_mask, dtype=np.uint8)

            oracle_stop_found = np.zeros_like(instantaneous_safe_mask, dtype=np.uint8)
            oracle_stop_end_bucket = np.full(instantaneous_safe_mask.shape, -1, dtype=np.int16)
            oracle_stop_elapsed_ms = np.full(instantaneous_safe_mask.shape, -1, dtype=np.int32)
            oracle_stop_observed_buckets_seen = np.full(
                instantaneous_safe_mask.shape, -1, dtype=np.int16
            )
            oracle_stop_abs_error_mbps = np.full(
                instantaneous_safe_mask.shape, np.nan, dtype=np.float32
            )
            oracle_stop_relative_error = np.full(
                instantaneous_safe_mask.shape, np.nan, dtype=np.float32
            )
            is_oracle_stop_window = np.zeros_like(instantaneous_safe_mask, dtype=np.uint8)

            for idx, (uuid, test_time) in enumerate(zip(uuids, test_times, strict=True)):
                record = oracle_map[(uuid, test_time)]
                if record.oracle_stop_found:
                    oracle_stop_found[idx] = 1
                    oracle_stop_end_bucket[idx] = record.oracle_stop_end_bucket
                    oracle_stop_elapsed_ms[idx] = record.oracle_stop_elapsed_ms
                    oracle_stop_observed_buckets_seen[idx] = (
                        record.oracle_stop_observed_buckets_seen
                    )
                    oracle_stop_abs_error_mbps[idx] = record.oracle_stop_abs_error_mbps
                    oracle_stop_relative_error[idx] = record.oracle_stop_relative_error
                    if int(end_bucket[idx]) >= record.oracle_stop_end_bucket:
                        stop_label[idx] = 1
                        continue_label[idx] = 0
                    if int(end_bucket[idx]) == record.oracle_stop_end_bucket:
                        is_oracle_stop_window[idx] = 1

            out_path = subset_dir / path.name
            np.savez(
                out_path,
                stop_label=stop_label,
                continue_label=continue_label,
                is_oracle_stop_window=is_oracle_stop_window,
                instantaneous_safe_window=instantaneous_safe_mask.astype(np.uint8),
                abs_error_mbps=abs_error,
                relative_error=relative_error,
                oracle_stop_found=oracle_stop_found,
                oracle_stop_end_bucket=oracle_stop_end_bucket,
                oracle_stop_elapsed_ms=oracle_stop_elapsed_ms,
                oracle_stop_observed_buckets_seen=oracle_stop_observed_buckets_seen,
                oracle_stop_abs_error_mbps=oracle_stop_abs_error_mbps,
                oracle_stop_relative_error=oracle_stop_relative_error,
                epsilon=np.array(epsilon, dtype=np.float32),
                error_kind=np.array(error_kind, dtype=np.str_),
                relative_epsilon_unit=np.array(relative_epsilon_unit, dtype=np.str_),
                effective_relative_epsilon_fraction=np.array(
                    (
                        epsilon / 100.0
                        if error_kind == "relative" and relative_epsilon_unit == "percent"
                        else (epsilon if error_kind == "relative" else np.nan)
                    ),
                    dtype=np.float32,
                ),
                oracle_definition=np.array("permanent_safe_suffix", dtype=np.str_),
                stop_label_definition=np.array("monotonic_suffix_from_oracle", dtype=np.str_),
                source_prediction_npz=np.array(str(path), dtype=np.str_),
                model_path=data["model_path"],
                best_iteration=data["best_iteration"],
            )

            subset_windows += int(stop_label.shape[0])
            subset_positive_windows += int(stop_label.sum())
            subset_oracle_windows += int(is_oracle_stop_window.sum())

            manifest_entries.append(
                {
                    "subset": subset,
                    "source_prediction_npz": str(path),
                    "label_npz": str(out_path),
                    "examples": int(stop_label.shape[0]),
                }
            )
            print(f"labeled {out_path}")

    path_bar.close()
    return (
        {
            "windows": subset_windows,
            "positive_windows": subset_positive_windows,
            "positive_window_rate": (
                float(subset_positive_windows / subset_windows) if subset_windows else 0.0
            ),
            "oracle_stop_windows": subset_oracle_windows,
        },
        manifest_entries,
    )


def main() -> None:
    args = parse_args()
    if args.epsilon < 0:
        raise SystemExit("--epsilon must be non-negative")
    if args.relative_denominator_floor <= 0:
        raise SystemExit("--relative-denominator-floor must be positive")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    effective_relative_epsilon_fraction = (
        (args.epsilon / 100.0)
        if args.error_kind == "relative" and args.relative_epsilon_unit == "percent"
        else (args.epsilon if args.error_kind == "relative" else None)
    )

    overall_summary = {
        "input_root": str(args.input_root),
        "output_root": str(output_root),
        "epsilon": args.epsilon,
        "error_kind": args.error_kind,
        "relative_epsilon_unit": args.relative_epsilon_unit,
        "effective_relative_epsilon_fraction": effective_relative_epsilon_fraction,
        "relative_denominator_floor": args.relative_denominator_floor,
        "oracle_definition": "permanent_safe_suffix",
        "stop_label_definition": "monotonic_suffix_from_oracle",
        "subsets": {},
    }
    manifest_entries: list[dict[str, object]] = []

    for subset in args.subsets:
        paths = list_subset_paths(args.input_root, subset, args.input_glob)
        oracle_map = build_oracle_map(
            subset=subset,
            paths=paths,
            epsilon=args.epsilon,
            error_kind=args.error_kind,
            relative_epsilon_unit=args.relative_epsilon_unit,
            relative_denominator_floor=args.relative_denominator_floor,
        )

        oracle_index_path = write_oracle_index_file(
            output_root=output_root,
            subset=subset,
            oracle_map=oracle_map,
            epsilon=args.epsilon,
            error_kind=args.error_kind,
            relative_epsilon_unit=args.relative_epsilon_unit,
        )
        oracle_summary = summarize_oracle_map(oracle_map)
        label_summary, subset_manifest = write_label_shards(
            subset=subset,
            paths=paths,
            output_root=output_root,
            oracle_map=oracle_map,
            epsilon=args.epsilon,
            error_kind=args.error_kind,
            relative_epsilon_unit=args.relative_epsilon_unit,
            relative_denominator_floor=args.relative_denominator_floor,
        )

        overall_summary["subsets"][subset] = {
            **oracle_summary,
            **label_summary,
            "oracle_index_npz": str(oracle_index_path),
        }
        manifest_entries.extend(subset_manifest)

    summary_path = output_root / "stage2_label_summary.json"
    summary_path.write_text(json.dumps(overall_summary, indent=2) + "\n")
    manifest_path = output_root / "manifest_stage2_labels.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2) + "\n")

    print(f"wrote Stage 2 label summary to {summary_path}")
    print(f"wrote Stage 2 label manifest to {manifest_path}")


if __name__ == "__main__":
    main()
