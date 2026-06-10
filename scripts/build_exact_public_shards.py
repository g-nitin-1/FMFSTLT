#!/usr/bin/env python3
"""Convert exact public feature CSV exports into per-file tensor shards."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MAX_BUCKETS = 100
FEATURE_COLUMNS = [
    "inst_throughput_mbps",
    "cumavg_throughput_mbps",
    "pipe_full_samples",
    "mean_rtt_us",
    "std_rtt_us",
    "mean_snd_cwnd",
    "std_snd_cwnd",
    "mean_bytes_in_flight",
    "std_bytes_in_flight",
    "mean_total_retrans",
    "std_total_retrans",
    "mean_dsack_dups",
    "std_dsack_dups",
]
DEFAULT_SPLITS = ("train", "test", "robustness")


@dataclass
class Example:
    uuid: str
    date: str
    speed_tier: str
    test_time: str
    y_true_mbps: float
    x: np.ndarray
    bucket_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Build raw exact-public tensor shards from exported CSVs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root_dir / "exports_exact_public",
        help="Root directory containing exported exact-public CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "raw_shards",
        help="Output directory for raw tensor shards.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits to process.",
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help="Optional glob evaluated relative to input root. Useful for probes.",
    )
    parser.add_argument(
        "--split",
        dest="forced_split",
        default=None,
        help="Override split inference when using --input-glob on loose files.",
    )
    parser.add_argument(
        "--fill-policy",
        choices=("zero", "forward_fill"),
        default="forward_fill",
        help="How to densify missing buckets before saving tensors.",
    )
    return parser.parse_args()


def safe_float(value: str, default: float = 0.0) -> float:
    if value is None:
        return default
    stripped = value.strip()
    if stripped == "":
        return default
    return float(stripped)


def forward_fill_in_place(x: np.ndarray, bucket_mask: np.ndarray) -> None:
    observed = np.flatnonzero(bucket_mask)
    if observed.size == 0:
        return

    first = int(observed[0])
    if first > 0:
        x[:first] = x[first]

    last = first
    for idx in observed[1:]:
        idx = int(idx)
        if idx > last + 1:
            x[last + 1 : idx] = x[last]
        last = idx

    if last < MAX_BUCKETS - 1:
        x[last + 1 :] = x[last]


def finalize_example(
    current_uuid: str | None,
    metadata: dict[str, str],
    x: np.ndarray,
    bucket_mask: np.ndarray,
    fill_policy: str,
) -> Example | None:
    if current_uuid is None:
        return None

    x_out = x.copy()
    if fill_policy == "forward_fill":
        forward_fill_in_place(x_out, bucket_mask)

    return Example(
        uuid=current_uuid,
        date=metadata["date"],
        speed_tier=metadata["speed_tier"],
        test_time=metadata["test_time"],
        y_true_mbps=float(metadata["y_true_mbps"]),
        x=x_out.astype(np.float32, copy=False),
        bucket_mask=bucket_mask.copy(),
    )


def infer_split(csv_path: Path, forced_split: str | None) -> str:
    if forced_split:
        return forced_split
    parent = csv_path.parent.name
    if parent in DEFAULT_SPLITS:
        return parent
    raise ValueError(f"could not infer split for {csv_path}; pass --split explicitly")


def iter_csv_files(input_root: Path, splits: Iterable[str], input_glob: str | None) -> list[Path]:
    if input_glob:
        return sorted(input_root.glob(input_glob))

    csvs: list[Path] = []
    for split in splits:
        csvs.extend(sorted((input_root / split).glob("*.csv")))
    return csvs


def process_csv(csv_path: Path, output_root: Path, split: str, fill_policy: str) -> dict:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"no rows found in {csv_path}")

    frame = frame.sort_values(["uuid", "bucket_100ms"], kind="stable")
    examples: list[Example] = []
    current_uuid: str | None = None
    current_metadata: dict[str, str] = {}
    current_x = np.zeros((MAX_BUCKETS, len(FEATURE_COLUMNS)), dtype=np.float32)
    current_mask = np.zeros((MAX_BUCKETS,), dtype=bool)
    total_rows = 0

    for row in frame.to_dict(orient="records"):
        total_rows += 1
        uuid = row["uuid"]
        if current_uuid is None:
            current_uuid = uuid
            current_metadata = {
                "date": row["date"],
                "speed_tier": row["speed_tier"],
                "test_time": row["test_time"],
                "y_true_mbps": row["y_true_mbps"],
            }
        elif uuid != current_uuid:
            example = finalize_example(
                current_uuid, current_metadata, current_x, current_mask, fill_policy
            )
            if example is not None:
                examples.append(example)

            current_uuid = uuid
            current_metadata = {
                "date": row["date"],
                "speed_tier": row["speed_tier"],
                "test_time": row["test_time"],
                "y_true_mbps": row["y_true_mbps"],
            }
            current_x = np.zeros((MAX_BUCKETS, len(FEATURE_COLUMNS)), dtype=np.float32)
            current_mask = np.zeros((MAX_BUCKETS,), dtype=bool)

        bucket = int(row["bucket_100ms"])
        if bucket < 0 or bucket >= MAX_BUCKETS:
            raise ValueError(f"bucket out of range in {csv_path}: {bucket}")

        feature_vector = np.array(
            [safe_float(str(row[col]), default=0.0) for col in FEATURE_COLUMNS],
            dtype=np.float32,
        )
        current_x[bucket] = feature_vector
        current_mask[bucket] = True

    example = finalize_example(current_uuid, current_metadata, current_x, current_mask, fill_policy)
    if example is not None:
        examples.append(example)

    if not examples:
        raise ValueError(f"no examples found in {csv_path}")

    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / f"{csv_path.stem}.npz"

    x = np.stack([ex.x for ex in examples], axis=0)
    bucket_mask = np.stack([ex.bucket_mask for ex in examples], axis=0)
    observed_bucket_count = bucket_mask.sum(axis=1).astype(np.int16, copy=False)
    y = np.array([ex.y_true_mbps for ex in examples], dtype=np.float32)
    uuids = np.array([ex.uuid for ex in examples], dtype=np.str_)
    dates = np.array([ex.date for ex in examples], dtype=np.str_)
    speed_tiers = np.array([ex.speed_tier for ex in examples], dtype=np.str_)
    test_times = np.array([ex.test_time for ex in examples], dtype=np.str_)

    np.savez(
        out_path,
        x=x,
        bucket_mask=bucket_mask,
        observed_bucket_count=observed_bucket_count,
        y_true_mbps=y,
        uuid=uuids,
        date=dates,
        speed_tier=speed_tiers,
        test_time=test_times,
        feature_names=np.array(FEATURE_COLUMNS, dtype=np.str_),
        fill_policy=np.array(fill_policy, dtype=np.str_),
    )

    return {
        "split": split,
        "source_csv": str(csv_path),
        "output_npz": str(out_path),
        "examples": int(x.shape[0]),
        "rows": int(total_rows),
        "tensor_shape": [int(v) for v in x.shape],
        "fill_policy": fill_policy,
    }


def main() -> None:
    args = parse_args()
    csv_files = iter_csv_files(args.input_root, args.splits, args.input_glob)
    if not csv_files:
        raise SystemExit("no input CSV files found")

    manifest: list[dict] = []
    for csv_path in csv_files:
        split = infer_split(csv_path, args.forced_split)
        entry = process_csv(csv_path, args.output_root, split, args.fill_policy)
        manifest.append(entry)
        print(f"built {entry['output_npz']} ({entry['examples']} tests, {entry['rows']} rows)")

    summary = {
        "feature_names": FEATURE_COLUMNS,
        "fill_policy": args.fill_policy,
        "files": manifest,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote manifest {manifest_path}")


if __name__ == "__main__":
    main()
