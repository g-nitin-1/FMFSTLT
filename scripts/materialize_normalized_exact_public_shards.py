#!/usr/bin/env python3
"""Apply train-only normalization statistics to exact-public raw shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_SPLITS = ("train", "test", "robustness")


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Normalize exact-public raw shards with train-only statistics."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "raw_shards",
        help="Root directory containing raw shard split subdirectories.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "train_stats.npz",
        help="Path to train-only normalization statistics.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "normalized_shards",
        help="Root directory for normalized shards.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits to normalize.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.stats_path, allow_pickle=False) as stats:
        mean = stats["mean"].astype(np.float32, copy=False)
        std = stats["std"].astype(np.float32, copy=False)
        stats_feature_names = stats["feature_names"]

    for split in args.splits:
        in_dir = args.input_root / split
        shard_paths = sorted(in_dir.glob("*.npz"))
        if not shard_paths:
            print(f"skip {split}: no shards under {in_dir}")
            continue

        out_dir = args.output_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for shard_path in shard_paths:
            with np.load(shard_path, allow_pickle=False) as data:
                x = data["x"].astype(np.float32, copy=False)
                feature_names = data["feature_names"]
                if not np.array_equal(feature_names, stats_feature_names):
                    raise ValueError(f"feature mismatch between {shard_path} and {args.stats_path}")

                x_norm = (x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
                out_path = out_dir / shard_path.name
                np.savez(
                    out_path,
                    x=x_norm.astype(np.float32, copy=False),
                    bucket_mask=data["bucket_mask"],
                    observed_bucket_count=data["observed_bucket_count"],
                    y_true_mbps=data["y_true_mbps"],
                    uuid=data["uuid"],
                    date=data["date"],
                    speed_tier=data["speed_tier"],
                    test_time=data["test_time"],
                    feature_names=feature_names,
                    fill_policy=data["fill_policy"],
                )
                print(f"normalized {out_path}")


if __name__ == "__main__":
    main()
