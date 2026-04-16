#!/usr/bin/env python3
"""Compute train-only normalization statistics for exact-public shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics from train exact-public shards."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "raw_shards" / "train",
        help="Directory containing raw train shard NPZ files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "train_stats.npz",
        help="Where to write the normalization statistics.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Minimum standard deviation clamp.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_paths = sorted(args.input_root.glob("*.npz"))
    if not shard_paths:
        raise SystemExit(f"no train shards found under {args.input_root}")

    feature_names: np.ndarray | None = None
    fill_policy: str | None = None
    total_count = 0
    total_tests = 0
    sum_x = None
    sumsq_x = None

    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as data:
            x = data["x"].astype(np.float64, copy=False)
            if feature_names is None:
                feature_names = data["feature_names"]
                fill_policy = str(data["fill_policy"])
                sum_x = np.zeros((x.shape[-1],), dtype=np.float64)
                sumsq_x = np.zeros((x.shape[-1],), dtype=np.float64)

            sum_x += x.sum(axis=(0, 1))
            sumsq_x += np.square(x, dtype=np.float64).sum(axis=(0, 1))
            total_count += int(x.shape[0] * x.shape[1])
            total_tests += int(x.shape[0])

    if feature_names is None or fill_policy is None or sum_x is None or sumsq_x is None:
        raise SystemExit("failed to accumulate train statistics")

    mean = sum_x / total_count
    var = np.maximum(sumsq_x / total_count - np.square(mean), 0.0)
    std = np.sqrt(var)
    std = np.where(std < args.epsilon, 1.0, std)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_path,
        feature_names=feature_names,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        total_bucket_count=np.array(total_count, dtype=np.int64),
        total_test_count=np.array(total_tests, dtype=np.int64),
        fill_policy=np.array(fill_policy, dtype=np.str_),
    )
    print(f"wrote train statistics to {args.output_path}")


if __name__ == "__main__":
    main()
