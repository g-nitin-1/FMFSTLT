#!/usr/bin/env python3
"""Materialize Stage 1 regression windows from normalized exact-public shards."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
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

        def update(self, n: int = 1) -> None:
            return None

        def close(self) -> None:
            return None


WINDOW_DEFAULT = 20
DEFAULT_SOURCE_SPLITS = ("train", "test", "robustness")


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Build Stage 1 regression windows from normalized shards."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "normalized_shards",
        help="Root directory containing normalized shard split subdirectories.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage1_uuid_split.npz",
        help="UUID-level train/validation split produced by make_stage1_uuid_split.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage1_windows",
        help="Output directory for Stage 1 window shards.",
    )
    parser.add_argument(
        "--source-splits",
        nargs="+",
        default=list(DEFAULT_SOURCE_SPLITS),
        help="Source normalized split directories to process.",
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help="Optional glob evaluated relative to input root. Useful for probes.",
    )
    parser.add_argument(
        "--window-size-buckets",
        type=int,
        default=WINDOW_DEFAULT,
        help="Number of 100 ms buckets in each regression window.",
    )
    parser.add_argument(
        "--stride-buckets",
        type=int,
        default=1,
        help="Stride between successive decision times.",
    )
    parser.add_argument(
        "--min-end-bucket",
        type=int,
        default=0,
        help="Smallest end bucket to materialize.",
    )
    parser.add_argument(
        "--end-bucket-source",
        choices=("observed", "all"),
        default="observed",
        help="Whether to create windows only at observed buckets or at every bucket.",
    )
    parser.add_argument(
        "--output-format",
        choices=("flat", "tensor"),
        default="flat",
        help="Store windows as flattened vectors or [T, F] tensors.",
    )
    parser.add_argument(
        "--max-examples-per-file",
        type=int,
        default=100000,
        help="Maximum number of window examples per output NPZ shard.",
    )
    return parser.parse_args()


@dataclass
class Buffer:
    x: list[np.ndarray] = field(default_factory=list)
    y_true_mbps: list[float] = field(default_factory=list)
    uuid: list[str] = field(default_factory=list)
    date: list[str] = field(default_factory=list)
    speed_tier: list[str] = field(default_factory=list)
    test_time: list[str] = field(default_factory=list)
    end_bucket: list[int] = field(default_factory=list)
    elapsed_ms: list[int] = field(default_factory=list)
    last_observed_bucket: list[int] = field(default_factory=list)
    observed_buckets_seen: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.uuid)

    def clear(self) -> None:
        self.x.clear()
        self.y_true_mbps.clear()
        self.uuid.clear()
        self.date.clear()
        self.speed_tier.clear()
        self.test_time.clear()
        self.end_bucket.clear()
        self.elapsed_ms.clear()
        self.last_observed_bucket.clear()
        self.observed_buckets_seen.clear()


def load_split_map(split_path: Path) -> dict[str, str]:
    with np.load(split_path, allow_pickle=False) as data:
        uuids = data["uuid"].tolist()
        subsets = data["subset"].tolist()
    return dict(zip(uuids, subsets, strict=True))


def iter_source_paths(
    input_root: Path, source_splits: Iterable[str], input_glob: str | None
) -> list[Path]:
    if input_glob:
        paths = sorted(input_root.glob(input_glob))
        if not paths:
            raise SystemExit(f"no normalized shard paths matched {input_glob!r}")
        return paths

    paths: list[Path] = []
    for split in source_splits:
        split_dir = input_root / split
        split_paths = sorted(split_dir.glob("*.npz"))
        if not split_paths:
            raise SystemExit(f"no normalized shards found under {split_dir}")
        paths.extend(split_paths)
    return paths


def make_window(x: np.ndarray, end_bucket: int, window_size: int) -> np.ndarray:
    start = max(0, end_bucket - window_size + 1)
    window = x[start : end_bucket + 1]
    if window.shape[0] < window_size:
        pad = np.repeat(window[-1:], window_size - window.shape[0], axis=0)
        window = np.concatenate([pad, window], axis=0)
    return window.astype(np.float32, copy=False)


def selected_end_buckets(
    bucket_mask: np.ndarray,
    end_bucket_source: str,
    min_end_bucket: int,
    stride_buckets: int,
) -> np.ndarray:
    observed = np.flatnonzero(bucket_mask)
    if observed.size == 0:
        return np.empty((0,), dtype=np.int16)

    min_end_bucket = max(0, min_end_bucket)
    last_observed = int(observed[-1])
    if min_end_bucket > last_observed:
        return np.empty((0,), dtype=np.int16)

    if end_bucket_source == "observed":
        selected = observed[observed >= min_end_bucket]
        if stride_buckets > 1:
            selected = selected[::stride_buckets]
        return selected.astype(np.int16, copy=False)

    return np.arange(min_end_bucket, last_observed + 1, stride_buckets, dtype=np.int16)


def append_example(
    buffer: Buffer,
    example_x: np.ndarray,
    *,
    y_true_mbps: float,
    uuid: str,
    date: str,
    speed_tier: str,
    test_time: str,
    end_bucket: int,
    observed_buckets_seen: int,
    last_observed_bucket: int,
) -> None:
    buffer.x.append(example_x)
    buffer.y_true_mbps.append(float(y_true_mbps))
    buffer.uuid.append(uuid)
    buffer.date.append(date)
    buffer.speed_tier.append(speed_tier)
    buffer.test_time.append(test_time)
    buffer.end_bucket.append(int(end_bucket))
    buffer.elapsed_ms.append(int((end_bucket + 1) * 100))
    buffer.observed_buckets_seen.append(int(observed_buckets_seen))
    buffer.last_observed_bucket.append(int(last_observed_bucket))


def flush_buffer(
    buffer: Buffer,
    *,
    output_root: Path,
    subset: str,
    source_split: str,
    source_shard_name: str,
    output_format: str,
    feature_names: np.ndarray,
    window_size_buckets: int,
    stride_buckets: int,
    end_bucket_source: str,
    counters: dict[tuple[str, str], int],
) -> dict[str, object] | None:
    if len(buffer) == 0:
        return None

    key = (subset, source_shard_name)
    part_index = counters[key]
    counters[key] += 1

    subset_dir = output_root / subset
    subset_dir.mkdir(parents=True, exist_ok=True)
    out_path = subset_dir / f"{source_shard_name}__part{part_index:04d}.npz"

    x = np.stack(buffer.x, axis=0).astype(np.float32, copy=False)
    if output_format == "flat":
        x_to_save = x.reshape(x.shape[0], -1)
    else:
        x_to_save = x

    np.savez(
        out_path,
        x=x_to_save,
        y_true_mbps=np.array(buffer.y_true_mbps, dtype=np.float32),
        uuid=np.array(buffer.uuid, dtype=np.str_),
        date=np.array(buffer.date, dtype=np.str_),
        speed_tier=np.array(buffer.speed_tier, dtype=np.str_),
        test_time=np.array(buffer.test_time, dtype=np.str_),
        end_bucket=np.array(buffer.end_bucket, dtype=np.int16),
        elapsed_ms=np.array(buffer.elapsed_ms, dtype=np.int32),
        observed_buckets_seen=np.array(buffer.observed_buckets_seen, dtype=np.int16),
        last_observed_bucket=np.array(buffer.last_observed_bucket, dtype=np.int16),
        feature_names=feature_names,
        output_format=np.array(output_format, dtype=np.str_),
        source_split=np.array(source_split, dtype=np.str_),
        source_shard=np.array(source_shard_name, dtype=np.str_),
        window_size_buckets=np.array(window_size_buckets, dtype=np.int16),
        stride_buckets=np.array(stride_buckets, dtype=np.int16),
        end_bucket_source=np.array(end_bucket_source, dtype=np.str_),
    )

    manifest_entry = {
        "subset": subset,
        "source_split": source_split,
        "source_shard": source_shard_name,
        "output_npz": str(out_path),
        "examples": int(x_to_save.shape[0]),
        "x_shape": [int(v) for v in x_to_save.shape],
    }
    buffer.clear()
    return manifest_entry


def main() -> None:
    args = parse_args()
    if args.window_size_buckets <= 0:
        raise SystemExit("--window-size-buckets must be positive")
    if args.stride_buckets <= 0:
        raise SystemExit("--stride-buckets must be positive")
    if args.max_examples_per_file <= 0:
        raise SystemExit("--max-examples-per-file must be positive")

    split_map = load_split_map(args.split_path)
    source_paths = iter_source_paths(args.input_root, args.source_splits, args.input_glob)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    buffers: dict[tuple[str, str, str], Buffer] = defaultdict(Buffer)
    counters: dict[tuple[str, str], int] = defaultdict(int)
    manifest_entries: list[dict[str, object]] = []

    shard_bar = tqdm(source_paths, desc="stage1 windows", unit="shard", dynamic_ncols=True)
    for shard_path in shard_bar:
        source_split = shard_path.parent.name
        source_shard_name = shard_path.stem
        shard_bar.set_postfix_str(f"{source_split}/{source_shard_name}")
        with np.load(shard_path, allow_pickle=False) as data:
            x = data["x"].astype(np.float32, copy=False)
            bucket_mask = data["bucket_mask"]
            y_true_mbps = data["y_true_mbps"].astype(np.float32, copy=False)
            uuids = data["uuid"].tolist()
            dates = data["date"].tolist()
            speed_tiers = data["speed_tier"].tolist()
            test_times = data["test_time"].tolist()
            feature_names = data["feature_names"]

            for idx in range(x.shape[0]):
                uuid = uuids[idx]
                if source_split == "train":
                    subset = split_map.get(uuid)
                    if subset is None:
                        raise KeyError(f"uuid {uuid} missing from {args.split_path}")
                else:
                    subset = source_split

                selected = selected_end_buckets(
                    bucket_mask=bucket_mask[idx],
                    end_bucket_source=args.end_bucket_source,
                    min_end_bucket=args.min_end_bucket,
                    stride_buckets=args.stride_buckets,
                )
                if selected.size == 0:
                    continue

                last_observed_bucket = int(np.flatnonzero(bucket_mask[idx])[-1])
                buffer_key = (subset, source_split, source_shard_name)
                buffer = buffers[buffer_key]

                for end_bucket in selected.tolist():
                    window = make_window(x[idx], end_bucket, args.window_size_buckets)
                    observed_buckets_seen = int(bucket_mask[idx, : end_bucket + 1].sum())
                    append_example(
                        buffer,
                        window,
                        y_true_mbps=float(y_true_mbps[idx]),
                        uuid=uuid,
                        date=dates[idx],
                        speed_tier=speed_tiers[idx],
                        test_time=test_times[idx],
                        end_bucket=int(end_bucket),
                        observed_buckets_seen=observed_buckets_seen,
                        last_observed_bucket=last_observed_bucket,
                    )

                    if len(buffer) >= args.max_examples_per_file:
                        manifest_entry = flush_buffer(
                            buffer,
                            output_root=output_root,
                            subset=subset,
                            source_split=source_split,
                            source_shard_name=source_shard_name,
                            output_format=args.output_format,
                            feature_names=feature_names,
                            window_size_buckets=args.window_size_buckets,
                            stride_buckets=args.stride_buckets,
                            end_bucket_source=args.end_bucket_source,
                            counters=counters,
                        )
                        if manifest_entry is not None:
                            manifest_entries.append(manifest_entry)
                            print(
                                f"built {manifest_entry['output_npz']} "
                                f"({manifest_entry['examples']} examples)"
                            )

    for (subset, source_split, source_shard_name), buffer in sorted(buffers.items()):
        with np.load(
            args.input_root / source_split / f"{source_shard_name}.npz", allow_pickle=False
        ) as data:
            feature_names = data["feature_names"]
        manifest_entry = flush_buffer(
            buffer,
            output_root=output_root,
            subset=subset,
            source_split=source_split,
            source_shard_name=source_shard_name,
            output_format=args.output_format,
            feature_names=feature_names,
            window_size_buckets=args.window_size_buckets,
            stride_buckets=args.stride_buckets,
            end_bucket_source=args.end_bucket_source,
            counters=counters,
        )
        if manifest_entry is not None:
            manifest_entries.append(manifest_entry)
            print(f"built {manifest_entry['output_npz']} ({manifest_entry['examples']} examples)")

    shard_bar.close()

    manifest_path = output_root / "manifest_stage1_windows.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2) + "\n")
    print(f"wrote manifest {manifest_path}")


if __name__ == "__main__":
    main()
