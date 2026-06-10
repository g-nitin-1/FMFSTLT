#!/usr/bin/env python3
"""Create a deterministic UUID-level train/validation split for Stage 1."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Build a deterministic UUID-level train/validation split."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "normalized_shards" / "train",
        help="Directory containing normalized train shards.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage1_uuid_split.npz",
        help="Output NPZ containing UUID-level subset assignments.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage1_uuid_split_summary.json",
        help="Output JSON summary for quick inspection.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation fraction within each speed tier.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260416,
        help="Random seed used for deterministic tier-wise assignment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be between 0 and 1")

    shard_paths = sorted(args.input_root.glob("*.npz"))
    if not shard_paths:
        raise SystemExit(f"no normalized train shards found under {args.input_root}")

    uuid_to_tier: dict[str, str] = {}
    uuid_to_date: dict[str, str] = {}
    uuid_to_speed_tier: dict[str, str] = {}
    tier_to_uuids: dict[str, list[str]] = defaultdict(list)

    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as data:
            uuids = data["uuid"]
            speed_tiers = data["speed_tier"]
            dates = data["date"]
            for uuid, speed_tier, date in zip(
                uuids.tolist(),
                speed_tiers.tolist(),
                dates.tolist(),
                strict=True,
            ):
                if uuid in uuid_to_tier:
                    if uuid_to_tier[uuid] != speed_tier:
                        raise ValueError(
                            f"uuid {uuid} appears with conflicting tiers: "
                            f"{uuid_to_tier[uuid]} vs {speed_tier}"
                        )
                    continue
                uuid_to_tier[uuid] = speed_tier
                uuid_to_speed_tier[uuid] = speed_tier
                uuid_to_date[uuid] = date
                tier_to_uuids[speed_tier].append(uuid)

    rng = np.random.default_rng(args.seed)
    assignments: dict[str, str] = {}
    summary: dict[str, object] = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "total_uuids": len(uuid_to_tier),
        "tiers": {},
    }

    for tier, uuids in sorted(tier_to_uuids.items()):
        ordered = np.array(sorted(uuids), dtype=np.str_)
        permuted = ordered[rng.permutation(ordered.shape[0])]
        val_count = int(round(ordered.shape[0] * args.val_fraction))
        if val_count <= 0 and ordered.shape[0] > 1:
            val_count = 1
        val_set = set(permuted[:val_count].tolist())
        train_count = 0
        for uuid in ordered.tolist():
            subset = "val" if uuid in val_set else "train"
            assignments[uuid] = subset
            if subset == "train":
                train_count += 1

        summary["tiers"][tier] = {
            "total": int(ordered.shape[0]),
            "train": int(train_count),
            "val": int(val_count),
        }

    ordered_uuids = np.array(sorted(assignments), dtype=np.str_)
    ordered_subsets = np.array(
        [assignments[uuid] for uuid in ordered_uuids.tolist()], dtype=np.str_
    )
    ordered_tiers = np.array(
        [uuid_to_speed_tier[uuid] for uuid in ordered_uuids.tolist()], dtype=np.str_
    )
    ordered_dates = np.array([uuid_to_date[uuid] for uuid in ordered_uuids.tolist()], dtype=np.str_)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_path,
        uuid=ordered_uuids,
        subset=ordered_subsets,
        speed_tier=ordered_tiers,
        date=ordered_dates,
        seed=np.array(args.seed, dtype=np.int64),
        val_fraction=np.array(args.val_fraction, dtype=np.float64),
    )

    subset_counts = Counter(ordered_subsets.tolist())
    summary["subset_counts"] = {
        "train": int(subset_counts.get("train", 0)),
        "val": int(subset_counts.get("val", 0)),
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"wrote UUID split to {args.output_path}")
    print(f"wrote summary to {args.summary_path}")
    print(
        "subset counts: "
        f"train={summary['subset_counts']['train']} "
        f"val={summary['subset_counts']['val']}"
    )


if __name__ == "__main__":
    main()
