#!/usr/bin/env python3
"""Split full-table exact-public CSV exports into per-date, per-tier files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_SPLITS = ("train", "test", "robustness")


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Split full-table exact-public exports into per-date, per-tier CSVs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root_dir / "exports_exact_public_full",
        help="Directory containing full-table CSV exports.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root_dir / "exports_exact_public",
        help="Directory for per-date, per-tier CSV files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits to process.",
    )
    return parser.parse_args()


def tier_slug(tier: str) -> str:
    if tier == "400+":
        return "400_plus"
    return tier.replace("-", "_")


def split_one(input_csv: Path, output_root: Path, split: str) -> None:
    out_dir = output_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[Path, tuple[object, csv.DictWriter]] = {}
    try:
      with input_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
          raise ValueError(f"missing header in {input_csv}")

        for row in reader:
          date_slug = row["date"].replace("-", "_")
          file_tier = tier_slug(row["speed_tier"])
          out_path = out_dir / f"{split}_{date_slug}_features_100ms_{file_tier}.csv"

          if out_path not in writers:
            out_handle = out_path.open("w", newline="")
            writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            writers[out_path] = (out_handle, writer)

          _, writer = writers[out_path]
          writer.writerow(row)
    finally:
      for out_handle, _ in writers.values():
        out_handle.close()


def main() -> None:
    args = parse_args()
    for split in args.splits:
        input_csv = args.input_root / f"{split}_features_100ms_public_full.csv"
        if not input_csv.exists():
            raise SystemExit(f"missing full export for {split}: {input_csv}")
        split_one(input_csv, args.output_root, split)
        print(f"split {input_csv}")


if __name__ == "__main__":
    main()
