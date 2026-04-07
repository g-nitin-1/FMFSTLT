# Dataset Construction Summary

This workspace contains a paper-like reproduction of the TURBOTEST dataset pipeline using M-Lab BigQuery plus local CSV exports.

## Construction choices

- Region filter: `client.Geo.CountryCode = 'US'`
- Sampled dates: 12 deterministic monthly dates from `2024-04-23` through `2025-03-04`
- Split policy:
  - Train: first 10 sampled days, `800,000` metadata rows, balanced to `160,000` per speed tier
  - Test: first 10 sampled days, `40,000` metadata rows, natural distribution
  - Robustness: last 2 sampled days, `133,000` metadata rows, natural distribution
- Temporal preprocessing:
  - `tcpinfo` snapshots aggregated into `100 ms` buckets
  - only the first `10 s` after `raw.Metadata.StartTime`
  - bucket features are mean/std summaries over selected `TCPInfo` and `BBRInfo` fields

## Important implementation note

The initial `test_40k_meta` build was biased because it reused the same `FARM_FINGERPRINT(uuid)` ordering logic used for the train split. That selected only the `25-100` tier. The final `test_40k_meta` was rebuilt with a salted ordering key:

`FARM_FINGERPRINT(CONCAT(uuid, '|test_v2'))`

The exported test files and all counts in [`export_manifest.csv`](/mnt/e/FMFSTLT/export_manifest.csv) reflect the corrected split.

## Final coverage

- Train: `799,567 / 800,000` UUIDs matched to `tcpinfo` and produced `48,689,021` aggregated rows
- Test: `39,979 / 40,000` UUIDs matched to `tcpinfo` and produced `2,455,763` aggregated rows
- Robustness: `132,944 / 133,000` UUIDs matched to `tcpinfo` and produced `7,844,086` aggregated rows
- Overall: `972,490 / 973,000` UUIDs matched to `tcpinfo` and produced `58,988,870` aggregated rows

## Export format

- Local export root: [`exports`](/mnt/e/FMFSTLT/exports)
- File naming convention:
  - `train_<date>_tcpinfo_100ms_<tier>.csv`
  - `test_<date>_tcpinfo_100ms_<tier>.csv`
  - `robustness_<date>_tcpinfo_100ms_<tier>.csv`
- Each day is exported as 5 CSV files, one per speed tier:
  - `0_25`
  - `25_100`
  - `100_200`
  - `200_400`
  - `400_plus`

## Caveat for the report

This is a practical reproduction of the paper workflow, not a guarantee that the exported UUIDs exactly match the authors' original internal split. The pipeline follows the author clarification used during reproduction:

- about 12 sampled days total
- US-only filtering
- BigQuery-side preprocessing
- local export of compact processed outputs

Use [`export_manifest.csv`](/mnt/e/FMFSTLT/export_manifest.csv) as the authoritative summary of row counts, day coverage, and exported line counts.
