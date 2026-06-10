# Data Pipeline

## Source

The dataset is reconstructed from the public M-Lab BigQuery table:

```text
measurement-lab.ndt_intermediate.extended_ndt7_downloads
```

The SQL under `sql/` selects US NDT7 download tests using BBR congestion control and
filters incomplete, erroneous, anomalous, early-exit, and non-production measurements.

## Sampling and Splits

The project uses 12 sampled dates spanning April 2024 through March 2025:

- 10 dates for train, validation, and test;
- 2 later dates for robustness evaluation.

Approximate test counts:

| Split | Tests |
|---|---:|
| Training source | 800,000 |
| Train UUIDs | 720,000 |
| Validation UUIDs | 80,000 |
| Test | 40,000 |
| Robustness | 133,000 |

The train/validation split is deterministic, UUID-level, and stratified by speed tier.
Test and robustness data are never used for normalization or pretraining.

## Features

Measurements are aggregated into 100 ms buckets for the first 10 seconds. Each bucket
contains 13 features:

1. instantaneous throughput;
2. cumulative average throughput;
3. BBR pipe-full proxy;
4. RTT mean;
5. RTT standard deviation;
6. congestion-window mean;
7. congestion-window standard deviation;
8. bytes-in-flight mean;
9. bytes-in-flight standard deviation;
10. retransmission mean;
11. retransmission standard deviation;
12. DSACK duplicate mean;
13. DSACK duplicate standard deviation.

Missing buckets are forward-filled while an independent bucket mask records which
observations were originally present. Normalization statistics are computed from the
training split only.

## BigQuery Build and Download

Prerequisites:

- an authenticated Google Cloud CLI with the `bq` command;
- a Google Cloud project with BigQuery Job User permission;
- permission to create tables in the destination dataset;
- awareness that the source scans are large and may incur BigQuery charges.

The five SQL files under `sql/` are all required and run in this order:

```text
sql/01_exact_ndt7_base_meta.sql
sql/02_exact_ndt7_splits.sql
sql/03a_exact_train_features.sql
sql/03b_exact_test_features.sql
sql/03c_exact_robustness_features.sql
```

Run them through the portable wrapper. It creates the destination dataset if needed and
replaces the repository's reference project identifier with your project and dataset:

```bash
export PROJECT_ID="your-google-cloud-project"
export DATASET="fmfstlt"
DRY_RUN=1 bash scripts/build_exact_public_bigquery.sh
bash scripts/build_exact_public_bigquery.sh
```

The dry run validates the queries and reports estimated bytes before execution. Set
`MAXIMUM_BYTES_BILLED` when running the build to make BigQuery reject any query above a
chosen scan limit. On June 10, 2026, the dry-run estimates were approximately 619 GB for
metadata, 483 GB each for train and test features, and 120 GB for robustness features.

The SQL creates BigQuery tables; it does not download local files. Download the generated
feature tables as restartable per-date, per-speed-tier CSV slices, then verify every local
file against its BigQuery row count:

```bash
export PROJECT_ID="your-google-cloud-project"
export DATASET="fmfstlt"
bash scripts/export_exact_public_features.sh
bash scripts/verify_exact_public_features.sh
```

For large exports, prefer `scripts/export_exact_public_tables_to_gcs.sh`. It extracts each
feature table once to a user-owned Google Cloud Storage bucket instead of issuing many
filtered local-download queries.

After the CSV export:

```bash
python -m scripts.build_exact_public_shards
python -m scripts.compute_exact_public_train_stats
python -m scripts.materialize_normalized_exact_public_shards
python -m scripts.make_stage1_uuid_split
```

Build and train the reproduced TurboTest baseline:

```bash
python -m experimental_scripts.build_stage1_regression_windows
python -m experimental_scripts.train_stage1_xgboost
python -m experimental_scripts.score_stage1_xgboost
python -m experimental_scripts.build_stage2_stop_labels --epsilon 10
python -m experimental_scripts.build_stage2_transformer_dataset --epsilon 10
python -m experimental_scripts.train_stage2_transformer
```

These baseline commands are retained for comparison and are not part of the primary
foundation-model workflow.

Generated CSV, NPZ, model, and log files are excluded from Git because the complete
local workspace exceeds 100 GB.

The queries were validated against the public M-Lab schema on June 10, 2026. M-Lab
documents `ndt_intermediate` as an unstable intermediate schema, so future schema changes
can require updating the `_internal202402` raw-field path.

## Reproducibility Note

The public-data reconstruction follows the paper's sampled-day methodology but cannot
guarantee identical UUIDs to the authors' internal split. The final comparison is
anchored at the validated 10% error-tolerance dataset. Other locally generated epsilon
datasets were excluded from final baseline claims after an XGBoost runtime mismatch was
found during materialization.
