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

## Build Order

Run BigQuery SQL in this order:

```text
sql/01_exact_ndt7_base_meta.sql
sql/02_exact_ndt7_splits.sql
sql/03a_exact_train_features.sql
sql/03b_exact_test_features.sql
sql/03c_exact_robustness_features.sql
```

After exporting the generated tables:

```bash
python -m scripts.build_exact_public_shards
python -m scripts.compute_exact_public_train_stats
python -m scripts.materialize_normalized_exact_public_shards
python -m scripts.make_stage1_uuid_split
```

Build and train the reproduced TurboTest baseline:

```bash
python -m scripts.build_stage1_regression_windows
python -m scripts.train_stage1_xgboost
python -m scripts.score_stage1_xgboost
python -m scripts.build_stage2_stop_labels --epsilon 10
python -m scripts.build_stage2_transformer_dataset --epsilon 10
python -m scripts.train_stage2_transformer
```

Generated CSV, NPZ, model, and log files are excluded from Git because the complete
local workspace exceeds 100 GB.

## Reproducibility Note

The public-data reconstruction follows the paper's sampled-day methodology but cannot
guarantee identical UUIDs to the authors' internal split. The final comparison is
anchored at the validated 10% error-tolerance dataset. Other locally generated epsilon
datasets were excluded from final baseline claims after an XGBoost runtime mismatch was
found during materialization.
