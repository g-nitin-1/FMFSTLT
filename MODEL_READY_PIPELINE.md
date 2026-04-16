# Model-Ready Pipeline

This pipeline starts from the exact public-data feature exports in [`exports_exact_public/`](/mnt/e/fmfstlt/exports_exact_public), not the older intermediate [`exports/`](/mnt/e/fmfstlt/exports).

## Inputs

The exact public-data feature tables exported from BigQuery contain these tensor channels per `100 ms` bucket:

1. `inst_throughput_mbps`
2. `cumavg_throughput_mbps`
3. `pipe_full_samples`
4. `mean_rtt_us`
5. `std_rtt_us`
6. `mean_snd_cwnd`
7. `std_snd_cwnd`
8. `mean_bytes_in_flight`
9. `std_bytes_in_flight`
10. `mean_total_retrans`
11. `std_total_retrans`
12. `mean_dsack_dups`
13. `std_dsack_dups`

`pipe_full_samples` is now derived from public `BBRInfo.BW` using a BBR-style startup proxy:

- substantial growth reset at `1.25x`
- pipe-full reached after `3` consecutive non-growth signals
- per-bucket value is the count of such signals inside the `100 ms` window

## Step 1: Build Raw Shards

Convert the exported CSV files into shard-level NPZ files:

```bash
python3 /mnt/e/fmfstlt/scripts/build_exact_public_shards.py
```

Default output:

- [`artifacts_exact_public/raw_shards/`](/mnt/e/fmfstlt/artifacts_exact_public/raw_shards)

Each shard contains:

- `x`: dense tensor, shape `[N, 100, 13]`
- `bucket_mask`: original observed buckets before densification
- `observed_bucket_count`
- `y_true_mbps`
- `uuid`
- `date`
- `speed_tier`
- `test_time`

Default fill policy is `forward_fill`, which keeps the tensor dense for model input while preserving the original observation mask.

## Step 2: Compute Train-Only Normalization

```bash
python3 /mnt/e/fmfstlt/scripts/compute_exact_public_train_stats.py
```

Default output:

- [`artifacts_exact_public/train_stats.npz`](/mnt/e/fmfstlt/artifacts_exact_public/train_stats.npz)

This computes train-only mean and standard deviation over the dense train tensors.

## Step 3: Materialize Normalized Shards

```bash
python3 /mnt/e/fmfstlt/scripts/materialize_normalized_exact_public_shards.py
```

Default output:

- [`artifacts_exact_public/normalized_shards/`](/mnt/e/fmfstlt/artifacts_exact_public/normalized_shards)

These normalized shards are the correct base for model training.

## Step 4: Freeze A UUID-Level Validation Split

Stage 1 and Stage 2 should use a UUID-level validation split carved only from the train split.

```bash
python3 /mnt/e/fmfstlt/scripts/make_stage1_uuid_split.py
```

Default outputs:

- [`artifacts_exact_public/stage1_uuid_split.npz`](/mnt/e/fmfstlt/artifacts_exact_public/stage1_uuid_split.npz)
- [`artifacts_exact_public/stage1_uuid_split_summary.json`](/mnt/e/fmfstlt/artifacts_exact_public/stage1_uuid_split_summary.json)

This split is deterministic, stratified by `speed_tier`, and keeps `test` and `robustness` untouched.

## Step 5: Build Stage 1 Regression Windows

The Stage 1 regressor consumes the most recent `2 s` of the normalized sequence. The script below materializes chunked window shards from the normalized split tensors.

```bash
python3 /mnt/e/fmfstlt/scripts/build_stage1_regression_windows.py
```

Default outputs:

- [`artifacts_exact_public/stage1_windows/`](/mnt/e/fmfstlt/artifacts_exact_public/stage1_windows)

Default behavior:

- window length: `20` buckets (`2 s`)
- stride: `1` bucket (`100 ms`)
- end-bucket source: `observed`
- output format: flattened vectors for XGBoost
- train UUIDs are routed into `train/` or `val/` using the frozen split file

Each Stage 1 output shard contains:

- `x`: `[N, 260]` by default (`20 x 13` flattened)
- `y_true_mbps`
- `uuid`
- `date`
- `speed_tier`
- `test_time`
- `end_bucket`
- `elapsed_ms`
- `observed_buckets_seen`
- `last_observed_bucket`

Useful variants:

```bash
python3 /mnt/e/fmfstlt/scripts/build_stage1_regression_windows.py --end-bucket-source all
```

This builds windows at every bucket up to the last observed bucket instead of only at observed buckets.

```bash
python3 /mnt/e/fmfstlt/scripts/build_stage1_regression_windows.py --output-format tensor
```

This stores each window as `[20, 13]` instead of flattening for XGBoost.

```bash
python3 /mnt/e/fmfstlt/scripts/build_stage1_regression_windows.py --input-glob 'train/train_2024_04_23_features_100ms_0_25.npz'
```

This is useful for probes before running the full materialization.

## What This Does Not Yet Build

This stage intentionally stops at normalized split tensors. It does not yet materialize:

- Stage 2 stop/continue labels
- epsilon-specific oracle stopping indices

Stage 1 windows are now covered by Step 5. Stage 2 labels and end-to-end evaluation artifacts still need to be built on top of the trained regressor outputs.

## Step 6: Train The Stage 1 XGBoost Regressor

Use external-memory quantile matrices so training does not require loading all Stage 1 windows into RAM at once.

```bash
python3 /mnt/e/fmfstlt/scripts/train_stage1_xgboost.py
```

For IITD HPC, `cd` into the repo on the cluster and submit the provided PBS helper after rebuilding the corrected Stage 1 windows on the cluster:

```bash
qsub scripts/train_stage1_xgboost_iitd.pbs
```

The HPC helper defaults to:

- queue: `standard`
- resources: `1` CPU node, `32` cores, `48` hours
- input root: `${SCRATCH}/fmfstlt_stage1_windows`
- output root: `${SCRATCH}/fmfstlt_stage1_xgboost`

If you have not rebuilt Stage 1 windows on HPC after the padding fix, submit this first:

```bash
qsub scripts/build_stage1_windows_iitd.pbs
```

Default outputs:

- [`artifacts_exact_public/stage1_xgboost_full_windows/stage1_xgboost_model.json`](/mnt/e/fmfstlt/artifacts_exact_public/stage1_xgboost_full_windows/stage1_xgboost_model.json)
- [`artifacts_exact_public/stage1_xgboost_full_windows/training_summary.json`](/mnt/e/fmfstlt/artifacts_exact_public/stage1_xgboost_full_windows/training_summary.json)

Defaults:

- train subset: `stage1_windows/train`
- eval subset: `stage1_windows/val`
- tree method: `hist`
- early stopping on validation

Useful probe:

```bash
python3 /mnt/e/fmfstlt/scripts/train_stage1_xgboost.py --input-root /mnt/e/fmfstlt/artifacts_exact_public/stage1_windows_probe --output-root /mnt/e/fmfstlt/artifacts_exact_public/stage1_xgboost_probe
```

## Step 7: Score Stage 1 Windows

Generate per-window throughput predictions and summarize regression error over `train`, `val`, `test`, and `robustness`.

```bash
python3 /mnt/e/fmfstlt/scripts/score_stage1_xgboost.py
```

Default outputs:

- prediction shards under [`artifacts_exact_public/stage1_predictions/`](/mnt/e/fmfstlt/artifacts_exact_public/stage1_predictions)
- summary metrics at [`artifacts_exact_public/stage1_predictions/metrics_summary.json`](/mnt/e/fmfstlt/artifacts_exact_public/stage1_predictions/metrics_summary.json)

The scorer saves:

- `y_pred_mbps`
- `y_true_mbps`
- `uuid`
- `speed_tier`
- `end_bucket`
- `elapsed_ms`

These scored windows are the correct input for Stage 2 oracle label construction.

## Step 8: Build Stage 2 Stop/Continue Labels

Generate epsilon-specific stop/continue labels from the scored Stage 1 windows, along with per-test oracle stopping indices.

```bash
python3 /mnt/e/fmfstlt/scripts/build_stage2_stop_labels.py --input-root /mnt/e/fmfstlt/artifacts_exact_public/stage1_predictions_full_windows --output-root /mnt/e/fmfstlt/artifacts_exact_public/stage2_labels_full_windows_eps_10 --epsilon 10
```

Default outputs:

- per-window Stage 2 label shards under `stage2_labels/.../<subset>/*.npz`
- per-test oracle stop tables like `stage2_labels/oracle_stop_indices_train.npz`
- summary metrics at `stage2_labels/stage2_label_summary.json`

Each label shard stores:

- `stop_label`
- `continue_label`
- `is_oracle_stop_window`
- `instantaneous_safe_window`
- `abs_error_mbps`
- `relative_error`
- repeated oracle stop metadata for the corresponding test

The label builder uses a monotonic-safe suffix oracle:

- first, it finds the earliest decision time `t*` such that all later windows for that test remain within `epsilon`
- then it writes `stop_label = 1` for all windows at `t >= t*`
- `is_oracle_stop_window` marks only the single oracle entry point at `t*`
- `instantaneous_safe_window` keeps the per-window instantaneous `error <= epsilon` indicator for debugging and analysis

For `--error-kind relative`, the default is paper-style percentage units:

- `--epsilon 5` means 5% relative error
- `--epsilon 10` means 10% relative error

This aligns with paper-style evaluation sweeps like `epsilon in {5, 10, 15, 20, 25, 30, 35}`.

## Step 9: Materialize The Stage 2 Transformer Dataset

Materialize a paper-faithful Stage 2 dataset directly from the normalized full-test shards and the trained Stage 1 regressor.

```bash
python3 /mnt/e/fmfstlt/scripts/build_stage2_transformer_dataset.py --epsilon 10 --output-root /mnt/e/fmfstlt/artifacts_exact_public/stage2_transformer_dataset_eps_10
```

Default outputs:

- tensor-ready shards under `stage2_transformer_dataset/.../<subset>/*.npz`
- summary metrics at `stage2_transformer_dataset/stage2_dataset_summary.json`

This step does not join the earlier `stage1_predictions_full_windows` or `stage2_labels_full_windows_eps_*` artifacts. Instead it:

- reads the normalized dense test histories from `normalized_shards`
- rescoring only the paper decision points at `500 ms` stride by default
- applies the corrected permanent-safe suffix oracle on those decision points
- writes one example per test with the full history plus per-decision tables

Each Stage 2 dataset shard stores:

- `x_full` as `[N, 100, 13]`
- `bucket_mask`
- `decision_valid_mask`
- `decision_end_bucket`
- `decision_elapsed_ms`
- `decision_observed_buckets_seen`
- `y_pred_mbps`
- `abs_error_mbps`
- `relative_error`
- `instantaneous_safe_window`
- `stop_label`
- `continue_label`
- `is_oracle_stop_window`
- repeated oracle metadata per test

## Step 10: Train The Stage 2 Transformer

Train one stop-policy model per epsilon-specific Stage 2 dataset.

```bash
python3 /mnt/e/fmfstlt/scripts/train_stage2_transformer.py --input-root /mnt/e/fmfstlt/artifacts_exact_public/stage2_transformer_dataset_eps_10 --output-root /mnt/e/fmfstlt/artifacts_exact_public/stage2_transformer_eps_10
```

Default outputs:

- `stage2_transformer_model.pt`
- `training_summary.json`

The trainer:

- consumes the full-history Stage 2 dataset shards
- trains a masked Transformer encoder on `stop_label` by default
- tunes the decision threshold on `val`
- evaluates both window-level classification quality and per-test early-exit policy metrics on `val`, `test`, and `robustness`
- uses paper-aligned defaults for the main architecture and optimizer:
  - `batch_size = 4096`
  - `learning_rate = 1e-3`
  - `num_heads = 8`
  - `num_layers = 8`
  - optimizer: `Adam`
- reports `within_epsilon_rate` from the realized per-decision error signal, not from the training label alone

Useful probe:

```bash
python3 /mnt/e/fmfstlt/scripts/train_stage2_transformer.py --input-root /mnt/e/fmfstlt/artifacts_exact_public/stage2_transformer_dataset_eps_10 --output-root /mnt/e/fmfstlt/artifacts_exact_public/stage2_transformer_eps_10_probe --max-train-shards 2 --max-eval-shards 1 --max-train-batches-per-epoch 10 --epochs 1
```

If `torch` complains about not finding a usable temp directory in a Linux or WSL shell, set `TMPDIR` before running. The trainer will also try to select a writable temp directory automatically if `TMPDIR` is unset:

```bash
env TMPDIR=/dev/shm python3 /mnt/e/fmfstlt/scripts/train_stage2_transformer.py ...
```
