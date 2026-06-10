# Main Workflow

This folder contains the supported end-to-end FMFSTLT workflow:

1. build and export the public M-Lab BigQuery dataset;
2. convert CSV exports into normalized tensor shards;
3. create the deterministic training/validation split;
4. pretrain FMNet-v3 with causal next-bucket prediction;
5. train the single two-stage foundation model;
6. evaluate it against stored baseline artifacts.

The model implementations are maintained as importable production code under
[`fmfstlt/models/`](../fmfstlt/models/). Baseline reproduction and research-only analysis
commands are isolated under [`experimental_scripts/`](../experimental_scripts/).

## Data Preparation

```bash
export PROJECT_ID="your-google-cloud-project"
DRY_RUN=1 bash scripts/build_exact_public_bigquery.sh
bash scripts/build_exact_public_bigquery.sh
bash scripts/export_exact_public_features.sh
bash scripts/verify_exact_public_features.sh

python -m scripts.build_exact_public_shards
python -m scripts.compute_exact_public_train_stats
python -m scripts.materialize_normalized_exact_public_shards
python -m scripts.make_stage1_uuid_split
```

## Model Training

```bash
fmfstlt-pretrain --epochs 10 --device cuda
fmfstlt-train --phase-1-epochs 15 --phase-2-epochs 5 --device cuda
```

## Evaluation

```bash
fmfstlt-evaluate --device cuda
```
