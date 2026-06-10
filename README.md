# FMFSTLT: Foundation Model for Speed-Test Learning Tasks

[![CI](https://github.com/g-nitin-1/FMFSTLT/actions/workflows/ci.yml/badge.svg)](https://github.com/g-nitin-1/FMFSTLT/actions/workflows/ci.yml)

A networking and machine-learning system that predicts final download throughput from
partial M-Lab NDT7 traces and decides when a speed test can terminate without waiting
for the full 10-second measurement.

The project reproduces the two-stage structure used by TurboTest and investigates
whether both stages can be implemented with one pretrained causal foundation model:

```text
100 ms TCP/BBR trace buckets
          |
          v
causal FMNet-v3 encoder
          |
          v
Stage 1: per-decision throughput + uncertainty
          |
          v
Stage 2: causal stop/continue policy
          |
          v
chosen stop time + reported Mbps
```

## Why This Project

Conventional Internet speed tests consume the full measurement duration even when the
throughput estimate has already stabilized. Early termination is difficult because a
system must optimize two competing objectives:

- report an accurate final-throughput estimate;
- stop as early as possible.

This repository covers the complete engineering path: public-data extraction,
100 ms TCP/BBR feature construction, reproducible dataset splits, XGBoost and
Transformer baselines, self-supervised pretraining, phased fine-tuning, threshold
selection, and time-shifted robustness evaluation.

## Final Architecture

FMNet-v3 processes all 100 buckets causally instead of compressing them into patches.
At decision buckets `4, 9, ..., 99` (every 500 ms), Stage 1 produces:

- `mu_d`: predicted `log(1 + final Mbps)`;
- `log_var_d`: learned prediction uncertainty;
- a causal hidden representation of the observed trace.

The Stage 2 policy consumes those Stage 1 outputs, elapsed time, observed-bucket count,
and the decision representation. It returns one stop probability per decision. Stage 1
is fully fine-tuned for throughput prediction; Stage 2 is then trained with Stage 1
frozen so policy gradients cannot corrupt the reported Mbps estimate.

See [Architecture](docs/architecture.md) for tensor shapes and training phases.

## Results

Evaluation uses held-out tests from the training date range and a separate
February-March 2025 robustness split.

### Throughput prediction

| Split | XGBoost prefix MAE | FMNet-v3 prefix MAE | FMNet-v3 final MAE |
|---|---:|---:|---:|
| Validation | 18.98 | 22.96 | 9.68 |
| Test | 17.72 | 21.06 | 8.56 |
| Robustness | 32.14 | 49.61 | 35.21 |

### Early termination at 10% error tolerance

| Split | System | Stop-policy F1 | Deployed within 10% | Mean savings |
|---|---|---:|---:|---:|
| Validation | TurboTest reproduction | 0.8696 | 0.6684 | 5,241 ms |
| Validation | FMNet-v3 two-stage | 0.8703 | 0.5867 | 5,087 ms |
| Test | TurboTest reproduction | 0.8608 | 0.6786 | 5,145 ms |
| Test | FMNet-v3 two-stage | 0.8604 | 0.5908 | 4,968 ms |
| Robustness | TurboTest reproduction | 0.8565 | 0.6608 | 5,245 ms |
| Robustness | FMNet-v3 two-stage | 0.8552 | 0.5785 | 5,073 ms |

The final model learns a competitive stopping policy, but it does not outperform the
specialized XGBoost prefix regressor. The main negative result is technically useful:
full-trace accuracy is not sufficient for early termination; prediction quality at
intermediate prefixes is the deployment bottleneck.

See [Experiment History](docs/experiments.md) for the v1, v2, and v3 progression.

## Repository Layout

```text
fmfstlt/models/          Importable FMNet-v3 and two-stage model implementations
scripts/                 Supported data, pretraining, training, and evaluation workflow
experimental_scripts/    TurboTest baseline reproduction and research-only analysis
sql/                     M-Lab BigQuery extraction and feature engineering
tests/                   Fast unit tests for causality, shapes, and gradient isolation
docs/                    Architecture, data pipeline, experiment history, and report
```

See [Main Workflow](scripts/README.md) for the supported path and
[Experimental Scripts](experimental_scripts/README.md) for baseline and ablation utilities.

Large datasets and checkpoints are intentionally excluded from Git. The public-data
pipeline rebuilds them from M-Lab BigQuery exports.

## Quick Start

Python 3.10+ is required.

```bash
git clone https://github.com/g-nitin-1/FMFSTLT.git
cd FMFSTLT
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Run a small architecture check without downloading the dataset:

```bash
python3 - <<'PY'
import torch
from fmfstlt.models import FMNetV3, FMNetV3Config

model = FMNetV3(FMNetV3Config(num_layers=2))
x = torch.randn(2, 100, 13)
decisions = torch.tensor([[4, 9, 14, 19], [4, 9, 14, 19]])
outputs = model(x, decisions)
print(outputs["throughput_mu"].shape)
PY
```

Expected output:

```text
torch.Size([2, 4])
```

## Reproduction Workflow

Build and export the M-Lab BigQuery tables as described in
[Data Pipeline](docs/data_pipeline.md), then run:

```bash
python -m scripts.build_exact_public_shards
python -m scripts.compute_exact_public_train_stats
python -m scripts.materialize_normalized_exact_public_shards
python -m scripts.make_stage1_uuid_split
```

Pretrain and fine-tune the final model:

```bash
fmfstlt-pretrain \
  --epochs 10 \
  --device cuda

fmfstlt-train \
  --pretrained-encoder artifacts_exact_public/foundation_v3_pretrain/fmnet_v3_pretrain.pt \
  --include-h-decision \
  --encoder-num-layers 10 \
  --phase-1-epochs 15 \
  --phase-1-cosine-lr \
  --phase-1-ema \
  --phase-2-epochs 5 \
  --device cuda
```

Evaluate the foundation pipeline against the reproduced baseline:

```bash
fmfstlt-evaluate \
  --foundation-checkpoint artifacts_exact_public/foundation_twostage_eps_10/phase2_checkpoint.pt \
  --baseline-checkpoint artifacts_exact_public/stage2_transformer_eps_10/stage2_transformer_model.pt \
  --input-root artifacts_exact_public/stage2_transformer_dataset_eps_10 \
  --device cuda
```

## Engineering Highlights

- Built a 59-million-row feature pipeline over public M-Lab NDT7 measurements.
- Engineered 13 time-series features from TCP and BBR telemetry at 100 ms resolution.
- Implemented causal attention and verified that future buckets cannot affect earlier
  representations.
- Used self-supervised next-bucket prediction before supervised throughput fine-tuning.
- Isolated Stage 1 and Stage 2 gradients to protect user-facing prediction quality.
- Evaluated temporal robustness on dates strictly later than the training period.
- Recorded negative results and ablations instead of reporting policy-only metrics as
  deployed accuracy.

## Documentation

- [Architecture](docs/architecture.md)
- [Data Pipeline](docs/data_pipeline.md)
- [Experiment History](docs/experiments.md)
- [Two-page report](docs/final_report.pdf)

## License

MIT
