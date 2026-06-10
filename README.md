# FMFSTLT: Foundation Model for Speed-Test Learning Tasks

[![CI](https://github.com/g-nitin-1/FMFSTLT/actions/workflows/ci.yml/badge.svg)](https://github.com/g-nitin-1/FMFSTLT/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

FMFSTLT is an end-to-end networking and machine-learning project for predicting final
download throughput from a partial M-Lab NDT7 trace and deciding when the speed test can
terminate safely.

The research question is:

> Can a self-supervised causal trace model replace TurboTest's separate XGBoost
> throughput regressor and Transformer stop classifier with one deployable neural
> checkpoint?

The final FMNet-v3 system follows the same Stage 1 then Stage 2 ordering:

```text
NDT7 TCP/BBR telemetry at 100 ms resolution
                    |
                    v
        causal FMNet-v3 trace encoder
                    |
                    v
 Stage 1: throughput estimate at each decision
                    |
                    v
 Stage 2: causal stop/continue policy
                    |
                    v
       selected stop time + reported Mbps
```

The final result is mixed but informative. FMNet-v3 learns a stop policy with nearly the
same classification F1 as the reproduced specialized baseline, but its own early-prefix
throughput estimates remain less accurate. On the held-out test split, the self-contained
model reports a value within 10% of the final throughput on **59.08%** of tests while
saving **4.97 seconds** on average.

## Project Status

The academic experiment is complete. This repository contains:

- the supported FMNet-v3 data, pretraining, fine-tuning, and evaluation workflow;
- the frozen TurboTest-style baseline reproduction at `epsilon = 10%`;
- v1, v2, and v3 experiment history and negative-result analysis;
- SQL for rebuilding the public M-Lab dataset;
- tests for causality, tensor shapes, and Stage 1/Stage 2 gradient isolation;
- the final ACM two-page report.

Large BigQuery exports, tensor shards, and model checkpoints are excluded from Git. A
complete local experimental workspace exceeds 100 GB.

## Why Early Termination Is Difficult

A normal NDT7 download test can run for roughly 10 seconds. Stopping early is useful only
if the reported throughput remains accurate. The system therefore has two coupled goals:

1. predict the final throughput from incomplete TCP/BBR telemetry;
2. stop as early as possible without exceeding the allowed error.

These objectives must be evaluated together. A stop classifier can have strong F1 while
the deployed system still reports an inaccurate Mbps value. For that reason, the primary
deployment metric in this repository uses the model's **own throughput prediction at its
own selected stop point**.

## Data

### Source and scale

The dataset is reconstructed from the public BigQuery table:

```text
measurement-lab.ndt_intermediate.extended_ndt7_downloads
```

It follows the paper's sampled-day scale: 12 dates spanning April 2024 through March
2025, rather than every day in that interval.

| Purpose | Dates | Approximate tests |
|---|---:|---:|
| Training source | First 10 sampled dates | 800,000 |
| Train UUIDs | Stratified subset of training source | 720,000 |
| Validation UUIDs | Stratified subset of training source | 80,000 |
| Test | Same first 10 dates, disjoint tests | 40,000 |
| Robustness | February 6 and March 4, 2025 | 133,000 |

The resulting feature pipeline contains approximately 59 million 100 ms rows. Test and
robustness measurements are excluded from normalization and self-supervised pretraining.

### Input representation

Each speed test is represented as:

```text
x_full:      float32 [100, 13]
bucket_mask: boolean [100]
```

The 100 rows cover the first 10 seconds in 100 ms buckets. Missing buckets are
forward-filled, while `bucket_mask` preserves whether a bucket was actually observed.
Normalization statistics are computed from training UUIDs only.

The 13 features are:

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

The SQL filters to US NDT7 download measurements using BBR and removes incomplete,
erroneous, anomalous, early-exit, and non-production tests. See
[Data Pipeline](docs/data_pipeline.md) for query order, export options, and reproducibility
constraints.

## Baseline Reproduction

The comparison baseline preserves TurboTest's two-model structure.

### Baseline Stage 1

At every 100 ms position with enough history, the latest 20 buckets are flattened:

```text
[20 buckets, 13 features] -> [260 features] -> XGBoost -> predicted final Mbps
```

The window is 2 seconds long and advances with stride 1 during Stage 1 materialization.
The reproduced XGBoost configuration uses depth 7, learning rate 0.03, and 1,500 rounds.

### Baseline Stage 2

Stage 2 considers 20 decisions at 500 ms intervals. Decision `d` ends at bucket:

```text
4, 9, 14, ..., 99
```

For each decision, the Stage 1 estimate is compared with the true final throughput. The
permanent-safe suffix oracle marks the earliest decision after which all remaining
predictions stay within `epsilon`. An 8-layer Transformer learns the resulting
stop/continue labels.

The frozen comparison point is:

```text
epsilon = 10%
validation-selected threshold = 0.25
best epoch = 3
```

Other epsilon-specific baseline datasets were excluded from final claims after an
XGBoost runtime mismatch was discovered in their materialization. The final comparison
therefore uses the validated `epsilon = 10%` baseline.

## FMNet-v3 Architecture

FMNet-v3 is one deployable checkpoint containing:

- one causal bucket-level encoder;
- Stage 1 throughput heads;
- one smaller causal Stage 2 policy Transformer.

It is a single neural system, but not literally the same Transformer block executed
twice. Stage 1 and Stage 2 are separate modules stored and deployed together. No XGBoost
prediction is required by the foundation path at inference time.

### Selected encoder

| Component | Selected value |
|---|---:|
| Input | `[batch, 100, 13]` |
| Bucket projection | `13 -> 256` |
| Position embeddings | 100 learned positions |
| Attention | Strictly causal |
| Encoder width | 256 |
| Attention heads | 8 |
| Encoder layers | 10 |
| Feed-forward width | 1,024 |
| Dropout | 0.15 |

The public class defaults to 8 encoder layers for lighter experiments. The selected
final run uses 10. Its pretraining checkpoint used 8 layers; shape-compatible pretrained
weights were loaded into the deeper model, and the additional layers were initialized
for supervised fine-tuning.

### Self-supervised pretraining

Pretraining uses only the 720,000 training UUIDs. Given buckets `0..t`, the model predicts
normalized bucket `t+1`:

```text
hidden[t] = causal_encoder(x[0:t])
next_bucket_prediction[t] = prediction_head(hidden[t])
```

A masked Smooth-L1 loss is computed only when the target bucket is valid:

```text
L_pretrain = SmoothL1(next_bucket_prediction[t], x[t + 1])
```

The selected pretraining run used 10 epochs, batch size 256, AdamW, learning rate
`3e-4`, weight decay `0.02`, and 5% post-normalization feature-vector dropout. Its best
average loss was `0.05448` at epoch 8.

### Stage 1: per-decision throughput

The encoder state is gathered at the 20 decision buckets:

```text
decision_hidden: [batch, 20, 256]
stage1_mu:       [batch, 20]
stage1_logvar:   [batch, 20]
```

`stage1_mu[d]` predicts:

```text
log(1 + final_throughput_mbps)
```

The reported Mbps value is recovered with:

```text
predicted_mbps[d] = exp(stage1_mu[d]) - 1
```

The selected run fully fine-tunes the encoder and throughput-mean head using:

```text
L_stage1 = 2 * MSE(prefix_mu, log1p(y_true))
         + 1 * MSE(final_mu,  log1p(y_true))
```

It uses AdamW, cosine learning-rate decay, and exponential moving average weights with
decay `0.999`. The best Stage 1 checkpoint was selected by validation prefix MAE at epoch
13 of 15.

`stage1_logvar` is present in the architecture and is passed to Stage 2, but the selected
run does **not** train that head with Gaussian NLL or another direct uncertainty loss.
Its parameters therefore remain an unsupervised auxiliary projection of the changing
encoder state, not a calibrated probabilistic confidence interval.

### Stage 2: stop policy

The strict architecture accepts four values per decision:

```text
[
  stage1_mu[d],
  stage1_logvar[d],
  elapsed_ms[d] / 10000,
  observed_bucket_count[d] / 100
]
```

The selected richer run also includes `decision_hidden[d]`, producing 260 input values
per decision. A 4-layer causal policy Transformer maps the 20-decision sequence to:

```text
stop_logit:       [batch, 20]
stop_probability: sigmoid(stop_logit)
```

Stage 2 is trained with masked binary cross-entropy against the permanent-safe suffix
label:

```text
L_stage2 = BCEWithLogits(stop_logit[d], stop_label[d])
```

These labels are inherited from the frozen epsilon-10 Stage 2 dataset and were generated
from the reproduced XGBoost prediction errors. During Phase 2, however, the policy input
contains FMNet-v3's Stage 1 outputs, not XGBoost outputs. At deployment, neither the
labels nor XGBoost are needed.

The validation-selected threshold is `0.45`. At inference, the first valid decision at
or above the threshold is selected; otherwise the last valid decision is used. The
system reports the Stage 1 Mbps estimate from that same decision.

### Why training is phased

The selected checkpoint uses:

1. **Pretraining:** learn causal TCP/BBR trace dynamics without throughput labels.
2. **Phase 1:** fully fine-tune the encoder and throughput heads.
3. **Phase 2:** freeze Stage 1, detach its outputs, and train only the stop policy.

An optional Phase 3 end-to-end run was tested but not selected. Allowing stop-policy
gradients to modify the encoder slightly improved some policy behavior while increasing
Stage 1 error by roughly 3-4x. The final deployed checkpoint is therefore the Phase 2
checkpoint.

## Results

All results below use held-out tests. The robustness split contains only the two later
2025 dates.

### Stage 1 throughput accuracy

| Split | XGBoost prefix MAE | FMNet-v3 prefix MAE | FMNet-v3 final-bucket MAE |
|---|---:|---:|---:|
| Validation | 18.98 Mbps | 22.96 Mbps | 9.68 Mbps |
| Test | 17.72 Mbps | 21.06 Mbps | 8.56 Mbps |
| Robustness | 32.14 Mbps | 49.61 Mbps | 35.21 Mbps |

`Prefix MAE` aggregates predictions across valid intermediate decision points.
`Final-bucket MAE` evaluates FMNet-v3 after the complete trace. There is no separate
XGBoost final-bucket column in the recorded comparison; the XGBoost values above are its
per-decision prefix metrics.

The model is strong after observing the full trace but is weaker at the intermediate
prefixes that determine early-termination quality.

### Stage 2 and deployed accuracy at 10% tolerance

| Split | System | Stop-policy F1 | Within 10% using its own prediction | Mean savings |
|---|---|---:|---:|---:|
| Validation | Reproduced baseline | 0.8696 | 0.6684 | 5,241 ms |
| Validation | FMNet-v3 | 0.8703 | 0.5867 | 5,087 ms |
| Test | Reproduced baseline | 0.8608 | 0.6786 | 5,145 ms |
| Test | FMNet-v3 | 0.8604 | 0.5908 | 4,968 ms |
| Robustness | Reproduced baseline | 0.8565 | 0.6608 | 5,245 ms |
| Robustness | FMNet-v3 | 0.8552 | 0.5785 | 5,073 ms |

The F1 values measure classification against the stop labels. They do not by themselves
measure the accuracy of the Mbps value returned to a user.

A diagnostic evaluation kept FMNet-v3's selected stop points but substituted the stored
XGBoost estimate at those same points. The within-10% rates became `0.6650`, `0.6734`,
and `0.6598` on validation, test, and robustness, respectively. This is close to the
baseline and isolates the main bottleneck: the learned stopping policy is competitive,
while FMNet-v3's own prefix-throughput estimate is not yet as accurate.

## Experiment Progression

### v1: patch encoder with separate downstream models

v1 grouped the trace into 20 non-overlapping 500 ms patches and used masked-patch
pretraining. Separate models handled full-trace regression, stop classification, and
speed-tier classification. The stop model did not consume the throughput model's output.

A later self-contained check combined the v1 stop decisions with v1 throughput estimates:

| Split | Within 10% | Mean savings | Stop MAE |
|---|---:|---:|---:|
| Validation | 0.2564 | 5,718 ms | 57.63 Mbps |
| Test | 0.2653 | 5,720 ms | 49.25 Mbps |
| Robustness | 0.2600 | 5,736 ms | 77.81 Mbps |

### v2: joint per-decision multitask models

v1.5 and v2 added per-decision throughput, uncertainty, stop, and optional speed-tier
heads. They tested multitask losses, policy-aware costs, causal decision encoders, Huber
and MSE objectives, and XGBoost distillation. Optimization remained unstable; the best
recorded v2 self-contained test reached `0.1134` within 10% while saving 4,330 ms.

### v3: causal 100-bucket encoder

v3 removed patch compression, retained all 100 bucket tokens, and aligned pretraining
with causal fine-tuning. Separating throughput optimization from policy optimization
made v3 the strongest self-contained foundation pipeline in this project.

See [Experiment History](docs/experiments.md) for the full progression.

## Repository Layout

```text
fmfstlt/
  models/                   Importable FMNet-v3 and two-stage model classes
scripts/                    Supported end-to-end foundation-model workflow
experimental_scripts/       Frozen baseline reproduction and research utilities
sql/                        Five canonical M-Lab BigQuery queries
tests/                      Unit tests for shapes, causality, and gradient isolation
docs/                       Architecture, data pipeline, experiments, and final report
```

The supported workflow is documented in [scripts/README.md](scripts/README.md).
Baseline and ablation commands are documented in
[experimental_scripts/README.md](experimental_scripts/README.md).

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/g-nitin-1/FMFSTLT.git
cd FMFSTLT
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

The installed command-line entry points are:

```text
fmfstlt-pretrain
fmfstlt-train
fmfstlt-evaluate
```

### Architecture smoke test

This check does not require the dataset:

```bash
python3 - <<'PY'
import torch
from fmfstlt.models import FMNetV3, FMNetV3Config

model = FMNetV3(FMNetV3Config(num_layers=2))
x = torch.randn(2, 100, 13)
decisions = torch.tensor([[4, 9, 14, 19], [4, 9, 14, 19]])
outputs = model(x, decisions)

print("decision throughput:", outputs["throughput_mu"].shape)
print("final throughput:", outputs["final_throughput_mu"].shape)
PY
```

Expected shapes:

```text
decision throughput: torch.Size([2, 4])
final throughput: torch.Size([2])
```

## Reproduction

### 1. Build the public BigQuery dataset

This step requires Google Cloud credentials, BigQuery permissions, and a billing-aware
project. Always run the dry run first.

```bash
export PROJECT_ID="your-google-cloud-project"
export DATASET="fmfstlt"
export MAXIMUM_BYTES_BILLED="700000000000"

DRY_RUN=1 bash scripts/build_exact_public_bigquery.sh
bash scripts/build_exact_public_bigquery.sh
bash scripts/export_exact_public_features.sh
bash scripts/verify_exact_public_features.sh
```

For large exports, prefer:

```bash
bash scripts/export_exact_public_tables_to_gcs.sh
```

### 2. Build normalized local shards

```bash
python -m scripts.build_exact_public_shards
python -m scripts.compute_exact_public_train_stats
python -m scripts.materialize_normalized_exact_public_shards
python -m scripts.make_stage1_uuid_split
```

### 3. Pretrain FMNet-v3

The following command matches the selected 10-epoch pretraining configuration:

```bash
fmfstlt-pretrain \
  --output-root artifacts_exact_public/foundation_v3_pretrain_10ep \
  --epochs 10 \
  --batch-size 256 \
  --d-model 256 \
  --num-heads 8 \
  --num-layers 8 \
  --ff-dim 1024 \
  --dropout 0.15 \
  --device cuda
```

### 4. Fine-tune the selected two-stage model

```bash
fmfstlt-train \
  --input-root artifacts_exact_public/stage2_transformer_dataset_eps_10 \
  --pretrained-encoder artifacts_exact_public/foundation_v3_pretrain_10ep/fmnet_v3_pretrain.pt \
  --output-root artifacts_exact_public/foundation_twostage_run3_cosine_ema \
  --include-h-decision \
  --encoder-d-model 256 \
  --encoder-num-heads 8 \
  --encoder-num-layers 10 \
  --encoder-ff-dim 1024 \
  --encoder-dropout 0.15 \
  --phase-1-epochs 15 \
  --phase-1-lr 5e-5 \
  --phase-1-cosine-lr \
  --phase-1-ema \
  --phase-2-epochs 5 \
  --batch-size 128 \
  --gradient-accumulation-steps 8 \
  --device cuda
```

The selected deployable artifact is:

```text
artifacts_exact_public/foundation_twostage_run3_cosine_ema/phase2_checkpoint.pt
```

### 5. Run the foundation-versus-baseline evaluation

This comparison requires the frozen baseline model and epsilon-10 Stage 2 dataset
artifacts in addition to the foundation checkpoint:

```bash
fmfstlt-evaluate \
  --foundation-checkpoint artifacts_exact_public/foundation_twostage_run3_cosine_ema/phase2_checkpoint.pt \
  --foundation-threshold 0.45 \
  --baseline-checkpoint artifacts_exact_public/stage2_transformer_eps_10_local_gpu_bs1024_acc4/stage2_transformer_model.pt \
  --baseline-threshold 0.25 \
  --input-root artifacts_exact_public/stage2_transformer_dataset_eps_10 \
  --output-root artifacts_exact_public/foundation_vs_baseline_multi_epsilon \
  --batch-size 128 \
  --device cuda
```

The comparison writes:

```text
artifacts_exact_public/foundation_vs_baseline_multi_epsilon/multi_epsilon_comparison.json
```

### Optional: rebuild the frozen baseline

```bash
python -m experimental_scripts.build_stage1_regression_windows
python -m experimental_scripts.train_stage1_xgboost
python -m experimental_scripts.score_stage1_xgboost
python -m experimental_scripts.build_stage2_stop_labels --epsilon 10
python -m experimental_scripts.build_stage2_transformer_dataset --epsilon 10
python -m experimental_scripts.train_stage2_transformer
```

## Testing and Quality Checks

```bash
pytest
ruff check .
ruff format --check .
bash -n scripts/*.sh
```

The test suite verifies:

- output tensor shapes;
- causal invariance of earlier representations to future-bucket changes;
- strict versus richer Stage 2 input dimensions;
- detached Stage 1 gradients during Phase 2;
- valid model configuration serialization.

## Important Limitations

- The public reconstruction follows the paper's sampled-day methodology but cannot
  guarantee the authors' exact UUID split.
- The primary baseline comparison is frozen at `epsilon = 10%`.
- FMNet-v3 does not beat the specialized XGBoost regressor at intermediate prefixes.
- The selected `stage1_logvar` head is not directly supervised or calibrated by an
  uncertainty loss.
- Phase 2 uses teacher-derived permanent-safe labels generated from the frozen XGBoost
  baseline rather than rebuilding the oracle from FMNet-v3's Stage 1 predictions.
- The selected richer Stage 2 policy receives the encoder decision state in addition to
  Stage 1 scalar outputs, so it is not the strictest possible TurboTest analogue.
- The evaluation command needs stored baseline artifacts for comparison, although the
  deployed FMNet-v3 inference path itself does not use XGBoost.

## Engineering Highlights

- Built a restartable public-data pipeline from BigQuery SQL to normalized tensor shards.
- Processed approximately 59 million TCP/BBR feature rows.
- Implemented causal PyTorch Transformers for both trace encoding and stop decisions.
- Added self-supervised next-bucket pretraining on unlabeled network traces.
- Preserved a single deployable Stage 1 to Stage 2 neural checkpoint.
- Used train-only statistics, UUID-level splits, and later-date robustness evaluation.
- Added threshold sweeps and self-contained deployed evaluation rather than relying only
  on classifier F1.
- Recorded unsuccessful architectures and optimization choices as reproducible research
  evidence.

## Documentation

- [Architecture](docs/architecture.md)
- [Data Pipeline](docs/data_pipeline.md)
- [Experiment History](docs/experiments.md)
- [Main Workflow](scripts/README.md)
- [Experimental Scripts](experimental_scripts/README.md)
- [Final ACM two-page report](docs/final_report.pdf)

## License

This project is licensed under the [MIT License](LICENSE).
