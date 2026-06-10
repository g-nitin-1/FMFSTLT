# Architecture

## Problem

Given a partial 10-second download-speed trace, predict the final throughput and decide
whether the measurement can stop. The model receives 13 normalized features for each
100 ms bucket.

```text
x_full: [batch, 100 buckets, 13 features]
```

The 20 decision points occur every 500 ms and end at buckets:

```text
4, 9, 14, ..., 99
```

## FMNet-v3 Encoder

FMNet-v3 projects each 13-dimensional bucket into a 256-dimensional token and adds a
learned position embedding. A 10-layer Transformer processes the 100 tokens with a
strict causal mask:

```text
hidden[t] may attend only to buckets 0..t
```

This avoids future leakage and retains the 100 ms structure that was lost in earlier
5-bucket patch variants.

Default final-run encoder:

| Parameter | Value |
|---|---:|
| Input features | 13 |
| Maximum buckets | 100 |
| Hidden width | 256 |
| Attention heads | 8 |
| Encoder layers | 10 |
| Feed-forward width | 1,024 |

## Self-Supervised Pretraining

The pretraining objective predicts bucket `t+1` from buckets `0..t`. Training is
restricted to the 720,000 training UUIDs.

```text
next_bucket_pred[t] = head(hidden[t])
target[t] = x_full[t + 1]
```

A masked Smooth-L1 loss is applied to valid consecutive buckets. This teaches the
encoder short-term TCP/BBR dynamics without using throughput or stop labels.

## Stage 1: Throughput and Uncertainty

The encoder states at the 20 decision buckets are gathered into:

```text
decision_hidden: [batch, 20, 256]
```

Two heads operate at every decision:

```text
stage1_mu:     [batch, 20]
stage1_logvar: [batch, 20]
```

`stage1_mu[d]` predicts `log(1 + final Mbps)` at decision `d`.
`stage1_logvar[d]` is exposed as an auxiliary policy input. In the selected run, this
head is not directly optimized with Gaussian negative log likelihood or another
uncertainty loss. It must not be interpreted as a calibrated variance estimate.

Stage 1 fully fine-tunes the pretrained encoder and throughput-mean head. The
log-variance head receives no direct supervision. The final run uses cosine
learning-rate decay and exponential moving average weights to reduce epoch-to-epoch
instability.

## Stage 2: Stop Policy

Stage 2 receives one feature vector per decision:

```text
[
  stage1_mu[d],
  stage1_logvar[d],
  elapsed_ms[d] / 10000,
  observed_bucket_count[d] / 100,
  decision_hidden[d]
]
```

These vectors are processed by a smaller 4-layer causal Transformer. Its output is:

```text
stop_logit: [batch, 20]
stop_probability = sigmoid(stop_logit)
```

The first valid probability above the selected threshold chooses the stop point. The
system reports `expm1(stage1_mu[d])` as the predicted Mbps.

The permanent-safe suffix targets are inherited from the frozen epsilon-10 baseline
dataset and were generated from XGBoost prediction errors. The Stage 2 policy itself is
trained on FMNet Stage 1 outputs and does not require XGBoost at deployment.

## Phased Fine-Tuning

### Phase 1

Train the encoder and Stage 1 heads for per-decision and final throughput prediction.
The Stage 2 policy is unused.

### Phase 2

Freeze the encoder and Stage 1 heads. Detach Stage 1 outputs and train only the Stage 2
policy with binary cross-entropy against the permanent-safe suffix target.

This creates a stable interface:

```text
Stage 1 predictions -> detached Stage 2 input
```

### Phase 3

An optional end-to-end phase was evaluated, but not selected. Allowing Stage 2 gradients
to update the encoder slightly improved policy metrics while increasing Stage 1 error
by roughly 3-4x. The final checkpoint is therefore the Phase 2 model.

## Deployment

At decision `d`:

1. Encode observed buckets causally.
2. Predict final throughput and uncertainty.
3. Update the Stage 2 decision sequence.
4. Stop if the probability crosses the threshold.
5. Report the Stage 1 prediction from the same decision.

No XGBoost output, relative-error label, or future bucket is required by the deployed
foundation path.
