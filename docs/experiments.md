# Experiment History

The project evolved through three architecture families. Each failure narrowed the
deployment bottleneck.

## v1: Patch Encoder and Separate Heads

v1 converted the 100 buckets into 20 non-overlapping 500 ms patches and pretrained the
encoder with masked-patch reconstruction.

Separate downstream models were trained for:

- full-trace throughput regression;
- prefix stop/continue classification;
- speed-tier classification.

The throughput regressor performed well on complete traces, but the Stage 2 classifier
did not consume its predictions. A post-hoc self-contained check paired the v1 stop
classifier with the v1 throughput regressor at the selected prefix:

| Split | Within 10% | Mean savings | Stop MAE |
|---|---:|---:|---:|
| Validation | 0.2564 | 5,718 ms | 57.63 Mbps |
| Test | 0.2653 | 5,720 ms | 49.25 Mbps |
| Robustness | 0.2600 | 5,736 ms | 77.81 Mbps |

This showed that a plausible stopping classifier is insufficient when its own prefix
throughput estimate is inaccurate.

## v2: Joint Per-Decision Multitask Heads

v1.5 and v2 introduced per-decision throughput, uncertainty, stop, and optional
speed-tier heads. Training combined throughput losses, binary stop loss, and
policy-aware costs.

The architecture was closer to the desired single-model system, but optimization was
unstable. The best v2 deployed self-check reached only:

```text
test within 10% = 0.1134
mean savings = 4,330 ms
```

Several variants either stopped aggressively with inaccurate Mbps or achieved high
accuracy only by waiting until the test was nearly complete.

## v3: Causal Bucket Encoder

v3 removed 500 ms patch compression and retained all 100 bucket tokens. Causal
next-bucket pretraining aligned the pretraining and fine-tuning attention patterns.

The final two-stage version:

1. fully fine-tunes Stage 1 for per-decision throughput;
2. freezes and detaches Stage 1;
3. trains a causal Stage 2 policy over Stage 1 outputs and decision context.

On the test split:

```text
deployed within 10% = 0.5908
mean savings = 4,968 ms
stop-policy F1 = 0.8604
```

v3 is the strongest self-contained foundation pipeline in the project. Its stop policy
is competitive with the reproduced specialized policy, but its prefix throughput
predictions remain the primary source of deployed error.

## Main Lesson

The central result is not that a larger Transformer automatically replaces the
specialized system. The experiments show that:

- self-supervised trace representations transfer well to stop-policy learning;
- full-trace throughput accuracy can hide weak early-prefix behavior;
- a deployed evaluation must use the model's own value at its own chosen stop;
- protecting Stage 1 from Stage 2 gradients is necessary when Stage 1 produces the
  user-facing measurement.
