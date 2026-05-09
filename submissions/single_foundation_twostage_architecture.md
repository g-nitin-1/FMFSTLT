# Single Foundation Two-Stage Architecture

Date: 2026-05-06

This document defines the next architecture to implement. The goal is to test the clarified project idea:

```text
raw speed-test trace
-> foundation Stage 1 prefix throughput predictions
-> foundation Stage 2 stop-policy over those Stage 1 outputs
```

This preserves the TurboTest Stage 1 -> Stage 2 order, but replaces the separate TurboTest models with one foundation-model system.

## 1. Baseline Being Replaced

TurboTest uses two separate task-specific models:

```text
Stage 1:
  XGBoost regressor
  input: latest 2-second / 20-bucket window
  output: prefix throughput prediction y_pred_mbps[d]

Stage 2:
  Transformer stop-policy model
  input: sequence of Stage 1 predictions and decision features
  output: stop probability per decision
```

The proposed architecture keeps the same logical order:

```text
Stage 1:
  foundation encoder + throughput head
  input: raw normalized trace prefix
  output: foundation prefix throughput prediction f_pred_mbps[d]

Stage 2:
  foundation policy module
  input: sequence built from foundation Stage 1 outputs
  output: stop probability per decision
```

The key requirement is that Stage 2 explicitly consumes the foundation Stage 1 outputs. It must not be only a direct hidden-state classifier.

## 2. Dataset Inputs

Training uses the valid frozen epsilon=10 Stage 2 dataset:

```text
artifacts_exact_public/stage2_transformer_dataset_eps_10/
```

Each batch contains:

| Name | Shape | Meaning |
|---|---:|---|
| `x_full` | `[B, 100, 13]` | normalized 100 ms bucket features |
| `bucket_mask` | `[B, 100]` | observed bucket mask |
| `decision_valid_mask` | `[B, 20]` | valid Stage 2 decisions |
| `decision_end_bucket` | `[B, 20]` | decision end bucket, normally `4,9,...,99` |
| `decision_elapsed_ms` | `[B, 20]` | elapsed milliseconds at each decision |
| `decision_observed_buckets_seen` | `[B, 20]` | observed buckets available at each decision |
| `y_true_mbps` | `[B]` | final measured throughput |
| `stop_label` | `[B, 20]` | permanent-safe suffix oracle target |
| `instantaneous_safe_window` | `[B, 20]` | diagnostic/evaluation safety label |
| `y_pred_mbps` | `[B, 20]` | TurboTest/XGBoost prefix prediction, used only for comparison or teacher ablation |
| `relative_error` | `[B, 20]` | XGBoost relative error, diagnostic only; not deployable input |
| `speed_tier` | `[B]` | speed-tier label, optional auxiliary task |

Deployment inputs must not include `relative_error`, `instantaneous_safe_window`, or any target-derived field.

## 3. High-Level Flow

```text
                           x_full [B,100,13]
                                  |
                                  v
                  shared causal foundation encoder
                                  |
                    hidden states H [B,100,256]
                                  |
                 gather decisions at buckets 4,9,...,99
                                  |
                    decision states Hd [B,20,256]
                                  |
       +--------------------------+--------------------------+
       |                                                     |
       v                                                     v
Stage 1 throughput head                         optional final/speed heads
mu[d], logvar[d]                                final_mu, speed_logits
       |
       v
foundation Stage 1 output sequence
[pred_log[d], pred_mbps[d], uncertainty[d], elapsed_norm[d], observed_norm[d], Hd[d]]
       |
       v
causal Stage 2 policy module
       |
       v
stop_logit[d] / stop_prob[d]
```

## 4. Shared Causal Foundation Encoder

Use the FMNet v3 bucket-level causal encoder as the base, with stricter mask handling.

Default configuration:

| Component | Value |
|---|---:|
| input features | `13` |
| buckets | `100` |
| bucket projection | `13 -> 256` |
| positional embedding | `[1, 100, 256]` |
| Transformer layers | `8` |
| attention heads | `8` |
| feed-forward dim | `1024` |
| dropout | `0.15` |
| head dropout | `0.10` |

Encoder details:

```text
x_full: [B,100,13]
bucket_mask: [B,100]

1. zero invalid bucket vectors after normalization
2. project buckets: [B,100,13] -> [B,100,256]
3. add learned position embeddings
4. apply causal Transformer attention
5. output hidden states H: [B,100,256]
```

Causality rule:

```text
H[:, t, :] may attend only to buckets <= t
```

Decision states:

```text
decision_end_bucket[d] = 5d + 4
Hd[d] = H[:, decision_end_bucket[d], :]
Hd shape: [B,20,256]
```

Invalid decisions are excluded from all decision-level losses and metrics.

## 5. Stage 1 Foundation Throughput Head

For each decision state `Hd[d]`:

```text
throughput_head(Hd[d]) -> stage1_mu[d]
uncertainty_head(Hd[d]) -> stage1_logvar[d]
```

Shapes:

```text
stage1_mu:     [B,20]
stage1_logvar: [B,20]
stage1_pred_mbps = expm1(stage1_mu)
```

Head architecture:

```text
MLP:
  Linear(256, 256)
  GELU
  Dropout(0.10)
  Linear(256, 1)
```

Stage 1 targets:

```text
target_log = log1p(y_true_mbps)
target_log_decision[d] = target_log for every valid decision d
```

Primary Stage 1 loss:

```text
prefix_mse = mean_valid((stage1_mu[d] - log1p(y_true_mbps))^2)
```

Optional uncertainty loss:

```text
prefix_nll = 0.5 * exp(-logvar[d]) * (target_log - mu[d])^2 + 0.5 * logvar[d]
```

Initial implementation should use MSE first. Add NLL only after MSE is stable.

Stage 1 metrics:

```text
prefix MAE/RMSE over all valid decisions
decision-index MAE/RMSE for d = 0..19
final-decision MAE/RMSE at d = 19
within-10%-error rate
```

Compare against TurboTest Stage 1:

```text
foundation stage1_pred_mbps[d] vs y_true_mbps
XGBoost y_pred_mbps[d] vs y_true_mbps
```

## 6. Stage 2 Foundation Policy Input

The Stage 2 policy must be built from foundation Stage 1 outputs.

For each decision `d`, construct:

```text
policy_token[d] =
  concat(
    stage1_mu[d],
    stage1_pred_norm[d],
    stage1_uncertainty[d],
    elapsed_norm[d],
    observed_norm[d],
    Hd[d]                       optional but default on
  )
```

Recommended features:

| Feature | Shape | Deployable? | Notes |
|---|---:|---|---|
| `stage1_mu[d]` | scalar | yes | log prediction |
| `stage1_pred_norm[d]` | scalar | yes | normalized Mbps prediction |
| `stage1_uncertainty[d]` | scalar | yes | from logvar or learned confidence |
| `elapsed_norm[d]` | scalar | yes | `elapsed_ms / 10000` |
| `observed_norm[d]` | scalar | yes | observed buckets / 100 |
| `Hd[d]` | `256` | yes | foundation decision representation |

Do not include:

```text
relative_error[d]
instantaneous_safe_window[d]
stop_label[d]
y_true_mbps
```

These are labels/evaluation fields, not deployment inputs.

Detach rule for the first version:

```text
policy_token uses detach(stage1_mu, stage1_logvar) during Phase 2
```

Reason: previous experiments showed stop-policy gradients can destabilize throughput learning. Detaching Stage 1 outputs tests the two-stage design cleanly.

## 7. Stage 2 Foundation Policy Module

The policy module consumes the 20 decision tokens.

Initial policy architecture:

```text
policy_input_dim = 256 + 5   # hidden state plus scalar Stage 1/meta features

policy_projection:
  Linear(policy_input_dim, 256)
  GELU
  Dropout(0.10)

causal decision Transformer:
  layers: 2
  heads: 4
  d_model: 256
  ff_dim: 512
  dropout: 0.10

stop_head:
  Linear(256, 128)
  GELU
  Dropout(0.10)
  Linear(128, 1)
```

Output:

```text
stop_logit: [B,20]
stop_prob = sigmoid(stop_logit)
```

Causality:

```text
stop_logit[d] may see only policy tokens <= d
```

This mirrors TurboTest Stage 2 sequence behavior.

## 8. Optional Heads

Final throughput head:

```text
final_mu = stage1_mu at last valid decision
or a separate MLP over last valid hidden state
```

Speed-tier head:

```text
speed_logits = MLP(last_valid_hidden)
```

These are auxiliary. They should not block Stage 2 policy work.

## 9. Training Schedule

### Phase 0: Initialization

Use one of:

```text
FMNet v3 pretrained encoder:
  artifacts_exact_public/foundation_v3_pretrain_10ep/fmnet_v3_pretrain.pt

or random init ablation
```

### Phase 1: Stage 1 Training

Goal: train foundation Stage 1 prefix throughput predictions.

Loss:

```text
L_phase1 =
  3.0 * prefix_mse
  + 1.0 * final_mse
```

Weights:

```text
stage1 prefix: 3.0
final throughput: 1.0
stop BCE: 0.0
speed tier: 0.0
```

Train:

```text
encoder + Stage 1 heads
```

Phase 1 success gate:

```text
prefix MAE should move toward XGBoost prefix MAE.
if prefix MAE remains > 2x XGBoost after enough epochs, policy success is unlikely.
```

### Phase 2: Detached Stage 2 Policy Training

Goal: train policy on top of foundation Stage 1 outputs without damaging Stage 1.

Forward:

```text
stage1 outputs = Stage1Head(Encoder(x_full))
policy_input = build_policy_tokens(detach(stage1 outputs), hidden states, metadata)
stop_logits = PolicyModule(policy_input)
```

Loss:

```text
L_phase2 =
  1.0 * stop_bce
  + 1.0 * prefix_mse
  + 0.5 * final_mse
```

The prefix/final losses remain active to prevent drift.

### Phase 3: Optional End-to-End Fine-Tuning

Only run if Phase 2 is competitive.

Forward:

```text
policy_input = build_policy_tokens(stage1 outputs, hidden states, metadata)
```

No detach.

Use lower learning rate:

```text
lr = 1e-5 or 2e-5
```

Keep Stage 1 loss weight high:

```text
prefix_mse weight >= stop_bce weight
```

## 10. Losses

Decision validity mask:

```text
valid = decision_valid_mask.bool()
```

Prefix throughput:

```text
prefix_mse =
  sum(valid * (stage1_mu - log1p(y_true_mbps))^2) / sum(valid)
```

Final throughput:

```text
final_mse =
  mean((final_mu - log1p(y_true_mbps))^2)
```

Stop policy:

```text
stop_bce =
  BCEWithLogits(stop_logit[d], stop_label[d])
  averaged over valid decisions
```

Optional speed tier:

```text
speed_ce = CrossEntropy(speed_logits, speed_tier)
```

Total:

```text
total =
  lambda_prefix * prefix_mse
  + lambda_final * final_mse
  + lambda_stop * stop_bce
  + lambda_speed * speed_ce
```

## 11. Evaluation

### Stage 1 Evaluation

Report:

```text
val/test/robustness final MAE and RMSE
val/test/robustness prefix MAE and RMSE
decision-index prefix MAE for d=0..19
within-10%-error rate
```

Compare against:

```text
Stage 1 XGBoost final and prefix predictions
foundation v1 throughput regressor
```

### Stage 2 Evaluation

Threshold sweep:

```text
thresholds: 0.01 to 0.99 or 0.05 to 0.95
selection: validation policy_constrained_savings
constraint: within_epsilon_rate >= 0.66
```

Report:

```text
window F1
policy within-epsilon rate
emitted-stop rate
mean savings vs full test
median savings vs full test
mean stop elapsed ms
mean relative error at stop
```

Compare against frozen epsilon=10 Stage 2 baseline:

```text
val:        F1 0.8696, within 0.6684, savings 5240.9 ms
test:       F1 0.8608, within 0.6786, savings 5144.7 ms
robustness: F1 0.8565, within 0.6608, savings 5245.0 ms
```

## 12. Success Criteria

Strict success:

```text
single foundation two-stage model beats or Pareto-dominates TurboTest Stage 2 on test
and is competitive with Stage 1 XGBoost/foundation v1 throughput.
```

Stage 2 Pareto win:

```text
within_epsilon_rate >= baseline
mean_savings_ms >= baseline
```

Acceptable research success:

```text
single foundation model matches Stage 2 within statistical noise
while preserving the Stage 1 -> Stage 2 foundation structure.
```

Negative result:

```text
if Stage 1 foundation prefix MAE remains far worse than XGBoost,
then the foundation Stage 2 policy is expected to lose.
```

## 13. First Implementation Checklist

Implement:

```text
scripts/foundation_model_twostage.py
scripts/train_foundation_twostage.py
```

Minimum features:

```text
FMNet v3 encoder with bucket_mask handling
Stage 1 mu/logvar heads
explicit policy-token builder from Stage 1 outputs
2-layer causal decision policy Transformer
Phase 1 and Phase 2 training modes
detach flag between Stage 1 and Stage 2
threshold sweep and policy metrics
Stage 1 prefix comparison vs XGBoost
```

First run:

```text
Phase 1 only probe
Phase 1 full
Phase 2 detached probe
Phase 2 detached full
```

Only after detached Phase 2 is competitive:

```text
Phase 3 end-to-end no-detach fine-tune
```

## 14. Expected Project Claim

If successful:

```text
We preserve TurboTest's two-stage logic but implement both stages in one foundation-model system.
The foundation model first predicts prefix throughput and then uses its own predictions
to decide early termination.
```

If not successful:

```text
The experiment directly tests the intended project hypothesis and shows where it fails:
the bottleneck is whether foundation Stage 1 prefix predictions can match the specialized
TurboTest/XGBoost regressor strongly enough for downstream stopping.
```

