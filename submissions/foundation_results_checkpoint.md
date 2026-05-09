# Foundation Model Results Checkpoint

Date: 2026-05-01

This checkpoint records the current implemented foundation-model line after completing:

- masked-patch pretraining on train UUIDs only
- throughput regression fine-tuning
- epsilon=10 early-stop fine-tuning
- speed-tier classification fine-tuning

The frozen baseline scope remains unchanged: baseline early-stopping comparisons are anchored only at epsilon=10.

## Artifacts

| Component | Artifact |
|---|---|
| Pretrained foundation encoder | `artifacts_exact_public/foundation_pretrain_masked_patch_v1/foundation_pretrain_model.pt` |
| Throughput regressor | `artifacts_exact_public/foundation_throughput_regressor_v1/foundation_throughput_model.pt` |
| Early-stop classifier | `artifacts_exact_public/foundation_early_stop_eps_10_v1/foundation_early_stop_model.pt` |
| Speed-tier classifier | `artifacts_exact_public/foundation_speed_classifier_v1/foundation_speed_classifier_model.pt` |

## Pretraining

| Metric | Value |
|---|---:|
| Train UUID examples | 720000 |
| Shards | 50 |
| Epochs | 5 |
| Mask ratio | 0.50 |
| Epoch 1 loss | 0.6103 |
| Epoch 5 loss | 0.4732 |

## Throughput Regression vs Stage 1 XGBoost

Lower is better.

| Subset | Stage 1 XGBoost MAE | Foundation MAE | Stage 1 XGBoost RMSE | Foundation RMSE |
|---|---:|---:|---:|---:|
| Val | 19.88 | 16.23 | 52.48 | 44.36 |
| Test | 18.55 | 14.24 | 46.20 | 34.04 |
| Robustness | 35.23 | 38.76 | 173.52 | 245.63 |

Interpretation: the foundation regressor improves val/test throughput prediction, but robustness is worse than the XGBoost baseline.

## Early Stopping at Epsilon 10 vs Frozen Stage 2 Transformer

| Subset | Baseline F1 | Foundation F1 | Baseline Within Eps | Foundation Within Eps | Baseline Stop ms | Foundation Stop ms |
|---|---:|---:|---:|---:|---:|---:|
| Val | 0.870 | 0.842 | 0.668 | 0.629 | 4141 | 3664 |
| Test | 0.861 | 0.830 | 0.679 | 0.630 | 4258 | 3682 |
| Robustness | 0.856 | 0.831 | 0.661 | 0.619 | 4144 | 3653 |

Interpretation: the foundation early-stop head is more aggressive and saves roughly 0.46-0.58 seconds more, but it loses F1 and within-epsilon policy accuracy against the frozen epsilon=10 Stage 2 Transformer baseline.

## Early-Stopping Policy-Selected Ablation

This ablation initializes the early-stop encoder from the throughput-regression checkpoint and selects threshold/epoch by validation `policy_within_epsilon_rate`, not window F1.

| Subset | F1 | Within Eps | Stop Rate | Mean Stop ms | Mean Savings ms | Mean Stop Abs Error Mbps |
|---|---:|---:|---:|---:|---:|---:|
| Val | 0.371 | 0.800 | 0.583 | 8429 | 954 | 6.63 |
| Test | 0.400 | 0.802 | 0.595 | 8378 | 1024 | 6.19 |
| Robustness | 0.382 | 0.801 | 0.580 | 8427 | 962 | 17.42 |

Best epoch: 1. Selected threshold: 0.80.

Interpretation: this is a conservative operating point. It beats the frozen Stage 2 baseline on within-epsilon rate, but only by waiting much longer, so it loses most of the time-saving benefit and has poor window-level F1.

## Early-Stopping Threshold Sweep

Two trained foundation early-stop checkpoints were rescored over 99 thresholds from 0.01 to 0.99:

- `f1_from_pretrain`: `artifacts_exact_public/foundation_early_stop_eps_10_v1/foundation_early_stop_model.pt`
- `policy_from_throughput`: `artifacts_exact_public/foundation_early_stop_eps_10_policy_from_throughput_v1/foundation_early_stop_model.pt`

The best near-baseline operating points are:

| Run | Subset | Threshold | F1 | Within Eps | Mean Savings ms | Mean Stop ms |
|---|---|---:|---:|---:|---:|---:|
| f1_from_pretrain | Val | 0.08 | 0.841 | 0.671 | 5232 | 4151 |
| f1_from_pretrain | Test | 0.09 | 0.830 | 0.682 | 5105 | 4298 |
| f1_from_pretrain | Robustness | 0.09 | 0.833 | 0.670 | 5098 | 4291 |
| policy_from_throughput | Val | 0.04 | 0.730 | 0.669 | 4637 | 4745 |
| policy_from_throughput | Test | 0.06 | 0.721 | 0.680 | 4487 | 4916 |
| policy_from_throughput | Robustness | 0.05 | 0.719 | 0.668 | 4490 | 4899 |

Interpretation: threshold sweeping helps identify balanced operating points, but no foundation early-stop checkpoint strictly dominates the frozen Stage 2 baseline on both within-epsilon rate and time savings. The `f1_from_pretrain` checkpoint is closest to baseline; the `policy_from_throughput` checkpoint is generally too conservative unless maximizing within-epsilon rate is the only objective.

## Speed-Tier Classification

There is no frozen paper baseline for this task in the current reproduction artifacts, so this result is reported as a standalone downstream foundation-model task.

| Subset | Accuracy | Macro F1 |
|---|---:|---:|
| Val | 0.7897 | 0.7863 |
| Test | 0.7653 | 0.7706 |
| Robustness | 0.7610 | 0.7634 |

Best epoch: 2 selected by validation macro F1.

## Current Bottom Line

The foundation model is useful as a shared representation:

- It improves throughput prediction on val/test compared with Stage 1 XGBoost.
- It produces a working standalone speed-tier classifier with about 0.76-0.79 accuracy.
- It does not yet beat the frozen epsilon=10 Stage 2 Transformer for early stopping.
- The policy-selected early-stop ablation can reach about 0.80 within-epsilon rate, but it is too conservative for the main early-stopping objective.
- Threshold sweeps show the best foundation early-stop operating points are competitive but not dominant over the frozen baseline.

Recommended next work is not another blind full rerun. The next high-value step is report writing plus, if time permits, one targeted early-stop modeling change rather than another threshold-only experiment.
