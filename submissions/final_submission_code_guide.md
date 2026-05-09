# Final Submission Code Guide

Date: 2026-05-09

This guide records the final code and artifact state for submission. Experiments are frozen. Do not run Run 4 for the final report.

## Main Claim

The submitted system tests a single foundation-model replacement for the two-model TurboTest pipeline while preserving TurboTest's order:

```text
raw 100 ms trace -> foundation Stage 1 prefix throughput predictions -> foundation Stage 2 stop policy
```

The final result is mixed: the foundation model matches the specialized Stage 2 policy closely in policy-only comparisons, but deployed user-facing accuracy still lags TurboTest because foundation prefix throughput predictions are weaker than XGBoost at early stop points.

## Final Code Paths

| Purpose | File |
|---|---|
| Two-stage foundation architecture | `scripts/foundation_model_twostage.py` |
| Two-stage foundation training | `scripts/train_foundation_twostage.py` |
| Causal foundation encoder used by two-stage model | `scripts/foundation_model_v3.py` |
| Foundation pretraining | `scripts/pretrain_foundation_v3.py` |
| Multi-epsilon deployed diagnostic | `scripts/compare_foundation_vs_baseline_multi_epsilon.py` |
| Frozen Stage 2 baseline trainer | `scripts/train_stage2_transformer.py` |
| Stage 2 dataset builder | `scripts/build_stage2_transformer_dataset.py` |

## Final Artifacts

| Component | Path |
|---|---|
| Foundation Run 3 phase-2 checkpoint | `artifacts_exact_public/foundation_twostage_run3_cosine_ema/phase2_checkpoint.pt` |
| Foundation Run 3 summary | `artifacts_exact_public/foundation_twostage_run3_cosine_ema/training_summary_phase2.json` |
| Frozen TurboTest Stage 2 baseline | `artifacts_exact_public/stage2_transformer_eps_10_local_gpu_bs1024_acc4/stage2_transformer_model.pt` |
| Multi-epsilon diagnostic output | `artifacts_exact_public/foundation_vs_baseline_multi_epsilon/multi_epsilon_comparison.json` |
| Valid Stage 2 dataset | `artifacts_exact_public/stage2_transformer_dataset_eps_10/` |

## Frozen Baseline Scope

Only epsilon=10 is used as the frozen reproduced baseline. Other epsilon datasets were found stale because their Stage 2 materialization used an incompatible XGBoost runtime. They are not used as final trained baselines.

## Final Two-Stage Foundation Run

The final foundation model is Run 3:

```bash
python3 /mnt/e/fmfstlt/scripts/train_foundation_twostage.py --input-root /mnt/e/fmfstlt/artifacts_exact_public/stage2_transformer_dataset_eps_10 --pretrained-encoder /mnt/e/fmfstlt/artifacts_exact_public/foundation_v3_pretrain_10ep/fmnet_v3_pretrain.pt --output-root /mnt/e/fmfstlt/artifacts_exact_public/foundation_twostage_run3_cosine_ema --include-h-decision --encoder-num-layers 10 --phase-1-epochs 15 --phase-1-lr 5e-5 --phase-1-prefix-weight 2.0 --phase-1-final-weight 1.0 --phase-1-cosine-lr --phase-1-ema --phase-1-ema-decay 0.999 --phase-2-epochs 5 --phase-2-lr 1e-3 --batch-size 128 --gradient-accumulation-steps 8 --device cuda --enable-phase-3
```

The final report uses the phase-2 checkpoint, not phase 3, because phase 3 damaged Stage 1 throughput quality.

## Final Deployed Diagnostic

The deployed comparison is:

```bash
python3 /mnt/e/fmfstlt/scripts/compare_foundation_vs_baseline_multi_epsilon.py --foundation-checkpoint /mnt/e/fmfstlt/artifacts_exact_public/foundation_twostage_run3_cosine_ema/phase2_checkpoint.pt --foundation-threshold 0.45 --baseline-checkpoint /mnt/e/fmfstlt/artifacts_exact_public/stage2_transformer_eps_10_local_gpu_bs1024_acc4/stage2_transformer_model.pt --baseline-threshold 0.25 --input-root /mnt/e/fmfstlt/artifacts_exact_public/stage2_transformer_dataset_eps_10 --output-root /mnt/e/fmfstlt/artifacts_exact_public/foundation_vs_baseline_multi_epsilon --batch-size 128 --device cuda
```

Interpretation of diagnostic rows:

| Row | Meaning |
|---|---|
| F+F | Foundation chooses stop and deploys foundation prediction. This is the real foundation user experience. |
| F+X | Foundation chooses stop and deploys XGBoost prediction. This isolates policy quality only. |
| B+X | Baseline chooses stop and deploys XGBoost prediction. This is real TurboTest behavior. |
| B+F | Baseline chooses stop and deploys foundation prediction. This is diagnostic only. |

## Final Report

The ACM-style two-column report source is:

```text
submissions/final_report_acm_2page.tex
```

Build from the `submissions/` directory:

```bash
pdflatex final_report_acm_2page.tex
```

