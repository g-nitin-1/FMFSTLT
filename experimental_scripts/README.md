# Experimental Scripts

These commands support the reproduced TurboTest baseline and research analysis. They are
kept for transparency and comparison, but they are not part of the primary FMFSTLT
workflow.

The supported foundation-model workflow lives in [`scripts/`](../scripts/), while the
importable model implementations live in [`fmfstlt/models/`](../fmfstlt/models/).

## Baseline Reproduction

```bash
python -m experimental_scripts.build_stage1_regression_windows
python -m experimental_scripts.train_stage1_xgboost
python -m experimental_scripts.score_stage1_xgboost
python -m experimental_scripts.build_stage2_stop_labels --epsilon 10
python -m experimental_scripts.build_stage2_transformer_dataset --epsilon 10
python -m experimental_scripts.train_stage2_transformer
```

## Analysis

```bash
python -m experimental_scripts.rescore_stage2_thresholds
python -m experimental_scripts.plot_pareto_frontier
```
