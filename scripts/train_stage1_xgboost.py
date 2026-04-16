#!/usr/bin/env python3
"""Train the Stage 1 XGBoost regressor from Stage 1 window shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import xgboost as xgb

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:  # type: ignore[override]
        def __init__(self, iterable=None, total=None, **kwargs) -> None:
            self.iterable = iterable
            self.total = total

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def update(self, n: int = 1) -> None:
            return None

        def set_postfix(self, *args, **kwargs) -> None:
            return None

        def close(self) -> None:
            return None


DEFAULT_TRAIN_SUBSET = "train"
DEFAULT_EVAL_SUBSET = "val"


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Train Stage 1 XGBoost from chunked Stage 1 window shards."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage1_windows",
        help="Root directory containing Stage 1 window subsets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage1_xgboost",
        help="Directory for model, metrics, and cache files.",
    )
    parser.add_argument(
        "--train-subset",
        default=DEFAULT_TRAIN_SUBSET,
        help="Subset name used for fitting.",
    )
    parser.add_argument(
        "--eval-subset",
        default=DEFAULT_EVAL_SUBSET,
        help="Subset name used for validation and early stopping.",
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help="Optional glob relative to input root for a probe run.",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=1500,
        help="Maximum number of boosting rounds (paper-scale default: 1500).",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Early stopping rounds on the eval subset (default: 50).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
        help="XGBoost learning rate (paper: 0.03).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=7,
        help="Maximum tree depth (paper: 7).",
    )
    parser.add_argument(
        "--min-child-weight",
        type=float,
        default=10.0,
        help="Minimum child weight.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Row subsampling fraction.",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Column subsampling fraction.",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=1.0,
        help="L2 regularization.",
    )
    parser.add_argument(
        "--max-bin",
        type=int,
        default=256,
        help="Histogram max_bin setting.",
    )
    parser.add_argument(
        "--nthread",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="CPU threads for XGBoost.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="XGBoost verbosity.",
    )
    return parser.parse_args()


def list_subset_paths(input_root: Path, subset: str, input_glob: str | None) -> list[Path]:
    subset_dir = input_root / subset
    if input_glob:
        paths = sorted(subset_dir.glob(input_glob))
    else:
        paths = sorted(subset_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no Stage 1 shards found for subset {subset} under {subset_dir}")
    return paths


def inspect_shards(paths: list[Path]) -> dict[str, object]:
    total_examples = 0
    feature_dim: int | None = None
    output_format: str | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            x = data["x"]
            total_examples += int(x.shape[0])
            current_dim = int(x.shape[1])
            if feature_dim is None:
                feature_dim = current_dim
                output_format = str(data["output_format"].item())
            elif current_dim != feature_dim:
                raise ValueError(
                    f"inconsistent feature dimension in {path}: {current_dim} vs {feature_dim}"
                )
    return {
        "examples": total_examples,
        "feature_dim": feature_dim,
        "output_format": output_format,
    }


class NpzShardIter(xgb.DataIter):
    """Feed Stage 1 NPZ shards into XGBoost without loading all windows at once."""

    def __init__(self, shard_paths: list[Path], cache_prefix: Path) -> None:
        super().__init__(cache_prefix=str(cache_prefix), release_data=True)
        self.shard_paths = shard_paths
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def next(self, input_data) -> bool:
        if self._cursor >= len(self.shard_paths):
            return False
        shard_path = self.shard_paths[self._cursor]
        self._cursor += 1
        with np.load(shard_path, allow_pickle=False) as data:
            x = data["x"].astype(np.float32, copy=False)
            y = data["y_true_mbps"].astype(np.float32, copy=False)
            input_data(data=x, label=y)
        return True


class TqdmTrainingCallback(xgb.callback.TrainingCallback):
    """Progress bar callback for XGBoost training."""

    def __init__(self, total_rounds: int) -> None:
        self.total_rounds = total_rounds
        self.pbar = None

    def before_training(self, model: xgb.Booster) -> xgb.Booster:
        self.pbar = tqdm(
            total=self.total_rounds,
            desc="stage1 xgboost",
            unit="round",
            dynamic_ncols=True,
        )
        return model

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log) -> bool:
        if self.pbar is None:
            return False
        self.pbar.update(1)
        postfix = {}
        for dataset_name, metrics in evals_log.items():
            for metric_name, values in metrics.items():
                if not values:
                    continue
                value = values[-1]
                if isinstance(value, tuple):
                    value = value[0]
                postfix[f"{dataset_name}_{metric_name}"] = f"{float(value):.4f}"
        if postfix:
            self.pbar.set_postfix(postfix, refresh=False)
        return False

    def after_training(self, model: xgb.Booster) -> xgb.Booster:
        if self.pbar is not None:
            self.pbar.close()
        return model


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = output_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_paths = list_subset_paths(args.input_root, args.train_subset, args.input_glob)
    eval_paths = list_subset_paths(args.input_root, args.eval_subset, args.input_glob)

    train_info = inspect_shards(train_paths)
    eval_info = inspect_shards(eval_paths)

    train_iter = NpzShardIter(train_paths, cache_dir / f"{args.train_subset}.cache")
    eval_iter = NpzShardIter(eval_paths, cache_dir / f"{args.eval_subset}.cache")

    dtrain = xgb.ExtMemQuantileDMatrix(
        train_iter,
        max_bin=args.max_bin,
        nthread=args.nthread,
    )
    deval = xgb.ExtMemQuantileDMatrix(
        eval_iter,
        max_bin=args.max_bin,
        nthread=args.nthread,
        ref=dtrain,
    )

    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "max_bin": args.max_bin,
        "nthread": args.nthread,
        "verbosity": args.verbosity,
        "eval_metric": ["rmse", "mae"],
    }

    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain, args.train_subset), (deval, args.eval_subset)],
        early_stopping_rounds=args.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False,
        callbacks=[TqdmTrainingCallback(args.num_boost_round)],
    )

    model_path = output_root / "stage1_xgboost_model.json"
    booster.save_model(model_path)

    best_iteration = getattr(booster, "best_iteration", None)
    best_score = getattr(booster, "best_score", None)
    summary = {
        "params": params,
        "train_subset": args.train_subset,
        "eval_subset": args.eval_subset,
        "train_shards": len(train_paths),
        "eval_shards": len(eval_paths),
        "train_examples": train_info["examples"],
        "eval_examples": eval_info["examples"],
        "feature_dim": train_info["feature_dim"],
        "output_format": train_info["output_format"],
        "num_boost_round": args.num_boost_round,
        "early_stopping_rounds": args.early_stopping_rounds,
        "best_iteration": int(best_iteration) if best_iteration is not None else None,
        "best_score": float(best_score) if best_score is not None else None,
        "evals_result": evals_result,
        "model_path": str(model_path),
    }

    summary_path = output_root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"saved model to {model_path}")
    print(f"saved training summary to {summary_path}")


if __name__ == "__main__":
    main()
