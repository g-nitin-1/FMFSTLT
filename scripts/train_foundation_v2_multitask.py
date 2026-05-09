#!/usr/bin/env python3
"""Train foundation v2: causal bucket-stem all-decision multitask model."""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import fields
from pathlib import Path

if "TMPDIR" not in os.environ:
    for candidate in ("/dev/shm", "/tmp", "/var/tmp", "/usr/tmp"):
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            os.environ["TMPDIR"] = candidate
            break

import numpy as np
import torch

from foundation_model import TraceFoundationConfig
from foundation_model_v2 import CausalBucketFoundationV2, load_v1_throughput_head_weights
from train_foundation_v15_multitask import (
    DEFAULT_EVAL_SUBSETS,
    POLICY_METRIC_CHOICES,
    SPEED_TIERS,
    WINDOW_METRIC_CHOICES,
    choose_threshold,
    compute_pos_weight,
    evaluate_model_outputs,
    list_subset_paths,
    maybe_limit,
    metric_score,
    read_dataset_summary,
    resolve_device,
    seed_everything,
    train_one_epoch,
    validate_args,
)
from train_stage2_transformer import evaluate_subset_with_threshold


THROUGHPUT_SELECTION_METRICS = (
    "throughput_mae",
    "throughput_rmse",
    "prefix_throughput_mae",
    "prefix_throughput_rmse",
)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    artifacts = root_dir / "artifacts_exact_public"
    parser = argparse.ArgumentParser(
        description=(
            "Train causal bucket-stem foundation v2 on the frozen epsilon=10 Stage 2 dataset. "
            "Defaults are throughput-focused so prefix quality is measured before stop-head tuning."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=artifacts / "stage2_transformer_dataset_eps_10",
        help="Root containing valid frozen epsilon=10 Stage 2 shards.",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=Path,
        default=artifacts / "foundation_throughput_regressor_v1" / "foundation_throughput_model.pt",
        help="v1 throughput checkpoint used to initialize compatible scalar heads.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=artifacts / "foundation_v2_causal_bucket_throughput_eps_10",
        help="Directory for the v2 model and summaries.",
    )
    parser.add_argument("--train-subset", default="train")
    parser.add_argument("--val-subset", default="val")
    parser.add_argument("--eval-subsets", nargs="+", default=list(DEFAULT_EVAL_SUBSETS))
    parser.add_argument("--input-glob", default=None, help="Optional glob relative to each subset directory.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--dropout", type=float, default=0.15, help="Override config dropout for v2 fine-tuning.")
    parser.add_argument("--num-layers", type=int, default=6, help="Decision-token Transformer layer count.")
    parser.add_argument("--bucket-hidden-dim", type=int, default=128)
    parser.add_argument("--stem-kernel-size", type=int, default=5)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--prefix-throughput-weight", type=float, default=3.0)
    parser.add_argument("--full-throughput-weight", type=float, default=1.0)
    parser.add_argument("--prefix-huber-weight", type=float, default=0.0)
    parser.add_argument("--full-huber-weight", type=float, default=0.0)
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--prefix-mse-weight", type=float, default=0.0)
    parser.add_argument("--full-mse-weight", type=float, default=0.0)
    parser.add_argument(
        "--prefix-target-source",
        choices=("true", "xgboost"),
        default="true",
        help="Target for per-decision throughput heads. Final throughput still targets y_true_mbps.",
    )
    parser.add_argument("--stop-bce-weight", type=float, default=0.0)
    parser.add_argument("--policy-weight", type=float, default=0.0)
    parser.add_argument("--speed-weight", type=float, default=0.0)
    parser.add_argument("--lambda-wrong", type=float, default=2.0)
    parser.add_argument("--lambda-wait", type=float, default=1.0)
    parser.add_argument(
        "--lambda-safe-stop",
        type=float,
        default=0.0,
        help="Extra reward-like loss weight that pushes stop probability up when the model says the prefix is safe.",
    )
    parser.add_argument("--epsilon-fraction", type=float, default=0.10)
    parser.add_argument("--relative-denominator-floor", type=float, default=1e-6)
    parser.add_argument(
        "--stop-target-source",
        choices=("stop_label", "instantaneous_safe_window", "model_instant_safe", "model_suffix_safe"),
        default="stop_label",
    )
    parser.add_argument(
        "--policy-safety-source",
        choices=("dataset", "model_prediction"),
        default="dataset",
    )
    parser.add_argument("--bucket-feature-dropout", type=float, default=0.05)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--init-throughput-heads",
        action="store_true",
        help="Copy v1 throughput regression head weights into v2 throughput mean heads.",
    )
    parser.add_argument("--fixed-threshold", type=float, default=0.50)
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--min-within-epsilon-rate", type=float, default=0.66)
    parser.add_argument("--savings-score-scale-ms", type=float, default=10000.0)
    parser.add_argument(
        "--threshold-metric",
        choices=WINDOW_METRIC_CHOICES + POLICY_METRIC_CHOICES,
        default="f1",
        help="Metric used to tune a secondary stop threshold on validation.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=WINDOW_METRIC_CHOICES + POLICY_METRIC_CHOICES + THROUGHPUT_SELECTION_METRICS,
        default="throughput_mae",
        help="Validation metric used to choose the best epoch.",
    )
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=19)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-shards", type=int, default=None)
    parser.add_argument("--max-eval-shards", type=int, default=None)
    parser.add_argument("--max-train-batches-per-epoch", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def config_from_checkpoint(checkpoint: dict[str, object], args: argparse.Namespace) -> TraceFoundationConfig:
    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint is missing model_config")
    allowed = {field.name for field in fields(TraceFoundationConfig)}
    kwargs = {key: raw_config[key] for key in allowed if key in raw_config}
    kwargs["dropout"] = args.dropout
    kwargs["num_layers"] = args.num_layers
    return TraceFoundationConfig(**kwargs)


def load_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[CausalBucketFoundationV2, TraceFoundationConfig, dict[str, object]]:
    checkpoint = torch.load(args.pretrained_model_path, map_location=device)
    config = config_from_checkpoint(checkpoint, args)
    model = CausalBucketFoundationV2(
        config,
        bucket_hidden_dim=args.bucket_hidden_dim,
        stem_kernel_size=args.stem_kernel_size,
        num_speed_tiers=len(SPEED_TIERS),
    ).to(device)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing model_state_dict")

    init_report: dict[str, object] = {
        "source_checkpoint": str(args.pretrained_model_path),
        "encoder_initialized": False,
        "throughput_heads_initialized": False,
    }
    if args.init_throughput_heads:
        missing_head = load_v1_throughput_head_weights(model, state)
        if missing_head:
            raise ValueError(
                "--init-throughput-heads was set, but checkpoint is missing throughput head keys: "
                f"{missing_head}"
            )
        init_report["throughput_heads_initialized"] = True
    return model, config, init_report


def selection_score(
    metric_name: str,
    *,
    val_outputs: dict[str, object],
    tuned_metrics: dict[str, object],
) -> float:
    if metric_name == "throughput_mae":
        return -float(val_outputs["throughput"]["overall"]["mae"])  # type: ignore[index]
    if metric_name == "throughput_rmse":
        return -float(val_outputs["throughput"]["overall"]["rmse"])  # type: ignore[index]
    if metric_name == "prefix_throughput_mae":
        y_true = np.asarray(val_outputs["y_true_mbps"], dtype=np.float64)
        y_pred = np.asarray(val_outputs["y_pred_mbps"], dtype=np.float64)
        if y_true.size == 0:
            return float("-inf")
        return -float(np.abs(y_pred - y_true).mean())
    if metric_name == "prefix_throughput_rmse":
        y_true = np.asarray(val_outputs["y_true_mbps"], dtype=np.float64)
        y_pred = np.asarray(val_outputs["y_pred_mbps"], dtype=np.float64)
        if y_true.size == 0:
            return float("-inf")
        return -float(np.sqrt(np.square(y_pred - y_true).mean()))
    return metric_score(metric_name, tuned_metrics)


def prefix_throughput_metrics(outputs: dict[str, object]) -> dict[str, float | int]:
    y_true = np.asarray(outputs["y_true_mbps"], dtype=np.float64)
    y_pred = np.asarray(outputs["y_pred_mbps"], dtype=np.float64)
    if y_true.size == 0:
        return {"count": 0, "mae": 0.0, "rmse": 0.0, "mean_relative_error": 0.0, "within_10pct_rate": 0.0}
    error = y_pred - y_true
    relative_error = np.abs(error) / np.maximum(np.abs(y_true), 1e-6)
    return {
        "count": int(y_true.size),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mean_relative_error": float(relative_error.mean()),
        "within_10pct_rate": float((relative_error <= 0.10).mean()),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.train_desc = "foundation v2 train"

    dataset_summary = read_dataset_summary(args.input_root)
    model, config, init_report = load_model(args, device)
    if args.freeze_encoder:
        for name, parameter in model.named_parameters():
            if not any(
                name.startswith(prefix)
                for prefix in (
                    "throughput_mu_head.",
                    "throughput_log_var_head.",
                    "stop_head.",
                    "final_mu_head.",
                    "final_log_var_head.",
                    "speed_head.",
                )
            ):
                parameter.requires_grad = False

    train_paths = maybe_limit(list_subset_paths(args.input_root, args.train_subset, args.input_glob), args.max_train_shards)
    val_paths = maybe_limit(list_subset_paths(args.input_root, args.val_subset, args.input_glob), args.max_eval_shards)
    final_eval_paths = {
        subset: maybe_limit(list_subset_paths(args.input_root, subset, args.input_glob), args.max_eval_shards)
        for subset in args.eval_subsets
    }

    pos_weight = compute_pos_weight(args, dataset_summary)
    stop_pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed)

    best_score = float("-inf")
    best_epoch = None
    best_threshold = 0.5
    best_threshold_score = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    epoch_history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            paths=train_paths,
            optimizer=optimizer,
            device=device,
            args=args,
            rng=rng,
            stop_pos_weight=stop_pos_weight,
        )
        val_outputs = evaluate_model_outputs(
            model=model,
            paths=val_paths,
            device=device,
            args=args,
            stop_pos_weight=stop_pos_weight,
            desc="foundation v2 eval val",
        )
        threshold, threshold_score, tuned_metrics = choose_threshold(
            args=args,
            outputs=val_outputs,
            subset_name=args.val_subset,
        )
        fixed_metrics = evaluate_subset_with_threshold(
            name=args.val_subset,
            outputs=val_outputs,
            threshold=args.fixed_threshold,
        )
        score = selection_score(args.selection_metric, val_outputs=val_outputs, tuned_metrics=tuned_metrics)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_threshold = threshold
            best_threshold_score = threshold_score
            best_state = copy.deepcopy(model.state_dict())
        epoch_history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation_throughput": val_outputs["throughput"],
                "validation_prefix_throughput": prefix_throughput_metrics(val_outputs),
                "fixed_threshold": args.fixed_threshold,
                "fixed_threshold_validation": fixed_metrics,
                "tuned_threshold": threshold,
                "tuned_threshold_score": threshold_score,
                "tuned_threshold_validation": tuned_metrics,
                "selection_score": score,
            }
        )

    model.load_state_dict(best_state)
    final_metrics: dict[str, object] = {}
    for subset, paths in final_eval_paths.items():
        outputs = evaluate_model_outputs(
            model=model,
            paths=paths,
            device=device,
            args=args,
            stop_pos_weight=stop_pos_weight,
            desc=f"foundation v2 final {subset}",
        )
        final_metrics[subset] = {
            "throughput": outputs["throughput"],
            "prefix_throughput": prefix_throughput_metrics(outputs),
            "fixed_threshold": evaluate_subset_with_threshold(
                name=subset,
                outputs=outputs,
                threshold=args.fixed_threshold,
            ),
            "tuned_threshold": evaluate_subset_with_threshold(
                name=subset,
                outputs=outputs,
                threshold=best_threshold,
            ),
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": "foundation_v2_causal_bucket_multitask_eps_10",
        "input_root": str(args.input_root),
        "pretrained_model_path": str(args.pretrained_model_path),
        "initialization": init_report,
        "model_config": config.to_dict(),
        "v2_config": model.to_v2_config_dict(),
        "speed_tiers": list(SPEED_TIERS),
        "pos_weight": float(pos_weight),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "effective_batch_size": int(args.batch_size * args.gradient_accumulation_steps),
        "eval_batch_size": int(args.eval_batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "bucket_feature_dropout": float(args.bucket_feature_dropout),
        "loss_weights": {
            "prefix_throughput": float(args.prefix_throughput_weight),
            "full_throughput": float(args.full_throughput_weight),
            "prefix_huber": float(args.prefix_huber_weight),
            "full_huber": float(args.full_huber_weight),
            "huber_beta": float(args.huber_beta),
            "prefix_mse": float(args.prefix_mse_weight),
            "full_mse": float(args.full_mse_weight),
            "stop_bce": float(args.stop_bce_weight),
            "policy": float(args.policy_weight),
            "speed": float(args.speed_weight),
            "lambda_wrong": float(args.lambda_wrong),
            "lambda_wait": float(args.lambda_wait),
            "lambda_safe_stop": float(args.lambda_safe_stop),
        },
        "stop_target_source": args.stop_target_source,
        "policy_safety_source": args.policy_safety_source,
        "prefix_target_source": args.prefix_target_source,
        "epsilon_fraction": float(args.epsilon_fraction),
        "fixed_threshold": float(args.fixed_threshold),
        "min_within_epsilon_rate": float(args.min_within_epsilon_rate),
        "savings_score_scale_ms": float(args.savings_score_scale_ms),
        "threshold_metric": args.threshold_metric,
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "best_tuned_threshold": best_threshold,
        "best_tuned_threshold_score": best_threshold_score,
        "best_selection_score": best_score,
        "epoch_history": epoch_history,
        "final_metrics": final_metrics,
    }
    model_path = args.output_root / "foundation_v2_multitask_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "v2_config": model.to_v2_config_dict(),
            "speed_tiers": list(SPEED_TIERS),
            "summary": summary,
        },
        model_path,
    )
    summary_path = args.output_root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote foundation v2 model to {model_path}")
    print(f"wrote training summary to {summary_path}")


if __name__ == "__main__":
    main()
