#!/usr/bin/env python3
"""Train foundation v1.5: causal all-decision multitask fine-tuning."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import fields
from pathlib import Path

import numpy as np

if "TMPDIR" not in os.environ:
    for candidate in ("/dev/shm", "/tmp", "/var/tmp", "/usr/tmp"):
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            os.environ["TMPDIR"] = candidate
            break

import torch
import torch.nn.functional as F

from foundation_model import TraceFoundationConfig
from foundation_model_v15 import (
    CausalPatchFoundationV15,
    load_v1_encoder_weights,
    load_v1_throughput_head_weights,
)
from train_stage2_transformer import evaluate_subset_with_threshold, safe_divide

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:  # type: ignore[override]
        def __init__(self, iterable=None, **kwargs) -> None:
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())

        def set_postfix(self, *args, **kwargs) -> None:
            return None

        def close(self) -> None:
            return None


DEFAULT_EVAL_SUBSETS = ("val", "test", "robustness")
SPEED_TIERS = ("0-25", "25-100", "100-200", "200-400", "400+")
SPEED_TIER_TO_INDEX = {tier: idx for idx, tier in enumerate(SPEED_TIERS)}
SPEED_TIER_TO_INDEX["400_plus"] = SPEED_TIER_TO_INDEX["400+"]
WINDOW_METRIC_CHOICES = ("f1", "balanced_accuracy", "accuracy", "precision", "recall", "auroc", "average_precision")
POLICY_METRIC_CHOICES = (
    "policy_within_epsilon_rate",
    "policy_emitted_stop_rate",
    "policy_mean_savings_vs_full_ms",
    "policy_mean_stop_elapsed_ms",
    "policy_mean_stop_abs_error_mbps",
    "policy_correct_savings_score",
    "policy_constrained_savings_score",
)
LOWER_IS_BETTER_METRICS = {"policy_mean_stop_elapsed_ms", "policy_mean_stop_abs_error_mbps"}


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    artifacts = root_dir / "artifacts_exact_public"
    parser = argparse.ArgumentParser(
        description="Train causal patch-token foundation v1.5 on the frozen epsilon=10 Stage 2 dataset."
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
        default=artifacts / "foundation_pretrain_masked_patch_v1" / "foundation_pretrain_model.pt",
        help="v1 masked-patch or downstream checkpoint used to initialize the shared encoder.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=artifacts / "foundation_v15_multitask_eps_10",
        help="Directory for the v1.5 model and summaries.",
    )
    parser.add_argument("--train-subset", default="train")
    parser.add_argument("--val-subset", default="val")
    parser.add_argument("--eval-subsets", nargs="+", default=list(DEFAULT_EVAL_SUBSETS))
    parser.add_argument("--input-glob", default=None, help="Optional glob relative to each subset directory.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--dropout", type=float, default=0.15, help="Override config dropout for v1.5 fine-tuning.")
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--prefix-throughput-weight", type=float, default=1.0)
    parser.add_argument("--full-throughput-weight", type=float, default=0.5)
    parser.add_argument("--stop-bce-weight", type=float, default=1.0)
    parser.add_argument("--policy-weight", type=float, default=0.5)
    parser.add_argument("--speed-weight", type=float, default=0.2)
    parser.add_argument("--lambda-wrong", type=float, default=2.0)
    parser.add_argument("--lambda-wait", type=float, default=0.2)
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
        help=(
            "Target for the stop BCE. model_* targets are recomputed from detached foundation "
            "throughput predictions so stop supervision matches model-policy evaluation."
        ),
    )
    parser.add_argument(
        "--policy-safety-source",
        choices=("dataset", "model_prediction"),
        default="dataset",
        help="Safety signal used by the policy cost term.",
    )
    parser.add_argument("--bucket-feature-dropout", type=float, default=0.05)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--init-throughput-heads",
        action="store_true",
        help="If the checkpoint is a v1 throughput regressor, copy its head into v1.5 throughput mean heads.",
    )
    parser.add_argument("--fixed-threshold", type=float, default=0.50)
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument(
        "--min-within-epsilon-rate",
        type=float,
        default=0.66,
        help="Constraint used by policy_constrained_savings_score.",
    )
    parser.add_argument(
        "--savings-score-scale-ms",
        type=float,
        default=10000.0,
        help="Scale used to normalize savings in policy_correct_savings_score.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=WINDOW_METRIC_CHOICES + POLICY_METRIC_CHOICES,
        default="f1",
        help="Metric used to tune a secondary threshold on validation.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=WINDOW_METRIC_CHOICES + POLICY_METRIC_CHOICES,
        default="f1",
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_subset_paths(root: Path, subset: str, input_glob: str | None) -> list[Path]:
    subset_dir = root / subset
    paths = sorted(subset_dir.glob(input_glob or "*.npz"))
    if not paths:
        raise SystemExit(f"no Stage 2 shards found for subset {subset} under {subset_dir}")
    return paths


def maybe_limit(paths: list[Path], limit: int | None) -> list[Path]:
    return paths if limit is None else paths[:limit]


def read_dataset_summary(input_root: Path) -> dict[str, object] | None:
    path = input_root / "stage2_dataset_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def compute_pos_weight(args: argparse.Namespace, dataset_summary: dict[str, object] | None) -> float:
    if args.pos_weight is not None:
        return float(args.pos_weight)
    if dataset_summary is None:
        return 1.0
    subset_summary = dataset_summary["subsets"][args.train_subset]
    total = int(subset_summary["decisions"])
    positive = int(subset_summary["stop_positive_decisions"])
    negative = total - positive
    if positive <= 0 or negative <= 0:
        return 1.0
    return float(negative / positive)


def config_from_checkpoint(checkpoint: dict[str, object], dropout: float) -> TraceFoundationConfig:
    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint is missing model_config")
    allowed = {field.name for field in fields(TraceFoundationConfig)}
    kwargs = {key: raw_config[key] for key in allowed if key in raw_config}
    kwargs["dropout"] = dropout
    return TraceFoundationConfig(**kwargs)


def load_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[CausalPatchFoundationV15, TraceFoundationConfig, dict[str, object]]:
    checkpoint = torch.load(args.pretrained_model_path, map_location=device)
    config = config_from_checkpoint(checkpoint, args.dropout)
    model = CausalPatchFoundationV15(config, num_speed_tiers=len(SPEED_TIERS)).to(device)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing model_state_dict")
    missing, unexpected = load_v1_encoder_weights(model, state)
    if unexpected:
        raise ValueError(f"unexpected translated checkpoint keys: {unexpected[:10]}")
    if missing:
        raise ValueError(f"missing translated encoder keys: {missing[:10]}")
    init_report: dict[str, object] = {"encoder_initialized": True, "throughput_heads_initialized": False}
    if args.init_throughput_heads:
        missing_head = load_v1_throughput_head_weights(model, state)
        if missing_head:
            raise ValueError(
                "--init-throughput-heads was set, but checkpoint is missing throughput head keys: "
                f"{missing_head}"
            )
        init_report["throughput_heads_initialized"] = True
    return model, config, init_report


def speed_indices(speed_tier: np.ndarray) -> np.ndarray:
    return np.array([SPEED_TIER_TO_INDEX[str(item)] for item in speed_tier.tolist()], dtype=np.int64)


def iter_test_batches(
    path: Path,
    *,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        x_full = data["x_full"].astype(np.float32, copy=False)
        bucket_mask = data["bucket_mask"].astype(bool, copy=False)
        decision_valid_mask = data["decision_valid_mask"].astype(bool, copy=False)
        valid_tests = np.flatnonzero(decision_valid_mask.any(axis=1))
        ordering = np.array(valid_tests, copy=True)
        if shuffle:
            rng.shuffle(ordering)

        for start in range(0, ordering.shape[0], batch_size):
            idx = ordering[start : start + batch_size]
            yield {
                "x_full": x_full[idx],
                "bucket_mask": bucket_mask[idx],
                "decision_valid_mask": decision_valid_mask[idx],
                "decision_end_bucket": data["decision_end_bucket"][idx].astype(np.int16, copy=False),
                "decision_elapsed_ms": data["decision_elapsed_ms"][idx].astype(np.int32, copy=False),
                "y_true_mbps": data["y_true_mbps"][idx].astype(np.float32, copy=False),
                "xgboost_y_pred_mbps": data["y_pred_mbps"][idx].astype(np.float32, copy=False),
                "stop_label": data["stop_label"][idx].astype(np.float32, copy=False),
                "instantaneous_safe_window": data["instantaneous_safe_window"][idx].astype(np.float32, copy=False),
                "uuid": data["uuid"][idx],
                "test_time": data["test_time"][idx],
                "oracle_stop_found": data["oracle_stop_found"][idx].astype(np.uint8, copy=False),
                "oracle_stop_elapsed_ms": data["oracle_stop_elapsed_ms"][idx].astype(np.int32, copy=False),
                "speed_tier": data["speed_tier"][idx],
                "speed_index": speed_indices(data["speed_tier"][idx]),
            }


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = values * mask.to(values.dtype)
    return weighted.sum() / torch.clamp(mask.to(values.dtype).sum(), min=1.0)


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_var = torch.clamp(log_var, min=-8.0, max=8.0)
    return 0.5 * torch.exp(-log_var) * torch.square(target - mu) + 0.5 * log_var


def smooth_l1_loss_values(prediction: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    if beta <= 0:
        return torch.abs(prediction - target)
    diff = torch.abs(prediction - target)
    return torch.where(diff < beta, 0.5 * torch.square(diff) / beta, diff - 0.5 * beta)


def mbps_from_log_prediction(mu: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.expm1(mu), min=0.0)


def compute_loss(
    *,
    model_outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    stop_pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = batch["decision_valid_mask"].bool()
    y_true = batch["y_true_mbps"]
    target_log = torch.log1p(torch.clamp(y_true, min=0.0)).unsqueeze(1)
    if getattr(args, "prefix_target_source", "true") == "xgboost":
        prefix_target_mbps = torch.clamp(batch["xgboost_y_pred_mbps"], min=0.0)
        target_log_by_decision = torch.log1p(prefix_target_mbps)
    else:
        target_log_by_decision = target_log.expand_as(model_outputs["throughput_mu"])

    prefix_nll = masked_mean(
        gaussian_nll(
            model_outputs["throughput_mu"],
            model_outputs["throughput_log_var"],
            target_log_by_decision,
        ),
        valid,
    )
    final_nll = gaussian_nll(
        model_outputs["final_mu"],
        model_outputs["final_log_var"],
        target_log.squeeze(1),
    ).mean()
    huber_beta = float(getattr(args, "huber_beta", 0.1))
    prefix_huber = masked_mean(
        smooth_l1_loss_values(model_outputs["throughput_mu"], target_log_by_decision, huber_beta),
        valid,
    )
    final_huber = smooth_l1_loss_values(model_outputs["final_mu"], target_log.squeeze(1), huber_beta).mean()
    prefix_mse = masked_mean(torch.square(model_outputs["throughput_mu"] - target_log_by_decision), valid)
    final_mse = torch.square(model_outputs["final_mu"] - target_log.squeeze(1)).mean()

    pred_mbps = mbps_from_log_prediction(model_outputs["throughput_mu"])
    denom = torch.clamp(torch.abs(y_true).unsqueeze(1), min=args.relative_denominator_floor)
    model_relative_error = torch.abs(pred_mbps - y_true.unsqueeze(1)) / denom
    model_instant_safe = (model_relative_error.detach() <= args.epsilon_fraction).to(torch.float32)

    if args.stop_target_source == "stop_label":
        stop_target = batch["stop_label"]
        bce_pos_weight = stop_pos_weight
    elif args.stop_target_source == "instantaneous_safe_window":
        stop_target = batch["instantaneous_safe_window"]
        bce_pos_weight = None
    elif args.stop_target_source == "model_instant_safe":
        stop_target = model_instant_safe
        bce_pos_weight = None
    else:
        safe_for_suffix = torch.where(valid, model_instant_safe.bool(), torch.ones_like(valid, dtype=torch.bool))
        suffix_safe = torch.flip(
            torch.cumprod(torch.flip(safe_for_suffix.to(torch.long), dims=[1]), dim=1),
            dims=[1],
        ).to(torch.float32)
        stop_target = suffix_safe.masked_fill(~valid, 0.0)
        bce_pos_weight = None

    stop_loss_all = F.binary_cross_entropy_with_logits(
        model_outputs["stop_logits"],
        stop_target,
        pos_weight=bce_pos_weight,
        reduction="none",
    )
    stop_bce = masked_mean(stop_loss_all, valid)

    if args.policy_safety_source == "dataset":
        safe = batch["instantaneous_safe_window"]
    else:
        safe = model_instant_safe
    elapsed_norm = torch.clamp(batch["decision_elapsed_ms"].to(torch.float32) / 10000.0, min=0.0, max=1.0)
    p_stop = torch.sigmoid(model_outputs["stop_logits"])
    wrong_stop_cost = p_stop * (1.0 - safe)
    wait_cost = (1.0 - p_stop) * safe * elapsed_norm
    safe_stop_reward = -torch.log(torch.clamp(p_stop, min=1e-6)) * safe * (1.0 - elapsed_norm)
    policy_loss = masked_mean(
        args.lambda_wrong * wrong_stop_cost
        + args.lambda_wait * wait_cost
        + args.lambda_safe_stop * safe_stop_reward,
        valid,
    )

    speed_loss = F.cross_entropy(model_outputs["speed_logits"], batch["speed_index"])
    total = (
        args.prefix_throughput_weight * prefix_nll
        + args.full_throughput_weight * final_nll
        + float(getattr(args, "prefix_huber_weight", 0.0)) * prefix_huber
        + float(getattr(args, "full_huber_weight", 0.0)) * final_huber
        + float(getattr(args, "prefix_mse_weight", 0.0)) * prefix_mse
        + float(getattr(args, "full_mse_weight", 0.0)) * final_mse
        + args.stop_bce_weight * stop_bce
        + args.policy_weight * policy_loss
        + args.speed_weight * speed_loss
    )
    return total, {
        "prefix_nll": float(prefix_nll.detach().item()),
        "final_nll": float(final_nll.detach().item()),
        "prefix_huber": float(prefix_huber.detach().item()),
        "final_huber": float(final_huber.detach().item()),
        "prefix_mse": float(prefix_mse.detach().item()),
        "final_mse": float(final_mse.detach().item()),
        "stop_bce": float(stop_bce.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "speed_ce": float(speed_loss.detach().item()),
        "model_safe_rate": float(masked_mean(model_instant_safe, valid).detach().item()),
    }


def tensor_batch(
    batch: dict[str, np.ndarray],
    device: torch.device,
    *,
    bucket_feature_dropout: float = 0.0,
    training: bool = False,
) -> dict[str, torch.Tensor]:
    x_full = torch.as_tensor(batch["x_full"], dtype=torch.float32, device=device)
    bucket_mask = torch.as_tensor(batch["bucket_mask"], dtype=torch.bool, device=device)
    if training and bucket_feature_dropout > 0:
        drop = torch.rand(bucket_mask.shape, device=device) < bucket_feature_dropout
        drop = drop & bucket_mask
        x_full = x_full.masked_fill(drop.unsqueeze(-1), 0.0)
    return {
        "x_full": x_full,
        "bucket_mask": bucket_mask,
        "decision_valid_mask": torch.as_tensor(batch["decision_valid_mask"], dtype=torch.bool, device=device),
        "decision_elapsed_ms": torch.as_tensor(batch["decision_elapsed_ms"], dtype=torch.float32, device=device),
        "y_true_mbps": torch.as_tensor(batch["y_true_mbps"], dtype=torch.float32, device=device),
        "xgboost_y_pred_mbps": torch.as_tensor(batch["xgboost_y_pred_mbps"], dtype=torch.float32, device=device),
        "stop_label": torch.as_tensor(batch["stop_label"], dtype=torch.float32, device=device),
        "instantaneous_safe_window": torch.as_tensor(
            batch["instantaneous_safe_window"], dtype=torch.float32, device=device
        ),
        "speed_index": torch.as_tensor(batch["speed_index"], dtype=torch.long, device=device),
    }


def train_one_epoch(
    *,
    model: CausalPatchFoundationV15,
    paths: list[Path],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    rng: np.random.Generator,
    stop_pos_weight: torch.Tensor,
) -> dict[str, float | int]:
    model.train()
    shuffled_paths = list(paths)
    rng.shuffle(shuffled_paths)

    total_loss = 0.0
    total_tests = 0
    optimizer_steps = 0
    microbatches = 0
    pending_microbatches = 0
    component_sums: dict[str, float] = {}
    stop_early = False

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(shuffled_paths, desc=getattr(args, "train_desc", "foundation v1.5 train"), unit="shard")
    for path in progress:
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=safe_divide(total_loss, total_tests), batches=optimizer_steps)
        for np_batch in iter_test_batches(path, batch_size=args.batch_size, shuffle=True, rng=rng):
            batch = tensor_batch(
                np_batch,
                device,
                bucket_feature_dropout=args.bucket_feature_dropout,
                training=True,
            )
            outputs = model(
                batch["x_full"],
                batch["bucket_mask"],
                decision_valid_mask=batch["decision_valid_mask"],
            )
            loss, components = compute_loss(
                model_outputs=outputs,
                batch=batch,
                args=args,
                stop_pos_weight=stop_pos_weight,
            )
            (loss / args.gradient_accumulation_steps).backward()

            tests = int(batch["x_full"].shape[0])
            total_loss += float(loss.item()) * tests
            total_tests += tests
            microbatches += 1
            pending_microbatches += 1
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + value * tests

            if pending_microbatches >= args.gradient_accumulation_steps:
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                pending_microbatches = 0
                if args.max_train_batches_per_epoch is not None and optimizer_steps >= args.max_train_batches_per_epoch:
                    stop_early = True
                    break
        if stop_early:
            break

    if pending_microbatches > 0 and (
        args.max_train_batches_per_epoch is None or optimizer_steps < args.max_train_batches_per_epoch
    ):
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    progress.close()
    result: dict[str, float | int] = {
        "loss": safe_divide(total_loss, total_tests),
        "tests": int(total_tests),
        "batches": int(optimizer_steps),
        "microbatches": int(microbatches),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "effective_batch_size": int(args.batch_size * args.gradient_accumulation_steps),
    }
    for key, value in component_sums.items():
        result[key] = safe_divide(value, total_tests)
    return result


def init_metric_bucket() -> dict[str, float]:
    return {"count": 0.0, "squared_error_sum": 0.0, "absolute_error_sum": 0.0, "relative_error_sum": 0.0}


def update_metric_bucket(bucket: dict[str, float], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    y_true64 = y_true.astype(np.float64, copy=False)
    y_pred64 = y_pred.astype(np.float64, copy=False)
    error = y_pred64 - y_true64
    bucket["count"] += float(y_true64.shape[0])
    bucket["squared_error_sum"] += float(np.square(error).sum())
    bucket["absolute_error_sum"] += float(np.abs(error).sum())
    denom = np.maximum(np.abs(y_true64), 1e-6)
    bucket["relative_error_sum"] += float((np.abs(error) / denom).sum())


def finalize_metric_bucket(bucket: dict[str, float]) -> dict[str, float | int]:
    count = int(bucket["count"])
    if count <= 0:
        return {"count": 0, "rmse": 0.0, "mae": 0.0, "mean_relative_error": 0.0}
    return {
        "count": count,
        "rmse": float(np.sqrt(bucket["squared_error_sum"] / count)),
        "mae": float(bucket["absolute_error_sum"] / count),
        "mean_relative_error": float(bucket["relative_error_sum"] / count),
    }


@torch.no_grad()
def evaluate_model_outputs(
    *,
    model: CausalPatchFoundationV15,
    paths: list[Path],
    device: torch.device,
    args: argparse.Namespace,
    stop_pos_weight: torch.Tensor,
    desc: str,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_tests = 0
    total_decisions = 0
    batches = 0
    overall = init_metric_bucket()
    by_speed_tier: dict[str, dict[str, float]] = {}

    probabilities_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    instantaneous_safe_list: list[np.ndarray] = []
    stop_label_list: list[np.ndarray] = []
    uuid_list: list[np.ndarray] = []
    test_time_list: list[np.ndarray] = []
    end_bucket_list: list[np.ndarray] = []
    elapsed_ms_list: list[np.ndarray] = []
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    relative_error_list: list[np.ndarray] = []
    oracle_stop_found_list: list[np.ndarray] = []
    oracle_stop_elapsed_ms_list: list[np.ndarray] = []

    progress = tqdm(paths, desc=desc, unit="shard")
    stop_early = False
    for path in progress:
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=safe_divide(total_loss, total_tests), batches=batches)
        for np_batch in iter_test_batches(path, batch_size=args.eval_batch_size, shuffle=False, rng=np.random.default_rng(0)):
            batch = tensor_batch(np_batch, device)
            outputs = model(
                batch["x_full"],
                batch["bucket_mask"],
                decision_valid_mask=batch["decision_valid_mask"],
            )
            loss, _ = compute_loss(
                model_outputs=outputs,
                batch=batch,
                args=args,
                stop_pos_weight=stop_pos_weight,
            )

            tests = int(batch["x_full"].shape[0])
            total_loss += float(loss.item()) * tests
            total_tests += tests
            batches += 1

            final_pred = mbps_from_log_prediction(outputs["final_mu"]).cpu().numpy().astype(np.float32)
            y_true = np_batch["y_true_mbps"].astype(np.float32, copy=False)
            update_metric_bucket(overall, y_true, final_pred)
            for tier in sorted(set(np_batch["speed_tier"].tolist())):
                tier_mask = np.asarray(np_batch["speed_tier"]) == tier
                bucket = by_speed_tier.setdefault(str(tier), init_metric_bucket())
                update_metric_bucket(bucket, y_true[tier_mask], final_pred[tier_mask])

            valid = np_batch["decision_valid_mask"].astype(bool, copy=False)
            test_idx, decision_idx = np.nonzero(valid)
            total_decisions += int(test_idx.shape[0])

            probs = torch.sigmoid(outputs["stop_logits"]).cpu().numpy().astype(np.float32)
            pred_mbps = mbps_from_log_prediction(outputs["throughput_mu"]).cpu().numpy().astype(np.float32)
            rel_error = (
                np.abs(pred_mbps - y_true.reshape(-1, 1))
                / np.maximum(np.abs(y_true.reshape(-1, 1)), args.relative_denominator_floor)
            ).astype(np.float32)
            model_safe = (rel_error <= args.epsilon_fraction).astype(np.uint8)

            probabilities_list.append(probs[test_idx, decision_idx])
            labels_list.append(np_batch["stop_label"][test_idx, decision_idx].astype(np.float32, copy=False))
            instantaneous_safe_list.append(model_safe[test_idx, decision_idx])
            stop_label_list.append(np_batch["stop_label"][test_idx, decision_idx].astype(np.uint8, copy=False))
            uuid_list.append(np_batch["uuid"][test_idx])
            test_time_list.append(np_batch["test_time"][test_idx])
            end_bucket_list.append(np_batch["decision_end_bucket"][test_idx, decision_idx].astype(np.int16, copy=False))
            elapsed_ms_list.append(np_batch["decision_elapsed_ms"][test_idx, decision_idx].astype(np.int32, copy=False))
            y_true_list.append(y_true[test_idx])
            y_pred_list.append(pred_mbps[test_idx, decision_idx])
            relative_error_list.append(rel_error[test_idx, decision_idx])
            oracle_stop_found_list.append(np_batch["oracle_stop_found"][test_idx].astype(np.uint8, copy=False))
            oracle_stop_elapsed_ms_list.append(np_batch["oracle_stop_elapsed_ms"][test_idx].astype(np.int32, copy=False))

            if args.max_eval_batches is not None and batches >= args.max_eval_batches:
                stop_early = True
                break
        if stop_early:
            break

    progress.close()
    return {
        "loss": safe_divide(total_loss, total_tests),
        "examples": int(total_decisions),
        "tests": int(total_tests),
        "batches": int(batches),
        "throughput": {
            "overall": finalize_metric_bucket(overall),
            "by_speed_tier": {tier: finalize_metric_bucket(bucket) for tier, bucket in sorted(by_speed_tier.items())},
        },
        "probabilities": np.concatenate(probabilities_list, axis=0) if probabilities_list else np.empty((0,), dtype=np.float32),
        "labels": np.concatenate(labels_list, axis=0) if labels_list else np.empty((0,), dtype=np.float32),
        "instantaneous_safe_window": np.concatenate(instantaneous_safe_list, axis=0) if instantaneous_safe_list else np.empty((0,), dtype=np.uint8),
        "stop_label": np.concatenate(stop_label_list, axis=0) if stop_label_list else np.empty((0,), dtype=np.uint8),
        "uuid": np.concatenate(uuid_list, axis=0) if uuid_list else np.empty((0,), dtype=np.str_),
        "test_time": np.concatenate(test_time_list, axis=0) if test_time_list else np.empty((0,), dtype=np.str_),
        "end_bucket": np.concatenate(end_bucket_list, axis=0) if end_bucket_list else np.empty((0,), dtype=np.int16),
        "elapsed_ms": np.concatenate(elapsed_ms_list, axis=0) if elapsed_ms_list else np.empty((0,), dtype=np.int32),
        "y_true_mbps": np.concatenate(y_true_list, axis=0) if y_true_list else np.empty((0,), dtype=np.float32),
        "y_pred_mbps": np.concatenate(y_pred_list, axis=0) if y_pred_list else np.empty((0,), dtype=np.float32),
        "relative_error": np.concatenate(relative_error_list, axis=0) if relative_error_list else np.empty((0,), dtype=np.float32),
        "oracle_stop_found": np.concatenate(oracle_stop_found_list, axis=0) if oracle_stop_found_list else np.empty((0,), dtype=np.uint8),
        "oracle_stop_elapsed_ms": np.concatenate(oracle_stop_elapsed_ms_list, axis=0) if oracle_stop_elapsed_ms_list else np.empty((0,), dtype=np.int32),
    }


def metric_score(metric_name: str, subset_metrics: dict[str, object]) -> float:
    if metric_name == "policy_correct_savings_score":
        policy = subset_metrics["policy_metrics"]
        within = float(policy.get("within_epsilon_rate") or 0.0)
        savings = float(policy.get("mean_savings_vs_full_ms") or 0.0)
        scale = float(subset_metrics.get("savings_score_scale_ms", 10000.0))
        return within * max(savings, 0.0) / max(scale, 1.0)
    if metric_name == "policy_constrained_savings_score":
        policy = subset_metrics["policy_metrics"]
        within = float(policy.get("within_epsilon_rate") or 0.0)
        savings = float(policy.get("mean_savings_vs_full_ms") or 0.0)
        minimum = float(subset_metrics.get("min_within_epsilon_rate", 0.66))
        if within < minimum:
            return within - minimum
        return max(savings, 0.0)
    if metric_name.startswith("policy_"):
        value = subset_metrics["policy_metrics"].get(metric_name.removeprefix("policy_"))
    else:
        value = subset_metrics["window_metrics"].get(metric_name)
    if value is None:
        return float("-inf")
    if metric_name in LOWER_IS_BETTER_METRICS:
        return -float(value)
    return float(value)


def threshold_candidates(args: argparse.Namespace) -> np.ndarray:
    if args.threshold_steps <= 1:
        return np.array([0.5], dtype=np.float64)
    return np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps, dtype=np.float64)


def choose_threshold(
    *,
    args: argparse.Namespace,
    outputs: dict[str, object],
    subset_name: str,
) -> tuple[float, float, dict[str, object]]:
    if args.decision_threshold is not None:
        threshold = float(args.decision_threshold)
        metrics = evaluate_subset_with_threshold(name=subset_name, outputs=outputs, threshold=threshold)
        metrics["min_within_epsilon_rate"] = float(args.min_within_epsilon_rate)
        metrics["savings_score_scale_ms"] = float(args.savings_score_scale_ms)
        return threshold, metric_score(args.threshold_metric, metrics), metrics

    best_threshold = 0.5
    best_score = float("-inf")
    best_metrics: dict[str, object] | None = None
    for threshold in threshold_candidates(args).tolist():
        metrics = evaluate_subset_with_threshold(name=subset_name, outputs=outputs, threshold=float(threshold))
        metrics["min_within_epsilon_rate"] = float(args.min_within_epsilon_rate)
        metrics["savings_score_scale_ms"] = float(args.savings_score_scale_ms)
        score = metric_score(args.threshold_metric, metrics)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
            best_metrics = metrics
    if best_metrics is None:
        best_metrics = evaluate_subset_with_threshold(name=subset_name, outputs=outputs, threshold=best_threshold)
        best_metrics["min_within_epsilon_rate"] = float(args.min_within_epsilon_rate)
        best_metrics["savings_score_scale_ms"] = float(args.savings_score_scale_ms)
    return best_threshold, best_score, best_metrics


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise SystemExit("--batch-size and --eval-batch-size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise SystemExit("--gradient-accumulation-steps must be positive")
    if args.threshold_steps <= 0:
        raise SystemExit("--threshold-steps must be positive")
    if not 0.0 <= args.bucket_feature_dropout < 1.0:
        raise SystemExit("--bucket-feature-dropout must be in [0, 1)")
    if args.threshold_min <= 0 or args.threshold_max >= 1 or args.threshold_min >= args.threshold_max:
        raise SystemExit("--threshold-min/max must satisfy 0 < min < max < 1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    dataset_summary = read_dataset_summary(args.input_root)
    model, config, init_report = load_model(args, device)
    if args.freeze_encoder:
        for name, parameter in model.named_parameters():
            if not any(name.startswith(prefix) for prefix in (
                "throughput_mu_head.",
                "throughput_log_var_head.",
                "stop_head.",
                "final_mu_head.",
                "final_log_var_head.",
                "speed_head.",
            )):
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
            desc="foundation v1.5 eval val",
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
        score = metric_score(args.selection_metric, tuned_metrics)
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
            desc=f"foundation v1.5 final {subset}",
        )
        final_metrics[subset] = {
            "throughput": outputs["throughput"],
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
        "task": "foundation_v15_multitask_eps_10",
        "input_root": str(args.input_root),
        "pretrained_model_path": str(args.pretrained_model_path),
        "initialization": init_report,
        "model_config": config.to_dict(),
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
            "stop_bce": float(args.stop_bce_weight),
            "policy": float(args.policy_weight),
            "speed": float(args.speed_weight),
            "lambda_wrong": float(args.lambda_wrong),
            "lambda_wait": float(args.lambda_wait),
            "lambda_safe_stop": float(args.lambda_safe_stop),
        },
        "stop_target_source": args.stop_target_source,
        "policy_safety_source": args.policy_safety_source,
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
    model_path = args.output_root / "foundation_v15_multitask_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "speed_tiers": list(SPEED_TIERS),
            "summary": summary,
        },
        model_path,
    )
    summary_path = args.output_root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote foundation v1.5 model to {model_path}")
    print(f"wrote training summary to {summary_path}")


if __name__ == "__main__":
    main()
