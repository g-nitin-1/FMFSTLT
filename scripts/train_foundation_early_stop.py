#!/usr/bin/env python3
"""Fine-tune the trace foundation encoder for epsilon=10 early stopping."""

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
from torch import nn

from foundation_model import EarlyStopFoundationModel, TraceFoundationConfig
from train_stage2_transformer import evaluate_subset_with_threshold, safe_divide

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

        def set_postfix(self, *args, **kwargs) -> None:
            return None

        def close(self) -> None:
            return None


DEFAULT_EVAL_SUBSETS = ("val", "test", "robustness")
WINDOW_METRIC_CHOICES = ("f1", "balanced_accuracy", "accuracy", "precision", "recall", "auroc", "average_precision")
POLICY_METRIC_CHOICES = (
    "policy_within_epsilon_rate",
    "policy_emitted_stop_rate",
    "policy_mean_savings_vs_full_ms",
    "policy_mean_stop_elapsed_ms",
    "policy_mean_stop_abs_error_mbps",
)
LOWER_IS_BETTER_METRICS = {"policy_mean_stop_elapsed_ms", "policy_mean_stop_abs_error_mbps"}


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Fine-tune the pretrained patch foundation encoder for frozen epsilon=10 stopping."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage2_transformer_dataset_eps_10",
        help="Root containing the valid frozen epsilon=10 Stage 2 dataset.",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=Path,
        default=root_dir
        / "artifacts_exact_public"
        / "foundation_pretrain_masked_patch_v1"
        / "foundation_pretrain_model.pt",
        help="Masked-patch pretrained foundation checkpoint.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "foundation_early_stop_eps_10_v1",
        help="Directory for the fine-tuned early-stop model and metrics.",
    )
    parser.add_argument("--train-subset", default="train", help="Training subset.")
    parser.add_argument("--val-subset", default="val", help="Validation subset for threshold and epoch selection.")
    parser.add_argument(
        "--eval-subsets",
        nargs="+",
        default=list(DEFAULT_EVAL_SUBSETS),
        help="Subsets evaluated after training.",
    )
    parser.add_argument("--input-glob", default=None, help="Optional glob relative to each subset directory.")
    parser.add_argument(
        "--target-field",
        choices=("stop_label", "is_oracle_stop_window"),
        default="stop_label",
        help="Decision target field stored in the Stage 2 dataset.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Physical mini-batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate physical mini-batches before each optimizer step.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=2048, help="Evaluation batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument("--pos-weight", type=float, default=None, help="Optional BCE positive class weight.")
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="If set, use this threshold instead of tuning on validation.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=WINDOW_METRIC_CHOICES + POLICY_METRIC_CHOICES,
        default="f1",
        help="Metric used to tune the stop threshold on validation.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=WINDOW_METRIC_CHOICES + POLICY_METRIC_CHOICES,
        default="f1",
        help="Validation window metric used to choose the best epoch.",
    )
    parser.add_argument("--threshold-min", type=float, default=0.05, help="Smallest tuned threshold.")
    parser.add_argument("--threshold-max", type=float, default=0.95, help="Largest tuned threshold.")
    parser.add_argument("--threshold-steps", type=int, default=19, help="Number of threshold candidates.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Train only the classifier head.")
    parser.add_argument(
        "--load-head",
        action="store_true",
        help=(
            "Also load matching head weights from the checkpoint. By default only the shared encoder is loaded, "
            "which is safer when initializing from throughput-regression checkpoints."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Training device.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Gradient clipping norm; <=0 disables.")
    parser.add_argument("--max-train-shards", type=int, default=None, help="Optional train shard limit.")
    parser.add_argument("--max-eval-shards", type=int, default=None, help="Optional eval shard limit.")
    parser.add_argument(
        "--max-train-batches-per-epoch",
        type=int,
        default=None,
        help="Optional optimizer-step limit per epoch.",
    )
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Optional eval batch limit.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_subset_paths(root: Path, subset: str, input_glob: str | None) -> list[Path]:
    subset_dir = root / subset
    if input_glob:
        paths = sorted(subset_dir.glob(input_glob))
    else:
        paths = sorted(subset_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no Stage 2 dataset shards found for subset {subset} under {subset_dir}")
    return paths


def maybe_limit(paths: list[Path], limit: int | None) -> list[Path]:
    if limit is None:
        return paths
    return paths[:limit]


def read_dataset_summary(input_root: Path) -> dict[str, object] | None:
    summary_path = input_root / "stage2_dataset_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


def compute_pos_weight(args: argparse.Namespace, dataset_summary: dict[str, object] | None) -> float:
    if args.pos_weight is not None:
        return float(args.pos_weight)
    if dataset_summary is None:
        return 1.0
    subset_summary = dataset_summary["subsets"][args.train_subset]
    total = int(subset_summary["decisions"])
    if args.target_field == "stop_label":
        positive = int(subset_summary["stop_positive_decisions"])
    else:
        positive = int(subset_summary["oracle_stop_decisions"])
    negative = total - positive
    if positive <= 0 or negative <= 0:
        return 1.0
    return float(negative / positive)


def config_from_checkpoint(checkpoint: dict[str, object]) -> TraceFoundationConfig:
    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("pretrained checkpoint is missing model_config")
    allowed = {field.name for field in fields(TraceFoundationConfig)}
    config_kwargs = {key: raw_config[key] for key in allowed if key in raw_config}
    return TraceFoundationConfig(**config_kwargs)


def load_pretrained_model(
    path: Path,
    device: torch.device,
    *,
    load_head: bool,
) -> tuple[EarlyStopFoundationModel, TraceFoundationConfig]:
    checkpoint = torch.load(path, map_location=device)
    config = config_from_checkpoint(checkpoint)
    model = EarlyStopFoundationModel(config).to(device)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("pretrained checkpoint is missing model_state_dict")
    if not load_head:
        state = {key: value for key, value in state.items() if not key.startswith("head.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_without_pretrain_head = [key for key in unexpected if not key.startswith("reconstruction_head.")]
    if unexpected_without_pretrain_head:
        raise ValueError(f"unexpected checkpoint keys: {unexpected_without_pretrain_head[:10]}")
    missing_without_head = [key for key in missing if not key.startswith("head.")]
    if missing_without_head:
        raise ValueError(f"missing encoder keys while loading checkpoint: {missing_without_head[:10]}")
    return model, config


def iter_decision_batches(
    path: Path,
    *,
    target_field: str,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
):
    with np.load(path, allow_pickle=False) as data:
        x_full = data["x_full"].astype(np.float32, copy=False)
        bucket_mask = data["bucket_mask"].astype(bool, copy=False)
        decision_valid_mask = data["decision_valid_mask"].astype(bool, copy=False)
        decision_end_bucket = data["decision_end_bucket"].astype(np.int16, copy=False)
        targets = data[target_field].astype(np.float32, copy=False)
        instantaneous_safe = data["instantaneous_safe_window"].astype(np.uint8, copy=False)
        stop_label = data["stop_label"].astype(np.uint8, copy=False)
        y_true = data["y_true_mbps"].astype(np.float32, copy=False)
        y_pred = data["y_pred_mbps"].astype(np.float32, copy=False)
        relative_error = data["relative_error"].astype(np.float32, copy=False)
        uuid = data["uuid"]
        test_time = data["test_time"]
        oracle_stop_found = data["oracle_stop_found"].astype(np.uint8, copy=False)
        oracle_stop_elapsed_ms = data["oracle_stop_elapsed_ms"].astype(np.int32, copy=False)

        test_indices, decision_indices = np.nonzero(decision_valid_mask)
        example_count = int(test_indices.shape[0])
        ordering = np.arange(example_count)
        if shuffle:
            rng.shuffle(ordering)

        positions = np.arange(x_full.shape[1], dtype=np.int16)

        for start in range(0, example_count, batch_size):
            batch_order = ordering[start : start + batch_size]
            batch_tests = test_indices[batch_order]
            batch_decisions = decision_indices[batch_order]
            batch_end_bucket = decision_end_bucket[batch_tests, batch_decisions]
            batch_history_lengths = batch_end_bucket.astype(np.int32) + 1
            prefix_bucket_mask = np.logical_and(
                bucket_mask[batch_tests],
                positions.reshape(1, -1) < batch_history_lengths.reshape(-1, 1),
            )

            yield {
                "x_full": x_full[batch_tests],
                "prefix_bucket_mask": prefix_bucket_mask,
                "labels": targets[batch_tests, batch_decisions],
                "instantaneous_safe_window": instantaneous_safe[batch_tests, batch_decisions],
                "stop_label": stop_label[batch_tests, batch_decisions],
                "uuid": uuid[batch_tests],
                "test_time": test_time[batch_tests],
                "end_bucket": batch_end_bucket.astype(np.int16, copy=False),
                "elapsed_ms": batch_history_lengths.astype(np.int32, copy=False) * 100,
                "y_true_mbps": y_true[batch_tests],
                "y_pred_mbps": y_pred[batch_tests, batch_decisions],
                "relative_error": relative_error[batch_tests, batch_decisions],
                "oracle_stop_found": oracle_stop_found[batch_tests],
                "oracle_stop_elapsed_ms": oracle_stop_elapsed_ms[batch_tests],
            }


def train_one_epoch(
    *,
    model: EarlyStopFoundationModel,
    paths: list[Path],
    target_field: str,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    gradient_accumulation_steps: int,
    rng: np.random.Generator,
    max_batches: int | None,
    clip_grad_norm: float,
) -> dict[str, float | int]:
    model.train()
    total_examples = 0
    total_loss = 0.0
    optimizer_step_count = 0
    microbatch_count = 0
    pending_microbatches = 0

    shuffled_paths = list(paths)
    rng.shuffle(shuffled_paths)
    progress = tqdm(shuffled_paths, desc="foundation early-stop train", unit="shard")
    stop_early = False
    optimizer.zero_grad(set_to_none=True)

    for path in progress:
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                loss=0.0 if total_examples == 0 else total_loss / total_examples,
                batches=optimizer_step_count,
            )
        for batch in iter_decision_batches(
            path,
            target_field=target_field,
            batch_size=batch_size,
            shuffle=True,
            rng=rng,
        ):
            x_full = torch.as_tensor(batch["x_full"], dtype=torch.float32, device=device)
            prefix_bucket_mask = torch.as_tensor(batch["prefix_bucket_mask"], dtype=torch.bool, device=device)
            targets = torch.as_tensor(batch["labels"], dtype=torch.float32, device=device)

            logits = model(x_full, prefix_bucket_mask)
            loss = criterion(logits, targets)
            (loss / gradient_accumulation_steps).backward()

            batch_size_current = int(targets.shape[0])
            total_examples += batch_size_current
            total_loss += float(loss.item()) * batch_size_current
            microbatch_count += 1
            pending_microbatches += 1

            if pending_microbatches >= gradient_accumulation_steps:
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_count += 1
                pending_microbatches = 0
                if max_batches is not None and optimizer_step_count >= max_batches:
                    stop_early = True
                    break
        if stop_early:
            break

    if pending_microbatches > 0 and (max_batches is None or optimizer_step_count < max_batches):
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step_count += 1

    progress.close()
    return {
        "loss": safe_divide(total_loss, total_examples),
        "examples": total_examples,
        "batches": optimizer_step_count,
        "microbatches": microbatch_count,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
    }


@torch.no_grad()
def predict_subset(
    *,
    model: EarlyStopFoundationModel,
    paths: list[Path],
    target_field: str,
    batch_size: int,
    device: torch.device,
    criterion: nn.Module,
    max_batches: int | None,
) -> dict[str, object]:
    model.eval()
    total_examples = 0
    total_loss = 0.0
    batch_count = 0

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

    progress = tqdm(paths, desc="foundation early-stop eval", unit="shard")
    stop_early = False
    for path in progress:
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                loss=0.0 if total_examples == 0 else total_loss / total_examples,
                batches=batch_count,
            )
        for batch in iter_decision_batches(
            path,
            target_field=target_field,
            batch_size=batch_size,
            shuffle=False,
            rng=np.random.default_rng(0),
        ):
            x_full = torch.as_tensor(batch["x_full"], dtype=torch.float32, device=device)
            prefix_bucket_mask = torch.as_tensor(batch["prefix_bucket_mask"], dtype=torch.bool, device=device)
            targets = torch.as_tensor(batch["labels"], dtype=torch.float32, device=device)

            logits = model(x_full, prefix_bucket_mask)
            loss = criterion(logits, targets)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False)

            probabilities_list.append(probs)
            labels_list.append(batch["labels"].astype(np.float32, copy=False))
            instantaneous_safe_list.append(batch["instantaneous_safe_window"].astype(np.uint8, copy=False))
            stop_label_list.append(batch["stop_label"].astype(np.uint8, copy=False))
            uuid_list.append(batch["uuid"])
            test_time_list.append(batch["test_time"])
            end_bucket_list.append(batch["end_bucket"].astype(np.int16, copy=False))
            elapsed_ms_list.append(batch["elapsed_ms"].astype(np.int32, copy=False))
            y_true_list.append(batch["y_true_mbps"].astype(np.float32, copy=False))
            y_pred_list.append(batch["y_pred_mbps"].astype(np.float32, copy=False))
            relative_error_list.append(batch["relative_error"].astype(np.float32, copy=False))
            oracle_stop_found_list.append(batch["oracle_stop_found"].astype(np.uint8, copy=False))
            oracle_stop_elapsed_ms_list.append(batch["oracle_stop_elapsed_ms"].astype(np.int32, copy=False))

            batch_size_current = int(targets.shape[0])
            total_examples += batch_size_current
            total_loss += float(loss.item()) * batch_size_current
            batch_count += 1

            if max_batches is not None and batch_count >= max_batches:
                stop_early = True
                break
        if stop_early:
            break

    progress.close()
    return {
        "loss": safe_divide(total_loss, total_examples),
        "examples": total_examples,
        "batches": batch_count,
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
    if metric_name.startswith("policy_"):
        raw_name = metric_name.removeprefix("policy_")
        value = subset_metrics["policy_metrics"].get(raw_name)
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


def choose_threshold_from_outputs(
    *,
    args: argparse.Namespace,
    outputs: dict[str, object],
    subset_name: str,
) -> tuple[float, float, dict[str, object]]:
    if args.decision_threshold is not None:
        threshold = float(args.decision_threshold)
        metrics = evaluate_subset_with_threshold(name=subset_name, outputs=outputs, threshold=threshold)
        return threshold, metric_score(args.threshold_metric, metrics), metrics

    best_threshold = 0.5
    best_score = float("-inf")
    best_metrics: dict[str, object] | None = None
    for threshold in threshold_candidates(args).tolist():
        metrics = evaluate_subset_with_threshold(name=subset_name, outputs=outputs, threshold=float(threshold))
        score = metric_score(args.threshold_metric, metrics)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
            best_metrics = metrics
    if best_metrics is None:
        best_metrics = evaluate_subset_with_threshold(name=subset_name, outputs=outputs, threshold=best_threshold)
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
    if args.threshold_min <= 0 or args.threshold_max >= 1 or args.threshold_min >= args.threshold_max:
        raise SystemExit("--threshold-min/max must satisfy 0 < min < max < 1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    dataset_summary = read_dataset_summary(args.input_root)
    model, config = load_pretrained_model(args.pretrained_model_path, device, load_head=args.load_head)
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    train_paths = maybe_limit(
        list_subset_paths(args.input_root, args.train_subset, args.input_glob),
        args.max_train_shards,
    )
    val_paths = maybe_limit(
        list_subset_paths(args.input_root, args.val_subset, args.input_glob),
        args.max_eval_shards,
    )
    final_eval_paths = {
        subset: maybe_limit(
            list_subset_paths(args.input_root, subset, args.input_glob),
            args.max_eval_shards,
        )
        for subset in args.eval_subsets
    }

    pos_weight = compute_pos_weight(args, dataset_summary)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
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
            target_field=args.target_field,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            rng=rng,
            max_batches=args.max_train_batches_per_epoch,
            clip_grad_norm=args.clip_grad_norm,
        )
        val_outputs = predict_subset(
            model=model,
            paths=val_paths,
            target_field=args.target_field,
            batch_size=args.eval_batch_size,
            device=device,
            criterion=criterion,
            max_batches=args.max_eval_batches,
        )
        threshold, threshold_score, val_metrics = choose_threshold_from_outputs(
            args=args,
            outputs=val_outputs,
            subset_name=args.val_subset,
        )
        score = metric_score(args.selection_metric, val_metrics)
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
                "threshold": threshold,
                "threshold_score": threshold_score,
                "validation": val_metrics,
                "selection_score": score,
            }
        )

    model.load_state_dict(best_state)
    final_metrics: dict[str, object] = {}
    for subset, paths in final_eval_paths.items():
        outputs = predict_subset(
            model=model,
            paths=paths,
            target_field=args.target_field,
            batch_size=args.eval_batch_size,
            device=device,
            criterion=criterion,
            max_batches=args.max_eval_batches,
        )
        final_metrics[subset] = evaluate_subset_with_threshold(
            name=subset,
            outputs=outputs,
            threshold=best_threshold,
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": "foundation_early_stop_eps_10",
        "input_root": str(args.input_root),
        "pretrained_model_path": str(args.pretrained_model_path),
        "model_config": config.to_dict(),
        "target_field": args.target_field,
        "prefix_alignment": "decision index d uses prefix patch count p=d+1 and stop_label[d]",
        "pos_weight": pos_weight,
        "freeze_encoder": bool(args.freeze_encoder),
        "load_head": bool(args.load_head),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "effective_batch_size": int(args.batch_size * args.gradient_accumulation_steps),
        "eval_batch_size": int(args.eval_batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "threshold_metric": args.threshold_metric,
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "best_threshold_score": best_threshold_score,
        "best_selection_score": best_score,
        "epoch_history": epoch_history,
        "final_metrics": final_metrics,
    }
    model_path = args.output_root / "foundation_early_stop_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "summary": summary,
        },
        model_path,
    )
    summary_path = args.output_root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote foundation early-stop model to {model_path}")
    print(f"wrote training summary to {summary_path}")


if __name__ == "__main__":
    main()
