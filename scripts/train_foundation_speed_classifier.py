#!/usr/bin/env python3
"""Fine-tune the trace foundation encoder for speed-tier classification."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import fields
from pathlib import Path
from typing import Iterator

import numpy as np

if "TMPDIR" not in os.environ:
    for candidate in ("/dev/shm", "/tmp", "/var/tmp", "/usr/tmp"):
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            os.environ["TMPDIR"] = candidate
            break

import torch
import torch.nn.functional as F

from foundation_model import SpeedTierClassificationModel, TraceFoundationConfig

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
DEFAULT_SPEED_TIERS = ("0-25", "25-100", "100-200", "200-400", "400+")


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Fine-tune the pretrained trace foundation model for speed-tier classification."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "normalized_shards",
        help="Root directory containing normalized shard split subdirectories.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage1_uuid_split.npz",
        help="UUID-level train/validation split used for normalized train shards.",
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
        default=root_dir / "artifacts_exact_public" / "foundation_speed_classifier_v1",
        help="Directory for fine-tuned model and metrics.",
    )
    parser.add_argument("--train-subset", default="train", help="Logical training subset.")
    parser.add_argument(
        "--eval-subsets",
        nargs="+",
        default=list(DEFAULT_EVAL_SUBSETS),
        help="Logical subsets evaluated after training.",
    )
    parser.add_argument("--input-glob", default=None, help="Optional glob relative to each physical subset directory.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of supervised fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Mini-batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=2048, help="Evaluation batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "macro_f1"),
        default="macro_f1",
        help="Validation metric used to choose the best epoch.",
    )
    parser.add_argument("--freeze-encoder", action="store_true", help="Train only the classification head.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Training device.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Gradient clipping norm; <=0 disables.")
    parser.add_argument("--max-train-shards", type=int, default=None, help="Optional train shard limit.")
    parser.add_argument("--max-eval-shards", type=int, default=None, help="Optional eval shard limit.")
    parser.add_argument("--max-train-batches-per-epoch", type=int, default=None, help="Optional train batch limit.")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Optional eval batch limit.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split_sets(split_path: Path) -> dict[str, set[str]]:
    with np.load(split_path, allow_pickle=False) as data:
        uuid = data["uuid"]
        subset = data["subset"]
        if uuid.shape != subset.shape:
            raise ValueError(f"uuid and subset arrays disagree in {split_path}: {uuid.shape} vs {subset.shape}")
        return {
            "train": set(uuid[subset == "train"].tolist()),
            "val": set(uuid[subset == "val"].tolist()),
        }


def physical_subset(logical_subset: str) -> str:
    if logical_subset in {"train", "val"}:
        return "train"
    return logical_subset


def list_subset_paths(input_root: Path, logical_subset: str, input_glob: str | None) -> list[Path]:
    subset_dir = input_root / physical_subset(logical_subset)
    if input_glob:
        paths = sorted(subset_dir.glob(input_glob))
    else:
        paths = sorted(subset_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no normalized shards found for logical subset {logical_subset} under {subset_dir}")
    return paths


def shard_indices(path: Path, uuid_filter: set[str] | None) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if uuid_filter is None:
            return np.arange(int(data["x"].shape[0]), dtype=np.int64)
        uuid = data["uuid"]
        keep = np.fromiter((str(item) in uuid_filter for item in uuid.tolist()), dtype=bool, count=len(uuid))
    return np.flatnonzero(keep)


def build_subset_indices(
    *,
    paths: list[Path],
    logical_subset: str,
    split_sets: dict[str, set[str]],
) -> dict[Path, np.ndarray]:
    uuid_filter = split_sets[logical_subset] if logical_subset in {"train", "val"} else None
    return {path: shard_indices(path, uuid_filter) for path in paths}


def iter_batches(
    path: Path,
    *,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> Iterator[dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        x = data["x"].astype(np.float32, copy=False)
        bucket_mask = data["bucket_mask"].astype(bool, copy=False)
        speed_tier = data["speed_tier"]

        ordering = np.array(indices, copy=True)
        if shuffle:
            rng.shuffle(ordering)

        for start in range(0, ordering.shape[0], batch_size):
            batch_indices = ordering[start : start + batch_size]
            yield {
                "x": x[batch_indices],
                "bucket_mask": bucket_mask[batch_indices],
                "speed_tier": speed_tier[batch_indices],
            }


def labels_from_tiers(speed_tier: np.ndarray, tier_to_label: dict[str, int]) -> np.ndarray:
    return np.array([tier_to_label[str(item)] for item in speed_tier.tolist()], dtype=np.int64)


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
    num_classes: int,
) -> tuple[SpeedTierClassificationModel, TraceFoundationConfig]:
    checkpoint = torch.load(path, map_location=device)
    config = config_from_checkpoint(checkpoint)
    model = SpeedTierClassificationModel(config, num_classes=num_classes).to(device)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("pretrained checkpoint is missing model_state_dict")
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_without_pretrain_head = [key for key in unexpected if not key.startswith("reconstruction_head.")]
    if unexpected_without_pretrain_head:
        raise ValueError(f"unexpected checkpoint keys: {unexpected_without_pretrain_head[:10]}")
    missing_without_head = [key for key in missing if not key.startswith("head.")]
    if missing_without_head:
        raise ValueError(f"missing encoder keys while loading checkpoint: {missing_without_head[:10]}")
    return model, config


def init_confusion(num_classes: int) -> np.ndarray:
    return np.zeros((num_classes, num_classes), dtype=np.int64)


def update_confusion(confusion: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    for true_label, pred_label in zip(y_true.tolist(), y_pred.tolist()):
        confusion[int(true_label), int(pred_label)] += 1


def compute_metrics(confusion: np.ndarray, label_to_tier: list[str]) -> dict[str, object]:
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = float(correct / total) if total else 0.0
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for idx, tier in enumerate(label_to_tier):
        tp = int(confusion[idx, idx])
        fp = int(confusion[:, idx].sum() - tp)
        fn = int(confusion[idx, :].sum() - tp)
        support = int(confusion[idx, :].sum())
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[tier] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "count": total,
        "accuracy": accuracy,
        "macro_f1": float(np.mean(np.asarray(f1_values, dtype=np.float64))) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "label_order": label_to_tier,
    }


def train_one_epoch(
    *,
    model: SpeedTierClassificationModel,
    paths: list[Path],
    indices_by_path: dict[Path, np.ndarray],
    tier_to_label: dict[str, int],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    clip_grad_norm: float,
    rng: np.random.Generator,
    max_batches: int | None,
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_batches = 0

    progress = tqdm(paths, desc="foundation speed train", unit="shard")
    for path in progress:
        indices = indices_by_path[path]
        if indices.size == 0:
            continue
        for batch in iter_batches(path, indices=indices, batch_size=batch_size, shuffle=True, rng=rng):
            labels_np = labels_from_tiers(batch["speed_tier"], tier_to_label)
            x = torch.as_tensor(batch["x"], dtype=torch.float32, device=device)
            bucket_mask = torch.as_tensor(batch["bucket_mask"], dtype=torch.bool, device=device)
            labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, bucket_mask)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            examples = int(labels.shape[0])
            total_loss += float(loss.item()) * examples
            total_examples += examples
            total_batches += 1
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(loss=f"{total_loss / max(total_examples, 1):.4g}", batches=total_batches)
            if max_batches is not None and total_batches >= max_batches:
                progress.close()
                return {
                    "loss": float(total_loss / max(total_examples, 1)),
                    "examples": int(total_examples),
                    "batches": int(total_batches),
                }
    progress.close()
    return {
        "loss": float(total_loss / max(total_examples, 1)),
        "examples": int(total_examples),
        "batches": int(total_batches),
    }


@torch.no_grad()
def evaluate_subset(
    *,
    model: SpeedTierClassificationModel,
    paths: list[Path],
    indices_by_path: dict[Path, np.ndarray],
    tier_to_label: dict[str, int],
    label_to_tier: list[str],
    device: torch.device,
    batch_size: int,
    max_batches: int | None,
    desc: str,
) -> dict[str, object]:
    model.eval()
    confusion = init_confusion(len(label_to_tier))
    total_loss = 0.0
    total_examples = 0
    total_batches = 0
    rng = np.random.default_rng(0)

    progress = tqdm(paths, desc=desc, unit="shard")
    for path in progress:
        indices = indices_by_path[path]
        if indices.size == 0:
            continue
        for batch in iter_batches(path, indices=indices, batch_size=batch_size, shuffle=False, rng=rng):
            labels_np = labels_from_tiers(batch["speed_tier"], tier_to_label)
            x = torch.as_tensor(batch["x"], dtype=torch.float32, device=device)
            bucket_mask = torch.as_tensor(batch["bucket_mask"], dtype=torch.bool, device=device)
            labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)

            logits = model(x, bucket_mask)
            loss = F.cross_entropy(logits, labels)
            pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
            update_confusion(confusion, labels_np, pred)

            examples = int(labels.shape[0])
            total_loss += float(loss.item()) * examples
            total_examples += examples
            total_batches += 1
            if hasattr(progress, "set_postfix"):
                metrics = compute_metrics(confusion, label_to_tier)
                progress.set_postfix(acc=f"{metrics['accuracy']:.4g}", batches=total_batches)
            if max_batches is not None and total_batches >= max_batches:
                progress.close()
                metrics = compute_metrics(confusion, label_to_tier)
                metrics["loss"] = float(total_loss / max(total_examples, 1))
                metrics["batches"] = int(total_batches)
                return metrics

    progress.close()
    metrics = compute_metrics(confusion, label_to_tier)
    metrics["loss"] = float(total_loss / max(total_examples, 1))
    metrics["batches"] = int(total_batches)
    return metrics


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    split_sets = load_split_sets(args.split_path)

    label_to_tier = list(DEFAULT_SPEED_TIERS)
    tier_to_label = {tier: idx for idx, tier in enumerate(label_to_tier)}
    model, config = load_pretrained_model(args.pretrained_model_path, device, num_classes=len(label_to_tier))
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    train_paths = list_subset_paths(args.input_root, args.train_subset, args.input_glob)
    if args.max_train_shards is not None:
        train_paths = train_paths[: args.max_train_shards]
    train_indices = build_subset_indices(paths=train_paths, logical_subset=args.train_subset, split_sets=split_sets)
    train_examples = sum(int(indices.size) for indices in train_indices.values())
    if train_examples <= 0:
        raise SystemExit("no training examples found after UUID filtering")

    eval_paths_by_subset: dict[str, list[Path]] = {}
    eval_indices_by_subset: dict[str, dict[Path, np.ndarray]] = {}
    for subset in args.eval_subsets:
        paths = list_subset_paths(args.input_root, subset, args.input_glob)
        if args.max_eval_shards is not None:
            paths = paths[: args.max_eval_shards]
        eval_paths_by_subset[subset] = paths
        eval_indices_by_subset[subset] = build_subset_indices(paths=paths, logical_subset=subset, split_sets=split_sets)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed)

    best_epoch = None
    best_score = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    epoch_history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            paths=train_paths,
            indices_by_path=train_indices,
            tier_to_label=tier_to_label,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
            clip_grad_norm=args.clip_grad_norm,
            rng=rng,
            max_batches=args.max_train_batches_per_epoch,
        )
        val_subset = "val" if "val" in eval_paths_by_subset else args.eval_subsets[0]
        val_metrics = evaluate_subset(
            model=model,
            paths=eval_paths_by_subset[val_subset],
            indices_by_path=eval_indices_by_subset[val_subset],
            tier_to_label=tier_to_label,
            label_to_tier=label_to_tier,
            device=device,
            batch_size=args.eval_batch_size,
            max_batches=args.max_eval_batches,
            desc=f"foundation speed eval {val_subset}",
        )
        score = float(val_metrics[args.selection_metric])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        epoch_history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation_subset": val_subset,
                "validation": val_metrics,
                "selection_score": score,
            }
        )

    model.load_state_dict(best_state)
    final_metrics: dict[str, object] = {}
    for subset in args.eval_subsets:
        final_metrics[subset] = evaluate_subset(
            model=model,
            paths=eval_paths_by_subset[subset],
            indices_by_path=eval_indices_by_subset[subset],
            tier_to_label=tier_to_label,
            label_to_tier=label_to_tier,
            device=device,
            batch_size=args.eval_batch_size,
            max_batches=args.max_eval_batches,
            desc=f"foundation speed final {subset}",
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": "foundation_speed_tier_classification",
        "pretrained_model_path": str(args.pretrained_model_path),
        "model_config": config.to_dict(),
        "label_order": label_to_tier,
        "freeze_encoder": bool(args.freeze_encoder),
        "train_subset": args.train_subset,
        "eval_subsets": args.eval_subsets,
        "train_examples": int(train_examples),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "best_selection_score": float(best_score),
        "epoch_history": epoch_history,
        "final_metrics": final_metrics,
    }
    model_path = args.output_root / "foundation_speed_classifier_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "label_order": label_to_tier,
            "summary": summary,
        },
        model_path,
    )
    summary_path = args.output_root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote foundation speed classifier model to {model_path}")
    print(f"wrote training summary to {summary_path}")


if __name__ == "__main__":
    main()
