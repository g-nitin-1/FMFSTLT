#!/usr/bin/env python3
"""Fine-tune the pretrained trace foundation encoder for throughput regression."""

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

from foundation_model import ThroughputRegressionModel, TraceFoundationConfig

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


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained trace foundation model for final throughput prediction."
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
        default=root_dir / "artifacts_exact_public" / "foundation_throughput_regressor_v1",
        help="Directory for fine-tuned model and metrics.",
    )
    parser.add_argument(
        "--train-subset",
        default="train",
        help="Logical training subset. For normalized shards this means stage1 split subset == train.",
    )
    parser.add_argument(
        "--eval-subsets",
        nargs="+",
        default=list(DEFAULT_EVAL_SUBSETS),
        help="Logical subsets evaluated after each/best epoch. Val is the 80k held-out UUID split.",
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help="Optional glob relative to each physical subset directory for probe runs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of supervised fine-tuning epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2048,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--target-transform",
        choices=("log1p", "raw"),
        default="log1p",
        help="Training target transform. Metrics are always reported back in Mbps.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Train only the regression head.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed.",
    )
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--max-train-shards",
        type=int,
        default=None,
        help="Optional train shard limit for probe runs.",
    )
    parser.add_argument(
        "--max-eval-shards",
        type=int,
        default=None,
        help="Optional eval shard limit for probe runs.",
    )
    parser.add_argument(
        "--max-train-batches-per-epoch",
        type=int,
        default=None,
        help="Optional train batch limit for probe runs.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Optional eval batch limit for probe runs.",
    )
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


def iter_batches(
    path: Path,
    *,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
    include_metadata: bool,
) -> Iterator[dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        x = data["x"].astype(np.float32, copy=False)
        bucket_mask = data["bucket_mask"].astype(bool, copy=False)
        y_true = data["y_true_mbps"].astype(np.float32, copy=False)
        speed_tier = data["speed_tier"] if include_metadata else None

        ordering = np.array(indices, copy=True)
        if shuffle:
            rng.shuffle(ordering)

        for start in range(0, ordering.shape[0], batch_size):
            batch_indices = ordering[start : start + batch_size]
            batch = {
                "x": x[batch_indices],
                "bucket_mask": bucket_mask[batch_indices],
                "y_true_mbps": y_true[batch_indices],
            }
            if speed_tier is not None:
                batch["speed_tier"] = speed_tier[batch_indices]
            yield batch


def target_from_mbps(y_true: torch.Tensor, target_transform: str) -> torch.Tensor:
    if target_transform == "log1p":
        return torch.log1p(torch.clamp(y_true, min=0.0))
    return y_true


def mbps_from_prediction(prediction: torch.Tensor, target_transform: str) -> torch.Tensor:
    if target_transform == "log1p":
        return torch.clamp(torch.expm1(prediction), min=0.0)
    return torch.clamp(prediction, min=0.0)


def init_metric_bucket() -> dict[str, float]:
    return {
        "count": 0.0,
        "squared_error_sum": 0.0,
        "absolute_error_sum": 0.0,
        "relative_error_sum": 0.0,
    }


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


def config_from_checkpoint(checkpoint: dict[str, object]) -> TraceFoundationConfig:
    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("pretrained checkpoint is missing model_config")
    allowed = {field.name for field in fields(TraceFoundationConfig)}
    config_kwargs = {key: raw_config[key] for key in allowed if key in raw_config}
    return TraceFoundationConfig(**config_kwargs)


def load_pretrained_model(path: Path, device: torch.device) -> tuple[ThroughputRegressionModel, TraceFoundationConfig]:
    checkpoint = torch.load(path, map_location=device)
    config = config_from_checkpoint(checkpoint)
    model = ThroughputRegressionModel(config).to(device)
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


def train_one_epoch(
    *,
    model: ThroughputRegressionModel,
    paths: list[Path],
    indices_by_path: dict[Path, np.ndarray],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    target_transform: str,
    clip_grad_norm: float,
    rng: np.random.Generator,
    max_batches: int | None,
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_batches = 0

    progress = tqdm(paths, desc="foundation throughput train", unit="shard")
    for path in progress:
        indices = indices_by_path[path]
        if indices.size == 0:
            continue
        for batch in iter_batches(
            path,
            indices=indices,
            batch_size=batch_size,
            shuffle=True,
            rng=rng,
            include_metadata=False,
        ):
            x = torch.as_tensor(batch["x"], dtype=torch.float32, device=device)
            bucket_mask = torch.as_tensor(batch["bucket_mask"], dtype=torch.bool, device=device)
            y_true = torch.as_tensor(batch["y_true_mbps"], dtype=torch.float32, device=device)
            target = target_from_mbps(y_true, target_transform)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, bucket_mask)
            loss = F.mse_loss(prediction, target)
            loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            examples = int(x.shape[0])
            total_loss += float(loss.item()) * examples
            total_examples += examples
            total_batches += 1

            if hasattr(progress, "set_postfix"):
                progress.set_postfix(loss=f"{total_loss / max(total_examples, 1):.4g}", batches=total_batches)

            if max_batches is not None and total_batches >= max_batches:
                progress.close()
                return {
                    "loss": float(total_loss / max(total_examples, 1)),
                    "batches": int(total_batches),
                    "examples": int(total_examples),
                }

    progress.close()
    return {
        "loss": float(total_loss / max(total_examples, 1)),
        "batches": int(total_batches),
        "examples": int(total_examples),
    }


@torch.no_grad()
def evaluate_subset(
    *,
    model: ThroughputRegressionModel,
    paths: list[Path],
    indices_by_path: dict[Path, np.ndarray],
    device: torch.device,
    batch_size: int,
    target_transform: str,
    max_batches: int | None,
    desc: str,
) -> dict[str, object]:
    model.eval()
    overall = init_metric_bucket()
    by_speed_tier: dict[str, dict[str, float]] = {}
    total_loss = 0.0
    total_examples = 0
    total_batches = 0
    rng = np.random.default_rng(0)

    progress = tqdm(paths, desc=desc, unit="shard")
    for path in progress:
        indices = indices_by_path[path]
        if indices.size == 0:
            continue
        for batch in iter_batches(
            path,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            rng=rng,
            include_metadata=True,
        ):
            x = torch.as_tensor(batch["x"], dtype=torch.float32, device=device)
            bucket_mask = torch.as_tensor(batch["bucket_mask"], dtype=torch.bool, device=device)
            y_true = torch.as_tensor(batch["y_true_mbps"], dtype=torch.float32, device=device)
            target = target_from_mbps(y_true, target_transform)
            prediction = model(x, bucket_mask)
            loss = F.mse_loss(prediction, target)
            y_pred_mbps = mbps_from_prediction(prediction, target_transform).detach().cpu().numpy().astype(np.float32)
            y_true_mbps = batch["y_true_mbps"].astype(np.float32, copy=False)

            update_metric_bucket(overall, y_true_mbps, y_pred_mbps)
            speed_tiers = batch["speed_tier"].tolist()
            speed_tiers_arr = np.array(speed_tiers, dtype=np.str_)
            for tier in sorted(set(speed_tiers)):
                mask = speed_tiers_arr == tier
                bucket = by_speed_tier.setdefault(tier, init_metric_bucket())
                update_metric_bucket(bucket, y_true_mbps[mask], y_pred_mbps[mask])

            examples = int(x.shape[0])
            total_loss += float(loss.item()) * examples
            total_examples += examples
            total_batches += 1

            if hasattr(progress, "set_postfix"):
                metrics = finalize_metric_bucket(overall)
                progress.set_postfix(mae=f"{metrics['mae']:.4g}", batches=total_batches)

            if max_batches is not None and total_batches >= max_batches:
                progress.close()
                return {
                    "loss": float(total_loss / max(total_examples, 1)),
                    "overall": finalize_metric_bucket(overall),
                    "by_speed_tier": {
                        tier: finalize_metric_bucket(bucket) for tier, bucket in sorted(by_speed_tier.items())
                    },
                    "batches": int(total_batches),
                }

    progress.close()
    return {
        "loss": float(total_loss / max(total_examples, 1)),
        "overall": finalize_metric_bucket(overall),
        "by_speed_tier": {
            tier: finalize_metric_bucket(bucket) for tier, bucket in sorted(by_speed_tier.items())
        },
        "batches": int(total_batches),
    }


def build_subset_indices(
    *,
    paths: list[Path],
    logical_subset: str,
    split_sets: dict[str, set[str]],
) -> dict[Path, np.ndarray]:
    uuid_filter = split_sets[logical_subset] if logical_subset in {"train", "val"} else None
    return {path: shard_indices(path, uuid_filter) for path in paths}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    split_sets = load_split_sets(args.split_path)

    model, config = load_pretrained_model(args.pretrained_model_path, device)
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
    best_val_mae = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epoch_history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            paths=train_paths,
            indices_by_path=train_indices,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
            target_transform=args.target_transform,
            clip_grad_norm=args.clip_grad_norm,
            rng=rng,
            max_batches=args.max_train_batches_per_epoch,
        )
        val_subset = "val" if "val" in eval_paths_by_subset else args.eval_subsets[0]
        val_metrics = evaluate_subset(
            model=model,
            paths=eval_paths_by_subset[val_subset],
            indices_by_path=eval_indices_by_subset[val_subset],
            device=device,
            batch_size=args.eval_batch_size,
            target_transform=args.target_transform,
            max_batches=args.max_eval_batches,
            desc=f"foundation throughput eval {val_subset}",
        )
        val_mae = float(val_metrics["overall"]["mae"])  # type: ignore[index]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        epoch_history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation_subset": val_subset,
                "validation": val_metrics,
            }
        )

    model.load_state_dict(best_state)
    final_metrics: dict[str, object] = {}
    for subset in args.eval_subsets:
        final_metrics[subset] = evaluate_subset(
            model=model,
            paths=eval_paths_by_subset[subset],
            indices_by_path=eval_indices_by_subset[subset],
            device=device,
            batch_size=args.eval_batch_size,
            target_transform=args.target_transform,
            max_batches=args.max_eval_batches,
            desc=f"foundation throughput final {subset}",
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": "foundation_throughput_regression",
        "pretrained_model_path": str(args.pretrained_model_path),
        "model_config": config.to_dict(),
        "target_transform": args.target_transform,
        "freeze_encoder": bool(args.freeze_encoder),
        "train_subset": args.train_subset,
        "eval_subsets": args.eval_subsets,
        "train_examples": int(train_examples),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "best_epoch": best_epoch,
        "best_val_mae": float(best_val_mae),
        "epoch_history": epoch_history,
    }
    model_path = args.output_root / "foundation_throughput_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "target_transform": args.target_transform,
            "summary": summary,
        },
        model_path,
    )
    training_summary_path = args.output_root / "training_summary.json"
    training_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    metrics_summary = {
        "model_path": str(model_path),
        "pretrained_model_path": str(args.pretrained_model_path),
        "best_epoch": best_epoch,
        "target_transform": args.target_transform,
        "overall_metrics": {
            subset: metrics["overall"] for subset, metrics in final_metrics.items()  # type: ignore[index]
        },
        "by_speed_tier": {
            subset: metrics["by_speed_tier"] for subset, metrics in final_metrics.items()  # type: ignore[index]
        },
    }
    metrics_summary_path = args.output_root / "metrics_summary.json"
    metrics_summary_path.write_text(json.dumps(metrics_summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote foundation throughput model to {model_path}")
    print(f"wrote training summary to {training_summary_path}")
    print(f"wrote metrics summary to {metrics_summary_path}")


if __name__ == "__main__":
    main()
