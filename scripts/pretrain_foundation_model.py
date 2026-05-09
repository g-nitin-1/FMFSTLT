#!/usr/bin/env python3
"""Pretrain the patch-based trace foundation model with masked reconstruction."""

from __future__ import annotations

import argparse
import json
import os
import random
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

from foundation_model import MaskedPatchReconstructionModel, TraceFoundationConfig, patchify_traces

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


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain the proposed patch-based foundation model using Objective A: "
            "masked-patch reconstruction on normalized train UUIDs only."
        )
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
        help="UUID-level Stage 1 split used to keep pretraining to train UUIDs only.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "foundation_pretrain_masked_patch_v1",
        help="Directory for the pretrained checkpoint and training summary.",
    )
    parser.add_argument(
        "--source-subset",
        default="train",
        help="Normalized shard subset to read. Keep this as train for leakage-safe pretraining.",
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help="Optional glob relative to the source subset directory for probe runs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of pretraining epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.5,
        help="Fraction of valid patches masked per trace. At least one valid patch remains visible.",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=13,
        help="Per-bucket feature dimension.",
    )
    parser.add_argument(
        "--max-sequence-buckets",
        type=int,
        default=100,
        help="Number of 100 ms buckets per trace.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=5,
        help="Buckets per foundation-model patch.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Transformer hidden size.",
    )
    parser.add_argument(
        "--patch-hidden-dim",
        type=int,
        default=128,
        help="Hidden size used by patch embedding and reconstruction MLPs.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Transformer attention heads.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=8,
        help="Number of Transformer encoder layers.",
    )
    parser.add_argument(
        "--ff-dim",
        type=int,
        default=1024,
        help="Transformer feed-forward dimension.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout in the Transformer encoder.",
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
        help="Optional limit on train shards for a probe run.",
    )
    parser.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=None,
        help="Optional limit on batches per epoch for a probe run.",
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


def list_source_paths(input_root: Path, source_subset: str, input_glob: str | None) -> list[Path]:
    subset_dir = input_root / source_subset
    if input_glob:
        paths = sorted(subset_dir.glob(input_glob))
    else:
        paths = sorted(subset_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no normalized shards found for subset {source_subset} under {subset_dir}")
    return paths


def load_train_uuid_set(split_path: Path) -> set[str]:
    with np.load(split_path, allow_pickle=False) as data:
        uuid = data["uuid"]
        subset = data["subset"]
        if uuid.shape != subset.shape:
            raise ValueError(f"uuid and subset arrays disagree in {split_path}: {uuid.shape} vs {subset.shape}")
        train_mask = subset == "train"
        return set(uuid[train_mask].tolist())


def shard_train_indices(path: Path, train_uuids: set[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        uuid = data["uuid"]
        keep = np.fromiter((str(item) in train_uuids for item in uuid.tolist()), dtype=bool, count=len(uuid))
    return np.flatnonzero(keep)


def iter_masked_pretrain_batches(
    path: Path,
    *,
    train_indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> Iterator[dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        x = data["x"].astype(np.float32, copy=False)
        bucket_mask = data["bucket_mask"].astype(bool, copy=False)

        ordering = np.array(train_indices, copy=True)
        if shuffle:
            rng.shuffle(ordering)

        for start in range(0, ordering.shape[0], batch_size):
            indices = ordering[start : start + batch_size]
            yield {
                "x": x[indices],
                "bucket_mask": bucket_mask[indices],
            }


def sample_masked_patch_mask(patch_valid: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0,1), got {mask_ratio}")

    masked = torch.zeros_like(patch_valid, dtype=torch.bool)
    for row_idx in range(patch_valid.shape[0]):
        valid_indices = torch.nonzero(patch_valid[row_idx], as_tuple=False).flatten()
        valid_count = int(valid_indices.numel())
        if valid_count <= 1:
            continue
        mask_count = int(round(valid_count * mask_ratio))
        mask_count = max(1, min(mask_count, valid_count - 1))
        permutation = torch.randperm(valid_count, device=patch_valid.device)
        selected = valid_indices[permutation[:mask_count]]
        masked[row_idx, selected] = True
    return masked


def build_config(args: argparse.Namespace) -> TraceFoundationConfig:
    return TraceFoundationConfig(
        feature_dim=args.feature_dim,
        max_sequence_buckets=args.max_sequence_buckets,
        patch_size=args.patch_size,
        d_model=args.d_model,
        patch_hidden_dim=args.patch_hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )


def train_one_epoch(
    *,
    model: MaskedPatchReconstructionModel,
    paths: list[Path],
    train_indices_by_path: dict[Path, np.ndarray],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    mask_ratio: float,
    clip_grad_norm: float,
    rng: np.random.Generator,
    max_batches: int | None,
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    total_masked_patches = 0
    total_batches = 0
    total_examples = 0

    progress = tqdm(paths, desc="foundation pretrain", unit="shard")
    for path in progress:
        train_indices = train_indices_by_path[path]
        if train_indices.size == 0:
            continue

        for batch in iter_masked_pretrain_batches(
            path,
            train_indices=train_indices,
            batch_size=batch_size,
            shuffle=True,
            rng=rng,
        ):
            x = torch.as_tensor(batch["x"], dtype=torch.float32, device=device)
            bucket_mask = torch.as_tensor(batch["bucket_mask"], dtype=torch.bool, device=device)

            with torch.no_grad():
                target_patches, patch_valid = patchify_traces(
                    x,
                    bucket_mask,
                    patch_size=model.config.patch_size,
                )
                masked_patch_mask = sample_masked_patch_mask(patch_valid, mask_ratio)

            if not bool(masked_patch_mask.any()):
                continue

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x, bucket_mask, masked_patch_mask=masked_patch_mask)
            prediction = outputs["reconstruction"]
            loss = F.mse_loss(prediction[masked_patch_mask], target_patches[masked_patch_mask])
            loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            masked_count = int(masked_patch_mask.sum().item())
            total_loss += float(loss.item()) * masked_count
            total_masked_patches += masked_count
            total_batches += 1
            total_examples += int(x.shape[0])

            if hasattr(progress, "set_postfix"):
                mean_loss = total_loss / max(total_masked_patches, 1)
                progress.set_postfix(loss=f"{mean_loss:.4g}", batches=total_batches)

            if max_batches is not None and total_batches >= max_batches:
                progress.close()
                mean_loss = total_loss / max(total_masked_patches, 1)
                return {
                    "loss": float(mean_loss),
                    "batches": int(total_batches),
                    "examples": int(total_examples),
                    "masked_patches": int(total_masked_patches),
                }

    progress.close()
    mean_loss = total_loss / max(total_masked_patches, 1)
    return {
        "loss": float(mean_loss),
        "batches": int(total_batches),
        "examples": int(total_examples),
        "masked_patches": int(total_masked_patches),
    }


def save_outputs(
    *,
    args: argparse.Namespace,
    model: MaskedPatchReconstructionModel,
    output_root: Path,
    config: TraceFoundationConfig,
    train_paths: list[Path],
    train_indices_by_path: dict[Path, np.ndarray],
    epoch_history: list[dict[str, float | int]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": "masked_patch_reconstruction_pretraining",
        "model_config": config.to_dict(),
        "input_root": str(args.input_root),
        "source_subset": args.source_subset,
        "split_path": str(args.split_path),
        "train_uuid_scope": "stage1_uuid_split subset == train",
        "mask_ratio": float(args.mask_ratio),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "train_shards": len(train_paths),
        "train_examples_after_uuid_filter": int(
            sum(int(indices.size) for indices in train_indices_by_path.values())
        ),
        "epoch_history": epoch_history,
    }

    model_path = output_root / "foundation_pretrain_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "summary": summary,
        },
        model_path,
    )
    summary_path = output_root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote foundation model checkpoint to {model_path}")
    print(f"wrote foundation pretraining summary to {summary_path}")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    train_uuids = load_train_uuid_set(args.split_path)
    paths = list_source_paths(args.input_root, args.source_subset, args.input_glob)
    if args.max_train_shards is not None:
        paths = paths[: args.max_train_shards]

    train_indices_by_path: dict[Path, np.ndarray] = {}
    for path in paths:
        train_indices_by_path[path] = shard_train_indices(path, train_uuids)

    kept_examples = sum(int(indices.size) for indices in train_indices_by_path.values())
    if kept_examples <= 0:
        raise SystemExit("no train UUID examples found in selected normalized shards")

    config = build_config(args)
    model = MaskedPatchReconstructionModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed)

    epoch_history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            model=model,
            paths=paths,
            train_indices_by_path=train_indices_by_path,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
            mask_ratio=args.mask_ratio,
            clip_grad_norm=args.clip_grad_norm,
            rng=rng,
            max_batches=args.max_batches_per_epoch,
        )
        metrics["epoch"] = epoch
        epoch_history.append(metrics)

    save_outputs(
        args=args,
        model=model,
        output_root=args.output_root,
        config=config,
        train_paths=paths,
        train_indices_by_path=train_indices_by_path,
        epoch_history=epoch_history,
    )


if __name__ == "__main__":
    main()
