#!/usr/bin/env python3
"""Causal next-bucket pretraining for FMNet v3.

Self-supervised objective: at each position t, predict bucket t+1 from
buckets 0..t.  Loss is masked Smooth-L1 over the 13 normalized features.

Restrictions:
  - Trains only on the 720k Stage 1 train UUIDs (no val/test/robustness leakage).
  - Reads normalized shards directly: no Stage 2 dataset / no XGBoost dependency.
  - Saves a single checkpoint with encoder weights compatible with FMNet v3
    fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np

if "TMPDIR" not in os.environ:
    for candidate in ("/dev/shm", "/tmp", "/var/tmp", "/usr/tmp"):
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            os.environ["TMPDIR"] = candidate
            break

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from fmfstlt.models.fmnet_v3 import FMNetV3, FMNetV3Config

try:
    from tqdm.auto import tqdm
except ImportError:

    class tqdm:  # type: ignore
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())

        def set_postfix(self, *a, **k):
            pass

        def close(self):
            pass


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    artifacts = root / "artifacts_exact_public"
    p = argparse.ArgumentParser(description="Causal next-bucket pretraining for FMNet v3")
    p.add_argument("--normalized-root", type=Path, default=artifacts / "normalized_shards")
    p.add_argument("--split-path", type=Path, default=artifacts / "stage1_uuid_split.npz")
    p.add_argument("--output-root", type=Path, default=artifacts / "foundation_v3_pretrain")
    p.add_argument("--source-splits", nargs="+", default=["train"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--feature-dropout",
        type=float,
        default=0.05,
        help="Probability of zeroing a bucket vector before encoding (regularization).",
    )
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-train-shards", type=int, default=None)
    p.add_argument("--max-train-batches-per-epoch", type=int, default=None)
    # model config
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--ff-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.15)
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda but CUDA not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_uuid_set(split_path: Path) -> set[str]:
    with np.load(split_path, allow_pickle=False) as data:
        uuids = data["uuid"].tolist()
        subsets = data["subset"].tolist()
    return {u for u, s in zip(uuids, subsets, strict=True) if s == "train"}


def list_shards(normalized_root: Path, splits: list[str]) -> list[Path]:
    paths: list[Path] = []
    for split in splits:
        d = normalized_root / split
        for p in sorted(d.glob("*.npz")):
            paths.append(p)
    if not paths:
        raise SystemExit(f"no normalized shards found under {normalized_root}")
    return paths


class NormalizedShardIterable(IterableDataset):
    """Iterable dataset that streams (x, mask) per train UUID across shards."""

    def __init__(
        self,
        shard_paths: list[Path],
        train_uuids: set[str],
        feature_dropout: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.shard_paths = list(shard_paths)
        self.train_uuids = train_uuids
        self.feature_dropout = feature_dropout
        self.seed = seed

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            paths = list(self.shard_paths)
            rng_seed = self.seed
        else:
            paths = self.shard_paths[worker.id :: worker.num_workers]
            rng_seed = self.seed + worker.id
        rng = np.random.default_rng(rng_seed)
        rng.shuffle(paths)
        for path in paths:
            with np.load(path, allow_pickle=False) as d:
                uuids = d["uuid"]
                x = d["x"].astype(np.float32, copy=False)
                bm = d["bucket_mask"].astype(np.uint8, copy=False)
            order = np.arange(x.shape[0])
            rng.shuffle(order)
            for i in order:
                u = str(uuids[i])
                if u not in self.train_uuids:
                    continue
                xi = x[i]
                bmi = bm[i]
                if self.feature_dropout > 0:
                    keep = rng.random(xi.shape[0]) >= self.feature_dropout
                    keep_mask = keep.astype(np.float32).reshape(-1, 1)
                    xi = xi * keep_mask
                yield xi, bmi


def build_loader(args: argparse.Namespace) -> DataLoader:
    train_uuids = load_train_uuid_set(args.split_path)
    paths = list_shards(args.normalized_root, args.source_splits)
    if args.max_train_shards is not None:
        paths = paths[: args.max_train_shards]
    print(f"[pretrain] {len(paths)} shards, {len(train_uuids):,} train UUIDs")
    ds = NormalizedShardIterable(
        shard_paths=paths,
        train_uuids=train_uuids,
        feature_dropout=args.feature_dropout,
        seed=args.seed,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def lr_at_step(step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_lr
    if step >= warmup_steps:
        return base_lr
    return base_lr * (step + 1) / warmup_steps


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    args.output_root.mkdir(parents=True, exist_ok=True)
    cfg = FMNetV3Config(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    model = FMNetV3(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] FMNet v3 with {n_params / 1e6:.2f}M parameters on {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)

    loader = build_loader(args)

    history = []
    global_step = 0
    best_loss = float("inf")
    best_path = args.output_root / "fmnet_v3_pretrain.pt"
    log_path = args.output_root / "pretrain_log.jsonl"
    log_f = open(log_path, "w")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0
        bar = tqdm(loader, desc=f"pretrain ep{epoch}", unit="batch", dynamic_ncols=True)
        for batch in bar:
            x_np, bm_np = batch
            x = x_np.to(device=device, dtype=torch.float32, non_blocking=True)
            bm = bm_np.to(device=device, dtype=torch.float32, non_blocking=True)

            for g in optimizer.param_groups:
                g["lr"] = lr_at_step(global_step, args.learning_rate, args.warmup_steps)

            out = model.forward_pretrain(x)
            pred = out["next_bucket_pred"][:, :-1, :]  # [B, T-1, F]
            target = x[:, 1:, :]  # [B, T-1, F]
            target_mask = bm[:, 1:].unsqueeze(-1)  # [B, T-1, 1]
            loss_per_elem = criterion(pred, target) * target_mask
            denom = target_mask.sum() * pred.shape[-1]
            loss = loss_per_elem.sum() / denom.clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            epoch_loss_sum += float(loss.item())
            epoch_count += 1
            global_step += 1
            if epoch_count % 50 == 0:
                bar.set_postfix(
                    loss=f"{epoch_loss_sum / max(1, epoch_count):.4f}", step=global_step
                )
            if (
                args.max_train_batches_per_epoch is not None
                and epoch_count >= args.max_train_batches_per_epoch
            ):
                break
        bar.close()
        avg = epoch_loss_sum / max(1, epoch_count)
        elapsed = time.time() - t0
        rec = {
            "epoch": epoch,
            "avg_loss": avg,
            "batches": epoch_count,
            "elapsed_s": elapsed,
            "global_step": global_step,
        }
        print(
            f"[pretrain] epoch {epoch}: avg_loss={avg:.4f}, batches={epoch_count}, "
            f"time={elapsed:.0f}s"
        )
        history.append(rec)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if avg < best_loss:
            best_loss = avg
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.to_dict(),
                    "epoch": epoch,
                    "loss": avg,
                    "tag": "fmnet_v3_pretrain",
                },
                best_path,
            )
            print(f"[pretrain] saved best to {best_path} (loss={avg:.4f})")

    log_f.close()
    summary = {
        "output_root": str(args.output_root),
        "model_path": str(best_path),
        "config": cfg.to_dict(),
        "epochs": args.epochs,
        "final_loss": history[-1]["avg_loss"] if history else None,
        "best_loss": best_loss,
        "history": history,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (args.output_root / "pretrain_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[pretrain] done. best_loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
