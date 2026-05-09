#!/usr/bin/env python3
"""Three-phase training for FMNetTwoStage.

Phase 1 (encoder + Stage 1 head):
  - trains on prefix throughput MSE + final throughput MSE
  - selection by validation prefix_throughput_mae
  - Stage 2 policy module is unused

Phase 2 (Stage 2 policy module on detached Stage 1 outputs):
  - encoder + Stage 1 head are frozen
  - Stage 1 outputs (and h_decision if richer variant) are detached at the
    Stage 2 input boundary
  - trains stop BCE only
  - selection by val policy_constrained_savings (within-eps >= --min-within-epsilon-rate)

Phase 3 (optional end-to-end fine-tune):
  - only runs if Phase 2 hits a competitive bar:
      val within-eps >= --phase3-gate-within-eps AND
      val mean savings >= --phase3-gate-savings-ms
  - everything unfrozen, very small LR
  - combined loss (small Stage 1 weight, full Stage 2 weight)

Final eval at end:
  - threshold sweep on val/test/robustness
  - report at best threshold (selected on val) and full sweep
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np

if "TMPDIR" not in os.environ:
    for c in ("/dev/shm", "/tmp", "/var/tmp", "/usr/tmp"):
        if os.path.isdir(c) and os.access(c, os.W_OK):
            os.environ["TMPDIR"] = c
            break

import torch
from torch import nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from foundation_model_twostage import (
    FMNetTwoStage, FMNetTwoStageConfig, FMNetV3Config, Stage2PolicyConfig,
    SPEED_TIER_TO_INDEX, log1p_mbps, expm1_mbps,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())
        def set_postfix(self, *a, **k): pass
        def close(self): pass


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    artifacts = root / "artifacts_exact_public"
    p = argparse.ArgumentParser(description="Three-phase training for FMNetTwoStage")
    p.add_argument("--input-root", type=Path,
                   default=artifacts / "stage2_transformer_dataset_eps_10")
    p.add_argument("--pretrained-encoder", type=Path, default=None,
                   help="Optional FMNet v3 pretraining checkpoint (loads encoder weights).")
    p.add_argument("--output-root", type=Path,
                   default=artifacts / "foundation_twostage_eps_10")
    p.add_argument("--train-subset", default="train")
    p.add_argument("--val-subset", default="val")
    p.add_argument("--eval-subsets", nargs="+", default=["val", "test", "robustness"])

    # Variant
    p.add_argument("--include-h-decision", action="store_true",
                   help="Richer variant: concat h_decision into Stage 2 input.")

    # Phase 1
    p.add_argument("--phase-1-epochs", type=int, default=8)
    p.add_argument("--phase-1-lr", type=float, default=5e-5)
    p.add_argument("--phase-1-prefix-weight", type=float, default=2.0)
    p.add_argument("--phase-1-final-weight", type=float, default=1.0)
    p.add_argument("--skip-phase-1", action="store_true",
                   help="Skip Phase 1 entirely (use phase-1 checkpoint via --resume-phase1).")
    p.add_argument("--resume-phase1", type=Path, default=None,
                   help="Path to a Phase 1 checkpoint to skip Phase 1 training.")

    # Phase 2
    p.add_argument("--phase-2-epochs", type=int, default=5)
    p.add_argument("--phase-2-lr", type=float, default=1e-3)

    # Phase 3
    p.add_argument("--enable-phase-3", action="store_true",
                   help="Enable optional Phase 3 end-to-end fine-tune (gated).")
    p.add_argument("--phase-3-epochs", type=int, default=3)
    p.add_argument("--phase-3-lr", type=float, default=2e-5)
    p.add_argument("--phase-3-prefix-weight", type=float, default=1.0)
    p.add_argument("--phase-3-final-weight", type=float, default=0.5)
    p.add_argument("--phase-3-stop-weight", type=float, default=1.0)
    p.add_argument("--phase3-gate-within-eps", type=float, default=0.65)
    p.add_argument("--phase3-gate-savings-ms", type=float, default=4500.0)

    # Common
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=1337)

    # Phase 1 stability options
    p.add_argument("--phase-1-cosine-lr", action="store_true",
                   help="Use cosine annealing for Phase 1 LR (after warmup).")
    p.add_argument("--phase-1-cosine-min-lr-frac", type=float, default=0.01,
                   help="Cosine decay floor as a fraction of base lr.")
    p.add_argument("--phase-1-ema", action="store_true",
                   help="Maintain exponential-moving-average weights during Phase 1; eval/save EMA.")
    p.add_argument("--phase-1-ema-decay", type=float, default=0.999,
                   help="EMA decay rate (closer to 1.0 = more averaging).")

    # Threshold sweep
    p.add_argument("--threshold-min", type=float, default=0.05)
    p.add_argument("--threshold-max", type=float, default=0.95)
    p.add_argument("--threshold-steps", type=int, default=19)
    p.add_argument("--min-within-epsilon-rate", type=float, default=0.66)

    # Probe knobs
    p.add_argument("--max-train-shards", type=int, default=None)
    p.add_argument("--max-eval-shards", type=int, default=None)
    p.add_argument("--max-train-batches-per-epoch", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)

    # Encoder config (used only if no pretrained encoder)
    p.add_argument("--encoder-d-model", type=int, default=256)
    p.add_argument("--encoder-num-heads", type=int, default=8)
    p.add_argument("--encoder-num-layers", type=int, default=8)
    p.add_argument("--encoder-ff-dim", type=int, default=1024)
    p.add_argument("--encoder-dropout", type=float, default=0.15)

    # Stage 2 policy config
    p.add_argument("--policy-d-model", type=int, default=64)
    p.add_argument("--policy-num-heads", type=int, default=4)
    p.add_argument("--policy-num-layers", type=int, default=4)
    p.add_argument("--policy-ff-dim", type=int, default=256)
    p.add_argument("--policy-dropout", type=float, default=0.15)

    return p.parse_args()


# ============================================================================
# Utilities
# ============================================================================

def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda but CUDA not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_subset_paths(root: Path, subset: str) -> list[Path]:
    paths = sorted((root / subset).glob("*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {root / subset}")
    return paths


def maybe_limit(paths: list[Path], limit: int | None) -> list[Path]:
    return paths if limit is None else paths[:limit]


def iter_batches(paths, batch_size, shuffle, rng):
    shard_order = list(range(len(paths)))
    if shuffle:
        rng.shuffle(shard_order)
    for shard_idx in shard_order:
        path = paths[shard_idx]
        with np.load(path, allow_pickle=False) as d:
            x_full = d["x_full"].astype(np.float32, copy=False)
            decision_valid_mask = d["decision_valid_mask"].astype(bool, copy=False)
            decision_end_bucket = d["decision_end_bucket"].astype(np.int64, copy=False)
            decision_elapsed_ms = d["decision_elapsed_ms"].astype(np.int32, copy=False)
            decision_observed = d["decision_observed_buckets_seen"].astype(np.int32, copy=False)
            stop_label = d["stop_label"].astype(np.float32, copy=False)
            instantaneous_safe = d["instantaneous_safe_window"].astype(np.uint8, copy=False)
            xgb_y_pred = d["y_pred_mbps"].astype(np.float32, copy=False)
            relative_error = d["relative_error"].astype(np.float32, copy=False)
            y_true = d["y_true_mbps"].astype(np.float32, copy=False)
            speed_tier = d["speed_tier"]
            uuid = d["uuid"]
            test_time = d["test_time"]
        N = x_full.shape[0]
        order = np.arange(N)
        if shuffle:
            rng.shuffle(order)
        for start in range(0, N, batch_size):
            idx = order[start: start + batch_size]
            tier_idx = np.array([SPEED_TIER_TO_INDEX[str(t)] for t in speed_tier[idx]],
                                dtype=np.int64)
            yield {
                "x_full": x_full[idx],
                "decision_valid_mask": decision_valid_mask[idx],
                "decision_end_bucket": decision_end_bucket[idx],
                "decision_elapsed_ms": decision_elapsed_ms[idx],
                "decision_observed": decision_observed[idx],
                "stop_label": stop_label[idx],
                "instantaneous_safe": instantaneous_safe[idx],
                "xgb_y_pred": xgb_y_pred[idx],
                "relative_error": relative_error[idx],
                "y_true": y_true[idx],
                "speed_tier_idx": tier_idx,
                "uuid": uuid[idx],
                "test_time": test_time[idx],
            }


# ============================================================================
# Metrics
# ============================================================================

def throughput_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.abs(y_pred - y_true)
    sq = (y_pred - y_true) ** 2
    rel = err / np.maximum(np.abs(y_true), 1e-6)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(sq.mean())),
        "mre": float(rel.mean()),
        "within_10pct": float((rel <= 0.10).mean()),
    }


def prefix_throughput_metrics(valid, pred, y_true_per_test, xgb_pred):
    mask = valid.astype(bool)
    if not mask.any():
        return {"foundation": {}, "xgboost": {}}
    target = np.broadcast_to(y_true_per_test[:, None], pred.shape)
    return {
        "foundation": throughput_metrics(target[mask], pred[mask]),
        "xgboost": throughput_metrics(target[mask], xgb_pred[mask]),
    }


def policy_metrics_at_threshold(eval_data: dict, threshold: float) -> dict:
    valid = eval_data["dec_valid"].astype(bool)
    end_bucket = eval_data["dec_end_bucket"]
    elapsed = eval_data["dec_elapsed_ms"]
    stop_prob = eval_data["dec_stop_prob"]
    stop_label = eval_data["dec_stop_label"]
    inst_safe = eval_data["dec_instantaneous_safe"]
    rel_err = eval_data["dec_relative_error"]

    pred_pos = (stop_prob >= threshold) & valid
    true_pos = stop_label.astype(bool) & valid
    tp = int(np.logical_and(pred_pos, true_pos).sum())
    fp = int(np.logical_and(pred_pos, ~true_pos).sum())
    fn = int(np.logical_and(~pred_pos, true_pos).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)

    N, D = valid.shape
    fired_mask = pred_pos
    chosen_idx = np.full(N, -1, dtype=np.int64)
    for i in range(N):
        fired = np.flatnonzero(fired_mask[i])
        if fired.size > 0:
            chosen_idx[i] = fired[0]
        else:
            v = np.flatnonzero(valid[i])
            chosen_idx[i] = v[-1] if v.size > 0 else -1

    rows = np.arange(N)
    valid_choose = chosen_idx >= 0
    chosen_idx_safe = np.clip(chosen_idx, 0, D - 1)
    sel_elapsed = elapsed[rows, chosen_idx_safe]
    sel_safe = inst_safe[rows, chosen_idx_safe]
    sel_rel = rel_err[rows, chosen_idx_safe]
    last_idx_per_test = np.array([
        (np.flatnonzero(valid[i])[-1] if valid[i].any() else 0) for i in range(N)
    ])
    full_elapsed = elapsed[rows, last_idx_per_test]
    fired_per_test = fired_mask.any(axis=1)

    n_used = int(valid_choose.sum())
    if n_used == 0:
        return {"threshold": threshold, "f1": f1, "precision": precision, "recall": recall,
                "tests": 0, "emitted_stop_rate": 0.0, "within_epsilon_rate": 0.0,
                "mean_savings_ms": 0.0, "median_savings_ms": 0.0,
                "mean_stop_elapsed_ms": 0.0, "median_stop_elapsed_ms": 0.0,
                "mean_relative_error_at_stop": 0.0}
    sel_elapsed = sel_elapsed[valid_choose]
    sel_safe = sel_safe[valid_choose]
    sel_rel = sel_rel[valid_choose]
    full_elapsed = full_elapsed[valid_choose]
    fired_per_test = fired_per_test[valid_choose]
    savings = full_elapsed.astype(np.float64) - sel_elapsed.astype(np.float64)
    return {
        "threshold": float(threshold),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tests": int(n_used),
        "emitted_stop_rate": float(fired_per_test.mean()),
        "within_epsilon_rate": float(sel_safe.mean()),
        "mean_savings_ms": float(savings.mean()),
        "median_savings_ms": float(np.median(savings)),
        "mean_stop_elapsed_ms": float(sel_elapsed.mean()),
        "median_stop_elapsed_ms": float(np.median(sel_elapsed)),
        "mean_relative_error_at_stop": float(sel_rel.mean()),
    }


def select_threshold(eval_data, thresholds, min_within_eps):
    sweep = [policy_metrics_at_threshold(eval_data, float(t)) for t in thresholds]
    feasible = [r for r in sweep if r["within_epsilon_rate"] >= min_within_eps]
    if feasible:
        best = max(feasible, key=lambda r: r["mean_savings_ms"])
    else:
        best = max(sweep, key=lambda r: r["within_epsilon_rate"])
    return best["threshold"], sweep


# ============================================================================
# Forward + collect
# ============================================================================

def gather_eval_predictions(model, paths, device, batch_size, max_batches, mode: str) -> dict:
    """mode is 'stage1' (Phase 1) or 'full' (Phase 2/3)."""
    model.eval()
    fields = {
        "uuid": [], "test_time": [], "speed_tier_idx": [], "y_true": [],
        "final_mu_mbps": [],
        "dec_valid": [], "dec_end_bucket": [], "dec_elapsed_ms": [],
        "dec_stage1_mu_mbps": [], "dec_stage1_logvar": [],
        "dec_stop_prob": [], "dec_stop_label": [],
        "dec_instantaneous_safe": [], "dec_xgb_y_pred": [],
        "dec_relative_error": [],
    }
    rng = np.random.default_rng(0)
    bcount = 0
    with torch.no_grad():
        for batch in iter_batches(paths, batch_size, shuffle=False, rng=rng):
            x = torch.from_numpy(batch["x_full"]).to(device, non_blocking=True)
            db = torch.from_numpy(batch["decision_end_bucket"]).to(device, non_blocking=True)
            if mode == "stage1":
                out = model.forward_stage1(x, db)
                stop_prob = np.zeros_like(batch["stop_label"])  # not used
            else:
                de_ms = torch.from_numpy(batch["decision_elapsed_ms"]).to(device, non_blocking=True)
                obs = torch.from_numpy(batch["decision_observed"]).to(device, non_blocking=True)
                out = model.forward_full(x, db, de_ms, obs, detach_stage1=True)
                stop_prob = torch.sigmoid(out["stop_logit"]).cpu().numpy()
            stage1_mu_mbps = expm1_mbps(out["stage1_mu"]).cpu().numpy()
            stage1_logvar = out["stage1_logvar"].cpu().numpy()
            final_mu_mbps = expm1_mbps(out["final_throughput_mu"]).cpu().numpy()

            fields["uuid"].append(batch["uuid"])
            fields["test_time"].append(batch["test_time"])
            fields["speed_tier_idx"].append(batch["speed_tier_idx"])
            fields["y_true"].append(batch["y_true"])
            fields["final_mu_mbps"].append(final_mu_mbps)
            fields["dec_valid"].append(batch["decision_valid_mask"])
            fields["dec_end_bucket"].append(batch["decision_end_bucket"])
            fields["dec_elapsed_ms"].append(batch["decision_elapsed_ms"])
            fields["dec_stage1_mu_mbps"].append(stage1_mu_mbps)
            fields["dec_stage1_logvar"].append(stage1_logvar)
            fields["dec_stop_prob"].append(stop_prob)
            fields["dec_stop_label"].append(batch["stop_label"])
            fields["dec_instantaneous_safe"].append(batch["instantaneous_safe"])
            fields["dec_xgb_y_pred"].append(batch["xgb_y_pred"])
            fields["dec_relative_error"].append(batch["relative_error"])
            bcount += 1
            if max_batches is not None and bcount >= max_batches:
                break
    return {k: np.concatenate(v, axis=0) if v and isinstance(v[0], np.ndarray)
            else np.concatenate(v) for k, v in fields.items()}


# ============================================================================
# Phase loops
# ============================================================================

def lr_at_step(step, base_lr, warmup):
    if warmup <= 0 or step >= warmup:
        return base_lr
    return base_lr * (step + 1) / warmup


def lr_cosine(step: int, total_steps: int, base_lr: float, warmup: int,
              min_lr_frac: float = 0.01) -> float:
    """Linear warmup then cosine decay from base_lr -> base_lr * min_lr_frac."""
    import math
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1.0 - min_lr_frac) * cos_factor)


class EMA:
    """Exponential moving average over model parameters.

    Holds shadow copies; can swap into model for evaluation and restore.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name] = p.detach().clone()

    def apply_to(self, model: nn.Module) -> None:
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


def train_phase_1(args, model, device, train_paths, val_paths, output_root):
    print("\n========== PHASE 1: encoder + Stage 1 head ==========")
    model.unfreeze_encoder()
    optimizer = torch.optim.AdamW(
        list(model.encoder_model.parameters()),
        lr=args.phase_1_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    rng = np.random.default_rng(args.seed)
    history = []
    best_val_mae = float("inf")
    best_state = None
    best_epoch = -1
    global_step = 0

    # Estimate total optimizer steps for cosine schedule
    if args.phase_1_cosine_lr and args.max_train_batches_per_epoch is None:
        # Approx: 5627 microbatches/epoch / accum_steps = ~700 opt steps/epoch
        approx_steps_per_epoch = 5627 // max(1, args.gradient_accumulation_steps)
        total_steps = approx_steps_per_epoch * args.phase_1_epochs
    else:
        total_steps = (args.max_train_batches_per_epoch or 700) * args.phase_1_epochs

    ema = EMA(model.encoder_model, decay=args.phase_1_ema_decay) if args.phase_1_ema else None
    if args.phase_1_cosine_lr:
        print(f"[phase1] cosine LR: base={args.phase_1_lr} -> {args.phase_1_lr*args.phase_1_cosine_min_lr_frac:.2e} over {total_steps} steps")
    if ema:
        print(f"[phase1] EMA averaging enabled, decay={args.phase_1_ema_decay}")

    for epoch in range(1, args.phase_1_epochs + 1):
        t0 = time.time()
        model.train()
        sums = {"loss": 0.0, "prefix_mse": 0.0, "final_mse": 0.0}
        microbatch = 0
        accum = 0
        opt_steps = 0
        optimizer.zero_grad(set_to_none=True)

        bar = tqdm(iter_batches(train_paths, args.batch_size, True, rng),
                   desc=f"phase1 ep{epoch}", unit="batch", dynamic_ncols=True)
        for batch in bar:
            x = torch.from_numpy(batch["x_full"]).to(device, non_blocking=True)
            db = torch.from_numpy(batch["decision_end_bucket"]).to(device, non_blocking=True)
            valid = torch.from_numpy(batch["decision_valid_mask"]).to(device)
            y_true = torch.from_numpy(batch["y_true"]).to(device)
            target_log = log1p_mbps(y_true)

            out = model.forward_stage1(x, db)
            target_dec = target_log.unsqueeze(1).expand_as(out["stage1_mu"])
            prefix_mse = (((out["stage1_mu"] - target_dec) ** 2) * valid.float()).sum() \
                / valid.float().sum().clamp_min(1.0)
            final_mse = ((out["final_throughput_mu"] - target_log) ** 2).mean()
            loss = (args.phase_1_prefix_weight * prefix_mse
                    + args.phase_1_final_weight * final_mse)
            (loss / args.gradient_accumulation_steps).backward()

            sums["loss"] += float(loss.item())
            sums["prefix_mse"] += float(prefix_mse.item())
            sums["final_mse"] += float(final_mse.item())
            microbatch += 1
            accum += 1
            if accum >= args.gradient_accumulation_steps:
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                if args.phase_1_cosine_lr:
                    new_lr = lr_cosine(global_step, total_steps,
                                        args.phase_1_lr, args.warmup_steps,
                                        args.phase_1_cosine_min_lr_frac)
                else:
                    new_lr = lr_at_step(global_step, args.phase_1_lr, args.warmup_steps)
                for g in optimizer.param_groups:
                    g["lr"] = new_lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model.encoder_model)
                opt_steps += 1
                accum = 0
                global_step += 1
            if opt_steps % 25 == 0 and opt_steps > 0:
                bar.set_postfix(
                    loss=f"{sums['loss']/max(1,microbatch):.3f}",
                    pmse=f"{sums['prefix_mse']/max(1,microbatch):.3f}",
                    fmse=f"{sums['final_mse']/max(1,microbatch):.3f}",
                    step=opt_steps,
                )
            if (args.max_train_batches_per_epoch is not None
                    and opt_steps >= args.max_train_batches_per_epoch):
                break
        bar.close()
        if accum > 0:
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model.encoder_model)

        # ---- validation (use EMA weights if enabled) ----
        if ema is not None:
            ema.apply_to(model.encoder_model)
        val_data = gather_eval_predictions(model, val_paths, device, args.batch_size,
                                           args.max_eval_batches, mode="stage1")
        val_final = throughput_metrics(val_data["y_true"], val_data["final_mu_mbps"])
        val_prefix = prefix_throughput_metrics(
            val_data["dec_valid"], val_data["dec_stage1_mu_mbps"],
            val_data["y_true"], val_data["dec_xgb_y_pred"]
        )
        train_avg = {k: v / max(1, microbatch) for k, v in sums.items()}
        train_time = time.time() - t0
        rec = {
            "phase": 1, "epoch": epoch, "train": train_avg,
            "val_final_throughput": val_final,
            "val_prefix_throughput": val_prefix,
            "train_time_s": train_time,
        }
        history.append(rec)
        f_mae = val_prefix["foundation"].get("mae", float("inf"))
        x_mae = val_prefix["xgboost"].get("mae", float("nan"))
        print(f"[phase1] ep{epoch} val: finalMAE {val_final['mae']:.2f} | "
              f"prefMAE F={f_mae:.2f} X={x_mae:.2f} | "
              f"finalRMSE {val_final['rmse']:.2f} | time {train_time:.0f}s")

        if f_mae < best_val_mae:
            best_val_mae = f_mae
            best_epoch = epoch
            # While EMA weights are applied, snapshot full state (encoder + heads)
            best_state = copy.deepcopy(model.state_dict())

        # Restore raw training weights for the next epoch's gradient updates
        if ema is not None:
            ema.restore(model.encoder_model)

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = output_root / "phase1_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "best_epoch": best_epoch,
        "best_val_prefix_mae": best_val_mae,
        "tag": "fmnet_twostage_phase1",
        "phase1_used_cosine_lr": bool(args.phase_1_cosine_lr),
        "phase1_used_ema": bool(args.phase_1_ema),
    }, ckpt_path)
    print(f"[phase1] saved best to {ckpt_path} (epoch={best_epoch}, val_prefMAE={best_val_mae:.2f})")
    return history, best_epoch


def train_phase_2(args, model, device, train_paths, val_paths, output_root, thresholds):
    print("\n========== PHASE 2: Stage 2 policy module (encoder frozen, Stage 1 detached) ==========")
    model.freeze_encoder()
    # Only Stage 2 policy module is trainable
    policy_params = list(model.policy.parameters())
    optimizer = torch.optim.AdamW(
        policy_params,
        lr=args.phase_2_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    rng = np.random.default_rng(args.seed + 1)
    history = []
    best_score = -float("inf")
    best_state = None
    best_threshold = None
    best_epoch = -1
    global_step = 0

    for epoch in range(1, args.phase_2_epochs + 1):
        t0 = time.time()
        model.train()
        # encoder is frozen but its dropout would still fire in train mode; force eval there
        model.encoder_model.eval()
        sums = {"loss": 0.0, "stop_bce": 0.0}
        microbatch = 0
        accum = 0
        opt_steps = 0
        optimizer.zero_grad(set_to_none=True)

        bar = tqdm(iter_batches(train_paths, args.batch_size, True, rng),
                   desc=f"phase2 ep{epoch}", unit="batch", dynamic_ncols=True)
        for batch in bar:
            x = torch.from_numpy(batch["x_full"]).to(device, non_blocking=True)
            db = torch.from_numpy(batch["decision_end_bucket"]).to(device, non_blocking=True)
            de_ms = torch.from_numpy(batch["decision_elapsed_ms"]).to(device, non_blocking=True)
            obs = torch.from_numpy(batch["decision_observed"]).to(device, non_blocking=True)
            valid = torch.from_numpy(batch["decision_valid_mask"]).to(device)
            stop_target = torch.from_numpy(batch["stop_label"]).to(device)

            out = model.forward_full(x, db, de_ms, obs, detach_stage1=True)
            bce = nn.functional.binary_cross_entropy_with_logits(
                out["stop_logit"], stop_target, reduction="none",
            )
            stop_loss = (bce * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
            loss = stop_loss
            (loss / args.gradient_accumulation_steps).backward()

            sums["loss"] += float(loss.item())
            sums["stop_bce"] += float(stop_loss.item())
            microbatch += 1
            accum += 1
            if accum >= args.gradient_accumulation_steps:
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(policy_params, args.clip_grad_norm)
                for g in optimizer.param_groups:
                    g["lr"] = lr_at_step(global_step, args.phase_2_lr, args.warmup_steps)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1
                accum = 0
                global_step += 1
            if opt_steps % 25 == 0 and opt_steps > 0:
                bar.set_postfix(loss=f"{sums['loss']/max(1,microbatch):.3f}", step=opt_steps)
            if (args.max_train_batches_per_epoch is not None
                    and opt_steps >= args.max_train_batches_per_epoch):
                break
        bar.close()
        if accum > 0:
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy_params, args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # ---- validation ----
        val_data = gather_eval_predictions(model, val_paths, device, args.batch_size,
                                           args.max_eval_batches, mode="full")
        threshold, sweep = select_threshold(val_data, thresholds, args.min_within_epsilon_rate)
        at_best = policy_metrics_at_threshold(val_data, threshold)
        score = (at_best["mean_savings_ms"]
                 if at_best["within_epsilon_rate"] >= args.min_within_epsilon_rate
                 else at_best["within_epsilon_rate"] - 10.0)
        train_time = time.time() - t0
        train_avg = {k: v / max(1, microbatch) for k, v in sums.items()}
        rec = {
            "phase": 2, "epoch": epoch, "train": train_avg,
            "val_threshold": threshold,
            "val_policy_at_threshold": at_best,
            "selection_score": score,
            "train_time_s": train_time,
        }
        history.append(rec)
        print(f"[phase2] ep{epoch} val: thr {threshold:.2f} | F1 {at_best['f1']:.4f} | "
              f"within_eps {at_best['within_epsilon_rate']:.4f} | "
              f"savings {at_best['mean_savings_ms']:.1f}ms | "
              f"emit {at_best['emitted_stop_rate']:.3f} | "
              f"time {train_time:.0f}s")

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = output_root / "phase2_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "best_score": best_score,
        "tag": "fmnet_twostage_phase2",
    }, ckpt_path)
    print(f"[phase2] saved best to {ckpt_path} (ep={best_epoch}, thr={best_threshold}, score={best_score:.2f})")
    return history, best_epoch, best_threshold, best_score


def train_phase_3(args, model, device, train_paths, val_paths, output_root, thresholds):
    print("\n========== PHASE 3: end-to-end fine-tune (low LR) ==========")
    model.unfreeze_encoder()
    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=args.phase_3_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    rng = np.random.default_rng(args.seed + 2)
    history = []
    best_score = -float("inf")
    best_state = None
    best_threshold = None
    best_epoch = -1
    global_step = 0

    for epoch in range(1, args.phase_3_epochs + 1):
        t0 = time.time()
        model.train()
        sums = {"loss": 0.0, "prefix_mse": 0.0, "final_mse": 0.0, "stop_bce": 0.0}
        microbatch = 0
        accum = 0
        opt_steps = 0
        optimizer.zero_grad(set_to_none=True)

        bar = tqdm(iter_batches(train_paths, args.batch_size, True, rng),
                   desc=f"phase3 ep{epoch}", unit="batch", dynamic_ncols=True)
        for batch in bar:
            x = torch.from_numpy(batch["x_full"]).to(device, non_blocking=True)
            db = torch.from_numpy(batch["decision_end_bucket"]).to(device, non_blocking=True)
            de_ms = torch.from_numpy(batch["decision_elapsed_ms"]).to(device, non_blocking=True)
            obs = torch.from_numpy(batch["decision_observed"]).to(device, non_blocking=True)
            valid = torch.from_numpy(batch["decision_valid_mask"]).to(device)
            stop_target = torch.from_numpy(batch["stop_label"]).to(device)
            y_true = torch.from_numpy(batch["y_true"]).to(device)
            target_log = log1p_mbps(y_true)

            out = model.forward_full(x, db, de_ms, obs, detach_stage1=False)
            target_dec = target_log.unsqueeze(1).expand_as(out["stage1_mu"])
            prefix_mse = (((out["stage1_mu"] - target_dec) ** 2) * valid.float()).sum() \
                / valid.float().sum().clamp_min(1.0)
            final_mse = ((out["final_throughput_mu"] - target_log) ** 2).mean()
            bce = nn.functional.binary_cross_entropy_with_logits(
                out["stop_logit"], stop_target, reduction="none",
            )
            stop_loss = (bce * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
            loss = (args.phase_3_prefix_weight * prefix_mse
                    + args.phase_3_final_weight * final_mse
                    + args.phase_3_stop_weight * stop_loss)
            (loss / args.gradient_accumulation_steps).backward()

            sums["loss"] += float(loss.item())
            sums["prefix_mse"] += float(prefix_mse.item())
            sums["final_mse"] += float(final_mse.item())
            sums["stop_bce"] += float(stop_loss.item())
            microbatch += 1
            accum += 1
            if accum >= args.gradient_accumulation_steps:
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                for g in optimizer.param_groups:
                    g["lr"] = lr_at_step(global_step, args.phase_3_lr, args.warmup_steps)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1
                accum = 0
                global_step += 1
            if (args.max_train_batches_per_epoch is not None
                    and opt_steps >= args.max_train_batches_per_epoch):
                break
        bar.close()
        if accum > 0:
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        val_data = gather_eval_predictions(model, val_paths, device, args.batch_size,
                                           args.max_eval_batches, mode="full")
        threshold, sweep = select_threshold(val_data, thresholds, args.min_within_epsilon_rate)
        at_best = policy_metrics_at_threshold(val_data, threshold)
        val_final = throughput_metrics(val_data["y_true"], val_data["final_mu_mbps"])
        val_prefix = prefix_throughput_metrics(
            val_data["dec_valid"], val_data["dec_stage1_mu_mbps"],
            val_data["y_true"], val_data["dec_xgb_y_pred"]
        )
        score = (at_best["mean_savings_ms"]
                 if at_best["within_epsilon_rate"] >= args.min_within_epsilon_rate
                 else at_best["within_epsilon_rate"] - 10.0)
        train_avg = {k: v / max(1, microbatch) for k, v in sums.items()}
        train_time = time.time() - t0
        rec = {
            "phase": 3, "epoch": epoch, "train": train_avg,
            "val_threshold": threshold,
            "val_policy_at_threshold": at_best,
            "val_final_throughput": val_final,
            "val_prefix_throughput": val_prefix,
            "selection_score": score,
            "train_time_s": train_time,
        }
        history.append(rec)
        print(f"[phase3] ep{epoch} val: thr {threshold:.2f} | F1 {at_best['f1']:.4f} | "
              f"within_eps {at_best['within_epsilon_rate']:.4f} | "
              f"savings {at_best['mean_savings_ms']:.1f}ms | "
              f"finalMAE {val_final['mae']:.2f} | "
              f"prefMAE F={val_prefix['foundation'].get('mae','?')} X={val_prefix['xgboost'].get('mae','?')}")

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = output_root / "phase3_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "tag": "fmnet_twostage_phase3",
    }, ckpt_path)
    print(f"[phase3] saved best to {ckpt_path}")
    return history, best_epoch, best_threshold, best_score


# ============================================================================
# Final eval
# ============================================================================

def final_evaluation(model, args, device, eval_paths_by_subset, threshold, thresholds):
    final = {}
    for subset, paths in eval_paths_by_subset.items():
        data = gather_eval_predictions(model, paths, device, args.batch_size,
                                       args.max_eval_batches, mode="full")
        sweep = [policy_metrics_at_threshold(data, float(t)) for t in thresholds]
        at_best = policy_metrics_at_threshold(data, threshold)
        f_throughput = throughput_metrics(data["y_true"], data["final_mu_mbps"])
        prefix = prefix_throughput_metrics(
            data["dec_valid"], data["dec_stage1_mu_mbps"],
            data["y_true"], data["dec_xgb_y_pred"]
        )
        final[subset] = {
            "policy_at_best_threshold": at_best,
            "policy_threshold_sweep": sweep,
            "final_throughput": f_throughput,
            "prefix_throughput": prefix,
        }
        print(f"[final] {subset}: thr {threshold:.2f} | F1 {at_best['f1']:.4f} | "
              f"within_eps {at_best['within_epsilon_rate']:.4f} | "
              f"savings {at_best['mean_savings_ms']:.1f}ms | "
              f"finalMAE {f_throughput['mae']:.2f}/{f_throughput['rmse']:.2f}")
    return final


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    train_paths = maybe_limit(list_subset_paths(args.input_root, args.train_subset),
                              args.max_train_shards)
    val_paths = maybe_limit(list_subset_paths(args.input_root, args.val_subset),
                            args.max_eval_shards)
    eval_paths = {s: maybe_limit(list_subset_paths(args.input_root, s), args.max_eval_shards)
                  for s in args.eval_subsets}
    thresholds = list(np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps))

    encoder_cfg = FMNetV3Config(
        d_model=args.encoder_d_model, num_heads=args.encoder_num_heads,
        num_layers=args.encoder_num_layers, ff_dim=args.encoder_ff_dim,
        dropout=args.encoder_dropout,
    )
    policy_cfg = Stage2PolicyConfig(
        d_model=args.policy_d_model, num_heads=args.policy_num_heads,
        num_layers=args.policy_num_layers, ff_dim=args.policy_ff_dim,
        dropout=args.policy_dropout,
    )
    cfg = FMNetTwoStageConfig(
        encoder=encoder_cfg, policy=policy_cfg, include_h_decision=args.include_h_decision,
    )
    model = FMNetTwoStage(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] FMNetTwoStage with {n_params/1e6:.2f}M params on {device} | "
          f"include_h_decision={args.include_h_decision}")

    if args.pretrained_encoder is not None and args.pretrained_encoder.exists():
        ckpt = torch.load(args.pretrained_encoder, map_location="cpu", weights_only=False)
        miss, unexp = model.load_encoder_state(ckpt["model_state_dict"])
        print(f"[init] loaded pretrain encoder ckpt {args.pretrained_encoder}: "
              f"missing={len(miss)} unexpected={len(unexp)}")

    # ---------- Phase 1 ----------
    phase1_hist = []
    if args.skip_phase_1:
        if args.resume_phase1 is None or not args.resume_phase1.exists():
            raise SystemExit("--skip-phase-1 set but --resume-phase1 missing/not found")
        rckpt = torch.load(args.resume_phase1, map_location="cpu", weights_only=False)
        state = rckpt["model_state_dict"]
        own = model.state_dict()
        loadable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        skipped_shape = [k for k, v in state.items() if k in own and own[k].shape != v.shape]
        skipped_missing = [k for k in state if k not in own]
        model.load_state_dict(loadable, strict=False)
        print(f"[init] loaded phase1 ckpt {args.resume_phase1}: "
              f"loaded={len(loadable)} skipped_shape={len(skipped_shape)} "
              f"skipped_missing={len(skipped_missing)}")
        if skipped_shape:
            print(f"  (shape-mismatch keys not loaded, will train from init: {skipped_shape[:4]}...)")
    else:
        phase1_hist, _ = train_phase_1(args, model, device, train_paths, val_paths,
                                       args.output_root)

    # ---------- Phase 2 ----------
    phase2_hist, phase2_best_epoch, phase2_thr, phase2_score = train_phase_2(
        args, model, device, train_paths, val_paths, args.output_root, thresholds,
    )
    # final eval after phase 2
    print("\n========== Final eval after Phase 2 ==========")
    phase2_final = final_evaluation(model, args, device, eval_paths, phase2_thr, thresholds)

    # Save phase2-only summary now in case Phase 3 is skipped
    summary_p2 = {
        "stage": "phase2",
        "config": cfg.to_dict(),
        "phase1_history": phase1_hist,
        "phase2_history": phase2_hist,
        "phase2_best_epoch": phase2_best_epoch,
        "phase2_threshold": phase2_thr,
        "final_metrics_after_phase2": phase2_final,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (args.output_root / "training_summary_phase2.json").write_text(
        json.dumps(summary_p2, indent=2) + "\n"
    )

    # ---------- Phase 3 (gated) ----------
    phase3_hist = []
    phase3_thr = None
    phase3_final = None
    val_at_p2 = phase2_final.get("val", {}).get("policy_at_best_threshold", {})
    p3_we = val_at_p2.get("within_epsilon_rate", 0.0)
    p3_sav = val_at_p2.get("mean_savings_ms", 0.0)
    p3_gate_passed = (p3_we >= args.phase3_gate_within_eps
                      and p3_sav >= args.phase3_gate_savings_ms)
    print(f"\nPhase 3 gate: val within_eps={p3_we:.4f} (need >= {args.phase3_gate_within_eps}), "
          f"savings={p3_sav:.1f} ms (need >= {args.phase3_gate_savings_ms})")

    if args.enable_phase_3 and p3_gate_passed:
        phase3_hist, _, phase3_thr, _ = train_phase_3(
            args, model, device, train_paths, val_paths, args.output_root, thresholds,
        )
        print("\n========== Final eval after Phase 3 ==========")
        phase3_final = final_evaluation(model, args, device, eval_paths, phase3_thr, thresholds)
    elif args.enable_phase_3:
        print("Phase 3 enabled but gate not passed; skipping.")
    else:
        print("Phase 3 not enabled.")

    # ---------- Save full summary ----------
    summary = {
        "stage": "phase3" if phase3_hist else "phase2",
        "config": cfg.to_dict(),
        "phase1_history": phase1_hist,
        "phase2_history": phase2_hist,
        "phase3_history": phase3_hist,
        "phase2_best_epoch": phase2_best_epoch,
        "phase2_threshold": phase2_thr,
        "phase3_threshold": phase3_thr,
        "phase3_gate_passed": bool(p3_gate_passed),
        "final_metrics_after_phase2": phase2_final,
        "final_metrics_after_phase3": phase3_final,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (args.output_root / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"\n[done] wrote {args.output_root/'training_summary.json'}")


if __name__ == "__main__":
    main()
