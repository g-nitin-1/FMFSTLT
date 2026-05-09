#!/usr/bin/env python3
"""Multi-task fine-tuning for FMNet v3.

Targets (from Stage 2 epsilon=10 dataset):
  - per-decision throughput  (log1p MSE on y_true_mbps broadcast to all decisions)
  - per-decision stop logit  (BCE on stop_label)
  - final throughput         (log1p MSE on y_true_mbps; pooled at last bucket)
  - speed-tier classifier    (CE on speed_tier)

Inference:
  - threshold sweep against frozen Stage 2 epsilon=10 baseline
  - per-decision throughput vs XGBoost prefix predictions stored in shards
  - final throughput MAE/RMSE against y_true_mbps
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
    for candidate in ("/dev/shm", "/tmp", "/var/tmp", "/usr/tmp"):
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            os.environ["TMPDIR"] = candidate
            break

import torch
from torch import nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from foundation_model_v3 import (
    FMNetV3, FMNetV3Config, SPEED_TIER_TO_INDEX, NUM_SPEED_TIERS,
    log1p_mbps, expm1_mbps,
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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    artifacts = root / "artifacts_exact_public"
    p = argparse.ArgumentParser(description="Multi-task fine-tuning for FMNet v3")
    p.add_argument("--input-root", type=Path,
                   default=artifacts / "stage2_transformer_dataset_eps_10")
    p.add_argument("--pretrained-checkpoint", type=Path, default=None,
                   help="Optional FMNet v3 pretraining checkpoint to initialize from.")
    p.add_argument("--output-root", type=Path,
                   default=artifacts / "foundation_v3_multitask_eps_10")
    p.add_argument("--train-subset", default="train")
    p.add_argument("--val-subset", default="val")
    p.add_argument("--eval-subsets", nargs="+", default=["val", "test", "robustness"])

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=1337)

    # Loss weights (phase 1 is the default phase; phase 2 kicks in after --phase-1-epochs)
    p.add_argument("--prefix-throughput-weight", type=float, default=2.0)
    p.add_argument("--stop-bce-weight", type=float, default=1.0)
    p.add_argument("--final-throughput-weight", type=float, default=0.5)
    p.add_argument("--speed-tier-weight", type=float, default=0.2)
    # Sequential training: optional second phase with different weights
    p.add_argument("--phase-1-epochs", type=int, default=None,
                   help="If set, switch to phase-2 weights after this many epochs.")
    p.add_argument("--phase-2-prefix-throughput-weight", type=float, default=None)
    p.add_argument("--phase-2-stop-bce-weight", type=float, default=None)
    p.add_argument("--phase-2-final-throughput-weight", type=float, default=None)
    p.add_argument("--phase-2-speed-tier-weight", type=float, default=None)

    # Threshold sweep
    p.add_argument("--threshold-min", type=float, default=0.05)
    p.add_argument("--threshold-max", type=float, default=0.95)
    p.add_argument("--threshold-steps", type=int, default=19)
    p.add_argument("--selection-metric",
                   choices=("policy_constrained_savings", "f1", "policy_within_epsilon",
                            "prefix_throughput_mae"),
                   default="policy_constrained_savings")
    p.add_argument("--min-within-epsilon-rate", type=float, default=0.66)

    # Probe knobs
    p.add_argument("--max-train-shards", type=int, default=None)
    p.add_argument("--max-eval-shards", type=int, default=None)
    p.add_argument("--max-train-batches-per-epoch", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)

    # Model config (used only if no pretrained checkpoint provided)
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


def list_subset_paths(root: Path, subset: str) -> list[Path]:
    paths = sorted((root / subset).glob("*.npz"))
    if not paths:
        raise SystemExit(f"no shards found under {root/subset}")
    return paths


# ----------------------------------------------------------------------------
# Streaming batch generator
# ----------------------------------------------------------------------------

def iter_batches(
    paths: list[Path],
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
):
    """Yield dict batches drawn from one shard at a time.

    Each batch has all per-test fields (full 100-bucket trace + all decision tables).
    """
    shard_order = list(range(len(paths)))
    if shuffle:
        rng.shuffle(shard_order)
    for shard_idx in shard_order:
        path = paths[shard_idx]
        with np.load(path, allow_pickle=False) as d:
            x_full = d["x_full"].astype(np.float32, copy=False)
            decision_valid_mask = d["decision_valid_mask"].astype(bool, copy=False)
            decision_end_bucket = d["decision_end_bucket"].astype(np.int64, copy=False)
            stop_label = d["stop_label"].astype(np.float32, copy=False)
            instantaneous_safe = d["instantaneous_safe_window"].astype(np.uint8, copy=False)
            xgb_y_pred = d["y_pred_mbps"].astype(np.float32, copy=False)
            relative_error = d["relative_error"].astype(np.float32, copy=False)
            y_true = d["y_true_mbps"].astype(np.float32, copy=False)
            speed_tier = d["speed_tier"]
            uuid = d["uuid"]
            test_time = d["test_time"]
            decision_elapsed_ms = d["decision_elapsed_ms"].astype(np.int32, copy=False)

        N = x_full.shape[0]
        order = np.arange(N)
        if shuffle:
            rng.shuffle(order)
        for start in range(0, N, batch_size):
            idx = order[start:start + batch_size]
            tier_idx = np.array(
                [SPEED_TIER_TO_INDEX[str(t)] for t in speed_tier[idx]], dtype=np.int64
            )
            yield {
                "x_full": x_full[idx],
                "decision_valid_mask": decision_valid_mask[idx],
                "decision_end_bucket": decision_end_bucket[idx],
                "stop_label": stop_label[idx],
                "instantaneous_safe": instantaneous_safe[idx],
                "xgb_y_pred": xgb_y_pred[idx],
                "relative_error": relative_error[idx],
                "y_true": y_true[idx],
                "speed_tier_idx": tier_idx,
                "uuid": uuid[idx],
                "test_time": test_time[idx],
                "decision_elapsed_ms": decision_elapsed_ms[idx],
            }


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------

def compute_loss(
    out: dict,
    batch: dict,
    weights: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    decision_valid_mask = torch.from_numpy(batch["decision_valid_mask"]).to(device)
    valid_count = decision_valid_mask.float().sum().clamp_min(1.0)

    y_true = torch.from_numpy(batch["y_true"]).to(device)        # [B]
    target_log = log1p_mbps(y_true)                              # [B]
    target_log_decisions = target_log.unsqueeze(1).expand_as(out["throughput_mu"])

    # Prefix throughput MSE in log1p space (only on valid decisions)
    prefix_mse_per = (out["throughput_mu"] - target_log_decisions) ** 2
    prefix_mse = (prefix_mse_per * decision_valid_mask.float()).sum() / valid_count

    # Stop BCE (only on valid decisions)
    stop_target = torch.from_numpy(batch["stop_label"]).to(device)
    bce = nn.functional.binary_cross_entropy_with_logits(
        out["stop_logit"], stop_target, reduction="none",
    )
    stop_loss = (bce * decision_valid_mask.float()).sum() / valid_count

    # Final throughput MSE in log1p space
    final_mse = ((out["final_throughput_mu"] - target_log) ** 2).mean()

    # Speed-tier CE
    speed_target = torch.from_numpy(batch["speed_tier_idx"]).to(device)
    speed_loss = nn.functional.cross_entropy(out["speed_tier_logits"], speed_target)

    total = (
        weights["prefix_throughput"] * prefix_mse
        + weights["stop"] * stop_loss
        + weights["final_throughput"] * final_mse
        + weights["speed_tier"] * speed_loss
    )
    components = {
        "loss": float(total.item()),
        "prefix_mse": float(prefix_mse.item()),
        "stop_bce": float(stop_loss.item()),
        "final_mse": float(final_mse.item()),
        "speed_ce": float(speed_loss.item()),
    }
    return total, components


# ----------------------------------------------------------------------------
# Eval / metrics
# ----------------------------------------------------------------------------

def gather_eval_predictions(
    model: nn.Module,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    max_batches: int | None,
) -> dict:
    model.eval()
    all_uuid: list = []
    all_test_time: list = []
    all_speed_idx: list = []
    all_y_true: list = []
    decisions = {
        "valid": [], "end_bucket": [], "elapsed_ms": [],
        "found_mu_mbps": [], "stop_prob": [], "stop_label": [],
        "instantaneous_safe": [], "xgb_y_pred": [], "relative_error": [],
    }
    final_mu: list = []
    speed_logits: list = []

    rng = np.random.default_rng(0)
    bcount = 0
    with torch.no_grad():
        for batch in iter_batches(paths, batch_size, shuffle=False, rng=rng):
            x = torch.from_numpy(batch["x_full"]).to(device, non_blocking=True)
            db = torch.from_numpy(batch["decision_end_bucket"]).to(device, non_blocking=True)
            out = model.forward_finetune(x, db)

            mu = expm1_mbps(out["throughput_mu"]).cpu().numpy()  # [B, D]
            stop_prob = torch.sigmoid(out["stop_logit"]).cpu().numpy()
            final = expm1_mbps(out["final_throughput_mu"]).cpu().numpy()  # [B]
            tier_logits = out["speed_tier_logits"].cpu().numpy()

            all_uuid.append(batch["uuid"])
            all_test_time.append(batch["test_time"])
            all_speed_idx.append(batch["speed_tier_idx"])
            all_y_true.append(batch["y_true"])
            decisions["valid"].append(batch["decision_valid_mask"])
            decisions["end_bucket"].append(batch["decision_end_bucket"])
            decisions["elapsed_ms"].append(batch["decision_elapsed_ms"])
            decisions["found_mu_mbps"].append(mu)
            decisions["stop_prob"].append(stop_prob)
            decisions["stop_label"].append(batch["stop_label"])
            decisions["instantaneous_safe"].append(batch["instantaneous_safe"])
            decisions["xgb_y_pred"].append(batch["xgb_y_pred"])
            decisions["relative_error"].append(batch["relative_error"])
            final_mu.append(final)
            speed_logits.append(tier_logits)
            bcount += 1
            if max_batches is not None and bcount >= max_batches:
                break

    out_dict = {
        "uuid": np.concatenate(all_uuid),
        "test_time": np.concatenate(all_test_time),
        "speed_tier_idx": np.concatenate(all_speed_idx),
        "y_true": np.concatenate(all_y_true),
        "final_mu": np.concatenate(final_mu),
        "speed_logits": np.concatenate(speed_logits),
    }
    for k, v in decisions.items():
        out_dict["dec_" + k] = np.concatenate(v, axis=0)
    return out_dict


def compute_throughput_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.abs(y_pred - y_true)
    sq = (y_pred - y_true) ** 2
    rel = err / np.maximum(np.abs(y_true), 1e-6)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(sq.mean())),
        "mre": float(rel.mean()),
        "within_10pct": float((rel <= 0.10).mean()),
    }


def compute_prefix_throughput_metrics(
    valid: np.ndarray,
    pred: np.ndarray,
    y_true_per_test: np.ndarray,
    xgb_pred: np.ndarray,
) -> dict:
    mask = valid.astype(bool)
    if not mask.any():
        return {"foundation": {}, "xgboost": {}}
    target = np.broadcast_to(y_true_per_test[:, None], pred.shape)
    return {
        "foundation": compute_throughput_metrics(target[mask], pred[mask]),
        "xgboost": compute_throughput_metrics(target[mask], xgb_pred[mask]),
    }


def policy_metrics_at_threshold(eval_data: dict, threshold: float) -> dict:
    """Compute window F1 + per-test policy metrics at a threshold.

    Per-test scan: take the first decision with prob >= threshold, otherwise
    fall through to the last valid decision (full test).
    """
    valid = eval_data["dec_valid"].astype(bool)
    end_bucket = eval_data["dec_end_bucket"]
    elapsed = eval_data["dec_elapsed_ms"]
    stop_prob = eval_data["dec_stop_prob"]
    stop_label = eval_data["dec_stop_label"]
    inst_safe = eval_data["dec_instantaneous_safe"]
    rel_err = eval_data["dec_relative_error"]

    # Window-level F1 over valid decisions
    pred_pos = (stop_prob >= threshold) & valid
    true_pos = (stop_label.astype(bool)) & valid
    tp = int(np.logical_and(pred_pos, true_pos).sum())
    fp = int(np.logical_and(pred_pos, ~true_pos).sum())
    fn = int(np.logical_and(~pred_pos, true_pos).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)

    # Per-test policy
    N, D = valid.shape
    fired_mask = pred_pos
    # For each test, find first fired index; if none, use last valid index
    chosen_idx = np.full(N, -1, dtype=np.int64)
    for i in range(N):
        fired = np.flatnonzero(fired_mask[i])
        if fired.size > 0:
            chosen_idx[i] = fired[0]
        else:
            valid_idx = np.flatnonzero(valid[i])
            chosen_idx[i] = valid_idx[-1] if valid_idx.size > 0 else -1

    # Compute metrics per test using chosen index
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
        return {
            "threshold": threshold, "f1": f1, "precision": precision, "recall": recall,
            "tests": 0, "emitted_stop_rate": 0.0, "within_epsilon_rate": 0.0,
            "mean_savings_ms": 0.0, "median_savings_ms": 0.0,
            "mean_stop_elapsed_ms": 0.0, "median_stop_elapsed_ms": 0.0,
            "mean_relative_error_at_stop": 0.0,
        }
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


def select_threshold(
    eval_data: dict,
    args: argparse.Namespace,
) -> tuple[float, list[dict]]:
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    sweep = [policy_metrics_at_threshold(eval_data, float(t)) for t in thresholds]

    if args.selection_metric == "f1":
        best = max(sweep, key=lambda r: r["f1"])
    elif args.selection_metric == "policy_within_epsilon":
        best = max(sweep, key=lambda r: r["within_epsilon_rate"])
    elif args.selection_metric == "policy_constrained_savings":
        feasible = [r for r in sweep if r["within_epsilon_rate"] >= args.min_within_epsilon_rate]
        if feasible:
            best = max(feasible, key=lambda r: r["mean_savings_ms"])
        else:
            best = max(sweep, key=lambda r: r["within_epsilon_rate"])
    else:  # prefix_throughput_mae has no threshold dependence
        best = sweep[len(sweep) // 2]
    return best["threshold"], sweep


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    weights_phase1 = {
        "prefix_throughput": args.prefix_throughput_weight,
        "stop": args.stop_bce_weight,
        "final_throughput": args.final_throughput_weight,
        "speed_tier": args.speed_tier_weight,
    }
    weights_phase2 = {
        "prefix_throughput": (args.phase_2_prefix_throughput_weight
                              if args.phase_2_prefix_throughput_weight is not None
                              else args.prefix_throughput_weight),
        "stop": (args.phase_2_stop_bce_weight
                 if args.phase_2_stop_bce_weight is not None
                 else args.stop_bce_weight),
        "final_throughput": (args.phase_2_final_throughput_weight
                             if args.phase_2_final_throughput_weight is not None
                             else args.final_throughput_weight),
        "speed_tier": (args.phase_2_speed_tier_weight
                       if args.phase_2_speed_tier_weight is not None
                       else args.speed_tier_weight),
    }
    weights = weights_phase1  # current phase reference; reassigned per epoch below

    train_paths = list_subset_paths(args.input_root, args.train_subset)
    val_paths = list_subset_paths(args.input_root, args.val_subset)
    if args.max_train_shards is not None:
        train_paths = train_paths[: args.max_train_shards]
    if args.max_eval_shards is not None:
        val_paths = val_paths[: args.max_eval_shards]

    cfg = FMNetV3Config(
        d_model=args.d_model, num_heads=args.num_heads, num_layers=args.num_layers,
        ff_dim=args.ff_dim, dropout=args.dropout,
    )

    if args.pretrained_checkpoint is not None and args.pretrained_checkpoint.exists():
        ckpt = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=False)
        if "config" in ckpt:
            cfg = FMNetV3Config.from_dict(ckpt["config"])
        model = FMNetV3(cfg).to(device)
        miss, unexp = model.load_encoder_state(ckpt["model_state_dict"], strict=False)
        print(f"[finetune] loaded pretrain ckpt {args.pretrained_checkpoint}: "
              f"missing={len(miss)} unexpected={len(unexp)}")
    else:
        model = FMNetV3(cfg).to(device)
        print(f"[finetune] no pretrain ckpt; training from random init")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[finetune] FMNet v3 with {n_params/1e6:.2f}M parameters on {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    history: list[dict] = []
    best_val_score = -float("inf")
    best_state = None
    best_threshold = None
    best_epoch = -1

    rng = np.random.default_rng(args.seed)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # Switch to phase 2 weights after phase-1 epochs
        if args.phase_1_epochs is not None and epoch > args.phase_1_epochs:
            weights = weights_phase2
            phase_tag = "phase2"
        else:
            weights = weights_phase1
            phase_tag = "phase1" if args.phase_1_epochs is not None else "single"
        print(f"[finetune] epoch {epoch} phase={phase_tag} weights={weights}")
        t0 = time.time()
        model.train()
        epoch_components: dict = {"loss": 0.0, "prefix_mse": 0.0, "stop_bce": 0.0,
                                  "final_mse": 0.0, "speed_ce": 0.0}
        accum_count = 0
        opt_steps = 0
        train_micro_batches = 0
        optimizer.zero_grad(set_to_none=True)

        bar = tqdm(iter_batches(train_paths, args.batch_size, True, rng),
                   desc=f"finetune ep{epoch}", unit="batch", dynamic_ncols=True)
        for batch in bar:
            x = torch.from_numpy(batch["x_full"]).to(device, non_blocking=True)
            db = torch.from_numpy(batch["decision_end_bucket"]).to(device, non_blocking=True)
            out = model.forward_finetune(x, db)
            loss, comps = compute_loss(out, batch, weights, device)
            scaled = loss / args.gradient_accumulation_steps
            scaled.backward()

            for k in epoch_components:
                epoch_components[k] += comps[k]
            train_micro_batches += 1
            accum_count += 1

            if accum_count >= args.gradient_accumulation_steps:
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                # warmup linear
                if args.warmup_steps > 0 and global_step < args.warmup_steps:
                    for g in optimizer.param_groups:
                        g["lr"] = args.learning_rate * (global_step + 1) / args.warmup_steps
                else:
                    for g in optimizer.param_groups:
                        g["lr"] = args.learning_rate
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1
                accum_count = 0
                global_step += 1
            if opt_steps % 25 == 0 and opt_steps > 0:
                bar.set_postfix(
                    loss=f"{epoch_components['loss']/max(1,train_micro_batches):.3f}",
                    pmse=f"{epoch_components['prefix_mse']/max(1,train_micro_batches):.3f}",
                    bce=f"{epoch_components['stop_bce']/max(1,train_micro_batches):.3f}",
                    fmse=f"{epoch_components['final_mse']/max(1,train_micro_batches):.3f}",
                    step=opt_steps,
                )
            if (args.max_train_batches_per_epoch is not None
                    and opt_steps >= args.max_train_batches_per_epoch):
                break
        bar.close()
        # flush remaining grads
        if accum_count > 0:
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_steps += 1

        train_time = time.time() - t0
        train_avg = {k: v / max(1, train_micro_batches) for k, v in epoch_components.items()}
        print(f"[finetune] epoch {epoch} train: "
              f"loss={train_avg['loss']:.4f} pmse={train_avg['prefix_mse']:.4f} "
              f"bce={train_avg['stop_bce']:.4f} fmse={train_avg['final_mse']:.4f} "
              f"sce={train_avg['speed_ce']:.4f} steps={opt_steps} time={train_time:.0f}s")

        # ---- validation ----
        val_data = gather_eval_predictions(
            model, val_paths, device, args.batch_size, args.max_eval_batches
        )
        threshold, val_sweep = select_threshold(val_data, args)
        val_policy = policy_metrics_at_threshold(val_data, threshold)
        val_final = compute_throughput_metrics(val_data["y_true"], val_data["final_mu"])
        val_prefix = compute_prefix_throughput_metrics(
            val_data["dec_valid"], np.log(np.maximum(val_data["dec_found_mu_mbps"], 1e-6)) * 0
            + val_data["dec_found_mu_mbps"],
            val_data["y_true"], val_data["dec_xgb_y_pred"]
        )
        val_speed_acc = float((val_data["speed_logits"].argmax(axis=1)
                                == val_data["speed_tier_idx"]).mean())

        if args.selection_metric == "f1":
            score = val_policy["f1"]
        elif args.selection_metric == "policy_within_epsilon":
            score = val_policy["within_epsilon_rate"]
        elif args.selection_metric == "policy_constrained_savings":
            score = val_policy["mean_savings_ms"] if val_policy["within_epsilon_rate"] >= args.min_within_epsilon_rate \
                else val_policy["within_epsilon_rate"] - 10.0
        else:  # prefix_throughput_mae — minimize
            score = -val_prefix["foundation"].get("mae", 1e9)

        rec = {
            "epoch": epoch,
            "train_avg": train_avg,
            "val_threshold": threshold,
            "val_policy_at_threshold": val_policy,
            "val_final_throughput": val_final,
            "val_prefix_throughput": val_prefix,
            "val_speed_tier_accuracy": val_speed_acc,
            "selection_score": score,
            "train_time_s": train_time,
        }
        history.append(rec)

        print(f"[finetune] epoch {epoch} val: "
              f"thr={threshold:.2f} "
              f"f1={val_policy['f1']:.4f} "
              f"within_eps={val_policy['within_epsilon_rate']:.4f} "
              f"savings={val_policy['mean_savings_ms']:.1f}ms "
              f"finalMAE={val_final['mae']:.2f} "
              f"prefMAE_F={val_prefix['foundation'].get('mae','?')} "
              f"prefMAE_X={val_prefix['xgboost'].get('mae','?')} "
              f"speedAcc={val_speed_acc:.4f} "
              f"score={score:.4f}")

        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch
            best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("training did not produce any checkpoint")
    model.load_state_dict(best_state)
    model_path = args.output_root / "fmnet_v3_finetuned.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "best_threshold": best_threshold,
        "best_epoch": best_epoch,
        "tag": "fmnet_v3_finetune",
    }, model_path)
    print(f"[finetune] saved best to {model_path} (epoch={best_epoch}, threshold={best_threshold})")

    # ---- final eval on all eval subsets ----
    final_metrics: dict = {}
    for subset in args.eval_subsets:
        paths = list_subset_paths(args.input_root, subset)
        if args.max_eval_shards is not None:
            paths = paths[: args.max_eval_shards]
        data = gather_eval_predictions(model, paths, device, args.batch_size, args.max_eval_batches)
        # threshold sweep on this subset
        sweep = [policy_metrics_at_threshold(data, float(t))
                 for t in np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)]
        at_best = policy_metrics_at_threshold(data, best_threshold)
        final = compute_throughput_metrics(data["y_true"], data["final_mu"])
        prefix = compute_prefix_throughput_metrics(
            data["dec_valid"], data["dec_found_mu_mbps"],
            data["y_true"], data["dec_xgb_y_pred"]
        )
        speed_acc = float(
            (data["speed_logits"].argmax(axis=1) == data["speed_tier_idx"]).mean()
        )
        final_metrics[subset] = {
            "policy_at_best_threshold": at_best,
            "policy_threshold_sweep": sweep,
            "final_throughput": final,
            "prefix_throughput": prefix,
            "speed_tier_accuracy": speed_acc,
        }
        print(f"[finetune] {subset}: "
              f"thr={best_threshold:.2f} f1={at_best['f1']:.4f} "
              f"within_eps={at_best['within_epsilon_rate']:.4f} "
              f"savings={at_best['mean_savings_ms']:.1f}ms "
              f"finalMAE={final['mae']:.2f}/{final['rmse']:.2f}")

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "model_path": str(model_path),
        "device": str(device),
        "config": cfg.to_dict(),
        "loss_weights_phase1": weights_phase1,
        "loss_weights_phase2": weights_phase2,
        "phase_1_epochs": args.phase_1_epochs,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "best_val_score": best_val_score,
        "history": history,
        "final_metrics": final_metrics,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (args.output_root / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[finetune] wrote training summary to {args.output_root/'training_summary.json'}")


if __name__ == "__main__":
    main()
