#!/usr/bin/env python3
"""Evaluate v1 early-stop decisions with v1 throughput predictions.

This answers the hybrid diagnostic question:

  If the v1 Stage 2 stop classifier chooses the stopping point, and the
  separate v1 Stage 1 throughput regressor supplies the Mbps prediction at
  that same prefix, what deployed accuracy and savings do we get?

Important: this is not the trained v1 architecture, because v1 early-stop did
not consume v1 Stage 1 predictions. It is a post-hoc composition of two v1
checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path

import numpy as np

if "TMPDIR" not in os.environ:
    for candidate in ("/dev/shm", "/tmp", "/var/tmp", "/usr/tmp"):
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            os.environ["TMPDIR"] = candidate
            break

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from foundation_model import (  # noqa: E402
    EarlyStopFoundationModel,
    ThroughputRegressionModel,
    TraceFoundationConfig,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())

        def set_postfix(self, *args, **kwargs):
            return None

        def close(self):
            return None


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    artifacts = root / "artifacts_exact_public"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--early-stop-model-path",
        type=Path,
        default=artifacts / "foundation_early_stop_eps_10_v1" / "foundation_early_stop_model.pt",
    )
    parser.add_argument(
        "--throughput-model-path",
        type=Path,
        default=artifacts / "foundation_throughput_regressor_v1" / "foundation_throughput_model.pt",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=artifacts / "stage2_transformer_dataset_eps_10",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=artifacts / "foundation_v1_hybrid_policy_with_v1_throughput",
    )
    parser.add_argument("--run-name", default="v1_early_stop_plus_v1_throughput")
    parser.add_argument("--subsets", nargs="+", default=["val", "test", "robustness"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--epsilon-fraction", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-shards", type=int, default=None)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_from_checkpoint(checkpoint: dict[str, object]) -> TraceFoundationConfig:
    raw = checkpoint.get("model_config")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint is missing model_config")
    allowed = {field.name for field in fields(TraceFoundationConfig)}
    return TraceFoundationConfig(**{key: raw[key] for key in allowed if key in raw})


def load_early_stop_model(path: Path, device: torch.device) -> tuple[EarlyStopFoundationModel, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = config_from_checkpoint(checkpoint)
    model = EarlyStopFoundationModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def load_throughput_model(path: Path, device: torch.device) -> tuple[ThroughputRegressionModel, str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = config_from_checkpoint(checkpoint)
    model = ThroughputRegressionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, str(checkpoint.get("target_transform", "log1p"))


def inverse_throughput(pred: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "log1p":
        return torch.expm1(pred).clamp_min(0.0)
    return pred.clamp_min(0.0)


def safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def update_choice_counts(counts: dict[str, int], chosen_decisions: np.ndarray) -> None:
    for decision in chosen_decisions.tolist():
        key = str(int(decision))
        counts[key] = counts.get(key, 0) + 1


@torch.no_grad()
def evaluate_subset(
    *,
    paths: list[Path],
    stop_model: EarlyStopFoundationModel,
    throughput_model: ThroughputRegressionModel,
    throughput_transform: str,
    threshold: float,
    epsilon_fraction: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    tests = 0
    emitted_stops = 0
    own_within = 0
    xgb_within = 0
    choice_counts: dict[str, int] = {}

    own_abs_errors: list[float] = []
    xgb_abs_errors: list[float] = []
    own_rel_errors: list[float] = []
    xgb_rel_errors: list[float] = []
    elapsed_values: list[float] = []
    savings_values: list[float] = []

    for path in tqdm(paths, desc="v1 hybrid", unit="shard"):
        with np.load(path, allow_pickle=False) as data:
            x_full = data["x_full"].astype(np.float32, copy=False)
            bucket_mask = data["bucket_mask"].astype(bool, copy=False)
            decision_valid = data["decision_valid_mask"].astype(bool, copy=False)
            decision_end_bucket = data["decision_end_bucket"].astype(np.int32, copy=False)
            xgb_pred = data["y_pred_mbps"].astype(np.float32, copy=False)
            y_true = data["y_true_mbps"].astype(np.float32, copy=False)

        n_tests, n_buckets, _ = x_full.shape
        n_decisions = decision_valid.shape[1]
        probs = np.full((n_tests, n_decisions), -np.inf, dtype=np.float32)
        own_pred = np.zeros((n_tests, n_decisions), dtype=np.float32)

        test_idx, decision_idx = np.nonzero(decision_valid)
        positions = np.arange(n_buckets, dtype=np.int32)
        for start in range(0, len(test_idx), batch_size):
            bt = test_idx[start : start + batch_size]
            bd = decision_idx[start : start + batch_size]
            end_bucket = decision_end_bucket[bt, bd]
            prefix_mask = bucket_mask[bt] & (positions[None, :] <= end_bucket[:, None])

            x_t = torch.as_tensor(x_full[bt], dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(prefix_mask, dtype=torch.bool, device=device)

            logits = stop_model(x_t, mask_t)
            pred = throughput_model(x_t, mask_t)
            probs[bt, bd] = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            own_pred[bt, bd] = inverse_throughput(pred, throughput_transform).cpu().numpy().astype(np.float32)

        for i in range(n_tests):
            valid_decisions = np.flatnonzero(decision_valid[i])
            if valid_decisions.size == 0:
                continue
            threshold_hits = valid_decisions[probs[i, valid_decisions] >= threshold]
            if threshold_hits.size:
                chosen = int(threshold_hits[0])
                emitted_stops += 1
            else:
                chosen = int(valid_decisions[-1])

            full = int(valid_decisions[-1])
            denom = max(abs(float(y_true[i])), 1e-6)
            own_abs = abs(float(own_pred[i, chosen]) - float(y_true[i]))
            xgb_abs = abs(float(xgb_pred[i, chosen]) - float(y_true[i]))
            own_rel = own_abs / denom
            xgb_rel = xgb_abs / denom
            elapsed = float((decision_end_bucket[i, chosen] + 1) * 100)
            full_elapsed = float((decision_end_bucket[i, full] + 1) * 100)

            tests += 1
            own_within += int(own_rel <= epsilon_fraction)
            xgb_within += int(xgb_rel <= epsilon_fraction)
            own_abs_errors.append(own_abs)
            xgb_abs_errors.append(xgb_abs)
            own_rel_errors.append(own_rel)
            xgb_rel_errors.append(xgb_rel)
            elapsed_values.append(elapsed)
            savings_values.append(full_elapsed - elapsed)
            update_choice_counts(choice_counts, np.asarray([chosen]))

    return {
        "tests": tests,
        "threshold": float(threshold),
        "emitted_stop_rate": emitted_stops / tests if tests else None,
        "own_stage1_within_epsilon_rate": own_within / tests if tests else None,
        "xgboost_at_same_stop_within_epsilon_rate": xgb_within / tests if tests else None,
        "own_stage1_mean_stop_abs_error_mbps": safe_mean(own_abs_errors),
        "xgboost_at_same_stop_mean_abs_error_mbps": safe_mean(xgb_abs_errors),
        "own_stage1_mean_stop_relative_error": safe_mean(own_rel_errors),
        "xgboost_at_same_stop_mean_relative_error": safe_mean(xgb_rel_errors),
        "mean_stop_elapsed_ms": safe_mean(elapsed_values),
        "mean_savings_vs_full_ms": safe_mean(savings_values),
        "chosen_decision_counts": dict(sorted(choice_counts.items(), key=lambda item: int(item[0]))),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    stop_model, stop_checkpoint = load_early_stop_model(args.early_stop_model_path, device)
    throughput_model, throughput_transform = load_throughput_model(args.throughput_model_path, device)

    threshold = args.threshold
    if threshold is None:
        summary = stop_checkpoint.get("summary")
        if isinstance(summary, dict) and "best_threshold" in summary:
            threshold = float(summary["best_threshold"])
        else:
            threshold = 0.5

    result: dict[str, object] = {
        "note": (
            "Post-hoc hybrid: v1 early-stop classifier chooses stop; separate v1 "
            "throughput regressor supplies Mbps at that prefix. v1 Stage 2 did not "
            "consume this Mbps during training or inference."
        ),
        "early_stop_model_path": str(args.early_stop_model_path),
        "throughput_model_path": str(args.throughput_model_path),
        "input_root": str(args.input_root),
        "threshold": float(threshold),
        "epsilon_fraction": float(args.epsilon_fraction),
        "stage2_inputs_in_v1": {
            "x_full": "[100, 13] normalized trace with forward-filled buckets",
            "prefix_bucket_mask": "boolean mask exposing only buckets up to the decision end bucket",
            "stage1_prediction_as_input": False,
        },
        "subsets": {},
    }

    for subset in args.subsets:
        paths = sorted((args.input_root / subset).glob("*.npz"))
        if args.max_shards is not None:
            paths = paths[: args.max_shards]
        if not paths:
            raise SystemExit(f"no shards found for subset {subset}")
        subset_metrics = evaluate_subset(
            paths=paths,
            stop_model=stop_model,
            throughput_model=throughput_model,
            throughput_transform=throughput_transform,
            threshold=float(threshold),
            epsilon_fraction=float(args.epsilon_fraction),
            batch_size=args.batch_size,
            device=device,
        )
        result["subsets"][subset] = subset_metrics
        print(
            f"{subset}: own within {subset_metrics['own_stage1_within_epsilon_rate']:.4f}, "
            f"xgb same-stop within {subset_metrics['xgboost_at_same_stop_within_epsilon_rate']:.4f}, "
            f"savings {subset_metrics['mean_savings_vs_full_ms']:.1f} ms, "
            f"own stop MAE {subset_metrics['own_stage1_mean_stop_abs_error_mbps']:.2f}"
        )

    out_path = args.output_root / f"{args.run_name}_hybrid_metrics.json"
    try:
        args.output_root.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out_path}")
    except OSError as exc:
        print(f"warning: could not write {out_path}: {exc}")


if __name__ == "__main__":
    main()
