#!/usr/bin/env python3
"""Evaluate foundation per-decision throughput predictions against Stage 2 XGBoost prefixes."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from foundation_model import TraceFoundationConfig
from foundation_model_v15 import CausalPatchFoundationV15
from foundation_model_v2 import CausalBucketFoundationV2
from train_foundation_v15_multitask import (
    DEFAULT_EVAL_SUBSETS,
    SPEED_TIERS,
    mbps_from_log_prediction,
    resolve_device,
    tensor_batch,
)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    artifacts = root_dir / "artifacts_exact_public"
    parser = argparse.ArgumentParser(
        description="Compare foundation prefix throughput predictions with Stage 2 XGBoost prefix predictions."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=artifacts / "stage2_transformer_dataset_eps_10",
        help="Root containing frozen epsilon=10 Stage 2 shards.",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--eval-subsets", nargs="+", default=list(DEFAULT_EVAL_SUBSETS))
    parser.add_argument("--input-glob", default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-eval-shards", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def config_from_checkpoint(checkpoint: dict[str, object]) -> TraceFoundationConfig:
    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint is missing model_config")
    allowed = {field.name for field in fields(TraceFoundationConfig)}
    kwargs = {key: raw_config[key] for key in allowed if key in raw_config}
    return TraceFoundationConfig(**kwargs)


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dict")
    config = config_from_checkpoint(checkpoint)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing model_state_dict")

    if "v2_config" in checkpoint:
        v2_config = checkpoint["v2_config"]
        if not isinstance(v2_config, dict):
            raise ValueError("v2_config must be a dict")
        model = CausalBucketFoundationV2(
            config,
            bucket_hidden_dim=int(v2_config.get("bucket_hidden_dim", 128)),
            stem_kernel_size=int(v2_config.get("stem_kernel_size", 5)),
            num_speed_tiers=len(SPEED_TIERS),
        )
    else:
        model = CausalPatchFoundationV15(config, num_speed_tiers=len(SPEED_TIERS))

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def list_subset_paths(input_root: Path, subset: str, input_glob: str | None, limit: int | None) -> list[Path]:
    paths = sorted((input_root / subset).glob(input_glob or "*.npz"))
    if not paths:
        raise SystemExit(f"no shards found for subset {subset} under {input_root / subset}")
    return paths if limit is None else paths[:limit]


def iter_eval_batches(path: Path, *, batch_size: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        x_full = data["x_full"].astype(np.float32, copy=False)
        decision_valid_mask = data["decision_valid_mask"].astype(bool, copy=False)
        valid_tests = np.flatnonzero(decision_valid_mask.any(axis=1))
        for start in range(0, valid_tests.shape[0], batch_size):
            idx = valid_tests[start : start + batch_size]
            yield {
                "x_full": x_full[idx],
                "bucket_mask": data["bucket_mask"][idx].astype(bool, copy=False),
                "decision_valid_mask": decision_valid_mask[idx],
                "decision_elapsed_ms": data["decision_elapsed_ms"][idx].astype(np.int32, copy=False),
                "y_true_mbps": data["y_true_mbps"][idx].astype(np.float32, copy=False),
                "y_pred_mbps": data["y_pred_mbps"][idx].astype(np.float32, copy=False),
                "xgboost_y_pred_mbps": data["y_pred_mbps"][idx].astype(np.float32, copy=False),
                "stop_label": data["stop_label"][idx].astype(np.float32, copy=False),
                "instantaneous_safe_window": data["instantaneous_safe_window"][idx].astype(np.float32, copy=False),
                "speed_index": np.zeros(idx.shape[0], dtype=np.int64),
            }


def empty_bucket() -> dict[str, float]:
    return {
        "count": 0.0,
        "absolute_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "relative_error_sum": 0.0,
        "within_10pct_sum": 0.0,
    }


def update_bucket(bucket: dict[str, float], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    y_true64 = y_true.astype(np.float64, copy=False)
    y_pred64 = y_pred.astype(np.float64, copy=False)
    error = y_pred64 - y_true64
    relative_error = np.abs(error) / np.maximum(np.abs(y_true64), 1e-6)
    bucket["count"] += float(y_true64.shape[0])
    bucket["absolute_error_sum"] += float(np.abs(error).sum())
    bucket["squared_error_sum"] += float(np.square(error).sum())
    bucket["relative_error_sum"] += float(relative_error.sum())
    bucket["within_10pct_sum"] += float((relative_error <= 0.10).sum())


def finalize_bucket(bucket: dict[str, float]) -> dict[str, float | int]:
    count = int(bucket["count"])
    if count <= 0:
        return {"count": 0, "mae": 0.0, "rmse": 0.0, "mean_relative_error": 0.0, "within_10pct_rate": 0.0}
    return {
        "count": count,
        "mae": float(bucket["absolute_error_sum"] / count),
        "rmse": float(np.sqrt(bucket["squared_error_sum"] / count)),
        "mean_relative_error": float(bucket["relative_error_sum"] / count),
        "within_10pct_rate": float(bucket["within_10pct_sum"] / count),
    }


@torch.no_grad()
def evaluate_subset(
    *,
    model: torch.nn.Module,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    max_eval_batches: int | None,
) -> dict[str, object]:
    foundation_by_decision = [empty_bucket() for _ in range(20)]
    xgboost_by_decision = [empty_bucket() for _ in range(20)]
    foundation_vs_xgboost_by_decision = [empty_bucket() for _ in range(20)]
    foundation_overall = empty_bucket()
    xgboost_overall = empty_bucket()
    foundation_vs_xgboost_overall = empty_bucket()
    batches = 0
    tests = 0

    for path in paths:
        for np_batch in iter_eval_batches(path, batch_size=batch_size):
            batch = tensor_batch(np_batch, device)
            outputs = model(
                batch["x_full"],
                batch["bucket_mask"],
                decision_valid_mask=batch["decision_valid_mask"],
            )
            foundation_pred = mbps_from_log_prediction(outputs["throughput_mu"]).cpu().numpy().astype(np.float32)
            xgboost_pred = np_batch["y_pred_mbps"].astype(np.float32, copy=False)
            valid = np_batch["decision_valid_mask"].astype(bool, copy=False)
            y_true_by_decision = np.repeat(np_batch["y_true_mbps"].reshape(-1, 1), valid.shape[1], axis=1)

            for decision_idx in range(valid.shape[1]):
                decision_mask = valid[:, decision_idx]
                if not decision_mask.any():
                    continue
                y_true = y_true_by_decision[decision_mask, decision_idx]
                update_bucket(
                    foundation_by_decision[decision_idx],
                    y_true,
                    foundation_pred[decision_mask, decision_idx],
                )
                update_bucket(
                    xgboost_by_decision[decision_idx],
                    y_true,
                    xgboost_pred[decision_mask, decision_idx],
                )
                update_bucket(
                    foundation_vs_xgboost_by_decision[decision_idx],
                    xgboost_pred[decision_mask, decision_idx],
                    foundation_pred[decision_mask, decision_idx],
                )

            test_idx, decision_idx = np.nonzero(valid)
            update_bucket(
                foundation_overall,
                y_true_by_decision[test_idx, decision_idx],
                foundation_pred[test_idx, decision_idx],
            )
            update_bucket(
                xgboost_overall,
                y_true_by_decision[test_idx, decision_idx],
                xgboost_pred[test_idx, decision_idx],
            )
            update_bucket(
                foundation_vs_xgboost_overall,
                xgboost_pred[test_idx, decision_idx],
                foundation_pred[test_idx, decision_idx],
            )
            tests += int(valid.shape[0])
            batches += 1
            if max_eval_batches is not None and batches >= max_eval_batches:
                return {
                    "tests": tests,
                    "batches": batches,
                    "foundation_overall": finalize_bucket(foundation_overall),
                    "xgboost_overall": finalize_bucket(xgboost_overall),
                    "foundation_vs_xgboost_overall": finalize_bucket(foundation_vs_xgboost_overall),
                    "by_decision": [
                        {
                            "decision_index": idx,
                            "end_bucket": 5 * idx + 4,
                            "elapsed_ms": 500 * (idx + 1),
                            "foundation": finalize_bucket(foundation_by_decision[idx]),
                            "xgboost": finalize_bucket(xgboost_by_decision[idx]),
                            "foundation_vs_xgboost": finalize_bucket(foundation_vs_xgboost_by_decision[idx]),
                        }
                        for idx in range(20)
                    ],
                }

    return {
        "tests": tests,
        "batches": batches,
        "foundation_overall": finalize_bucket(foundation_overall),
        "xgboost_overall": finalize_bucket(xgboost_overall),
        "foundation_vs_xgboost_overall": finalize_bucket(foundation_vs_xgboost_overall),
        "by_decision": [
            {
                "decision_index": idx,
                "end_bucket": 5 * idx + 4,
                "elapsed_ms": 500 * (idx + 1),
                "foundation": finalize_bucket(foundation_by_decision[idx]),
                "xgboost": finalize_bucket(xgboost_by_decision[idx]),
                "foundation_vs_xgboost": finalize_bucket(foundation_vs_xgboost_by_decision[idx]),
            }
            for idx in range(20)
        ],
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = load_model(args.model_path, device)

    results: dict[str, object] = {
        "model_path": str(args.model_path),
        "input_root": str(args.input_root),
        "batch_size": int(args.batch_size),
        "subsets": {},
    }
    for subset in args.eval_subsets:
        paths = list_subset_paths(args.input_root, subset, args.input_glob, args.max_eval_shards)
        results["subsets"][subset] = evaluate_subset(
            model=model,
            paths=paths,
            device=device,
            batch_size=args.batch_size,
            max_eval_batches=args.max_eval_batches,
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / f"{args.run_name}_prefix_throughput.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote prefix throughput evaluation to {output_path}")
    for subset, metrics in results["subsets"].items():
        foundation = metrics["foundation_overall"]
        xgboost = metrics["xgboost_overall"]
        mimic = metrics["foundation_vs_xgboost_overall"]
        print(
            f"{subset}: foundation MAE {foundation['mae']:.3f} RMSE {foundation['rmse']:.3f} "
            f"within {foundation['within_10pct_rate']:.3f}; "
            f"xgboost MAE {xgboost['mae']:.3f} RMSE {xgboost['rmse']:.3f} "
            f"within {xgboost['within_10pct_rate']:.3f}; "
            f"foundation-vs-xgboost MAE {mimic['mae']:.3f}"
        )


if __name__ == "__main__":
    main()
