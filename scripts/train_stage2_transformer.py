#!/usr/bin/env python3
"""Train and evaluate a Stage 2 Transformer stop-policy model."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np

if "TMPDIR" not in os.environ:
    for candidate in ("/dev/shm", "/tmp", "/var/tmp", "/usr/tmp"):
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            os.environ["TMPDIR"] = candidate
            break

import torch
from torch import nn

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:
    average_precision_score = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]

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
        description="Train a Transformer classifier for Stage 2 stop/continue decisions."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage2_transformer_dataset",
        help="Root directory containing materialized Stage 2 transformer dataset shards.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root_dir / "artifacts_exact_public" / "stage2_transformer",
        help="Directory for the trained model and metrics.",
    )
    parser.add_argument(
        "--train-subset",
        default="train",
        help="Subset used for fitting.",
    )
    parser.add_argument(
        "--val-subset",
        default="val",
        help="Subset used for threshold tuning and model selection.",
    )
    parser.add_argument(
        "--eval-subsets",
        nargs="+",
        default=list(DEFAULT_EVAL_SUBSETS),
        help="Subsets to evaluate after training.",
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help="Optional glob relative to each subset directory for a probe run.",
    )
    parser.add_argument(
        "--target-field",
        choices=("stop_label", "is_oracle_stop_window"),
        default="stop_label",
        help="Training target field stored in the materialized Stage 2 dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
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
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Transformer hidden size.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Transformer attention heads.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of Transformer encoder layers.",
    )
    parser.add_argument(
        "--ff-dim",
        type=int,
        default=256,
        help="Transformer feed-forward dimension.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout applied in the Transformer and head.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="If set, use this probability threshold instead of tuning on validation.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=("f1", "balanced_accuracy", "accuracy", "precision", "recall"),
        default="f1",
        help="Metric used to tune the decision threshold on validation.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("f1", "balanced_accuracy", "accuracy", "precision", "recall", "auroc", "average_precision"),
        default="f1",
        help="Validation metric used to choose the best epoch.",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.05,
        help="Smallest threshold considered when tuning on validation.",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.95,
        help="Largest threshold considered when tuning on validation.",
    )
    parser.add_argument(
        "--threshold-steps",
        type=int,
        default=19,
        help="Number of evenly spaced thresholds to test on validation.",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Optional BCE positive class weight. If omitted, infer from dataset summary.",
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
        "--max-eval-shards",
        type=int,
        default=None,
        help="Optional limit on eval shards for a probe run.",
    )
    parser.add_argument(
        "--max-train-batches-per-epoch",
        type=int,
        default=None,
        help="Optional limit on optimizer steps per epoch for a probe run.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Optional limit on eval batches for a probe run.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_subset_paths(root: Path, subset: str, input_glob: str | None) -> list[Path]:
    subset_dir = root / subset
    if input_glob:
        paths = sorted(subset_dir.glob(input_glob))
    else:
        paths = sorted(subset_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no Stage 2 dataset shards found for subset {subset} under {subset_dir}")
    return paths


def read_dataset_summary(input_root: Path) -> dict[str, object] | None:
    summary_path = input_root / "stage2_dataset_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


def infer_shape_from_first_shard(paths: list[Path]) -> tuple[int, int]:
    with np.load(paths[0], allow_pickle=False) as data:
        x = data["x"]
        if x.ndim != 3:
            raise ValueError(f"expected tensor input rank 3 in {paths[0]}, got {x.ndim}")
        return int(x.shape[1]), int(x.shape[2])


def compute_pos_weight(
    args: argparse.Namespace,
    dataset_summary: dict[str, object] | None,
) -> float:
    if args.pos_weight is not None:
        return float(args.pos_weight)

    if dataset_summary is None:
        raise ValueError("--pos-weight must be set when stage2_dataset_summary.json is missing")

    subset_summary = dataset_summary["subsets"][args.train_subset]
    if args.target_field == "stop_label":
        positive = int(subset_summary["stop_positive_windows"])
    else:
        positive = int(subset_summary["oracle_stop_windows"])
    total = int(subset_summary["windows"])
    negative = total - positive
    if positive <= 0 or negative <= 0:
        raise ValueError(
            f"cannot infer a stable pos_weight for {args.target_field}: positive={positive}, negative={negative}"
        )
    return float(negative / positive)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def confusion_from_predictions(labels: np.ndarray, predictions: np.ndarray) -> tuple[int, int, int, int]:
    labels_bool = labels.astype(bool, copy=False)
    preds_bool = predictions.astype(bool, copy=False)
    tp = int(np.logical_and(labels_bool, preds_bool).sum())
    tn = int(np.logical_and(~labels_bool, ~preds_bool).sum())
    fp = int(np.logical_and(~labels_bool, preds_bool).sum())
    fn = int(np.logical_and(labels_bool, ~preds_bool).sum())
    return tp, fp, tn, fn


def compute_window_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    predictions = probabilities >= threshold
    tp, fp, tn, fn = confusion_from_predictions(labels, predictions)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    accuracy = safe_divide(tp + tn, tp + fp + tn + fn)
    specificity = safe_divide(tn, tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)

    auroc = None
    average_precision = None
    if labels.size > 0 and labels.min() != labels.max():
        if roc_auc_score is not None:
            auroc = float(roc_auc_score(labels, probabilities))
        if average_precision_score is not None:
            average_precision = float(average_precision_score(labels, probabilities))

    return {
        "count": int(labels.shape[0]),
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "auroc": auroc,
        "average_precision": average_precision,
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "predicted_positive_rate": float(predictions.mean()) if predictions.size else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def threshold_metric_value(metric_name: str, labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    metrics = compute_window_metrics(labels, probabilities, threshold)
    value = metrics.get(metric_name)
    if value is None:
        return float("-inf")
    return float(value)


def choose_threshold(args: argparse.Namespace, labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    if args.decision_threshold is not None:
        return float(args.decision_threshold), threshold_metric_value(
            args.threshold_metric, labels, probabilities, float(args.decision_threshold)
        )

    if args.threshold_steps <= 1:
        candidates = np.array([0.5], dtype=np.float64)
    else:
        candidates = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps, dtype=np.float64)

    best_threshold = float(candidates[0])
    best_score = float("-inf")
    for threshold in candidates.tolist():
        score = threshold_metric_value(args.threshold_metric, labels, probabilities, float(threshold))
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


class Stage2Transformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        window_size_buckets: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, window_size_buckets, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x) + self.position_embedding[:, : x.shape[1], :]
        hidden = self.encoder(hidden)
        pooled = self.norm(hidden[:, -1, :])
        logits = self.head(pooled).squeeze(-1)
        return logits


def iterate_shard_batches(
    path: Path,
    *,
    target_field: str,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
):
    with np.load(path, allow_pickle=False) as data:
        x = data["x"].astype(np.float32, copy=False)
        y = data[target_field].astype(np.float32, copy=False)
        indices = np.arange(x.shape[0])
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, x.shape[0], batch_size):
            batch_index = indices[start : start + batch_size]
            yield x[batch_index], y[batch_index]


def train_one_epoch(
    *,
    model: nn.Module,
    paths: list[Path],
    target_field: str,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    rng: np.random.Generator,
    max_batches: int | None,
    clip_grad_norm: float,
) -> dict[str, float | int]:
    model.train()
    total_examples = 0
    total_loss = 0.0
    batch_count = 0

    shuffled_paths = list(paths)
    rng.shuffle(shuffled_paths)
    path_bar = tqdm(shuffled_paths, desc="stage2 train", unit="shard", dynamic_ncols=True)
    stop_early = False

    for path in path_bar:
        path_bar.set_postfix(loss=0.0 if total_examples == 0 else total_loss / total_examples, batches=batch_count)
        for batch_x, batch_y in iterate_shard_batches(
            path,
            target_field=target_field,
            batch_size=batch_size,
            shuffle=True,
            rng=rng,
        ):
            inputs = torch.from_numpy(batch_x).to(device=device, dtype=torch.float32)
            targets = torch.from_numpy(batch_y).to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            batch_size_current = int(targets.shape[0])
            total_examples += batch_size_current
            total_loss += float(loss.item()) * batch_size_current
            batch_count += 1

            if max_batches is not None and batch_count >= max_batches:
                stop_early = True
                break

        if stop_early:
            break

    path_bar.close()
    return {
        "loss": safe_divide(total_loss, total_examples),
        "examples": total_examples,
        "batches": batch_count,
    }


def predict_subset(
    *,
    model: nn.Module,
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
    uuid_list: list[np.ndarray] = []
    test_time_list: list[np.ndarray] = []
    end_bucket_list: list[np.ndarray] = []
    elapsed_ms_list: list[np.ndarray] = []
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    relative_error_list: list[np.ndarray] = []
    stop_label_list: list[np.ndarray] = []
    oracle_stop_found_list: list[np.ndarray] = []
    oracle_stop_elapsed_ms_list: list[np.ndarray] = []

    with torch.no_grad():
        path_bar = tqdm(paths, desc="stage2 eval", unit="shard", dynamic_ncols=True)
        stop_early = False
        for path in path_bar:
            path_bar.set_postfix(loss=0.0 if total_examples == 0 else total_loss / total_examples, batches=batch_count)
            with np.load(path, allow_pickle=False) as data:
                x = data["x"].astype(np.float32, copy=False)
                labels = data[target_field].astype(np.float32, copy=False)
                num_examples = int(x.shape[0])

                shard_probs: list[np.ndarray] = []
                shard_labels: list[np.ndarray] = []

                for start in range(0, num_examples, batch_size):
                    batch_x = x[start : start + batch_size]
                    batch_y = labels[start : start + batch_size]
                    inputs = torch.from_numpy(batch_x).to(device=device, dtype=torch.float32)
                    targets = torch.from_numpy(batch_y).to(device=device, dtype=torch.float32)
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                    probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False)

                    shard_probs.append(probs)
                    shard_labels.append(batch_y.astype(np.float32, copy=False))

                    batch_size_current = int(batch_y.shape[0])
                    total_examples += batch_size_current
                    total_loss += float(loss.item()) * batch_size_current
                    batch_count += 1

                    if max_batches is not None and batch_count >= max_batches:
                        stop_early = True
                        break

                probs_concat = np.concatenate(shard_probs, axis=0)
                labels_concat = np.concatenate(shard_labels, axis=0)

                probabilities_list.append(probs_concat)
                labels_list.append(labels_concat)
                uuid_list.append(data["uuid"][: probs_concat.shape[0]])
                test_time_list.append(data["test_time"][: probs_concat.shape[0]])
                end_bucket_list.append(data["end_bucket"][: probs_concat.shape[0]].astype(np.int16, copy=False))
                elapsed_ms_list.append(data["elapsed_ms"][: probs_concat.shape[0]].astype(np.int32, copy=False))
                y_true_list.append(data["y_true_mbps"][: probs_concat.shape[0]].astype(np.float32, copy=False))
                y_pred_list.append(data["y_pred_mbps"][: probs_concat.shape[0]].astype(np.float32, copy=False))
                relative_error_list.append(
                    data["relative_error"][: probs_concat.shape[0]].astype(np.float32, copy=False)
                )
                stop_label_list.append(data["stop_label"][: probs_concat.shape[0]].astype(np.uint8, copy=False))
                oracle_stop_found_list.append(
                    data["oracle_stop_found"][: probs_concat.shape[0]].astype(np.uint8, copy=False)
                )
                oracle_stop_elapsed_ms_list.append(
                    data["oracle_stop_elapsed_ms"][: probs_concat.shape[0]].astype(np.int32, copy=False)
                )

            if stop_early:
                break
        path_bar.close()

    return {
        "loss": safe_divide(total_loss, total_examples),
        "examples": total_examples,
        "batches": batch_count,
        "probabilities": np.concatenate(probabilities_list, axis=0) if probabilities_list else np.empty((0,), dtype=np.float32),
        "labels": np.concatenate(labels_list, axis=0) if labels_list else np.empty((0,), dtype=np.float32),
        "uuid": np.concatenate(uuid_list, axis=0) if uuid_list else np.empty((0,), dtype=np.str_),
        "test_time": np.concatenate(test_time_list, axis=0) if test_time_list else np.empty((0,), dtype=np.str_),
        "end_bucket": np.concatenate(end_bucket_list, axis=0) if end_bucket_list else np.empty((0,), dtype=np.int16),
        "elapsed_ms": np.concatenate(elapsed_ms_list, axis=0) if elapsed_ms_list else np.empty((0,), dtype=np.int32),
        "y_true_mbps": np.concatenate(y_true_list, axis=0) if y_true_list else np.empty((0,), dtype=np.float32),
        "y_pred_mbps": np.concatenate(y_pred_list, axis=0) if y_pred_list else np.empty((0,), dtype=np.float32),
        "relative_error": np.concatenate(relative_error_list, axis=0) if relative_error_list else np.empty((0,), dtype=np.float32),
        "stop_label": np.concatenate(stop_label_list, axis=0) if stop_label_list else np.empty((0,), dtype=np.uint8),
        "oracle_stop_found": np.concatenate(oracle_stop_found_list, axis=0) if oracle_stop_found_list else np.empty((0,), dtype=np.uint8),
        "oracle_stop_elapsed_ms": np.concatenate(oracle_stop_elapsed_ms_list, axis=0) if oracle_stop_elapsed_ms_list else np.empty((0,), dtype=np.int32),
    }


def compute_policy_metrics(outputs: dict[str, object], threshold: float) -> dict[str, float | int | None]:
    probabilities = np.asarray(outputs["probabilities"], dtype=np.float32)
    uuid = np.asarray(outputs["uuid"])
    test_time = np.asarray(outputs["test_time"])
    end_bucket = np.asarray(outputs["end_bucket"], dtype=np.int16)
    elapsed_ms = np.asarray(outputs["elapsed_ms"], dtype=np.int32)
    y_true_mbps = np.asarray(outputs["y_true_mbps"], dtype=np.float32)
    y_pred_mbps = np.asarray(outputs["y_pred_mbps"], dtype=np.float32)
    relative_error = np.asarray(outputs["relative_error"], dtype=np.float32)
    stop_label = np.asarray(outputs["stop_label"], dtype=np.uint8)
    oracle_stop_found = np.asarray(outputs["oracle_stop_found"], dtype=np.uint8)
    oracle_stop_elapsed_ms = np.asarray(outputs["oracle_stop_elapsed_ms"], dtype=np.int32)

    grouped_rows: dict[tuple[str, str], list[int]] = {}
    for idx, key in enumerate(zip(uuid.tolist(), test_time.tolist())):
        grouped_rows.setdefault(key, []).append(idx)

    emitted_stops = 0
    within_epsilon = 0
    stop_elapsed_values: list[float] = []
    stop_relative_error_values: list[float] = []
    stop_abs_error_values: list[float] = []
    savings_vs_full_values: list[float] = []
    excess_vs_oracle_values: list[float] = []
    oracle_tests = 0

    for row_indices in grouped_rows.values():
        row_indices.sort(key=lambda idx: (int(end_bucket[idx]), int(elapsed_ms[idx])))
        chosen_idx = row_indices[-1]

        for idx in row_indices:
            if probabilities[idx] >= threshold:
                chosen_idx = idx
                emitted_stops += 1
                break

        full_idx = row_indices[-1]
        stop_elapsed = float(elapsed_ms[chosen_idx])
        full_elapsed = float(elapsed_ms[full_idx])
        stop_rel_error = float(relative_error[chosen_idx])
        stop_abs_error = float(abs(y_pred_mbps[chosen_idx] - y_true_mbps[chosen_idx]))

        stop_elapsed_values.append(stop_elapsed)
        stop_relative_error_values.append(stop_rel_error)
        stop_abs_error_values.append(stop_abs_error)
        savings_vs_full_values.append(full_elapsed - stop_elapsed)

        if int(stop_label[chosen_idx]) == 1:
            within_epsilon += 1

        if int(oracle_stop_found[row_indices[0]]) == 1:
            oracle_tests += 1
            excess_vs_oracle_values.append(stop_elapsed - float(oracle_stop_elapsed_ms[row_indices[0]]))

    total_tests = len(grouped_rows)
    return {
        "tests": total_tests,
        "emitted_stop_rate": safe_divide(emitted_stops, total_tests),
        "within_epsilon_rate": safe_divide(within_epsilon, total_tests),
        "mean_stop_elapsed_ms": safe_mean(stop_elapsed_values),
        "median_stop_elapsed_ms": safe_median(stop_elapsed_values),
        "mean_stop_relative_error": safe_mean(stop_relative_error_values),
        "mean_stop_abs_error_mbps": safe_mean(stop_abs_error_values),
        "mean_savings_vs_full_ms": safe_mean(savings_vs_full_values),
        "median_savings_vs_full_ms": safe_median(savings_vs_full_values),
        "tests_with_oracle_stop": oracle_tests,
        "mean_excess_vs_oracle_ms": safe_mean(excess_vs_oracle_values),
        "median_excess_vs_oracle_ms": safe_median(excess_vs_oracle_values),
    }


def evaluate_subset_with_threshold(
    *,
    name: str,
    outputs: dict[str, object],
    threshold: float,
) -> dict[str, object]:
    labels = np.asarray(outputs["labels"], dtype=np.float32)
    probabilities = np.asarray(outputs["probabilities"], dtype=np.float32)
    window_metrics = compute_window_metrics(labels, probabilities, threshold)
    policy_metrics = compute_policy_metrics(outputs, threshold)
    return {
        "subset": name,
        "loss": float(outputs["loss"]),
        "examples": int(outputs["examples"]),
        "batches": int(outputs["batches"]),
        "window_metrics": window_metrics,
        "policy_metrics": policy_metrics,
    }


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def selection_score(selection_metric: str, subset_metrics: dict[str, object]) -> float:
    value = subset_metrics["window_metrics"].get(selection_metric)
    if value is None:
        return float("-inf")
    return float(value)


def maybe_limit(paths: list[Path], limit: int | None) -> list[Path]:
    if limit is None:
        return paths
    return paths[:limit]


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.max_train_batches_per_epoch is not None and args.max_train_batches_per_epoch <= 0:
        raise SystemExit("--max-train-batches-per-epoch must be positive when set")
    if args.max_eval_batches is not None and args.max_eval_batches <= 0:
        raise SystemExit("--max-eval-batches must be positive when set")
    if args.max_train_shards is not None and args.max_train_shards <= 0:
        raise SystemExit("--max-train-shards must be positive when set")
    if args.max_eval_shards is not None and args.max_eval_shards <= 0:
        raise SystemExit("--max-eval-shards must be positive when set")
    if args.threshold_steps <= 0:
        raise SystemExit("--threshold-steps must be positive")
    if args.threshold_min <= 0 or args.threshold_max >= 1 or args.threshold_min >= args.threshold_max:
        raise SystemExit("--threshold-min/max must satisfy 0 < min < max < 1")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    dataset_summary = read_dataset_summary(args.input_root)

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

    window_size_buckets, feature_dim = infer_shape_from_first_shard(train_paths)
    pos_weight = compute_pos_weight(args, dataset_summary)

    model = Stage2Transformer(
        input_dim=feature_dim,
        window_size_buckets=window_size_buckets,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = output_root / "stage2_transformer_model.pt"
    summary_path = output_root / "training_summary.json"

    rng = np.random.default_rng(args.seed)
    history: list[dict[str, object]] = []
    best_state = None
    best_epoch = -1
    best_threshold = None
    best_score = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            paths=train_paths,
            target_field=args.target_field,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
            rng=rng,
            max_batches=args.max_train_batches_per_epoch,
            clip_grad_norm=args.clip_grad_norm,
        )

        val_outputs = predict_subset(
            model=model,
            paths=val_paths,
            target_field=args.target_field,
            batch_size=args.batch_size,
            device=device,
            criterion=criterion,
            max_batches=args.max_eval_batches,
        )
        threshold, tuned_threshold_score = choose_threshold(
            args,
            np.asarray(val_outputs["labels"], dtype=np.float32),
            np.asarray(val_outputs["probabilities"], dtype=np.float32),
        )
        val_metrics = evaluate_subset_with_threshold(
            name=args.val_subset,
            outputs=val_outputs,
            threshold=threshold,
        )

        record = {
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_threshold": threshold,
            "val_threshold_metric_score": tuned_threshold_score,
            "val_metrics": val_metrics,
        }
        history.append(record)

        current_score = selection_score(args.selection_metric, val_metrics)
        if best_state is None or current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_threshold = threshold
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "epoch": epoch,
                "threshold": threshold,
            }

    if best_state is None or best_threshold is None:
        raise RuntimeError("training finished without a valid best checkpoint")

    model.load_state_dict(best_state["model_state_dict"])
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "window_size_buckets": window_size_buckets,
                "feature_dim": feature_dim,
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "ff_dim": args.ff_dim,
                "dropout": args.dropout,
                "target_field": args.target_field,
                "best_threshold": best_threshold,
                "best_epoch": best_epoch,
            },
        },
        model_path,
    )

    final_metrics: dict[str, object] = {}
    for subset, paths in final_eval_paths.items():
        outputs = predict_subset(
            model=model,
            paths=paths,
            target_field=args.target_field,
            batch_size=args.batch_size,
            device=device,
            criterion=criterion,
            max_batches=args.max_eval_batches,
        )
        final_metrics[subset] = evaluate_subset_with_threshold(
            name=subset,
            outputs=outputs,
            threshold=best_threshold,
        )

    training_summary = {
        "input_root": str(args.input_root),
        "output_root": str(output_root),
        "model_path": str(model_path),
        "train_subset": args.train_subset,
        "val_subset": args.val_subset,
        "eval_subsets": args.eval_subsets,
        "target_field": args.target_field,
        "device": str(device),
        "seed": args.seed,
        "pos_weight": pos_weight,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "best_selection_metric": args.selection_metric,
        "best_selection_score": best_score,
        "window_size_buckets": window_size_buckets,
        "feature_dim": feature_dim,
        "history": history,
        "final_metrics": final_metrics,
    }
    summary_path.write_text(json.dumps(training_summary, indent=2) + "\n")

    print(f"wrote Stage 2 Transformer model to {model_path}")
    print(f"wrote Stage 2 training summary to {summary_path}")


if __name__ == "__main__":
    main()
