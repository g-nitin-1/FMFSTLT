#!/usr/bin/env python3
"""Plot the TURBOTEST Stage 2 Pareto frontier from threshold sweep results.

Reads threshold_sweep_all.json produced by rescore_stage2_thresholds.py
and generates two figures:

  Figure A — full sweep curves (all thresholds, all epsilon):
      X: median % data transferred
      Y: within-epsilon rate

  Figure B — Pareto frontier points (one per epsilon, best threshold):
      X: median % data transferred
      Y: within-epsilon rate
      (mirrors paper Figure 3)

Usage:
    python3 scripts/plot_pareto_frontier.py
    python3 scripts/plot_pareto_frontier.py --subset test --min-within-epsilon 0.7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


EPSILON_VALUES = [5, 10, 15, 20, 25, 30, 35]


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    artifacts = root_dir / "artifacts_exact_public"
    p = argparse.ArgumentParser(description="Plot Pareto frontier from Stage 2 threshold sweep.")
    p.add_argument(
        "--sweep-path",
        type=Path,
        default=artifacts / "stage2_threshold_sweep" / "threshold_sweep_all.json",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=artifacts / "stage2_threshold_sweep",
    )
    p.add_argument(
        "--subset",
        default="test",
        help="Which subset's results to plot (val or test).",
    )
    p.add_argument(
        "--min-emitted-stop-rate",
        type=float,
        default=0.05,
        help=(
            "Minimum fraction of tests that must emit an early stop for a "
            "threshold to be included in the Pareto selection. Prevents "
            "degenerate all-full-test operating points."
        ),
    )
    p.add_argument(
        "--x-axis",
        choices=(
            "median_pct_data_transferred",
            "mean_pct_data_transferred",
            "median_stop_elapsed_ms",
            "mean_stop_elapsed_ms",
        ),
        default="median_pct_data_transferred",
        help="X-axis metric.",
    )
    p.add_argument(
        "--y-axis",
        choices=(
            "within_epsilon_rate",
            "mean_relative_error_at_stop",
            "median_relative_error_at_stop",
        ),
        default="within_epsilon_rate",
        help="Y-axis metric.",
    )
    return p.parse_args()


def load_sweep(sweep_path: Path, subset: str) -> dict[int, list[dict]]:
    data = json.loads(sweep_path.read_text())
    result: dict[int, list[dict]] = {}
    for eps_str, eps_data in data.items():
        eps = int(eps_str)
        subsets = eps_data.get("subsets", {})
        if subset not in subsets:
            print(f"[warn] subset '{subset}' not found for eps={eps}, skipping")
            continue
        result[eps] = subsets[subset]
    return result


def select_pareto_point(
    sweep: list[dict],
    min_emitted_stop_rate: float,
    x_field: str,
    y_field: str,
) -> dict | None:
    """Pick the threshold with highest within_epsilon_rate subject to min emitted-stop rate.

    Falls back to lowest x-axis value if all points fail the emitted-stop constraint.
    """
    candidates = [pt for pt in sweep if pt["emitted_stop_rate"] >= min_emitted_stop_rate]
    pool = candidates if candidates else sweep

    if y_field == "within_epsilon_rate":
        best = max(pool, key=lambda pt: (pt[y_field], -(pt[x_field] or 1.0)))
    else:
        # for error metrics, lower is better
        best = min(pool, key=lambda pt: (pt[y_field] or 1.0, pt[x_field] or 1.0))
    return best


def print_summary_table(
    sweep_by_eps: dict[int, list[dict]],
    min_emitted_stop_rate: float,
    x_field: str,
    y_field: str,
    subset: str,
) -> None:
    print(f"\n{'=' * 70}")
    print(
        f"Pareto frontier summary  |  subset={subset}  |  min_emitted={min_emitted_stop_rate:.2f}"
    )
    print(f"{'=' * 70}")
    print(
        f"{'eps':>5}  {'threshold':>10}  {'emitted%':>9}  {'within_eps%':>12}  {'med_%xfer':>10}  {'med_savings_s':>14}"
    )
    print(f"{'-' * 70}")
    for eps in sorted(sweep_by_eps.keys()):
        pt = select_pareto_point(sweep_by_eps[eps], min_emitted_stop_rate, x_field, y_field)
        if pt is None:
            continue
        med_xfer = pt.get("median_pct_data_transferred")
        med_save = pt.get("median_savings_vs_full_ms")
        print(
            f"{eps:>5}  {pt['threshold']:>10.3f}  "
            f"{pt['emitted_stop_rate'] * 100:>8.1f}%  "
            f"{pt['within_epsilon_rate'] * 100:>11.1f}%  "
            f"{(med_xfer or 0) * 100:>9.1f}%  "
            f"{(med_save or 0) / 1000:>13.2f}s"
        )
    print()

    # full sweep info for each epsilon
    print(f"\nFull sweep range per epsilon (subset={subset}):")
    print(f"{'eps':>5}  {'thresholds':>10}  {'max_within_eps%':>16}  {'at_emitted_rate':>16}")
    print(f"{'-' * 55}")
    for eps in sorted(sweep_by_eps.keys()):
        sweep = sweep_by_eps[eps]
        max_within = max(pt["within_epsilon_rate"] for pt in sweep)
        # find threshold that gives max within_epsilon_rate
        best_pt = max(sweep, key=lambda pt: pt["within_epsilon_rate"])
        print(
            f"{eps:>5}  {len(sweep):>10}  "
            f"{max_within * 100:>15.1f}%  "
            f"thr={best_pt['threshold']:.2f} emit={best_pt['emitted_stop_rate'] * 100:.1f}%"
        )


def plot_sweep_curves(
    sweep_by_eps: dict[int, list[dict]],
    x_field: str,
    y_field: str,
    subset: str,
    output_path: Path,
    min_emitted_stop_rate: float,
) -> None:
    if not HAS_MPL:
        print("[skip] matplotlib not available, skipping plots")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = cm.viridis(np.linspace(0.1, 0.9, len(EPSILON_VALUES)))

    for color, eps in zip(colors, EPSILON_VALUES, strict=True):
        if eps not in sweep_by_eps:
            continue
        sweep = sweep_by_eps[eps]
        xs = [pt[x_field] for pt in sweep if pt[x_field] is not None]
        ys = [pt[y_field] for pt in sweep if pt[x_field] is not None]
        ax.plot(xs, ys, "-o", color=color, markersize=3, alpha=0.7, label=f"ε={eps}%")

        # mark the selected Pareto point
        pt = select_pareto_point(sweep, min_emitted_stop_rate, x_field, y_field)
        if pt and pt[x_field] is not None:
            ax.plot(
                pt[x_field],
                pt[y_field],
                "*",
                color=color,
                markersize=14,
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=5,
            )

    ax.set_xlabel(_axis_label(x_field), fontsize=12)
    ax.set_ylabel(_axis_label(y_field), fontsize=12)
    ax.set_title(
        f"Stage 2 Threshold Sweep — {subset} set\n(★ = selected operating point)", fontsize=13
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    if "pct" in x_field:
        ax.set_xlim(0, 1.05)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    if y_field == "within_epsilon_rate":
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"saved sweep curves plot to {output_path}")


def plot_pareto_frontier(
    sweep_by_eps: dict[int, list[dict]],
    x_field: str,
    y_field: str,
    subset: str,
    output_path: Path,
    min_emitted_stop_rate: float,
) -> None:
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = cm.viridis(np.linspace(0.1, 0.9, len(EPSILON_VALUES)))

    xs_pareto, ys_pareto, labels_pareto = [], [], []
    for color, eps in zip(colors, EPSILON_VALUES, strict=True):
        if eps not in sweep_by_eps:
            continue
        pt = select_pareto_point(sweep_by_eps[eps], min_emitted_stop_rate, x_field, y_field)
        if pt is None or pt[x_field] is None:
            continue
        xs_pareto.append(pt[x_field])
        ys_pareto.append(pt[y_field])
        labels_pareto.append(f"ε={eps}%")
        ax.scatter(
            pt[x_field],
            pt[y_field],
            color=color,
            s=100,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(f" ε={eps}%", (pt[x_field], pt[y_field]), fontsize=8, va="center")

    # draw the frontier line (sorted by x)
    if xs_pareto:
        order = np.argsort(xs_pareto)
        ax.plot(
            [xs_pareto[i] for i in order],
            [ys_pareto[i] for i in order],
            "--",
            color="gray",
            alpha=0.5,
            zorder=1,
        )

    ax.set_xlabel(_axis_label(x_field), fontsize=12)
    ax.set_ylabel(_axis_label(y_field), fontsize=12)
    ax.set_title(f"TURBOTEST Stage 2 Pareto Frontier — {subset} set", fontsize=13)
    ax.grid(True, alpha=0.3)
    if "pct" in x_field:
        ax.set_xlim(0, 1.05)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    if y_field == "within_epsilon_rate":
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"saved Pareto frontier plot to {output_path}")


def _axis_label(field: str) -> str:
    labels = {
        "median_pct_data_transferred": "Median % Data Transferred",
        "mean_pct_data_transferred": "Mean % Data Transferred",
        "median_stop_elapsed_ms": "Median Stop Time (ms)",
        "mean_stop_elapsed_ms": "Mean Stop Time (ms)",
        "within_epsilon_rate": "Within-ε Rate (fraction of tests)",
        "mean_relative_error_at_stop": "Mean Relative Error at Stop",
        "median_relative_error_at_stop": "Median Relative Error at Stop",
    }
    return labels.get(field, field)


def save_pareto_json(
    sweep_by_eps: dict[int, list[dict]],
    min_emitted_stop_rate: float,
    x_field: str,
    y_field: str,
    subset: str,
    output_path: Path,
) -> None:
    points = []
    for eps in sorted(sweep_by_eps.keys()):
        pt = select_pareto_point(sweep_by_eps[eps], min_emitted_stop_rate, x_field, y_field)
        if pt is None:
            continue
        points.append({"epsilon": eps, **pt})
    output_path.write_text(json.dumps({"subset": subset, "points": points}, indent=2) + "\n")
    print(f"saved Pareto frontier JSON to {output_path}")


def main() -> None:
    args = parse_args()

    if not args.sweep_path.exists():
        raise SystemExit(
            f"sweep file not found: {args.sweep_path}\nRun rescore_stage2_thresholds.py first."
        )

    sweep_by_eps = load_sweep(args.sweep_path, args.subset)
    if not sweep_by_eps:
        raise SystemExit(f"no data found for subset '{args.subset}' in {args.sweep_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print_summary_table(
        sweep_by_eps,
        min_emitted_stop_rate=args.min_emitted_stop_rate,
        x_field=args.x_axis,
        y_field=args.y_axis,
        subset=args.subset,
    )

    save_pareto_json(
        sweep_by_eps,
        min_emitted_stop_rate=args.min_emitted_stop_rate,
        x_field=args.x_axis,
        y_field=args.y_axis,
        subset=args.subset,
        output_path=args.output_dir / f"pareto_frontier_{args.subset}.json",
    )

    plot_sweep_curves(
        sweep_by_eps,
        x_field=args.x_axis,
        y_field=args.y_axis,
        subset=args.subset,
        output_path=args.output_dir / f"sweep_curves_{args.subset}.png",
        min_emitted_stop_rate=args.min_emitted_stop_rate,
    )

    plot_pareto_frontier(
        sweep_by_eps,
        x_field=args.x_axis,
        y_field=args.y_axis,
        subset=args.subset,
        output_path=args.output_dir / f"pareto_frontier_{args.subset}.png",
        min_emitted_stop_rate=args.min_emitted_stop_rate,
    )


if __name__ == "__main__":
    main()
