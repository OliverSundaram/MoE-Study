import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = Path(__file__).parent
GRAPHS_DIR = EVAL_DIR / "graphs"

MODELS = [
    ("Dense", EVAL_DIR / "dense_checkpoint_results.json", "#2a78d6"),
    ("MoE", EVAL_DIR / "moe_checkpoint_results.json", "#eb6834"),
]

BG = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def load_results(path):
    with open(path) as f:
        return json.load(f)


def metric_names(benchmark_data):
    names = []
    for key in benchmark_data:
        if key in ("name", "alias", "sample_len"):
            continue
        if key.endswith("_stderr,none"):
            continue
        if key.endswith(",none"):
            names.append(key[: -len(",none")])
    return names


def get_value(benchmark_data, metric):
    return benchmark_data.get(f"{metric},none")


def get_stderr(benchmark_data, metric):
    v = benchmark_data.get(f"{metric}_stderr,none")
    return v if isinstance(v, (int, float)) else None


def format_value(v):
    return f"{v:,.1f}" if abs(v) >= 100 else f"{v:.3f}"


def plot_benchmark(benchmark, dense_data, moe_data):
    metrics = [m for m in metric_names(dense_data) if m in metric_names(moe_data)]
    if not metrics:
        return

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), facecolor=BG)
    if n == 1:
        axes = [axes]

    fig.suptitle(benchmark, fontsize=16, fontweight="bold", color=INK_PRIMARY, y=1.03)

    x = np.array([0, 1.0])
    bar_width = 0.5

    for ax, metric in zip(axes, metrics):
        ax.set_facecolor(BG)
        values = [get_value(dense_data, metric), get_value(moe_data, metric)]
        errors = [get_stderr(dense_data, metric) or 0, get_stderr(moe_data, metric) or 0]
        colors = [c for _, _, c in MODELS]

        bars = ax.bar(
            x, values, width=bar_width, color=colors,
            yerr=errors, ecolor=INK_SECONDARY, capsize=4,
            edgecolor="none", zorder=3,
        )

        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                format_value(v), ha="center", va="bottom",
                fontsize=10, color=INK_PRIMARY,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([name for name, _, _ in MODELS], fontsize=10, color=INK_SECONDARY)
        ax.set_title(metric, fontsize=11, color=INK_SECONDARY, pad=10)

        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", colors=INK_SECONDARY, labelsize=9)
        ax.tick_params(axis="x", length=0)

        top = max(values) if values else 0
        ax.set_ylim(0, top * 1.2 if top > 0 else 1)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in MODELS]
    labels = [name for name, _, _ in MODELS]
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=2, frameon=False, fontsize=10, labelcolor=INK_SECONDARY,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    GRAPHS_DIR.mkdir(exist_ok=True)
    out_path = GRAPHS_DIR / f"{benchmark}.png"
    fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    dense = load_results(EVAL_DIR / "dense_checkpoint_results.json")
    moe = load_results(EVAL_DIR / "moe_checkpoint_results.json")

    for benchmark in (b for b in dense if b in moe):
        plot_benchmark(benchmark, dense[benchmark], moe[benchmark])


if __name__ == "__main__":
    main()