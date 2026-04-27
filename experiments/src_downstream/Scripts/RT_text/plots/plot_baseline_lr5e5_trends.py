from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DATASET_FILES = [
    "DermSynth_knowledgebase.json",
    "MedMCQA_RT.json",
    "MedQuAD_dermatology_qa_retrieval_doclt300.json",
    "sce_retrieval.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot RT nonhomo-full step trends for selected baseline lr=5e-5 result groups."
    )
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_group_rows(group_dir: Path) -> list[dict]:
    rows = []
    for cp_dir in sorted(
        [path for path in group_dir.iterdir() if path.is_dir() and path.name.startswith("cp")],
        key=lambda path: int(path.name.replace("cp", "")),
    ):
        metrics = []
        for file_name in DATASET_FILES:
            file_path = cp_dir / file_name
            if not file_path.exists():
                metrics = []
                break
            metrics.append(load_json(file_path))
        if not metrics:
            continue
        avg_ndcg = sum(item["NDCG@10"] for item in metrics) / len(metrics) * 100.0
        avg_recall = sum(item["Recall@10"] for item in metrics) / len(metrics) * 100.0
        rows.append(
            {
                "step": int(cp_dir.name.replace("cp", "")),
                "avg_ndcg": avg_ndcg,
                "avg_recall": avg_recall,
                "avg": (avg_ndcg + avg_recall) / 2.0,
            }
        )
    return rows


def label_for_group(group_name: str) -> str:
    return group_name.replace("_lr5e-5", "").replace("lossv", "lossv")


def plot_groups(series: dict[str, list[dict]], output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    metric_specs = [
        ("avg", "Avg (%)"),
        ("avg_ndcg", "Avg NDCG@10 (%)"),
        ("avg_recall", "Avg Recall@10 (%)"),
    ]

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    for idx, (group_name, rows) in enumerate(series.items()):
        steps = [row["step"] for row in rows]
        color = colors[idx % len(colors)]
        label = label_for_group(group_name)
        for axis, (metric_key, ylabel) in zip(axes, metric_specs):
            values = [row[metric_key] for row in rows]
            axis.plot(steps, values, marker="o", linewidth=2.0, markersize=5, label=label, color=color)
            axis.set_ylabel(ylabel)
            axis.grid(True, linestyle="--", alpha=0.35)

        best_row = max(rows, key=lambda row: (row["avg"], row["avg_ndcg"], row["step"]))
        axes[0].scatter(best_row["step"], best_row["avg"], color=color, s=55, zorder=4)
        axes[0].annotate(
            f"{label}: cp{best_row['step']} ({best_row['avg']:.2f})",
            xy=(best_row["step"], best_row["avg"]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )

    axes[0].set_title("3p2_1p3b Baseline lr=5e-5 RT-full Trends")
    axes[-1].set_xlabel("Checkpoint Step")
    axes[0].legend(loc="best", frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    series = {}
    for group_name in args.groups:
        group_dir = baseline_dir / group_name
        rows = collect_group_rows(group_dir)
        if not rows:
            raise FileNotFoundError(f"No complete cp* results found under {group_dir}")
        series[group_name] = rows
    plot_groups(series, Path(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
