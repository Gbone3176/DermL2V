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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows(method_dir: Path) -> list[tuple[int, float, float, float]]:
    rows: list[tuple[int, float, float, float]] = []
    cp_dirs = sorted(
        [path for path in method_dir.iterdir() if path.is_dir() and path.name.startswith("cp")],
        key=lambda path: int(path.name.replace("cp", "")),
    )
    for cp_dir in cp_dirs:
        metrics = []
        for file_name in DATASET_FILES:
            path = cp_dir / file_name
            if not path.exists():
                metrics = []
                break
            metrics.append(load_json(path))
        if not metrics:
            continue
        step = int(cp_dir.name.replace("cp", ""))
        avg_ndcg = sum(item["NDCG@10"] for item in metrics) / len(metrics)
        avg_recall = sum(item["Recall@10"] for item in metrics) / len(metrics)
        avg = (avg_ndcg + avg_recall) / 2.0
        rows.append((step, avg, avg_ndcg, avg_recall))
    return rows


def build_plot(method_dir: Path, output_path: Path, title: str | None) -> None:
    rows = collect_rows(method_dir)
    if not rows:
        raise FileNotFoundError(f"No complete cp* rows found under {method_dir}")

    steps = [row[0] for row in rows]
    avg = [row[1] * 100.0 for row in rows]
    avg_ndcg = [row[2] * 100.0 for row in rows]
    avg_recall = [row[3] * 100.0 for row in rows]

    best_idx = max(range(len(rows)), key=lambda idx: (rows[idx][1], rows[idx][0]))

    plt.figure(figsize=(9, 5.5))
    plt.plot(steps, avg, marker="o", linewidth=2.2, color="#1f77b4", label="Avg (%)")
    plt.plot(
        steps,
        avg_ndcg,
        marker="s",
        linewidth=1.6,
        linestyle="--",
        color="#ff7f0e",
        alpha=0.9,
        label="Avg NDCG@10 (%)",
    )
    plt.plot(
        steps,
        avg_recall,
        marker="^",
        linewidth=1.6,
        linestyle="--",
        color="#2ca02c",
        alpha=0.9,
        label="Avg Recall@10 (%)",
    )
    plt.scatter(
        [steps[best_idx]],
        [avg[best_idx]],
        color="#d62728",
        s=80,
        zorder=5,
        label=f"Best Avg: cp{steps[best_idx]}",
    )
    plt.annotate(
        f"cp{steps[best_idx]}: {avg[best_idx]:.2f}",
        xy=(steps[best_idx], avg[best_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        color="#d62728",
    )

    plt.xlabel("Step")
    plt.ylabel("Metric (%)")
    plt.title(title or method_dir.name)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Avg / Avg_NDCG@10 / Avg_Recall@10 vs step for one RT nonhomo full method directory"
    )
    parser.add_argument("--method-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    method_dir = Path(args.method_dir)
    output_path = Path(args.output) if args.output else method_dir / "avg_vs_step.png"
    build_plot(method_dir, output_path, args.title)
    print(output_path)


if __name__ == "__main__":
    main()
