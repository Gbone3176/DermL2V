from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float) * 100.0


def build_probe_hop_figure(probe_results_path: Path, output_path: Path) -> None:
    payload = load_json(probe_results_path)
    diversity = payload["diversity_stats"]
    strategy_metrics = payload["strategy_metrics"]
    best_hop_stats = payload["best_hop_stats"]

    view_labels = [f"mean" if idx == 0 else f"hop_{idx-1}" for idx in range(len(diversity["mean_merge_weight_per_view"]))]
    view_weights = _pct(diversity["mean_merge_weight_per_view"])
    view_weight_std = _pct(diversity["std_merge_weight_per_view"])
    top1_view_fraction = _pct([diversity["top1_view_fraction"].get(f"view_{idx}", 0.0) for idx in range(len(view_labels))])

    hop_labels = [f"hop_{idx}" for idx in range(8)]
    best_hop_fraction = _pct([best_hop_stats["best_single_hop_fraction"].get(label, 0.0) for label in hop_labels])
    hop_ndcg10 = _pct([strategy_metrics[f"single_hop_{idx}"]["NDCG@10"] for idx in range(8)])
    hop_recall10 = _pct([strategy_metrics[f"single_hop_{idx}"]["Recall@10"] for idx in range(8)])

    full_ndcg10 = float(strategy_metrics["full"]["NDCG@10"]) * 100.0
    full_recall10 = float(strategy_metrics["full"]["Recall@10"]) * 100.0
    mean_ndcg10 = float(strategy_metrics["mean_only"]["NDCG@10"]) * 100.0
    mean_recall10 = float(strategy_metrics["mean_only"]["Recall@10"]) * 100.0
    uniform_ndcg10 = float(strategy_metrics["uniform_sa"]["NDCG@10"]) * 100.0
    uniform_recall10 = float(strategy_metrics["uniform_sa"]["Recall@10"]) * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SA_Fusion Hop Probe Summary", fontsize=16)

    ax = axes[0, 0]
    x = np.arange(len(view_labels))
    ax.bar(x, view_weights, yerr=view_weight_std, color=["#4c78a8"] + ["#9ecae9"] * 8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(view_labels, rotation=30, ha="right")
    ax.set_ylabel("Mean merge weight (%)")
    ax.set_title("Router Mean Weight Per View")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(x, top1_view_fraction, color=["#e15759"] + ["#fdd0a2"] * 8)
    ax.set_xticks(x)
    ax.set_xticklabels(view_labels, rotation=30, ha="right")
    ax.set_ylabel("Top-1 fraction (%)")
    ax.set_title("Which View Wins Router Top-1")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax = axes[1, 0]
    hx = np.arange(len(hop_labels))
    ax.bar(hx, best_hop_fraction, color="#59a14f")
    ax.set_xticks(hx)
    ax.set_xticklabels(hop_labels, rotation=0)
    ax.set_ylabel("Best single-hop fraction (%)")
    ax.set_title("Best Single-Hop Winner Distribution")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.text(
        0.98,
        0.95,
        f"mean best-hop margin: {best_hop_stats['mean_best_hop_margin']*100:.2f}%\nstd: {best_hop_stats['std_best_hop_margin']*100:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    ax = axes[1, 1]
    width = 0.36
    ax.bar(hx - width / 2, hop_ndcg10, width=width, label="single-hop NDCG@10", color="#f28e2b")
    ax.bar(hx + width / 2, hop_recall10, width=width, label="single-hop Recall@10", color="#76b7b2")
    ax.axhline(full_ndcg10, color="#d62728", linestyle="-", linewidth=1.6, label=f"full NDCG@10 ({full_ndcg10:.2f})")
    ax.axhline(mean_ndcg10, color="#8c564b", linestyle="--", linewidth=1.4, label=f"mean-only NDCG@10 ({mean_ndcg10:.2f})")
    ax.axhline(uniform_ndcg10, color="#9467bd", linestyle=":", linewidth=1.4, label=f"uniform-SA NDCG@10 ({uniform_ndcg10:.2f})")
    ax.axhline(full_recall10, color="#2ca02c", linestyle="-", linewidth=1.2, alpha=0.85, label=f"full Recall@10 ({full_recall10:.2f})")
    ax.axhline(mean_recall10, color="#17becf", linestyle="--", linewidth=1.2, alpha=0.85, label=f"mean-only Recall@10 ({mean_recall10:.2f})")
    ax.axhline(uniform_recall10, color="#bcbd22", linestyle=":", linewidth=1.2, alpha=0.85, label=f"uniform-SA Recall@10 ({uniform_recall10:.2f})")
    ax.set_xticks(hx)
    ax.set_xticklabels(hop_labels, rotation=0)
    ax.set_ylabel("Metric (%)")
    ax.set_title("Single-Hop Retrieval Performance vs Fusion Baselines")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower right")

    footer = (
        f"pairwise view cosine mean={diversity['mean_pairwise_view_cosine']:.4f}, "
        f"pairwise hop cosine mean={diversity['mean_pairwise_hop_cosine']:.4f}, "
        f"top-k attention Jaccard mean={diversity['mean_topk_attention_jaccard']:.4f}, "
        f"router entropy mean={diversity['mean_router_entropy']:.4f}"
    )
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=10)

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SA_Fusion probe hop statistics from probe_results.json")
    parser.add_argument("--probe-results", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    probe_results_path = Path(args.probe_results)
    output_path = Path(args.output) if args.output else probe_results_path.with_name("hop_probe_summary.png")
    build_probe_hop_figure(probe_results_path, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
