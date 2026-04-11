from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path("/storage/BioMedNLP/llm2vec/output/downstream/DermL2V/RT_text/nonhomo_full")
OUTPUT_DIR = ROOT / "atK"
OUTPUT_MD = OUTPUT_DIR / "RT_Nonhomo_SA_K_Sweep_ByAt.md"
OUTPUT_ABLATION_MD = OUTPUT_DIR / "ablation_atK.md"
OUTPUT_PNG_TEMPLATE = "RT_Nonhomo_SA_K_Sweep_At{at}_FinalAvg.png"

DATASETS = [
    ("DermSynth_knowledgebase", "DermSynth"),
    ("MedMCQA_RT", "MedMCQA"),
    ("MedQuAD_dermatology_qa_retrieval_doclt300", "MedQuAD-doclt300"),
]
KS = [16, 32, 64, 128, 256]
ATS = [3, 5, 10]


def to_percent(value: float) -> float:
    return value * 100.0


def load_results() -> dict[int, dict[str, dict[str, float]]]:
    results: dict[int, dict[str, dict[str, float]]] = {}
    for k in KS:
        model_name = f"DermL2V_Baseline_SM_SA_K{k}_cp50"
        results[k] = {}
        for dataset_dir, dataset_label in DATASETS:
            result_path = ROOT / dataset_dir / f"{model_name}.json"
            results[k][dataset_label] = json.loads(result_path.read_text())
    return results


def summarize_by_at(results: dict[int, dict[str, dict[str, float]]]) -> dict[int, list[dict[str, float]]]:
    summary: dict[int, list[dict[str, float]]] = {}
    for at in ATS:
        rows = []
        for k in KS:
            avg_ndcg = sum(results[k][label][f"NDCG@{at}"] for _, label in DATASETS) / len(DATASETS)
            avg_recall = sum(results[k][label][f"Recall@{at}"] for _, label in DATASETS) / len(DATASETS)
            rows.append(
                {
                    "k": k,
                    "avg_ndcg": avg_ndcg,
                    "avg_recall": avg_recall,
                    "final_avg": (avg_ndcg + avg_recall) / 2.0,
                }
            )
        summary[at] = rows
    return summary


def build_markdown(results: dict[int, dict[str, dict[str, float]]]) -> str:
    by_at = summarize_by_at(results)
    lines = [
        "# RT Nonhomo SA K Sweep by @N",
        "",
        "Models included:",
        "`DermL2V_Baseline_SM_SA_K16_cp50`",
        "`DermL2V_Baseline_SM_SA_K32_cp50`",
        "`DermL2V_Baseline_SM_SA_K64_cp50`",
        "`DermL2V_Baseline_SM_SA_K128_cp50`",
        "`DermL2V_Baseline_SM_SA_K256_cp50`",
        "",
        "Datasets:",
        "`DermSynth_knowledgebase`",
        "`MedMCQA_RT`",
        "`MedQuAD_dermatology_qa_retrieval_doclt300`",
        "",
    ]

    for at in ATS:
        rows = [{"values": results[row["k"]], **row} for row in by_at[at]]
        rows.sort(key=lambda row: row["final_avg"], reverse=True)

        lines.extend(
            [
                f"## @{at}",
                "",
                "| Rank | K | DermSynth NDCG (%) | DermSynth Recall (%) | MedMCQA NDCG (%) | MedMCQA Recall (%) | MedQuAD-doclt300 NDCG (%) | MedQuAD-doclt300 Recall (%) | Avg NDCG (%) | Avg Recall (%) | Final Avg (%) |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for rank, row in enumerate(rows, start=1):
            values = row["values"]
            lines.append(
                "| {rank} | {k} | {d_ndcg:.2f} | {d_recall:.2f} | {m_ndcg:.2f} | {m_recall:.2f} | {q_ndcg:.2f} | {q_recall:.2f} | {avg_ndcg:.2f} | {avg_recall:.2f} | {final_avg:.2f} |".format(
                    rank=rank,
                    k=row["k"],
                    d_ndcg=to_percent(values["DermSynth"][f"NDCG@{at}"]),
                    d_recall=to_percent(values["DermSynth"][f"Recall@{at}"]),
                    m_ndcg=to_percent(values["MedMCQA"][f"NDCG@{at}"]),
                    m_recall=to_percent(values["MedMCQA"][f"Recall@{at}"]),
                    q_ndcg=to_percent(values["MedQuAD-doclt300"][f"NDCG@{at}"]),
                    q_recall=to_percent(values["MedQuAD-doclt300"][f"Recall@{at}"]),
                    avg_ndcg=to_percent(row["avg_ndcg"]),
                    avg_recall=to_percent(row["avg_recall"]),
                    final_avg=to_percent(row["final_avg"]),
                )
            )

        lines.append("")

    return "\n".join(lines)


def build_ablation_markdown(results: dict[int, dict[str, dict[str, float]]]) -> str:
    by_at = summarize_by_at(results)
    lines = [
        "# Ablation at K",
        "",
        "Datasets: `DermSynth_knowledgebase`, `MedMCQA_RT`, `MedQuAD_dermatology_qa_retrieval_doclt300`",
        "",
        "Metrics are reported as percentages.",
        "",
        "| K | Avg NDCG@3 (%) | Avg Recall@3 (%) | Avg Ave@3 (%) | Avg NDCG@5 (%) | Avg Recall@5 (%) | Avg Ave@5 (%) | Avg NDCG@10 (%) | Avg Recall@10 (%) | Avg Ave@10 (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in KS:
        row3 = next(row for row in by_at[3] if row["k"] == k)
        row5 = next(row for row in by_at[5] if row["k"] == k)
        row10 = next(row for row in by_at[10] if row["k"] == k)
        lines.append(
            "| {k} | {ndcg3:.2f} | {recall3:.2f} | {avg3:.2f} | {ndcg5:.2f} | {recall5:.2f} | {avg5:.2f} | {ndcg10:.2f} | {recall10:.2f} | {avg10:.2f} |".format(
                k=k,
                ndcg3=to_percent(row3["avg_ndcg"]),
                recall3=to_percent(row3["avg_recall"]),
                avg3=to_percent(row3["final_avg"]),
                ndcg5=to_percent(row5["avg_ndcg"]),
                recall5=to_percent(row5["avg_recall"]),
                avg5=to_percent(row5["final_avg"]),
                ndcg10=to_percent(row10["avg_ndcg"]),
                recall10=to_percent(row10["avg_recall"]),
                avg10=to_percent(row10["final_avg"]),
            )
        )
    return "\n".join(lines)


def plot_curves(results: dict[int, dict[str, dict[str, float]]]) -> None:
    by_at = summarize_by_at(results)
    avg_ndcg = {at: [to_percent(row["avg_ndcg"]) for row in by_at[at]] for at in ATS}
    avg_recall = {at: [to_percent(row["avg_recall"]) for row in by_at[at]] for at in ATS}
    ndcg_style = {"color": "#1565c0", "marker": "o"}
    recall_style = {"color": "#2e7d32", "marker": "s"}
    for at in ATS:
        fig, ax_left = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax_right = ax_left.twinx()

        ndcg_line = ax_left.plot(KS, avg_ndcg[at], linewidth=2.2, label=f"NDCG@{at}", **ndcg_style)
        recall_line = ax_right.plot(KS, avg_recall[at], linewidth=2.2, label=f"Recall@{at}", **recall_style)

        ax_left.set_title(f"Average Retrieval Metrics by K (@{at})")
        ax_left.set_xlabel("K")
        ax_left.set_ylabel("NDCG (%)", color=ndcg_style["color"])
        ax_right.set_ylabel("Recall (%)", color=recall_style["color"])
        ax_left.set_xticks(KS)
        ax_left.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
        ax_right.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
        ax_left.tick_params(axis="y", colors=ndcg_style["color"])
        ax_right.tick_params(axis="y", colors=recall_style["color"])
        ax_left.grid(True, linestyle="--", alpha=0.35)
        lines = ndcg_line + recall_line
        labels = [line.get_label() for line in lines]
        ax_left.legend(lines, labels, frameon=False, loc="best")
        out_path = OUTPUT_DIR / OUTPUT_PNG_TEMPLATE.format(at=at)
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    results = load_results()
    OUTPUT_MD.write_text(build_markdown(results))
    OUTPUT_ABLATION_MD.write_text(build_ablation_markdown(results))
    plot_curves(results)
    print(OUTPUT_MD)
    print(OUTPUT_ABLATION_MD)
    for at in ATS:
        print(OUTPUT_DIR / OUTPUT_PNG_TEMPLATE.format(at=at))


if __name__ == "__main__":
    main()
