from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/storage/BioMedNLP/llm2vec/output/downstream/DermL2V/RT_text/nonhomo_full")
OUTPUT_MD = ROOT / "RT_Nonhomo_SA_K_Sweep_ByAt.md"
OUTPUT_PNG_TEMPLATE = "RT_Nonhomo_SA_K_Sweep_At{at}_FinalAvg.png"

DATASETS = [
    ("DermSynth_knowledgebase", "DermSynth"),
    ("MedMCQA_RT", "MedMCQA"),
    ("MedQuAD_dermatology_qa_retrieval_doclt300", "MedQuAD-doclt300"),
]
KS = [8, 16, 32, 64, 128, 256]
ATS = [3, 5, 10]


def load_results() -> dict[int, dict[str, dict[str, float]]]:
    results: dict[int, dict[str, dict[str, float]]] = {}
    for k in KS:
        model_name = f"DermL2V_Baseline_SM_SA_K{k}_cp50"
        results[k] = {}
        for dataset_dir, dataset_label in DATASETS:
            result_path = ROOT / dataset_dir / f"{model_name}.json"
            results[k][dataset_label] = json.loads(result_path.read_text())
    return results


def build_markdown(results: dict[int, dict[str, dict[str, float]]]) -> str:
    lines = [
        "# RT Nonhomo SA K Sweep by @N",
        "",
        "Models included:",
        "`DermL2V_Baseline_SM_SA_K8_cp50`",
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
        rows = []
        for k in KS:
            ndcg_values = [results[k][label][f"NDCG@{at}"] for _, label in DATASETS]
            recall_values = [results[k][label][f"Recall@{at}"] for _, label in DATASETS]
            avg_ndcg = sum(ndcg_values) / len(ndcg_values)
            avg_recall = sum(recall_values) / len(recall_values)
            rows.append(
                {
                    "k": k,
                    "avg_ndcg": avg_ndcg,
                    "avg_recall": avg_recall,
                    "final_avg": (avg_ndcg + avg_recall) / 2.0,
                    "values": results[k],
                }
            )

        rows.sort(key=lambda row: row["final_avg"], reverse=True)

        lines.extend(
            [
                f"## @{at}",
                "",
                "| Rank | K | DermSynth NDCG | DermSynth Recall | MedMCQA NDCG | MedMCQA Recall | MedQuAD-doclt300 NDCG | MedQuAD-doclt300 Recall | Avg NDCG | Avg Recall | Final Avg |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for rank, row in enumerate(rows, start=1):
            values = row["values"]
            lines.append(
                "| {rank} | {k} | {d_ndcg:.5f} | {d_recall:.5f} | {m_ndcg:.5f} | {m_recall:.5f} | {q_ndcg:.5f} | {q_recall:.5f} | {avg_ndcg:.5f} | {avg_recall:.5f} | {final_avg:.5f} |".format(
                    rank=rank,
                    k=row["k"],
                    d_ndcg=values["DermSynth"][f"NDCG@{at}"],
                    d_recall=values["DermSynth"][f"Recall@{at}"],
                    m_ndcg=values["MedMCQA"][f"NDCG@{at}"],
                    m_recall=values["MedMCQA"][f"Recall@{at}"],
                    q_ndcg=values["MedQuAD-doclt300"][f"NDCG@{at}"],
                    q_recall=values["MedQuAD-doclt300"][f"Recall@{at}"],
                    avg_ndcg=row["avg_ndcg"],
                    avg_recall=row["avg_recall"],
                    final_avg=row["final_avg"],
                )
            )

        lines.append("")

    return "\n".join(lines)


def plot_curves(results: dict[int, dict[str, dict[str, float]]]) -> None:
    final_avg = {at: [] for at in ATS}
    for at in ATS:
        for k in KS:
            avg_ndcg = sum(results[k][label][f"NDCG@{at}"] for _, label in DATASETS) / len(DATASETS)
            avg_recall = sum(results[k][label][f"Recall@{at}"] for _, label in DATASETS) / len(DATASETS)
            final_avg[at].append((avg_ndcg + avg_recall) / 2.0)

    colors = {3: "#1b5e20", 5: "#0d47a1", 10: "#b71c1c"}
    for at in ATS:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax.plot(KS, final_avg[at], marker="o", linewidth=2.2, color=colors[at])
        ax.set_title(f"Final Avg by K (@{at})")
        ax.set_xlabel("K")
        ax.set_ylabel("Final Avg")
        ax.set_xticks(KS)
        ax.grid(True, linestyle="--", alpha=0.35)
        out_path = ROOT / OUTPUT_PNG_TEMPLATE.format(at=at)
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    results = load_results()
    OUTPUT_MD.write_text(build_markdown(results))
    plot_curves(results)
    print(OUTPUT_MD)
    for at in ATS:
        print(ROOT / OUTPUT_PNG_TEMPLATE.format(at=at))


if __name__ == "__main__":
    main()
