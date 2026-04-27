from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_ROOT = Path("output/downstream/RT_text/nonhomo-full")
DATASET_ORDER = ["DermaSynth-E3", "MedMCQA", "MedQuAD", "SCE-Derma-SQ"]
SECTION_MODEL_ORDER = [
    ("General / Decoder Baselines", ["gpt2"]),
    ("Lexical Baselines", ["BM25"]),
    (
        "Biomedical Encoders",
        [
            "BioClinicalBERT",
            "Clinical_ModernBERT",
            "BioClinical-ModernBERT-large",
            "BioLinkBERT",
            "pubmedbert-base-embeddings",
        ],
    ),
    ("BME Retriever Models", ["BMRETRIEVER-1B", "MedCPT", "BMRETRIEVER-7B"]),
    (
        "LLM-based Encoders",
        [
            "NV-Embed-v2",
            "Qwen3-Embedding-0.6B",
            "LLM2Vec_Llama-31-8B",
            "LLM2Vec_Llama3-8B-inst",
            "Qwen3-Embedding-8B",
        ],
    ),
]
MODEL_DISPLAY = {
    "gpt2": "gpt2",
    "BM25": "BM25",
    "BioClinicalBERT": "BioClinicalBERT",
    "Clinical_ModernBERT": "Clinical_ModernBERT",
    "BioClinical-ModernBERT-large": "BioClinical-ModernBERT-large",
    "BioLinkBERT": "BioLinkBERT",
    "pubmedbert-base-embeddings": "pubmedbert-base-embeddings",
    "BMRETRIEVER-1B": "BMRetriever-1B",
    "MedCPT": "MedCPT",
    "NV-Embed-v2": "NV-Embed-v2",
    "Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B",
    "BMRETRIEVER-7B": "BMRetriever-7B",
    "LLM2Vec_Llama-31-8B": "LLM2Vec_Llama-31-8B",
    "LLM2Vec_Llama3-8B-inst": "LLM2Vec_Llama3-8B-inst",
    "Qwen3-Embedding-8B": "Qwen3-Embedding-8B",
}
DATASET_FILE_CANDIDATES = {
    "DermaSynth-E3": ["DermaSynth-E3.json", "DermSynth_knowledgebase.json"],
    "MedMCQA": ["MedMCQA.json", "MedMCQA_RT.json"],
    "MedQuAD": ["MedQuAD.json", "MedQuAD_dermatology_qa_retrieval_doclt300.json"],
    "SCE-Derma-SQ": ["SCE-Derma-SQ.json", "sce_retrieval.json"],
}


def load_model_metrics(root: Path, model_name: str) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}

    if model_name.startswith("BMRETRIEVER-"):
        model_dir = root / model_name
        if model_dir.exists():
            for dataset in DATASET_ORDER:
                path = model_dir / f"{dataset}.json"
                if not path.exists():
                    return {}
                metrics[dataset] = json.loads(path.read_text(encoding="utf-8"))
            return metrics

        for dataset in DATASET_ORDER:
            path = root / dataset / f"{model_name}.json"
            if not path.exists():
                return {}
            metrics[dataset] = json.loads(path.read_text(encoding="utf-8"))
        return metrics

    model_dir = root / model_name
    if not model_dir.exists():
        return {}

    for dataset in DATASET_ORDER:
        for candidate in DATASET_FILE_CANDIDATES[dataset]:
            path = model_dir / candidate
            if path.exists():
                metrics[dataset] = json.loads(path.read_text(encoding="utf-8"))
                break

    if len(metrics) != len(DATASET_ORDER):
        return {}
    return metrics


def collect_rows(root: Path) -> Dict[str, Tuple[str, Dict[str, Dict[str, float]], float, float, float]]:
    rows: Dict[str, Tuple[str, Dict[str, Dict[str, float]], float, float, float]] = {}
    for _, model_names in SECTION_MODEL_ORDER:
        for model_name in model_names:
            metrics = load_model_metrics(root, model_name)
            if not metrics:
                continue
            avg_ndcg = sum(metrics[dataset]["NDCG@10"] for dataset in DATASET_ORDER) / len(DATASET_ORDER)
            avg_recall = sum(metrics[dataset]["Recall@10"] for dataset in DATASET_ORDER) / len(DATASET_ORDER)
            avg = (avg_ndcg + avg_recall) / 2.0
            rows[model_name] = (model_name, metrics, avg_ndcg, avg_recall, avg)
    return rows


def build_section_lines(section_rows: List[Tuple[str, Dict[str, Dict[str, float]], float, float, float]], fmt) -> List[str]:
    lines = [
        "| Model | DermaSynth-E3 NDCG@10 (%) | DermaSynth-E3 Recall@10 (%) | MedMCQA NDCG@10 (%) | MedMCQA Recall@10 (%) | MedQuAD NDCG@10 (%) | MedQuAD Recall@10 (%) | SCE-Derma-SQ NDCG@10 (%) | SCE-Derma-SQ Recall@10 (%) | Avg NDCG@10 (%) | Avg Recall@10 (%) | Avg (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for model_name, metrics, avg_ndcg, avg_recall, avg in section_rows:
        lines.append(
            "| {model} | {d_ndcg} | {d_recall} | {m_ndcg} | {m_recall} | {mq_ndcg} | {mq_recall} | {s_ndcg} | {s_recall} | {avg_ndcg} | {avg_recall} | {avg} |".format(
                model=MODEL_DISPLAY.get(model_name, model_name),
                d_ndcg=fmt(metrics["DermaSynth-E3"]["NDCG@10"] * 100.0, "DermaSynth-E3_ndcg"),
                d_recall=fmt(metrics["DermaSynth-E3"]["Recall@10"] * 100.0, "DermaSynth-E3_recall"),
                m_ndcg=fmt(metrics["MedMCQA"]["NDCG@10"] * 100.0, "MedMCQA_ndcg"),
                m_recall=fmt(metrics["MedMCQA"]["Recall@10"] * 100.0, "MedMCQA_recall"),
                mq_ndcg=fmt(metrics["MedQuAD"]["NDCG@10"] * 100.0, "MedQuAD_ndcg"),
                mq_recall=fmt(metrics["MedQuAD"]["Recall@10"] * 100.0, "MedQuAD_recall"),
                s_ndcg=fmt(metrics["SCE-Derma-SQ"]["NDCG@10"] * 100.0, "SCE-Derma-SQ_ndcg"),
                s_recall=fmt(metrics["SCE-Derma-SQ"]["Recall@10"] * 100.0, "SCE-Derma-SQ_recall"),
                avg_ndcg=fmt(avg_ndcg * 100.0, "avg_ndcg"),
                avg_recall=fmt(avg_recall * 100.0, "avg_recall"),
                avg=fmt(avg * 100.0, "avg"),
            )
        )
    lines.append("")
    return lines


def build_markdown(rows_by_model: Dict[str, Tuple[str, Dict[str, Dict[str, float]], float, float, float]]) -> str:
    ordered_rows = list(rows_by_model.values())
    best_by_column: Dict[str, float] = {}
    column_keys = []
    for dataset in DATASET_ORDER:
        column_keys.extend([f"{dataset}_ndcg", f"{dataset}_recall"])
    column_keys.extend(["avg_ndcg", "avg_recall", "avg"])

    for key in column_keys:
        best_by_column[key] = float("-inf")

    for _, metrics, avg_ndcg, avg_recall, avg in ordered_rows:
        for dataset in DATASET_ORDER:
            best_by_column[f"{dataset}_ndcg"] = max(best_by_column[f"{dataset}_ndcg"], metrics[dataset]["NDCG@10"] * 100.0)
            best_by_column[f"{dataset}_recall"] = max(best_by_column[f"{dataset}_recall"], metrics[dataset]["Recall@10"] * 100.0)
        best_by_column["avg_ndcg"] = max(best_by_column["avg_ndcg"], avg_ndcg * 100.0)
        best_by_column["avg_recall"] = max(best_by_column["avg_recall"], avg_recall * 100.0)
        best_by_column["avg"] = max(best_by_column["avg"], avg * 100.0)

    def fmt(value: float, key: str) -> str:
        text = f"{value:.2f}"
        if abs(value - best_by_column[key]) < 1e-9:
            return f"**{text}**"
        return text

    lines = [
        "# RT Nonhomo Full Results",
        "",
        "Metrics are reported as percentages on the four-dataset setting: `DermaSynth-E3`, `MedMCQA`, `MedQuAD`, and `SCE-Derma-SQ`.",
        "`Avg (%) = (Avg NDCG@10 + Avg Recall@10) / 2`.",
        "",
    ]

    for section_title, model_names in SECTION_MODEL_ORDER:
        section_rows = [rows_by_model[model_name] for model_name in model_names if model_name in rows_by_model]
        if not section_rows:
            continue
        lines.append(f"## {section_title}")
        lines.append("")
        lines.extend(build_section_lines(section_rows, fmt))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render global RT nonhomo-full results markdown")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Root directory containing per-model result JSON files")
    parser.add_argument("--output", default=None, help="Output markdown path, defaults to <root>/results.md")
    args = parser.parse_args()

    root = Path(args.root)
    output_path = Path(args.output) if args.output else root / "results.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(collect_rows(root)), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
