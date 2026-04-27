from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


DATASET_CANDIDATES: List[Tuple[str, List[str]]] = [
    ("DermaSynth-E3", ["DermaSynth-E3.json", "DermSynth_knowledgebase.json"]),
    ("MedMCQA", ["MedMCQA.json", "MedMCQA_RT.json"]),
    ("MedQuAD", ["MedQuAD.json", "MedQuAD_dermatology_qa_retrieval_doclt300.json"]),
    ("SCE-Derma-SQ", ["SCE-Derma-SQ.json", "sce_retrieval.json"]),
]


def build_summary(model_dir: Path) -> str:
    metrics: Dict[str, Dict[str, float]] = {}
    for dataset, candidates in DATASET_CANDIDATES:
        for candidate in candidates:
            dataset_path = model_dir / candidate
            if dataset_path.exists():
                metrics[dataset] = json.loads(dataset_path.read_text(encoding="utf-8"))
                break

    if not metrics:
        raise FileNotFoundError(f"No dataset json files found under {model_dir}")

    rows = []
    ndcgs = []
    recalls = []
    for dataset, _ in DATASET_CANDIDATES:
        if dataset not in metrics:
            continue
        ndcg = metrics[dataset]["NDCG@10"] * 100.0
        recall = metrics[dataset]["Recall@10"] * 100.0
        ndcgs.append(ndcg)
        recalls.append(recall)
        rows.append(f"| {dataset} | {ndcg:.2f} | {recall:.2f} |")

    avg_ndcg = sum(ndcgs) / len(ndcgs)
    avg_recall = sum(recalls) / len(recalls)
    avg = (avg_ndcg + avg_recall) / 2.0

    lines = [
        f"# {model_dir.name} RT-Nonhomo Full Summary",
        "",
        "Metrics are reported as percentages.",
        "",
        "| Dataset | NDCG@10 (%) | Recall@10 (%) |",
        "|---|---:|---:|",
        *rows,
        f"| Avg | {avg_ndcg:.2f} | {avg_recall:.2f} |",
        "",
        f"Overall Avg (%) = ({avg_ndcg:.2f} + {avg_recall:.2f}) / 2 = {avg:.2f}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render summary for contrastive/non-DermL2V RT nonhomo full results")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_path = Path(args.output) if args.output else model_dir / "summary.md"
    output_path.write_text(build_summary(model_dir), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
