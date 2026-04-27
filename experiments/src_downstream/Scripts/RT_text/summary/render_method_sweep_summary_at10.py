from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DATASETS = [
    ("DermSynth_knowledgebase", "DermaSynth-E3"),
    ("MedMCQA_RT", "MedMCQA"),
    ("MedQuAD_dermatology_qa_retrieval_doclt300", "MedQuAD"),
    ("sce_retrieval", "SCE-Derma-SQ"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configured_datasets(config_path: Path | None) -> list[tuple[str, str]]:
    if config_path is None:
        return DATASETS
    cfg = load_json(config_path)
    configured = cfg.get("rt_full_datasets")
    if not configured:
        return DATASETS
    return [
        (str(item.get("output_stem") or item["name"]), str(item.get("display_name") or item["name"]))
        for item in configured
    ]


def cp_step(cp_dir: Path) -> int | None:
    if not cp_dir.name.startswith("cp"):
        return None
    try:
        return int(cp_dir.name[2:])
    except ValueError:
        return None


def collect_rows(method_dir: Path, datasets: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cp_dirs = sorted(
        (path for path in method_dir.iterdir() if path.is_dir() and cp_step(path) is not None),
        key=lambda path: cp_step(path) or -1,
    )
    for cp_dir in cp_dirs:
        metrics = {}
        missing_files = []
        for file_stem, _ in datasets:
            path = cp_dir / f"{file_stem}.json"
            if not path.exists():
                missing_files.append(path.name)
                break
            metrics[file_stem] = load_json(path)
        if missing_files:
            print(
                f"Skipping incomplete checkpoint {cp_dir}: missing {', '.join(missing_files)}",
                file=sys.stderr,
            )
            continue
        avg_ndcg = sum(metrics[name]["NDCG@10"] for name, _ in datasets) / len(datasets)
        avg_recall = sum(metrics[name]["Recall@10"] for name, _ in datasets) / len(datasets)
        rows.append(
            {
                "cp": cp_dir.name,
                "metrics": metrics,
                "avg_ndcg": avg_ndcg,
                "avg_recall": avg_recall,
                "avg": (avg_ndcg + avg_recall) / 2.0,
            }
        )
    return rows


def best_values(rows: list[dict[str, Any]], datasets: list[tuple[str, str]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for row in rows:
        metrics = row["metrics"]
        candidate_metrics = {
            "Avg_NDCG@10": float(row["avg_ndcg"]),
            "Avg_Recall@10": float(row["avg_recall"]),
            "Avg": float(row["avg"]),
        }
        for file_stem, _ in datasets:
            candidate_metrics[f"{file_stem}_NDCG@10"] = float(metrics[file_stem]["NDCG@10"])
            candidate_metrics[f"{file_stem}_Recall@10"] = float(metrics[file_stem]["Recall@10"])
        for key, value in candidate_metrics.items():
            best[key] = max(best.get(key, float("-inf")), value)
    return best


def pct(value: float) -> str:
    return f"{value * 100.0:.2f}"


def fmt_pct(value: float, is_best: bool) -> str:
    rendered = pct(value)
    return f"**{rendered}**" if is_best else rendered


def is_best(value: float, best: dict[str, float], key: str) -> bool:
    return abs(value - best[key]) < 1e-12


def build_markdown(method_dir: Path, rows: list[dict[str, Any]], run_root: str | None, datasets: list[tuple[str, str]]) -> str:
    best = best_values(rows, datasets)
    lines = [
        "# Nonhomo Full Sweep Summary at @10",
        "",
    ]
    if run_root:
        lines.append(f"Run root: `{run_root}`")
    lines.append(f"Output dir: `{method_dir}`")
    lines.extend(
        [
            "",
            "Metrics are reported as percentages.",
            "Avg means `(Avg NDCG@10 + Avg Recall@10) / 2` across the configured nonhomo-full datasets.",
            "",
            "| CP | "
            + " | ".join(
                f"{display} NDCG@10 (%) | {display} Recall@10 (%)" for _, display in datasets
            )
            + " | Avg_NDCG@10 (%) | Avg_Recall@10 (%) | Avg (%) |",
            "|---:|" + "---:|" * (len(datasets) * 2 + 3),
        ]
    )
    for row in rows:
        cp = str(row["cp"])
        metrics = row["metrics"]
        values = [cp]
        for file_stem, _ in datasets:
            ndcg = float(metrics[file_stem]["NDCG@10"])
            recall = float(metrics[file_stem]["Recall@10"])
            values.append(fmt_pct(ndcg, is_best(ndcg, best, f"{file_stem}_NDCG@10")))
            values.append(fmt_pct(recall, is_best(recall, best, f"{file_stem}_Recall@10")))
        values.extend(
            [
                fmt_pct(float(row["avg_ndcg"]), is_best(float(row["avg_ndcg"]), best, "Avg_NDCG@10")),
                fmt_pct(float(row["avg_recall"]), is_best(float(row["avg_recall"]), best, "Avg_Recall@10")),
                fmt_pct(float(row["avg"]), is_best(float(row["avg"]), best, "Avg")),
            ]
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render summary_at10.md for one new-layout RT-nonhomo-full method directory")
    parser.add_argument("--method-dir", required=True)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config-path", default=None)
    args = parser.parse_args()

    method_dir = Path(args.method_dir)
    datasets = configured_datasets(Path(args.config_path) if args.config_path else None)
    rows = collect_rows(method_dir, datasets)
    if not rows:
        raise FileNotFoundError(f"No complete cp* rows found under {method_dir}")
    output_path = Path(args.output) if args.output else method_dir / "summary_at10.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(method_dir, rows, args.run_root, datasets), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
