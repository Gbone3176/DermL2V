from __future__ import annotations

import argparse
import json
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


def collect_rows(method_dir: Path, datasets: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cp_dirs = sorted(
        (path for path in method_dir.iterdir() if path.is_dir() and path.name.startswith("cp")),
        key=lambda path: int(path.name.replace("cp", "")),
    )
    for cp_dir in cp_dirs:
        metrics = {}
        missing = False
        for file_stem, _ in datasets:
            path = cp_dir / f"{file_stem}.json"
            if not path.exists():
                missing = True
                break
            metrics[file_stem] = load_json(path)
        if missing:
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


def best_steps(rows: list[dict[str, Any]], datasets: list[tuple[str, str]]) -> dict[str, str]:
    best: dict[str, tuple[float, str]] = {}
    for row in rows:
        cp = str(row["cp"])
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
            if key not in best or value > best[key][0]:
                best[key] = (value, cp)
    return {key: cp for key, (_, cp) in best.items()}


def pct(value: float) -> str:
    return f"{value * 100.0:.2f}"


def fmt_pct(value: float, is_best: bool) -> str:
    rendered = pct(value)
    return f"**{rendered}**" if is_best else rendered


def build_markdown(method_dir: Path, rows: list[dict[str, Any]], run_root: str | None, datasets: list[tuple[str, str]]) -> str:
    best = best_steps(rows, datasets)
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
            values.append(fmt_pct(metrics[file_stem]["NDCG@10"], best.get(f"{file_stem}_NDCG@10") == cp))
            values.append(fmt_pct(metrics[file_stem]["Recall@10"], best.get(f"{file_stem}_Recall@10") == cp))
        values.extend(
            [
                fmt_pct(float(row["avg_ndcg"]), best.get("Avg_NDCG@10") == cp),
                fmt_pct(float(row["avg_recall"]), best.get("Avg_Recall@10") == cp),
                fmt_pct(float(row["avg"]), best.get("Avg") == cp),
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
