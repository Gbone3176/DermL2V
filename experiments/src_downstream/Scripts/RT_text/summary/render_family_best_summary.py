from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


DATASETS = [
    ("DermaSynth-E3", "DermSynth_knowledgebase.json"),
    ("MedMCQA", "MedMCQA_RT.json"),
    ("MedQuAD", "MedQuAD_dermatology_qa_retrieval_doclt300.json"),
    ("SCE-Derma-SQ", "sce_retrieval.json"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(config_path: Path) -> dict[str, Any]:
    return load_json(config_path)


def configured_datasets(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    configured = cfg.get("rt_full_datasets")
    if not configured:
        return DATASETS
    return [
        (
            str(item.get("display_name") or item["name"]),
            f"{item.get('output_stem') or item['name']}.json",
        )
        for item in configured
    ]


def method_dir(cfg: dict[str, Any], method: dict[str, Any]) -> Path:
    root = Path(cfg["rt_nonhomo_full_output_root"]) / str(method["output_family"]) / str(method["output_method"])
    output_params = method.get("output_params")
    if output_params:
        root = root / str(output_params)
    return root


def summarize_cp_dir(cp_dir: Path, datasets: list[tuple[str, str]]) -> dict[str, Any] | None:
    metrics = {}
    for dataset_display, file_name in datasets:
        path = cp_dir / file_name
        if not path.exists():
            return None
        metrics[dataset_display] = load_json(path)
    avg_ndcg = sum(metrics[dataset]["NDCG@10"] for dataset, _ in datasets) / len(datasets)
    avg_recall = sum(metrics[dataset]["Recall@10"] for dataset, _ in datasets) / len(datasets)
    return {
        "cp": cp_dir.name,
        "metrics": metrics,
        "avg_ndcg": avg_ndcg,
        "avg_recall": avg_recall,
        "avg": (avg_ndcg + avg_recall) / 2.0,
    }


def best_summary(base_dir: Path, datasets: list[tuple[str, str]]) -> dict[str, Any] | None:
    candidates = []
    if not base_dir.exists():
        return None
    for cp_dir in sorted(path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("cp")):
        summary = summarize_cp_dir(cp_dir, datasets)
        if summary is not None:
            candidates.append(summary)
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item["avg"], item["avg_ndcg"], int(str(item["cp"]).replace("cp", ""))))


def pct(value: float) -> str:
    return f"{value * 100.0:.2f}"


def fmt_pct(value: float, is_best: bool) -> str:
    rendered = pct(value)
    return f"**{rendered}**" if is_best else rendered


def render_table(rows: list[dict[str, Any]], datasets: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return ["No complete results found.", ""]
    best: dict[str, float] = {}
    for dataset_display, _ in datasets:
        best[f"{dataset_display}_ndcg"] = max(float(row["metrics"][dataset_display]["NDCG@10"]) for row in rows)
        best[f"{dataset_display}_recall"] = max(float(row["metrics"][dataset_display]["Recall@10"]) for row in rows)
    best["avg_ndcg"] = max(float(row["avg_ndcg"]) for row in rows)
    best["avg_recall"] = max(float(row["avg_recall"]) for row in rows)
    best["avg"] = max(float(row["avg"]) for row in rows)
    lines = [
        "| Setting | Best Step | "
        + " | ".join(
            f"{display} NDCG@10 (%) | {display} Recall@10 (%)" for display, _ in datasets
        )
        + " | Avg NDCG@10 (%) | Avg Recall@10 (%) | Avg (%) |",
        "|---|---|" + "---:|" * (len(datasets) * 2 + 3),
    ]
    for row in rows:
        metrics = row["metrics"]
        values = [row["label"], row["cp"]]
        for dataset_display, _ in datasets:
            values.append(
                fmt_pct(float(metrics[dataset_display]["NDCG@10"]), float(metrics[dataset_display]["NDCG@10"]) == best[f"{dataset_display}_ndcg"])
            )
            values.append(
                fmt_pct(float(metrics[dataset_display]["Recall@10"]), float(metrics[dataset_display]["Recall@10"]) == best[f"{dataset_display}_recall"])
            )
        values.extend(
            [
                fmt_pct(float(row["avg_ndcg"]), float(row["avg_ndcg"]) == best["avg_ndcg"]),
                fmt_pct(float(row["avg_recall"]), float(row["avg_recall"]) == best["avg_recall"]),
                fmt_pct(float(row["avg"]), float(row["avg"]) == best["avg"]),
            ]
        )
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return lines


def section_title(output_method: str) -> str:
    return output_method.replace("_", " ")


def row_label(method: dict[str, Any]) -> str:
    output_params = method.get("output_params")
    if output_params:
        return str(output_params)
    return str(method.get("display_name") or method.get("output_method") or "default")


def build_sections(cfg: dict[str, Any], output_family: str | None = None) -> list[str]:
    datasets = configured_datasets(cfg)
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for _, method in cfg.get("methods", {}).items():
        if "output_family" not in method or "output_method" not in method:
            continue
        if output_family is not None and str(method["output_family"]) != output_family:
            continue
        key = str(method["output_method"])
        grouped.setdefault(key, []).append(method)

    title_family = output_family or "DermL2V"
    sections = [
        f"# {title_family} RT Nonhomo Full Best-Step Summary",
        "",
        "Metrics are reported as percentages. All tables use the configured nonhomo-full dataset set.",
        "`Avg (%) = (Avg NDCG@10 + Avg Recall@10) / 2`.",
        "",
    ]

    for output_method, methods in grouped.items():
        rows = []
        for method in methods:
            summary = best_summary(method_dir(cfg, method), datasets)
            if summary is None:
                continue
            rows.append(
                {
                    "label": row_label(method),
                    "cp": summary["cp"],
                    "metrics": summary["metrics"],
                    "avg_ndcg": summary["avg_ndcg"],
                    "avg_recall": summary["avg_recall"],
                    "avg": summary["avg"],
                }
            )
        if not rows:
            continue
        sections.append(f"## {section_title(output_method)}")
        sections.append("")
        sections.extend(render_table(rows, datasets))
    return sections


def main() -> None:
    parser = argparse.ArgumentParser(description="Render sum_DermL2V.md from new-layout RT-nonhomo-full method directories")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--method-key", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config_path = Path(args.config_path)
    cfg = load_config(config_path)
    if args.method_key:
        family_method = cfg["methods"][args.method_key]
    else:
        family_method = next(iter(cfg["methods"].values()))
    output_family = str(family_method["output_family"])
    family_root = Path(cfg["rt_nonhomo_full_output_root"]) / output_family
    output_path = Path(args.output) if args.output else family_root / "sum_DermL2V.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(build_sections(cfg, output_family=output_family)), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
