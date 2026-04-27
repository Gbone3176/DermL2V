from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RT_TEXT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATHS = [
    RT_TEXT_ROOT / "configs" / "derml2v_loss02_rt_full_eval_paths.json",
]

DEFAULT_FULL_DATASETS = [
    ("DermSynth_knowledgebase", "eval3-text-benchmark_split_choices.jsonl"),
    ("MedMCQA_RT", "MedMCQA_RT_query_doc.jsonl"),
    ("MedQuAD_dermatology_qa_retrieval_doclt300", "MedQuAD_dermatology_qa_retrieval_doclt300.jsonl"),
    ("sce_retrieval", "sce_retrieval.jsonl"),
]


@dataclass
class ResolvedCheckpoint:
    config_path: Path
    config: dict[str, Any]
    method_key: str
    method: dict[str, Any]
    checkpoint_dir: Path
    checkpoint_step: int
    checkpoint_output_dir: Path
    model_name: str


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_step(checkpoint_dir: Path) -> int:
    name = checkpoint_dir.name
    if not name.startswith("checkpoint-"):
        raise ValueError(f"Checkpoint dir must end with checkpoint-<step>, got: {checkpoint_dir}")
    try:
        return int(name.split("-", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Invalid checkpoint step in {checkpoint_dir}") from exc


def _candidate_methods(config_path: Path) -> list[tuple[str, dict[str, Any]]]:
    cfg = load_json(config_path)
    methods = []
    for method_key, raw_method in cfg.get("methods", {}).items():
        method = dict(raw_method)
        run_root = method.get("run_root")
        if not run_root:
            continue
        method["run_root"] = Path(run_root)
        methods.append((method_key, method))
    return methods


def resolve_checkpoint(
    checkpoint_dir: Path,
    method_key: str | None = None,
    config_paths: list[Path] | None = None,
) -> ResolvedCheckpoint:
    checkpoint_dir = checkpoint_dir.resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir does not exist: {checkpoint_dir}")
    step = checkpoint_step(checkpoint_dir)
    config_paths = config_paths or DEFAULT_CONFIG_PATHS

    best_match: tuple[int, Path, dict[str, Any], str, dict[str, Any]] | None = None
    for config_path in config_paths:
        cfg = load_json(config_path)
        for candidate_key, candidate_method in _candidate_methods(config_path):
            if method_key is not None and candidate_key != method_key:
                continue
            run_root = candidate_method["run_root"]
            try:
                checkpoint_dir.relative_to(run_root)
            except ValueError:
                continue
            match_len = len(run_root.parts)
            if best_match is None or match_len > best_match[0]:
                best_match = (match_len, config_path, cfg, candidate_key, candidate_method)

    if best_match is None:
        raise KeyError(f"Could not resolve checkpoint to a configured experiment: {checkpoint_dir}")

    _, config_path, cfg, resolved_method_key, method = best_match
    output_family = method.get("output_family")
    output_method = method.get("output_method")
    output_params = method.get("output_params")
    if not output_family or not output_method:
        raise KeyError(
            f"Method {resolved_method_key} in {config_path} is missing output_family/output_method"
        )
    model_prefix = method.get("model_prefix")
    if not model_prefix:
        raise KeyError(f"Method {resolved_method_key} in {config_path} is missing model_prefix")

    output_root = Path(cfg["rt_nonhomo_full_output_root"]) / output_family / output_method
    if output_params:
        output_root = output_root / output_params
    cp_output_dir = output_root / f"cp{step}"
    return ResolvedCheckpoint(
        config_path=config_path,
        config=cfg,
        method_key=resolved_method_key,
        method=method,
        checkpoint_dir=checkpoint_dir,
        checkpoint_step=step,
        checkpoint_output_dir=cp_output_dir,
        model_name=f"{model_prefix}_cp{step}",
    )


def dataset_file_paths(config: dict[str, Any] | str | Path) -> list[Path]:
    if isinstance(config, dict):
        configured = config.get("rt_full_datasets")
        if configured:
            paths = []
            for item in configured:
                path = item.get("path") if isinstance(item, dict) else item
                paths.append(Path(path))
            return paths
        root = Path(config["rt_dataset_root"])
    else:
        root = Path(config)
    return [root / filename for _, filename in DEFAULT_FULL_DATASETS]
