#!/usr/bin/env python
"""Evaluate BMRetriever-7B LoRA checkpoints on RT nonhomo-full datasets."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_bmretriever_full import (  # noqa: E402
    build_results_dot,
    format_passage,
    format_query,
)
from experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_full_utils import (  # noqa: E402
    build_corpus_queries,
    dataset_output_name,
    evaluate_at_10,
    load_jsonl,
)


LOGGER = logging.getLogger("eval_bmretriever7b_lora_rt_full")

DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "ContrastiveModel/BMRetriever7B/output/20260501_021339_bmretriever7b_lora-r16_a32_b1_ga16_ep1.0_lr1e-05_tau1_fp16"
)
DEFAULT_BASE_MODEL = (
    "/cache/transformers_cache/models--BMRetriever--BMRetriever-7B/"
    "snapshots/13e6adb9273c5f254e037987d6b44e9e4b005b9a"
)
DEFAULT_DATASET_ROOT = Path("/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "ContrastiveModel/BMRetriever7B/rt_full_eval/output"
DEFAULT_QUERY_INSTRUCTION = (
    "Given a dermatologic question, return the answer that most closely corresponds to the information being asked for."
)
DEFAULT_DATASETS = [
    "eval3-text-benchmark_split_choices.jsonl",
    "MedMCQA_RT_query_doc.jsonl",
    "MedQuAD_dermatology_qa_retrieval_doclt300.jsonl",
    "sce_retrieval.jsonl",
]
DISPLAY_NAMES = {
    "DermSynth_knowledgebase": "DermaSynth-E3",
    "MedMCQA_RT": "MedMCQA",
    "MedQuAD_dermatology_qa_retrieval_doclt300": "MedQuAD",
    "SCE-Derma-SQ": "SCE-Derma-SQ",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", action="append", default=None, help="Dataset jsonl path or name under dataset_root.")
    parser.add_argument("--checkpoint", action="append", default=None, help="Checkpoint label/path. Defaults to all.")
    parser.add_argument("--query_instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary_name", default="summary_at10.md")
    parser.add_argument("--summary_title", default="BMRetriever-7B LoRA RT Nonhomo-Full Summary")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    output_dir = args.output_root / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = resolve_checkpoints(run_dir, args.checkpoint)
    datasets = resolve_datasets(args.dataset_root, args.dataset)
    device = torch.device(args.device)

    LOGGER.info("Evaluating %d checkpoints on %d datasets", len(checkpoints), len(datasets))
    LOGGER.info("Output=%s", output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for label, checkpoint_path in checkpoints:
        checkpoint_output = output_dir / label
        checkpoint_output.mkdir(parents=True, exist_ok=True)
        pending = [
            dataset
            for dataset in datasets
            if args.overwrite or not dataset_metric_file(checkpoint_output, dataset).exists()
        ]
        if not pending:
            LOGGER.info("All datasets already complete for %s", label)
            continue

        LOGGER.info("Loading base model %s with adapter %s", args.base_model, checkpoint_path)
        model_kwargs = {"local_files_only": True}
        if device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        base_model = AutoModel.from_pretrained(args.base_model, **model_kwargs)
        model = PeftModel.from_pretrained(base_model, checkpoint_path, local_files_only=True).to(device).eval()

        for dataset in pending:
            metrics = evaluate_dataset(
                dataset_path=dataset,
                tokenizer=tokenizer,
                model=model,
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_samples=args.max_samples,
                query_instruction=args.query_instruction,
                device=device,
            )
            out_file = dataset_metric_file(checkpoint_output, dataset)
            with out_file.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=4)
            LOGGER.info("%s %s %s", label, dataset_output_name(str(dataset)), json.dumps(metrics))

        del model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = output_dir / args.summary_name
    write_summary(summary_path, output_dir, checkpoints, datasets, args.summary_title)
    LOGGER.info("Wrote summary to %s", summary_path)


def resolve_checkpoints(run_dir: Path, requested: list[str] | None) -> list[tuple[str, Path]]:
    if requested:
        resolved = []
        for item in requested:
            path = Path(item)
            if not path.is_absolute():
                path = run_dir / item
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint does not exist: {path}")
            resolved.append((checkpoint_label(path), path.resolve()))
        return sorted(resolved, key=lambda pair: checkpoint_sort_key(pair[0]))

    checkpoints = [(checkpoint_label(path), path.resolve()) for path in run_dir.glob("checkpoint-*") if path.is_dir()]
    final_dir = run_dir / "final"
    if final_dir.is_dir():
        checkpoints.append(("final", final_dir.resolve()))
    if not checkpoints:
        raise RuntimeError(f"No checkpoint-* or final directories found under {run_dir}")
    return sorted(checkpoints, key=lambda pair: checkpoint_sort_key(pair[0]))


def checkpoint_label(path: Path) -> str:
    if path.name.startswith("checkpoint-"):
        return "step-" + path.name.split("checkpoint-", 1)[1]
    return path.name


def checkpoint_sort_key(label: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"step-(\d+)", label)
    if match:
        return (0, int(match.group(1)))
    if label == "final":
        return (1, math.inf)
    return (2, label)


def resolve_datasets(dataset_root: Path, requested: list[str] | None) -> list[Path]:
    names = requested or DEFAULT_DATASETS
    paths = []
    for name in names:
        path = Path(name)
        if not path.is_absolute():
            path = dataset_root / name
        if not path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {path}")
        paths.append(path.resolve())
    return list(dict.fromkeys(paths))


def dataset_metric_file(output_dir: Path, dataset_path: Path) -> Path:
    return output_dir / f"{dataset_output_name(str(dataset_path))}.json"


def evaluate_dataset(
    dataset_path: Path,
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    max_samples: int,
    query_instruction: str,
    device: torch.device,
) -> dict[str, float]:
    dataset = load_jsonl(str(dataset_path))
    if max_samples and max_samples > 0:
        dataset = dataset[:max_samples]
    corpus, queries, relevant_docs = build_corpus_queries(dataset)
    if not queries or not corpus or not relevant_docs:
        raise ValueError(f"No valid retrieval samples were built from {dataset_path}")

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [format_query(queries[qid], query_instruction) for qid in query_ids]
    corpus_texts = [format_passage(corpus[cid]["text"]) for cid in corpus_ids]

    LOGGER.info(
        "Encoding %s: %d queries, %d docs",
        dataset_output_name(str(dataset_path)),
        len(query_texts),
        len(corpus_texts),
    )
    q_emb = encode_batches(tokenizer, model, query_texts, batch_size, device, max_length, "queries")
    d_emb = encode_batches(tokenizer, model, corpus_texts, batch_size, device, max_length, "docs")
    return evaluate_at_10(relevant_docs, build_results_dot(q_emb, d_emb, query_ids, corpus_ids), len(corpus_ids))


def encode_batches(tokenizer, model, texts: list[str], batch_size: int, device: torch.device, max_length: int, desc: str):
    all_embeddings = []
    total = (len(texts) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, total=total):
        batch = texts[start : start + batch_size]
        all_embeddings.append(encode_texts(tokenizer, model, batch, device, max_length).cpu())
    return torch.cat(all_embeddings, dim=0)


def encode_texts(tokenizer, model, texts: list[str], device: torch.device, max_length: int) -> torch.Tensor:
    enc = tokenize_with_eos(tokenizer, texts, max_length=max_length, device=device)
    with torch.no_grad():
        outputs = model(**enc)
        embeddings = last_token_pool(outputs.last_hidden_state, enc["attention_mask"]).float()
    return embeddings


def tokenize_with_eos(tokenizer, texts: list[str], max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    rows = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=max_length - 1,
    )["input_ids"]
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("BMRetriever tokenizer does not define eos_token_id.")
    rows = [row + [eos_token_id] for row in rows]
    padded = tokenizer.pad({"input_ids": rows}, padding=True, return_attention_mask=True, return_tensors="pt")
    return {key: value.to(device) for key, value in padded.items()}


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0]).item()
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    batch_indices = torch.arange(batch_size, device=last_hidden_states.device)
    return last_hidden_states[batch_indices, sequence_lengths]


def write_summary(
    summary_path: Path,
    output_dir: Path,
    checkpoints: list[tuple[str, Path]],
    datasets: list[Path],
    title: str,
) -> None:
    rows = []
    for label, _ in checkpoints:
        checkpoint_output = output_dir / label
        metrics_by_dataset = {}
        for dataset in datasets:
            path = dataset_metric_file(checkpoint_output, dataset)
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    metrics_by_dataset[dataset_output_name(str(dataset))] = json.load(handle)
        if metrics_by_dataset:
            rows.append((label, metrics_by_dataset))

    lines = [f"# {title}", ""]
    if not rows:
        lines.extend(["No completed results found.", ""])
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        return

    dataset_names = [dataset_output_name(str(dataset)) for dataset in datasets]
    headers = ["Checkpoint"]
    for name in dataset_names:
        display = DISPLAY_NAMES.get(name, name)
        headers.extend([f"{display} NDCG@10 (%)", f"{display} Recall@10 (%)"])
    headers.extend(["Avg_NDCG@10 (%)", "Avg_Recall@10 (%)", "Avg (%)"])

    table = [headers]
    for label, metrics_by_dataset in rows:
        row = [label]
        ndcgs = []
        recalls = []
        for name in dataset_names:
            metrics = metrics_by_dataset.get(name)
            if not metrics:
                row.extend(["-", "-"])
                continue
            ndcg = float(metrics["NDCG@10"]) * 100
            recall = float(metrics["Recall@10"]) * 100
            ndcgs.append(ndcg)
            recalls.append(recall)
            row.extend([f"{ndcg:.2f}", f"{recall:.2f}"])
        avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        row.extend([f"{avg_ndcg:.2f}", f"{avg_recall:.2f}", f"{(avg_ndcg + avg_recall) / 2:.2f}"])
        table.append(row)

    lines.extend(render_markdown_table(table))
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def render_markdown_table(table: list[list[str]]) -> list[str]:
    widths = [max(len(str(row[i])) for row in table) for i in range(len(table[0]))]
    lines = []
    lines.append("| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(table[0])) + " |")
    lines.append("| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |")
    for row in table[1:]:
        lines.append("| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |")
    return lines


if __name__ == "__main__":
    main()
