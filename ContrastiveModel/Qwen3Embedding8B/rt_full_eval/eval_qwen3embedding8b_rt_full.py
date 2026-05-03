"""Evaluate Qwen3-Embedding-8B LoRA checkpoints on RT nonhomo-full datasets."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_full_utils import (  # noqa: E402
    build_corpus_queries,
    build_results,
    dataset_output_name,
    evaluate_at_10,
    load_jsonl,
)


LOGGER = logging.getLogger("eval_qwen3embedding8b_rt_full")

DEFAULT_RUN_DIR = REPO_ROOT / "ContrastiveModel/Qwen3Embedding8B/output"
DEFAULT_BASE_MODEL = "/cache/modelscope/hub/models/Qwen/Qwen3-Embedding-8B"
DEFAULT_DATASET_ROOT = Path("/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output/downstream/RT_text/nonhomo-full/Qwen3Embedding8B"
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
    "sce_retrieval": "SCE-Derma-SQ",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", action="append", default=None, help="Dataset jsonl path or name under dataset_root.")
    parser.add_argument("--checkpoint", action="append", default=None, help="Checkpoint label/path. Defaults to all.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--query_task_name", default="RT_text")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary_name", default="summary_at10.md")
    parser.add_argument("--summary_title", default="Qwen3-Embedding-8B LoRA RT Nonhomo-Full Summary")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    run_name = run_dir.name
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = resolve_checkpoints(run_dir, args.checkpoint)
    datasets = resolve_datasets(args.dataset_root, args.dataset)
    device = torch.device(args.device)
    normalize = not args.no_normalize

    LOGGER.info("Evaluating %d checkpoints on %d datasets", len(checkpoints), len(datasets))
    LOGGER.info("Run=%s normalize=%s output=%s", run_dir, normalize, output_dir)

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

        embedding_config = load_embedding_config(checkpoint_path, run_dir)
        base_model = embedding_config.get("base_model") or args.base_model
        LOGGER.info("Loading base=%s adapter=%s", base_model, checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            use_fast=True,
            local_files_only=args.local_files_only,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
        model = AutoModel.from_pretrained(
            base_model,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            local_files_only=args.local_files_only,
        )
        model.config.use_cache = False
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False).to(device).eval()

        for dataset in pending:
            metrics = evaluate_dataset(
                dataset_path=dataset,
                tokenizer=tokenizer,
                model=model,
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_samples=args.max_samples,
                query_task_name=args.query_task_name,
                normalize=normalize,
                device=device,
            )
            out_file = dataset_metric_file(checkpoint_output, dataset)
            with out_file.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=4)
            LOGGER.info("%s %s %s", label, dataset_output_name(str(dataset)), json.dumps(metrics))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = output_dir / args.summary_name
    write_summary(summary_path, output_dir, checkpoints, datasets, args.summary_title)
    LOGGER.info("Wrote summary to %s", summary_path)


def resolve_run_dir(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir() and ((path / "final").is_dir() or any(path.glob("checkpoint-*"))):
        return path
    if path.is_dir():
        runs = [
            child
            for child in path.iterdir()
            if child.is_dir() and ((child / "final").is_dir() or any(child.glob("checkpoint-*")))
        ]
        if runs:
            return max(runs, key=lambda item: item.stat().st_mtime).resolve()
    raise FileNotFoundError(f"No Qwen3 training run found at or under: {path}")


def load_embedding_config(checkpoint_path: Path, run_dir: Path) -> dict:
    for config_path in (checkpoint_path / "embedding_config.json", run_dir / "final" / "embedding_config.json"):
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return {}


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
    name = path.name
    if name.startswith("checkpoint-"):
        return "step-" + name.split("checkpoint-", 1)[1]
    return name


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
    query_task_name: str,
    normalize: bool,
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
    query_texts = [format_query(query_task_name, queries[qid]) for qid in query_ids]
    corpus_texts = [format_passage(corpus[cid]["text"]) for cid in corpus_ids]

    LOGGER.info(
        "Encoding %s: %d queries, %d docs",
        dataset_output_name(str(dataset_path)),
        len(query_texts),
        len(corpus_texts),
    )
    q_emb = encode_batches(tokenizer, model, query_texts, batch_size, device, max_length, normalize, "queries")
    d_emb = encode_batches(tokenizer, model, corpus_texts, batch_size, device, max_length, normalize, "docs")
    return evaluate_at_10(relevant_docs, build_results(q_emb, d_emb, query_ids, corpus_ids), len(corpus_ids))


def encode_batches(
    tokenizer,
    model,
    texts: list[str],
    batch_size: int,
    device: torch.device,
    max_length: int,
    normalize: bool,
    desc: str,
) -> torch.Tensor:
    embeddings = []
    total = (len(texts) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(texts), batch_size), total=total, desc=f"Encoding {desc}", leave=False):
        batch = texts[start : start + batch_size]
        embeddings.append(encode_texts(tokenizer, model, batch, device, max_length, normalize).cpu())
    return torch.cat(embeddings, dim=0)


@torch.inference_mode()
def encode_texts(
    tokenizer,
    model,
    texts: list[str],
    device: torch.device,
    max_length: int,
    normalize: bool,
) -> torch.Tensor:
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = model(**encoded)
    pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"]).float()
    if normalize:
        pooled = F.normalize(pooled, p=2, dim=-1)
    return pooled


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]


def format_query(task_name: str, text: str) -> str:
    instruction = {
        "SemVariants": "Retrieve the dermatology description that has the same clinical meaning as the query.",
        "VisVariants": "Retrieve the visual dermatology description that best matches the diagnosis-style query.",
        "DermQA": "Retrieve the answer that best responds to the dermatology question.",
        "SI1": "Retrieve the answer that best matches the dermatology clinical scenario.",
        "SI2": "Retrieve the medically correct answer passage for the dermatology question.",
    }.get(task_name, "Retrieve the most relevant dermatology passage for the query.")
    return f"{instruction}\nQuery: {text}"


def format_passage(text: str) -> str:
    return f"Represent this passage\npassage: {text}"


def iter_metric_rows(output_dir: Path, checkpoints: list[tuple[str, Path]], datasets: list[Path]) -> Iterable[dict]:
    for label, _ in checkpoints:
        row = {"checkpoint": label}
        complete = True
        for dataset in datasets:
            dataset_name = dataset_output_name(str(dataset))
            metric_file = output_dir / label / f"{dataset_name}.json"
            if not metric_file.exists():
                complete = False
                continue
            with metric_file.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            row[f"{dataset_name}:NDCG@10"] = metrics.get("NDCG@10")
            row[f"{dataset_name}:Recall@10"] = metrics.get("Recall@10")
        if complete:
            yield row


def write_summary(
    summary_path: Path,
    output_dir: Path,
    checkpoints: list[tuple[str, Path]],
    datasets: list[Path],
    title: str,
) -> None:
    rows = list(iter_metric_rows(output_dir, checkpoints, datasets))
    dataset_names = [dataset_output_name(str(dataset)) for dataset in datasets]
    display_names = [DISPLAY_NAMES.get(name, name) for name in dataset_names]

    headers = ["Checkpoint"]
    for name in display_names:
        headers.extend([f"{name} NDCG@10 (%)", f"{name} Recall@10 (%)"])
    headers.extend(["Avg_NDCG@10 (%)", "Avg_Recall@10 (%)", "Avg (%)"])

    table_rows = []
    for row in rows:
        ndcgs = [row[f"{name}:NDCG@10"] for name in dataset_names]
        recalls = [row[f"{name}:Recall@10"] for name in dataset_names]
        avg_ndcg = sum(ndcgs) / len(ndcgs)
        avg_recall = sum(recalls) / len(recalls)
        overall = (avg_ndcg + avg_recall) / 2
        values = [row["checkpoint"]]
        for ndcg, recall in zip(ndcgs, recalls):
            values.extend([ndcg * 100, recall * 100])
        values.extend([avg_ndcg * 100, avg_recall * 100, overall * 100])
        table_rows.append(values)

    best_by_col = {}
    for col_idx in range(1, len(headers)):
        numeric_values = [values[col_idx] for values in table_rows]
        best_by_col[col_idx] = max(numeric_values) if numeric_values else None

    lines = [
        f"# {title}",
        "",
        f"- Result root: `{output_dir}`",
        f"- Checkpoints evaluated: `{len(rows)}`",
        "- Values are percentages.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for values in table_rows:
        cells = [str(values[0])]
        for col_idx, value in enumerate(values[1:], start=1):
            formatted = f"{value:.2f}"
            if best_by_col[col_idx] is not None and abs(value - best_by_col[col_idx]) < 1e-12:
                formatted = f"**{formatted}**"
            cells.append(formatted)
        lines.append("| " + " | ".join(cells) + " |")

    if table_rows:
        best_idx = max(range(len(table_rows)), key=lambda idx: table_rows[idx][-1])
        best = table_rows[best_idx]
        lines.extend(["", f"Best overall checkpoint: `{best[0]}` with Avg={best[-1]:.2f}%."])

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
