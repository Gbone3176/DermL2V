#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm2vec.dataset.utils import load_dataset
from llm2vec.llm2vecV3 import LLM2Vec


SEPARATOR = "!@#$%^&*()"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen merged backbone with layer-wise mean pooling for retrieval."
    )
    parser.add_argument("config", type=str, help="Path to the training JSON config.")
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split used for layer selection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for encoding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Encoding batch size. Defaults to per_device_eval_batch_size from config.",
    )
    parser.add_argument(
        "--dataset-effective-batch-size",
        type=int,
        default=None,
        help="effective_batch_size used when loading the dataset. Defaults to batch size.",
    )
    parser.add_argument(
        "--eval-top-k",
        type=int,
        default=None,
        help="Top-k used for retrieval metrics. Defaults to eval_top_k from config.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of query-positive pairs evaluated.",
    )
    parser.add_argument(
        "--window-sizes",
        type=str,
        default="4,8,12,16",
        help="Comma-separated contiguous window sizes for range evaluation.",
    )
    parser.add_argument(
        "--range-mode",
        type=str,
        default="suffix",
        choices=["none", "suffix", "sliding", "both"],
        help="Which contiguous layer ranges to evaluate in addition to single layers.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=4,
        help="Stride for sliding windows when range-mode includes sliding.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for JSON results. Defaults to experiments/layer_selection/results.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="recall_at_1",
        help="Primary metric used for ranking best layers in the summary.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_torch_dtype(torch_dtype_name: Optional[str]):
    valid = {"auto", None, "bfloat16", "float16", "float32"}
    if torch_dtype_name not in valid:
        raise ValueError(
            f"Invalid torch_dtype {torch_dtype_name!r}. Expected one of {sorted(v for v in valid if v is not None)} or None."
        )
    if torch_dtype_name in {"auto", None}:
        return torch_dtype_name
    return getattr(torch, torch_dtype_name)


def parse_int_csv(raw_value: str) -> List[int]:
    values: List[int] = []
    for piece in raw_value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(int(piece))
    return sorted(set(v for v in values if v > 0))


def masked_mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    masked_hidden = hidden_states * mask
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return masked_hidden.sum(dim=1) / denom


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, p=2, dim=1)
    b = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a, b.transpose(0, 1))


def build_eval_examples(
    cfg: Dict,
    split: str,
    effective_batch_size: int,
    seed: int,
) -> List:
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = load_dataset(
        cfg["dataset_name"],
        split=split,
        file_path=cfg["dataset_file_path"],
        effective_batch_size=effective_batch_size,
        dermqa_upsample_ratio=1,
        shuffle_individual_datasets=False,
    )
    return [dataset[i] for i in range(len(dataset))]


def extract_query_doc_pairs(examples: Sequence, max_examples: Optional[int]) -> Tuple[List[str], List[str]]:
    queries: List[str] = []
    docs: List[str] = []
    for sample in examples:
        if not hasattr(sample, "texts") or len(sample.texts) < 2:
            continue
        queries.append(sample.texts[0])
        docs.append(sample.texts[1])
        if max_examples is not None and len(queries) >= max_examples:
            break
    if not queries:
        raise ValueError("No valid query-positive pairs found in the selected split.")
    return queries, docs


def prepare_text_batch(model: LLM2Vec, texts: Sequence[str], separator: str) -> List[str]:
    prepared: List[str] = []
    for text in texts:
        if separator not in text:
            raise ValueError(f"Expected separator {separator!r} in text: {text[:120]!r}")
        instruction, content = text.split(separator, 1)
        prepared.append(model._convert_to_str(instruction.strip(), content.strip()))
    return prepared


def encode_all_layers(
    model: LLM2Vec,
    texts: Sequence[str],
    batch_size: int,
    device: str,
    separator: str,
) -> List[torch.Tensor]:
    per_layer_chunks: Optional[List[List[torch.Tensor]]] = None
    model.eval()
    model.to(device)

    iterator = range(0, len(texts), batch_size)
    for start in tqdm(iterator, desc="Encoding", leave=False):
        batch_texts = texts[start : start + batch_size]
        prepared = prepare_text_batch(model, batch_texts, separator)
        tokenized = model.tokenize(prepared)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        model_inputs = {k: v for k, v in tokenized.items() if k != "embed_mask"}

        with torch.no_grad():
            outputs = model.model(**model_inputs, output_hidden_states=True)

        hidden_states = list(outputs.hidden_states)
        hidden_states = hidden_states[1:] if len(hidden_states) > 1 else hidden_states
        if per_layer_chunks is None:
            per_layer_chunks = [[] for _ in range(len(hidden_states))]

        pool_mask = tokenized["attention_mask"]
        if getattr(model, "skip_instruction", False) and tokenized.get("embed_mask") is not None:
            pool_mask = tokenized["embed_mask"]

        for layer_idx, layer_hidden in enumerate(hidden_states):
            pooled = masked_mean_pooling(layer_hidden, pool_mask)
            per_layer_chunks[layer_idx].append(pooled.to(torch.float16).cpu())

    if per_layer_chunks is None:
        raise ValueError("No embeddings were encoded.")

    return [torch.cat(chunks, dim=0) for chunks in per_layer_chunks]


def build_retrieval_state(num_pairs: int) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    corpus_ids = [f"{idx}_pos" for idx in range(num_pairs)]
    relevant_docs = {str(idx): {corpus_ids[idx]: 1} for idx in range(num_pairs)}
    return corpus_ids, relevant_docs


def run_retrieval_metrics(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    eval_top_k: int,
    corpus_ids: Sequence[str],
    relevant_docs: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    scores = cosine_similarity_matrix(query_embeddings.float(), doc_embeddings.float())
    scores[torch.isnan(scores)] = -1

    top_k = min(eval_top_k, len(corpus_ids))
    top_vals, top_idx = torch.topk(scores, top_k, dim=1, largest=True, sorted=True)
    top_vals = top_vals.cpu().tolist()
    top_idx = top_idx.cpu().tolist()

    results: Dict[str, Dict[str, float]] = {}
    for row_idx in range(len(top_idx)):
        query_id = str(row_idx)
        results[query_id] = {}
        for rank, doc_idx in enumerate(top_idx[row_idx]):
            results[query_id][corpus_ids[doc_idx]] = float(top_vals[row_idx][rank])

    retriever = EvaluateRetrieval(None, score_function="cos_sim")
    default_k_values = [1, 3, 5, 10, 100, 1000]
    k_values = [k for k in default_k_values if k <= top_k]
    if not k_values:
        k_values = [top_k]
    ndcg, _map, recall, precision = retriever.evaluate(
        relevant_docs, results, k_values, ignore_identical_ids=False
    )

    metrics = {
        **{f"ndcg_at_{k.split('@')[1]}": float(v) for k, v in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": float(v) for k, v in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": float(v) for k, v in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": float(v) for k, v in precision.items()},
    }
    return metrics


def rank_records(records: List[Dict], metric_key: str) -> List[Dict]:
    return sorted(records, key=lambda item: item["metrics"].get(metric_key, float("-inf")), reverse=True)


def evaluate_single_layers(
    query_layers: Sequence[torch.Tensor],
    doc_layers: Sequence[torch.Tensor],
    eval_top_k: int,
    corpus_ids: Sequence[str],
    relevant_docs: Dict[str, Dict[str, int]],
) -> List[Dict]:
    records: List[Dict] = []
    for idx, (q_layer, d_layer) in enumerate(zip(query_layers, doc_layers), start=1):
        metrics = run_retrieval_metrics(q_layer, d_layer, eval_top_k, corpus_ids, relevant_docs)
        records.append(
            {
                "kind": "single_layer",
                "layer": idx,
                "start_layer": idx,
                "end_layer": idx,
                "metrics": metrics,
            }
        )
    return records


def iter_window_candidates(
    num_layers: int,
    window_sizes: Sequence[int],
    range_mode: str,
    stride: int,
) -> Iterable[Tuple[int, int]]:
    emitted = set()
    for window_size in window_sizes:
        if window_size > num_layers:
            continue
        if range_mode in {"suffix", "both"}:
            start = num_layers - window_size + 1
            emitted.add((start, num_layers))
            yield start, num_layers
        if range_mode in {"sliding", "both"}:
            for start in range(1, num_layers - window_size + 2, max(1, stride)):
                end = start + window_size - 1
                key = (start, end)
                if key in emitted:
                    continue
                emitted.add(key)
                yield key


def average_layer_range(layer_embeddings: Sequence[torch.Tensor], start_layer: int, end_layer: int) -> torch.Tensor:
    selected = layer_embeddings[start_layer - 1 : end_layer]
    stacked = torch.stack([emb.float() for emb in selected], dim=0)
    return stacked.mean(dim=0)


def evaluate_layer_ranges(
    query_layers: Sequence[torch.Tensor],
    doc_layers: Sequence[torch.Tensor],
    window_sizes: Sequence[int],
    range_mode: str,
    stride: int,
    eval_top_k: int,
    corpus_ids: Sequence[str],
    relevant_docs: Dict[str, Dict[str, int]],
) -> List[Dict]:
    if range_mode == "none":
        return []

    num_layers = len(query_layers)
    records: List[Dict] = []
    for start_layer, end_layer in iter_window_candidates(num_layers, window_sizes, range_mode, stride):
        q_emb = average_layer_range(query_layers, start_layer, end_layer)
        d_emb = average_layer_range(doc_layers, start_layer, end_layer)
        metrics = run_retrieval_metrics(q_emb, d_emb, eval_top_k, corpus_ids, relevant_docs)
        records.append(
            {
                "kind": "layer_range",
                "start_layer": start_layer,
                "end_layer": end_layer,
                "window_size": end_layer - start_layer + 1,
                "metrics": metrics,
            }
        )
    return records


def summarize(records: List[Dict], metric_key: str, top_n: int = 5) -> List[Dict]:
    ranked = rank_records(records, metric_key)
    return ranked[: min(top_n, len(ranked))]


def print_summary(
    single_records: List[Dict],
    range_records: List[Dict],
    metric_key: str,
) -> None:
    print(f"\nTop single layers by {metric_key}:")
    for record in summarize(single_records, metric_key):
        value = record["metrics"].get(metric_key)
        print(
            f"  layer {record['layer']:>2}: {metric_key}={value:.4f}, "
            f"ndcg_at_10={record['metrics'].get('ndcg_at_10', float('nan')):.4f}, "
            f"map_at_10={record['metrics'].get('map_at_10', float('nan')):.4f}"
        )

    if range_records:
        print(f"\nTop layer ranges by {metric_key}:")
        for record in summarize(range_records, metric_key):
            value = record["metrics"].get(metric_key)
            print(
                f"  layers {record['start_layer']:>2}-{record['end_layer']:>2}: "
                f"{metric_key}={value:.4f}, "
                f"ndcg_at_10={record['metrics'].get('ndcg_at_10', float('nan')):.4f}, "
                f"map_at_10={record['metrics'].get('map_at_10', float('nan')):.4f}"
            )


def default_output_dir() -> Path:
    return REPO_ROOT / "experiments" / "layer_selection" / "results"


def build_output_payload(
    args: argparse.Namespace,
    cfg: Dict,
    num_examples: int,
    single_records: List[Dict],
    range_records: List[Dict],
) -> Dict:
    payload = {
        "config_path": os.path.abspath(args.config),
        "split": args.split,
        "device": args.device,
        "num_examples": num_examples,
        "primary_metric": args.metric,
        "model_init": {
            "model_name_or_path": cfg.get("model_name_or_path"),
            "peft_model_name_or_path": cfg.get("peft_model_name_or_path"),
            "extra_model_name_or_path": cfg.get("extra_model_name_or_path", []),
        },
        "single_layers": rank_records(single_records, args.metric),
        "layer_ranges": rank_records(range_records, args.metric),
    }

    if single_records:
        payload["best_single_layer"] = rank_records(single_records, args.metric)[0]
    if range_records:
        payload["best_layer_range"] = rank_records(range_records, args.metric)[0]
    return payload


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    batch_size = args.batch_size or cfg.get("per_device_eval_batch_size") or cfg.get("per_device_train_batch_size", 32)
    effective_batch_size = args.dataset_effective_batch_size or batch_size
    eval_top_k = args.eval_top_k or cfg.get("eval_top_k", 10)
    separator = cfg.get("eval_separator", SEPARATOR)
    window_sizes = parse_int_csv(args.window_sizes)
    if not window_sizes and args.range_mode != "none":
        raise ValueError("At least one window size is required when range-mode is not 'none'.")

    examples = build_eval_examples(
        cfg=cfg,
        split=args.split,
        effective_batch_size=effective_batch_size,
        seed=int(cfg.get("seed", 42)),
    )
    queries, docs = extract_query_doc_pairs(examples, args.max_examples)
    corpus_ids, relevant_docs = build_retrieval_state(len(queries))

    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=cfg["model_name_or_path"],
        enable_bidirectional=cfg.get("bidirectional", False),
        peft_model_name_or_path=cfg.get("peft_model_name_or_path"),
        extra_model_name_or_path=cfg.get("extra_model_name_or_path", []),
        merge_peft=True,
        pooling_mode="mean",
        max_length=cfg.get("max_seq_length"),
        torch_dtype=resolve_torch_dtype(cfg.get("torch_dtype")),
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )

    print(f"Loaded merged backbone for layer selection on split={args.split}, pairs={len(queries)}")
    query_layers = encode_all_layers(model, queries, batch_size, args.device, separator)
    doc_layers = encode_all_layers(model, docs, batch_size, args.device, separator)

    single_records = evaluate_single_layers(query_layers, doc_layers, eval_top_k, corpus_ids, relevant_docs)
    range_records = evaluate_layer_ranges(
        query_layers=query_layers,
        doc_layers=doc_layers,
        window_sizes=window_sizes,
        range_mode=args.range_mode,
        stride=args.window_stride,
        eval_top_k=eval_top_k,
        corpus_ids=corpus_ids,
        relevant_docs=relevant_docs,
    )

    print_summary(single_records, range_records, args.metric)

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(args.config).stem}_{args.split}_layerwise_eval.json"
    payload = build_output_payload(args, cfg, len(queries), single_records, range_records)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
