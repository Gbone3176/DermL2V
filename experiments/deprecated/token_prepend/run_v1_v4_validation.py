#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm
from transformers import GemmaConfig, LlamaConfig, MistralConfig, Qwen2Config


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm2vec.dataset.utils import load_dataset
from llm2vec.llm2vecV1 import LLM2Vec as LLM2VecV1
from llm2vec.llm2vec_prepend import LLM2Vec as LLM2VecV4


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare llm2vecV1 and llm2vecV4 retrieval performance on a validation split."
    )
    parser.add_argument("config", type=str, help="Path to the training JSON config.")
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split used for evaluation.",
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
        help="Encoding batch size. Defaults to eval_batch_size/per_device_eval_batch_size from config.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k used for retrieval metrics.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of query-positive pairs evaluated.",
    )
    parser.add_argument(
        "--pooling-mode",
        type=str,
        default="mean",
        choices=["mean", "weighted_mean", "eos_token", "last_token", "latent_pooling"],
        help="Pooling mode forced for both V1 and V4.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for JSON outputs. Defaults to experiments/v1_v4_validation/results.",
    )
    parser.add_argument(
        "--cross-layer-start-layer",
        type=int,
        default=0,
        help="First layer index where V4 starts prepending LF memory tokens.",
    )
    parser.add_argument(
        "--cross-layer-end-layer",
        type=int,
        default=None,
        help="Last exclusive layer index for V4 LF prepending. Defaults to all eligible layers.",
    )
    parser.add_argument(
        "--include-memory-in-pooling",
        action="store_true",
        help="If set, V4 includes prepended memory tokens in final pooling.",
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


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, p=2, dim=1)
    b = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a, b.transpose(0, 1))


def prepare_for_tokenization(model, text: str, pooling_mode: str = "mean") -> str:
    config_name = getattr(model.config, "_name_or_path", None)
    if config_name == "meta-llama/Meta-Llama-3-8B-Instruct" or isinstance(model.config, LlamaConfig):
        return "<|start_header_id|>user<|end_header_id|>\n\n" + text.strip() + "<|eot_id|>"
    if config_name in [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
    ]:
        text = "[INST] " + text.strip() + " [/INST]"
    if config_name in ["google/gemma-2-9b-it"]:
        text = "<bos><start_of_turn>user\n" + text.strip() + "<end_of_turn>"
    if config_name in [
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen3-8B-Embedding",
    ] or isinstance(model.config, Qwen2Config):
        text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"
    if pooling_mode == "eos_token":
        if config_name == "meta-llama/Meta-Llama-3-8B":
            text = text.strip() + "<|end_of_text|>"
        elif isinstance(model.config, (LlamaConfig, MistralConfig)):
            text = text.strip() + " </s>"
        elif isinstance(model.config, GemmaConfig):
            text = text.strip() + "<eos>"
        elif isinstance(model.config, Qwen2Config):
            text = text.strip() + "<|endoftext|>"
    return text


def build_corpus_queries(dataset, max_examples: Optional[int]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    relevant_docs: Dict[str, Dict[str, int]] = {}

    kept = 0
    for idx, sample in enumerate(dataset):
        if not hasattr(sample, "texts") or len(sample.texts) < 2:
            continue
        pair_id = str(idx)
        queries[pair_id] = sample.texts[0]
        doc_id = f"{pair_id}_pos"
        corpus[doc_id] = {"text": sample.texts[1]}
        relevant_docs[pair_id] = {doc_id: 1}
        kept += 1
        if max_examples is not None and kept >= max_examples:
            break

    if not queries:
        raise ValueError("No valid query-positive pairs found in the selected split.")
    return corpus, queries, relevant_docs


def encode_with_separator_batches(
    model,
    texts: List[str],
    batch_size: int,
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    all_embeddings = []
    iterator = range(0, len(texts), batch_size)
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(iterator, desc=desc, total=total_batches):
        batch = texts[i : i + batch_size]
        embs = model.encode_with_separator(batch, device=device)
        all_embeddings.append(embs.cpu())
    return torch.cat(all_embeddings, dim=0)


def build_results(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    corpus_ids: List[str],
    query_ids: List[str],
    top_k: int,
) -> Dict[str, Dict[str, float]]:
    scores = cos_sim(query_embeddings.float(), doc_embeddings.float())
    scores[torch.isnan(scores)] = -1
    top_k = min(top_k, len(corpus_ids))
    top_vals, top_idx = torch.topk(scores, top_k, dim=1, largest=True, sorted=True)
    top_vals = top_vals.cpu().tolist()
    top_idx = top_idx.cpu().tolist()

    results: Dict[str, Dict[str, float]] = {}
    for row_idx, qid in enumerate(query_ids):
        results[qid] = {}
        for rank, doc_idx in enumerate(top_idx[row_idx]):
            results[qid][corpus_ids[doc_idx]] = float(top_vals[row_idx][rank])
    return results


def evaluate_results(
    results: Dict[str, Dict[str, float]],
    relevant_docs: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    retriever = EvaluateRetrieval(None, score_function="cos_sim")
    ndcg, _map, recall, precision = retriever.evaluate(
        relevant_docs,
        results,
        [10],
        ignore_identical_ids=False,
    )
    metrics = {
        "ndcg_at_10": float(ndcg["NDCG@10"]),
        "recall_at_10": float(recall["Recall@10"]),
        "precision_at_10": float(precision["P@10"]),
    }
    metrics["avg_recall_ndcg_at_10"] = (metrics["ndcg_at_10"] + metrics["recall_at_10"]) / 2.0
    return metrics


def instantiate_model(
    model_cls: Type,
    cfg: Dict,
    args: argparse.Namespace,
):
    torch_dtype = resolve_torch_dtype(cfg.get("torch_dtype", "bfloat16"))
    attn_implementation = cfg.get("attn_implementation", "flash_attention_2")
    if str(args.device).startswith("cpu") and attn_implementation == "flash_attention_2":
        attn_implementation = "sdpa"

    common_kwargs = dict(
        base_model_name_or_path=cfg["model_name_or_path"],
        peft_model_name_or_path=cfg.get("peft_model_name_or_path"),
        extra_model_name_or_path=cfg.get("extra_model_name_or_path"),
        enable_bidirectional=cfg.get("bidirectional", True),
        merge_peft=True,
        pooling_mode=args.pooling_mode,
        max_length=cfg.get("max_seq_length", 512),
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    if model_cls is LLM2VecV4:
        common_kwargs.update(
            cross_layer_lf_prepend=True,
            cross_layer_lf_start_layer=args.cross_layer_start_layer,
            cross_layer_lf_end_layer=args.cross_layer_end_layer,
            cross_layer_lf_exclude_from_pooling=not args.include_memory_in_pooling,
            cross_layer_lf_use_embed_mask=True,
        )
    model = model_cls.from_pretrained(
        **common_kwargs,
    )
    return model


def evaluate_model(
    model_name: str,
    model_cls: Type,
    cfg: Dict,
    args: argparse.Namespace,
    queries: Dict[str, str],
    corpus: Dict[str, Dict[str, str]],
    relevant_docs: Dict[str, Dict[str, int]],
    device: torch.device,
) -> Dict:
    logger.info("Loading %s", model_name)
    model = instantiate_model(model_cls, cfg, args)
    model.to(device)
    model.eval()

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [
        prepare_for_tokenization(model, queries[qid], pooling_mode=args.pooling_mode)
        for qid in query_ids
    ]
    corpus_texts = [
        prepare_for_tokenization(model, corpus[cid]["text"], pooling_mode=args.pooling_mode)
        for cid in corpus_ids
    ]

    logger.info("Encoding %s queries with %s", len(query_texts), model_name)
    q_emb = encode_with_separator_batches(
        model, query_texts, args.batch_size, device=device, desc=f"{model_name} queries"
    )
    logger.info("Encoding %s documents with %s", len(corpus_texts), model_name)
    d_emb = encode_with_separator_batches(
        model, corpus_texts, args.batch_size, device=device, desc=f"{model_name} docs"
    )

    results = build_results(q_emb, d_emb, corpus_ids, query_ids, top_k=args.top_k)
    metrics = evaluate_results(results, relevant_docs)
    return {
        "model_name": model_name,
        "metrics": metrics,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = cfg.get("eval_batch_size") or cfg.get("per_device_eval_batch_size") or 16
    args.batch_size = batch_size

    dataset = load_dataset(
        cfg["dataset_name"],
        split=args.split,
        file_path=cfg.get("dataset_file_path"),
        effective_batch_size=batch_size,
        dermqa_upsample_ratio=cfg.get("dermqa_upsample_ratio", 1),
        shuffle_individual_datasets=False,
    )
    corpus, queries, relevant_docs = build_corpus_queries(dataset, args.max_examples)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "experiments" / "v1_v4_validation" / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    results = {
        "config": os.path.abspath(args.config),
        "split": args.split,
        "num_queries": len(queries),
        "num_docs": len(corpus),
        "pooling_mode": args.pooling_mode,
        "v4_cross_layer": {
            "start_layer": args.cross_layer_start_layer,
            "end_layer": args.cross_layer_end_layer,
            "exclude_memory_from_pooling": not args.include_memory_in_pooling,
        },
        "models": {},
    }

    for model_name, model_cls in [("v1", LLM2VecV1), ("v4", LLM2VecV4)]:
        model_result = evaluate_model(
            model_name=model_name,
            model_cls=model_cls,
            cfg=cfg,
            args=args,
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            device=device,
        )
        results["models"][model_name] = model_result

    v1_avg = results["models"]["v1"]["metrics"]["avg_recall_ndcg_at_10"]
    v4_avg = results["models"]["v4"]["metrics"]["avg_recall_ndcg_at_10"]
    results["summary"] = {
        "winner_by_avg_recall_ndcg_at_10": "v4" if v4_avg > v1_avg else "v1",
        "v4_minus_v1": {
            "recall_at_10": results["models"]["v4"]["metrics"]["recall_at_10"] - results["models"]["v1"]["metrics"]["recall_at_10"],
            "ndcg_at_10": results["models"]["v4"]["metrics"]["ndcg_at_10"] - results["models"]["v1"]["metrics"]["ndcg_at_10"],
            "avg_recall_ndcg_at_10": v4_avg - v1_avg,
        },
    }

    config_stem = Path(args.config).stem
    output_path = output_dir / f"{config_stem}_{args.split}_v1_v4_compare.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(json.dumps(results["summary"], indent=2, ensure_ascii=False))
    logger.info("Saved comparison results to %s", output_path)


if __name__ == "__main__":
    main()
