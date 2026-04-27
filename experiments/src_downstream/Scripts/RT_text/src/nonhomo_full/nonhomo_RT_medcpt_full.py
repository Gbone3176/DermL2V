import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_full_utils import (
    DATASET_NAME_MAPPING,
    build_corpus_queries,
    evaluate_at_10,
    load_jsonl,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def sanitize_path_component(name: str) -> str:
    sanitized = name.strip().replace(os.sep, "_")
    if os.altsep:
        sanitized = sanitized.replace(os.altsep, "_")
    return sanitized or "unknown"


def dataset_display_name(dataset_file_path: str) -> str:
    dataset_stem = os.path.splitext(os.path.basename(dataset_file_path))[0]
    return DATASET_NAME_MAPPING.get(dataset_stem, dataset_stem)


def build_model_output_file(output_root: str, dataset_file_path: str, model_name: str) -> str:
    output_dir = Path(output_root) / sanitize_path_component(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / f"{sanitize_path_component(dataset_display_name(dataset_file_path))}.json")


def cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    return last_hidden_state[:, 0, :]


def encode_queries(tokenizer, model, texts: Sequence[str], batch_size: int, device: torch.device, max_length: int) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    iterator = tqdm(range(0, len(texts), batch_size), desc="Encoding queries")
    for start in iterator:
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            embeddings = cls_pool(model(**encoded).last_hidden_state).float().cpu()
        outputs.append(embeddings)
    return torch.cat(outputs, dim=0)


def encode_documents(tokenizer, model, texts: Sequence[str], batch_size: int, device: torch.device, max_length: int) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    iterator = tqdm(range(0, len(texts), batch_size), desc="Encoding documents")
    for start in iterator:
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            embeddings = cls_pool(model(**encoded).last_hidden_state).float().cpu()
        outputs.append(embeddings)
    return torch.cat(outputs, dim=0)


def build_dense_candidates(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    retrieve_top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.cuda.is_available():
        score_device = torch.device("cuda")
        query_embeddings = query_embeddings.to(score_device, dtype=torch.float32, non_blocking=True)
        doc_embeddings = doc_embeddings.to(score_device, dtype=torch.float32, non_blocking=True)
    else:
        query_embeddings = query_embeddings.float()
        doc_embeddings = doc_embeddings.float()

    scores = torch.mm(query_embeddings, doc_embeddings.transpose(0, 1))
    scores[~torch.isfinite(scores)] = -1e9
    top_k = min(retrieve_top_k, doc_embeddings.shape[0])
    return torch.topk(scores, top_k, dim=1, largest=True, sorted=True)


def rerank_candidates(
    cross_tokenizer,
    cross_model,
    queries: Sequence[str],
    corpus_texts: Sequence[str],
    corpus_ids: Sequence[str],
    dense_top_idx: torch.Tensor,
    dense_top_scores: torch.Tensor,
    rerank_top_k: int,
    batch_size: int,
    device: torch.device,
    max_length: int,
) -> Dict[str, Dict[str, float]]:
    top_doc_count = dense_top_idx.shape[1]
    rerank_limit = min(rerank_top_k, top_doc_count)
    results: Dict[str, Dict[str, float]] = {}

    iterator = tqdm(range(len(queries)), desc="Cross-encoder reranking")
    for query_idx in iterator:
        query = queries[query_idx]
        candidate_indices = dense_top_idx[query_idx].tolist()
        candidate_scores = dense_top_scores[query_idx].tolist()
        candidate_indices = candidate_indices[:rerank_limit]
        candidate_scores = candidate_scores[:rerank_limit]

        pairs = [[query, corpus_texts[doc_idx]] for doc_idx in candidate_indices]
        rerank_scores: List[float] = []
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            encoded = cross_tokenizer(
                batch_pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = cross_model(**encoded).logits.squeeze(dim=-1).float().cpu().tolist()
            if isinstance(logits, float):
                logits = [logits]
            rerank_scores.extend(logits)

        merged = list(zip(candidate_indices, rerank_scores, candidate_scores))
        merged.sort(key=lambda item: (item[1], item[2]), reverse=True)

        final_docs: Dict[str, float] = {}
        for doc_idx, rerank_score, _ in merged[: min(10, len(merged))]:
            final_docs[corpus_ids[doc_idx]] = float(rerank_score)
        results[str(query_idx)] = final_docs

    return results


def maybe_half(model, device: torch.device):
    if device.type == "cuda":
        try:
            return model.half()
        except Exception:
            return model
    return model


def main():
    parser = argparse.ArgumentParser(description="Two-stage MedCPT retrieval evaluation for RT nonhomo full")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--query_model_path", type=str, required=True)
    parser.add_argument("--article_model_path", type=str, required=True)
    parser.add_argument("--cross_model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="MedCPT")
    parser.add_argument("--query_max_length", type=int, default=64)
    parser.add_argument("--article_max_length", type=int, default=512)
    parser.add_argument("--cross_max_length", type=int, default=512)
    parser.add_argument("--dense_batch_size", type=int, default=64)
    parser.add_argument("--cross_batch_size", type=int, default=32)
    parser.add_argument("--retrieve_top_k", type=int, default=100)
    parser.add_argument("--rerank_top_k", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    output_file = build_model_output_file(args.output, args.input, args.model_name)
    if os.path.exists(output_file):
        logger.info("Results already exist at %s, skipping...", output_file)
        return

    dataset = load_jsonl(args.input)
    if args.max_samples and args.max_samples > 0:
        dataset = dataset[: args.max_samples]
    corpus, queries, relevant_docs = build_corpus_queries(dataset)
    if not queries or not corpus or not relevant_docs:
        raise ValueError("No valid retrieval samples were built from the dataset.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading MedCPT encoders")
    query_tokenizer = AutoTokenizer.from_pretrained(args.query_model_path)
    article_tokenizer = AutoTokenizer.from_pretrained(args.article_model_path)
    cross_tokenizer = AutoTokenizer.from_pretrained(args.cross_model_path)

    query_model = maybe_half(AutoModel.from_pretrained(args.query_model_path).to(device).eval(), device)
    article_model = maybe_half(AutoModel.from_pretrained(args.article_model_path).to(device).eval(), device)
    cross_model = maybe_half(
        AutoModelForSequenceClassification.from_pretrained(args.cross_model_path).to(device).eval(),
        device,
    )

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

    logger.info("Dataset %s: %d queries, %d unique documents", dataset_display_name(args.input), len(query_texts), len(corpus_texts))
    query_embeddings = encode_queries(
        query_tokenizer,
        query_model,
        query_texts,
        args.dense_batch_size,
        device,
        args.query_max_length,
    )
    doc_embeddings = encode_documents(
        article_tokenizer,
        article_model,
        corpus_texts,
        args.dense_batch_size,
        device,
        args.article_max_length,
    )

    dense_top_scores, dense_top_idx = build_dense_candidates(
        query_embeddings=query_embeddings,
        doc_embeddings=doc_embeddings,
        retrieve_top_k=max(args.retrieve_top_k, 10),
    )

    results = rerank_candidates(
        cross_tokenizer=cross_tokenizer,
        cross_model=cross_model,
        queries=query_texts,
        corpus_texts=corpus_texts,
        corpus_ids=corpus_ids,
        dense_top_idx=dense_top_idx.cpu(),
        dense_top_scores=dense_top_scores.cpu(),
        rerank_top_k=max(args.rerank_top_k, 10),
        batch_size=args.cross_batch_size,
        device=device,
        max_length=args.cross_max_length,
    )

    keyed_results = {query_ids[int(query_idx)]: docs for query_idx, docs in results.items()}
    metrics = evaluate_at_10(relevant_docs, keyed_results, len(corpus_ids))
    logger.info(json.dumps(metrics, indent=4))
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
