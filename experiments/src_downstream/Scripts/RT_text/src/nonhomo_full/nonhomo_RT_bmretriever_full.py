import argparse
import json
import logging
import os
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_full_utils import (
    build_corpus_queries,
    build_output_file,
    evaluate_at_10,
    load_jsonl,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DEFAULT_QUERY_INSTRUCTION = "Given a biomedical question, retrieve the most relevant passage that answers it."


def format_query(query: str, task_description: str) -> str:
    return f"{task_description}\nQuery: {query}"


def format_passage(passage: str) -> str:
    return f"Represent this passage\npassage: {passage}"


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0]).item()
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    batch_indices = torch.arange(batch_size, device=last_hidden_states.device)
    return last_hidden_states[batch_indices, sequence_lengths]


def tokenize_with_eos(tokenizer, texts: List[str], max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    batch_dict = tokenizer(
        texts,
        max_length=max_length - 1,
        padding=True,
        truncation=True,
        return_attention_mask=False,
    )
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("BMRetriever tokenizer does not define eos_token_id.")
    batch_dict["input_ids"] = [input_ids + [eos_token_id] for input_ids in batch_dict["input_ids"]]
    padded = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")
    return {k: v.to(device) for k, v in padded.items()}


def encode_texts(tokenizer, model, texts: List[str], device: torch.device, max_length: int) -> torch.Tensor:
    enc = tokenize_with_eos(tokenizer, texts, max_length=max_length, device=device)
    with torch.no_grad():
        outputs = model(**enc)
        embeddings = last_token_pool(outputs.last_hidden_state, enc["attention_mask"]).float()
    return embeddings


def encode_batches(tokenizer, model, texts: List[str], batch_size: int, device: torch.device, max_length: int, desc=None):
    all_embeddings = []
    iterator = range(0, len(texts), batch_size)
    if desc is not None:
        total_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = tqdm(iterator, desc=desc, total=total_batches)
    for i in iterator:
        batch = texts[i : i + batch_size]
        all_embeddings.append(encode_texts(tokenizer, model, batch, device, max_length).cpu())
    return torch.cat(all_embeddings, dim=0)


def build_results_dot(
    q_emb: torch.Tensor,
    d_emb: torch.Tensor,
    query_ids: List[str],
    corpus_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        q_emb = q_emb.to(device, dtype=torch.float32, non_blocking=True)
        d_emb = d_emb.to(device, dtype=torch.float32, non_blocking=True)
    else:
        q_emb = q_emb.float()
        d_emb = d_emb.float()
    scores = torch.mm(q_emb, d_emb.transpose(0, 1))
    scores[~torch.isfinite(scores)] = -1
    top_k = min(10, len(corpus_ids))
    top_vals, top_idx = torch.topk(scores, top_k, dim=1, largest=True, sorted=True)
    top_vals = top_vals.cpu().tolist()
    top_idx = top_idx.cpu().tolist()

    results: Dict[str, Dict[str, float]] = {}
    for i, qid in enumerate(query_ids):
        results[qid] = {}
        for rank, idx in enumerate(top_idx[i]):
            results[qid][corpus_ids[idx]] = top_vals[i][rank]
    return results


def main():
    parser = argparse.ArgumentParser(description="Full retrieval evaluation using BMRetriever")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="BMRetriever/BMRetriever-1B")
    parser.add_argument("--model_name", type=str, default="BMRETRIEVER-1B")
    parser.add_argument("--query_instruction", type=str, default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    output_file = build_output_file(args.output, args.input, args.model_name)
    if os.path.exists(output_file):
        logger.info("Results already exist at %s, skipping...", output_file)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModel.from_pretrained(args.model_path, **model_kwargs).to(device).eval()

    dataset = load_jsonl(args.input)
    if args.max_samples and args.max_samples > 0:
        dataset = dataset[: args.max_samples]
    corpus, queries, relevant_docs = build_corpus_queries(dataset)
    if not queries or not corpus or not relevant_docs:
        raise ValueError("No valid retrieval samples were built from the dataset.")

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [format_query(queries[qid], args.query_instruction) for qid in query_ids]
    corpus_texts = [format_passage(corpus[cid]["text"]) for cid in corpus_ids]

    logger.info("Encoding %d queries", len(query_texts))
    q_emb = encode_batches(tokenizer, model, query_texts, args.batch_size, device, args.max_length, "Encoding queries")
    logger.info("Encoding %d documents", len(corpus_texts))
    d_emb = encode_batches(tokenizer, model, corpus_texts, args.batch_size, device, args.max_length, "Encoding documents")

    metrics = evaluate_at_10(relevant_docs, build_results_dot(q_emb, d_emb, query_ids, corpus_ids), len(corpus_ids))
    logger.info(json.dumps(metrics, indent=4))
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
