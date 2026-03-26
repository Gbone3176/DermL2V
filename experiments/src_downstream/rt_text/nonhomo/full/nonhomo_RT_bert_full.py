import argparse
import json
import logging
import os

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_full_utils import (
    build_corpus_queries,
    build_output_file,
    build_results,
    evaluate_at_10,
    load_jsonl,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def encode_texts(tokenizer, model, texts, device, max_length):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        hs = out.last_hidden_state
        batch_idx = torch.arange(hs.size(0), device=device)
        cls_id = tokenizer.cls_token_id
        if cls_id is None:
            pooled = hs[batch_idx, 0, :]
        else:
            input_ids = enc["input_ids"]
            positions = torch.arange(input_ids.size(1), device=device).unsqueeze(0).expand_as(input_ids)
            cls_mask = input_ids == cls_id
            idx_cls = (cls_mask * positions).max(dim=1).values
            has_cls = cls_mask.any(dim=1)
            final_idx = torch.where(has_cls, idx_cls, torch.zeros_like(idx_cls))
            pooled = hs[batch_idx, final_idx, :]
    return pooled


def encode_batches(tokenizer, model, texts, batch_size, device, max_length, desc=None):
    all_embeddings = []
    iterator = range(0, len(texts), batch_size)
    if desc is not None:
        total_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = tqdm(iterator, desc=desc, total=total_batches)
    for i in iterator:
        batch = texts[i : i + batch_size]
        all_embeddings.append(encode_texts(tokenizer, model, batch, device, max_length).cpu())
    return torch.cat(all_embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Full retrieval evaluation using a BERT-style encoder")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model_name or args.model_path.rstrip("/").split("/")[-1]
    output_file = build_output_file(args.output, args.input, model_name)
    if os.path.exists(output_file):
        logger.info("Results already exist at %s, skipping...", output_file)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        logger.warning("Failed to load tokenizer directly: %s", e)
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).to(device).eval()

    dataset = load_jsonl(args.input)
    if args.max_samples and args.max_samples > 0:
        dataset = dataset[: args.max_samples]
    corpus, queries, relevant_docs = build_corpus_queries(dataset)
    if not queries or not corpus or not relevant_docs:
        raise ValueError("No valid retrieval samples were built from the dataset.")

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

    logger.info("Encoding %d queries", len(query_texts))
    q_emb = encode_batches(tokenizer, model, query_texts, args.batch_size, device, args.max_length, "Encoding queries")
    logger.info("Encoding %d documents", len(corpus_texts))
    d_emb = encode_batches(tokenizer, model, corpus_texts, args.batch_size, device, args.max_length, "Encoding documents")

    metrics = evaluate_at_10(relevant_docs, build_results(q_emb, d_emb, query_ids, corpus_ids), len(corpus_ids))
    logger.info(json.dumps(metrics, indent=4))
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
