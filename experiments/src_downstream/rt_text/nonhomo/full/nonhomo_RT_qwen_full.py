import argparse
import json
import logging
import os

import torch
from sentence_transformers import SentenceTransformer

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


def encode_batches(model, texts, batch_size, prompt_name=None, desc=None):
    return model.encode(
        texts,
        prompt_name=prompt_name,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=desc is not None,
    )


def main():
    parser = argparse.ArgumentParser(description="Full retrieval evaluation using Qwen-style embedding models")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    args = parser.parse_args()

    model_name = args.model_name or args.model_name_or_path.rstrip("/").split("/")[-1]
    output_file = build_output_file(args.output, args.input, model_name)
    if os.path.exists(output_file):
        logger.info("Results already exist at %s, skipping...", output_file)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    tokenizer_kwargs = {"padding_side": "left", "trust_remote_code": True}
    try:
        model = SentenceTransformer(
            args.model_name_or_path,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            device=device,
            trust_remote_code=True,
        )
    except TypeError:
        model = SentenceTransformer(
            args.model_name_or_path,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            device=device,
        )
    model.max_seq_length = args.max_length

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
    q_emb = encode_batches(model, query_texts, args.batch_size, prompt_name="query", desc="Encoding queries")
    logger.info("Encoding %d documents", len(corpus_texts))
    d_emb = encode_batches(model, corpus_texts, args.batch_size, prompt_name=None, desc="Encoding documents")

    metrics = evaluate_at_10(relevant_docs, build_results(q_emb, d_emb, query_ids, corpus_ids), len(corpus_ids))
    logger.info(json.dumps(metrics, indent=4))
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
