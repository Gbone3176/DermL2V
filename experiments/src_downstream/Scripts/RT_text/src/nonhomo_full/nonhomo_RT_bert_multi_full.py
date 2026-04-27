import argparse
import json
import logging
import os

import torch

from experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_bert_full import encode_batches
from experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_full_utils import (
    build_corpus_queries,
    build_output_file,
    build_results,
    evaluate_at_10,
    load_jsonl,
)
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset full retrieval evaluation using a BERT-style encoder")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_name = args.model_name or args.model_path.rstrip("/").split("/")[-1]
    pending = [path for path in dict.fromkeys(args.input) if not os.path.exists(build_output_file(args.output, path, model_name))]
    if not pending:
        logger.info("All requested datasets already complete for %s", model_name)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).to(device).eval()

    for dataset_file in dict.fromkeys(args.input):
        output_file = build_output_file(args.output, dataset_file, model_name)
        if os.path.exists(output_file):
            continue
        dataset = load_jsonl(dataset_file)
        if args.max_samples and args.max_samples > 0:
            dataset = dataset[: args.max_samples]
        corpus, queries, relevant_docs = build_corpus_queries(dataset)
        query_ids = list(queries.keys())
        corpus_ids = list(corpus.keys())
        q_emb = encode_batches(tokenizer, model, [queries[qid] for qid in query_ids], args.batch_size, device, args.max_length, "Encoding queries")
        d_emb = encode_batches(tokenizer, model, [corpus[cid]["text"] for cid in corpus_ids], args.batch_size, device, args.max_length, "Encoding documents")
        metrics = evaluate_at_10(relevant_docs, build_results(q_emb, d_emb, query_ids, corpus_ids), len(corpus_ids))
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
