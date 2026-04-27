import argparse
import json
import os

import torch
from transformers import AutoModel, AutoTokenizer

from experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_bmretriever_full import (
    DEFAULT_QUERY_INSTRUCTION,
    build_results_dot,
    encode_batches,
    format_passage,
    format_query,
)
from experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_full_utils import (
    build_corpus_queries,
    build_output_file,
    evaluate_at_10,
    load_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset full retrieval evaluation using BMRetriever")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--model_path", default="BMRetriever/BMRetriever-1B")
    parser.add_argument("--model_name", default="BMRETRIEVER-1B")
    parser.add_argument("--query_instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pending = [path for path in dict.fromkeys(args.input) if not os.path.exists(build_output_file(args.output, path, args.model_name))]
    if not pending:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"trust_remote_code": True}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModel.from_pretrained(args.model_path, **model_kwargs).to(device).eval()

    for dataset_file in dict.fromkeys(args.input):
        output_file = build_output_file(args.output, dataset_file, args.model_name)
        if os.path.exists(output_file):
            continue
        dataset = load_jsonl(dataset_file)
        if args.max_samples and args.max_samples > 0:
            dataset = dataset[: args.max_samples]
        corpus, queries, relevant_docs = build_corpus_queries(dataset)
        query_ids = list(queries.keys())
        corpus_ids = list(corpus.keys())
        query_texts = [format_query(queries[qid], args.query_instruction) for qid in query_ids]
        corpus_texts = [format_passage(corpus[cid]["text"]) for cid in corpus_ids]
        q_emb = encode_batches(tokenizer, model, query_texts, args.batch_size, device, args.max_length, "Encoding queries")
        d_emb = encode_batches(tokenizer, model, corpus_texts, args.batch_size, device, args.max_length, "Encoding documents")
        metrics = evaluate_at_10(relevant_docs, build_results_dot(q_emb, d_emb, query_ids, corpus_ids), len(corpus_ids))
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
