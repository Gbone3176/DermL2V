import argparse
import json
import os

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_medcpt_full import (
    encode_documents,
    encode_queries,
    build_dense_candidates,
    maybe_half,
    rerank_candidates,
)
from experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_full_utils import (
    build_corpus_queries,
    build_output_file,
    evaluate_at_10,
    load_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset two-stage MedCPT retrieval evaluation")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--query_model_path", required=True)
    parser.add_argument("--article_model_path", required=True)
    parser.add_argument("--cross_model_path", required=True)
    parser.add_argument("--model_name", default="MedCPT")
    parser.add_argument("--query_max_length", type=int, default=64)
    parser.add_argument("--article_max_length", type=int, default=512)
    parser.add_argument("--cross_max_length", type=int, default=512)
    parser.add_argument("--dense_batch_size", type=int, default=64)
    parser.add_argument("--cross_batch_size", type=int, default=32)
    parser.add_argument("--retrieve_top_k", type=int, default=100)
    parser.add_argument("--rerank_top_k", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pending = [path for path in dict.fromkeys(args.input) if not os.path.exists(build_output_file(args.output, path, args.model_name))]
    if not pending:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_tokenizer = AutoTokenizer.from_pretrained(args.query_model_path)
    article_tokenizer = AutoTokenizer.from_pretrained(args.article_model_path)
    cross_tokenizer = AutoTokenizer.from_pretrained(args.cross_model_path)
    query_model = maybe_half(AutoModel.from_pretrained(args.query_model_path).to(device).eval(), device)
    article_model = maybe_half(AutoModel.from_pretrained(args.article_model_path).to(device).eval(), device)
    cross_model = maybe_half(AutoModelForSequenceClassification.from_pretrained(args.cross_model_path).to(device).eval(), device)

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
        query_texts = [queries[qid] for qid in query_ids]
        corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]
        query_embeddings = encode_queries(query_tokenizer, query_model, query_texts, args.dense_batch_size, device, args.query_max_length)
        doc_embeddings = encode_documents(article_tokenizer, article_model, corpus_texts, args.dense_batch_size, device, args.article_max_length)
        dense_top_scores, dense_top_idx = build_dense_candidates(query_embeddings=query_embeddings, doc_embeddings=doc_embeddings, retrieve_top_k=max(args.retrieve_top_k, 10))
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
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
