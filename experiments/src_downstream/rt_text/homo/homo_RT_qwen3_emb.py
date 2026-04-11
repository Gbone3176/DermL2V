import argparse
import json
import os

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from experiments.src_downstream.rt_text.homo.homo_RT_utils import (
    DEFAULT_DERMVARIANTS_DIR,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_RETRIEVAL_SUBSETS,
    DEFAULT_VIS_DATASET,
    build_mixed_retrieval_dataset,
    build_retrieval_dataset,
    build_results,
    build_vis_mcq_samples,
    evaluate_retrieval_metrics,
    load_retrieval_subset_datasets,
    load_jsonl,
    macro_average,
    output_matches_retrieval_mode,
    resolve_output_file,
)


def encode_queries(model, texts, batch_size):
    return model.encode(
        texts,
        prompt_name="query",
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
    ).cpu()


def encode_docs(model, texts, batch_size):
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
    ).cpu()


def evaluate_vis_mcq(model, vis_dataset_path, batch_size, max_samples):
    dataset = load_jsonl(vis_dataset_path)
    if max_samples > 0:
        dataset = dataset[:max_samples]
    queries, candidate_sets = build_vis_mcq_samples(dataset)
    if not queries:
        return {"dataset_path": vis_dataset_path, "count": 0, "accuracy": 0.0}

    q_embs = encode_queries(model, queries, batch_size)
    flat_candidates = [candidate for candidates in candidate_sets for candidate in candidates]
    cand_counts = [len(candidates) for candidates in candidate_sets]
    flat_c_embs = encode_docs(model, flat_candidates, batch_size)

    total = 0
    correct = 0
    start_idx = 0
    for idx, count in enumerate(cand_counts):
        end_idx = start_idx + count
        sims = F.cosine_similarity(q_embs[idx].unsqueeze(0), flat_c_embs[start_idx:end_idx], dim=1)
        pred = int(torch.argmax(sims).item())
        total += 1
        if pred == 0:
            correct += 1
        start_idx = end_idx

    return {"dataset_path": vis_dataset_path, "count": total, "accuracy": (correct / total) if total > 0 else 0.0}


def evaluate_retrieval_suite(model, dataset_dir, subsets, batch_size, max_samples, retrieval_mode):
    metrics_by_subset = {}
    dataset_paths, datasets_by_subset = load_retrieval_subset_datasets(dataset_dir, subsets, max_samples)

    if retrieval_mode == "mixed":
        corpus, queries_by_subset, relevant_docs_by_subset = build_mixed_retrieval_dataset(datasets_by_subset)
        corpus_ids = list(corpus.keys())
        if not corpus_ids:
            for subset in subsets:
                metrics_by_subset[subset] = {"dataset_path": dataset_paths[subset], "query_count": 0, "corpus_count": 0}
            return metrics_by_subset

        d_embs = encode_docs(model, [corpus[doc_id]["text"] for doc_id in corpus_ids], batch_size)
        for subset in subsets:
            queries = queries_by_subset[subset]
            relevant_docs = relevant_docs_by_subset[subset]
            query_ids = list(queries.keys())
            if not query_ids:
                metrics_by_subset[subset] = {"dataset_path": dataset_paths[subset], "query_count": 0, "corpus_count": len(corpus_ids)}
                continue

            q_embs = encode_queries(model, [queries[qid] for qid in query_ids], batch_size)
            results = build_results(q_embs, d_embs, query_ids, corpus_ids)
            metrics = evaluate_retrieval_metrics(relevant_docs, results, len(corpus_ids))
            metrics.update({"dataset_path": dataset_paths[subset], "query_count": len(query_ids), "corpus_count": len(corpus_ids)})
            metrics_by_subset[subset] = metrics
        return metrics_by_subset

    for subset in subsets:
        dataset_path = dataset_paths[subset]
        dataset = datasets_by_subset[subset]

        corpus, queries, relevant_docs = build_retrieval_dataset(dataset)
        if not corpus or not queries:
            metrics_by_subset[subset] = {"dataset_path": dataset_path, "query_count": 0, "corpus_count": 0}
            continue

        query_ids = list(queries.keys())
        corpus_ids = list(corpus.keys())
        q_embs = encode_queries(model, [queries[qid] for qid in query_ids], batch_size)
        d_embs = encode_docs(model, [corpus[doc_id]["text"] for doc_id in corpus_ids], batch_size)
        results = build_results(q_embs, d_embs, query_ids, corpus_ids)
        metrics = evaluate_retrieval_metrics(relevant_docs, results, len(corpus_ids))
        metrics.update({"dataset_path": dataset_path, "query_count": len(query_ids), "corpus_count": len(corpus_ids)})
        metrics_by_subset[subset] = metrics
    return metrics_by_subset


def main():
    parser = argparse.ArgumentParser(description="Homogeneous RT benchmark using Qwen-style embedding models")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--vis_dataset", type=str, default=DEFAULT_VIS_DATASET)
    parser.add_argument("--dermvariants_dir", type=str, default=DEFAULT_DERMVARIANTS_DIR)
    parser.add_argument("--retrieval_subsets", type=str, nargs="*", default=DEFAULT_RETRIEVAL_SUBSETS)
    parser.add_argument("--retrieval_mode", type=str, default=DEFAULT_RETRIEVAL_MODE, choices=["mixed", "separate"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    args = parser.parse_args()

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

    model_name = args.model_name or os.path.basename(args.model_name_or_path.rstrip("/"))
    output_path = resolve_output_file(args.output, model_name)
    if output_matches_retrieval_mode(output_path, args.retrieval_mode):
        print(f"Output already exists, skipping: {output_path}")
        return

    vis_metrics = evaluate_vis_mcq(model, args.vis_dataset, args.batch_size, args.max_samples)
    retrieval_metrics = evaluate_retrieval_suite(
        model,
        args.dermvariants_dir,
        args.retrieval_subsets,
        args.batch_size,
        args.max_samples,
        args.retrieval_mode,
    )

    results = {
        "model_name": model_name,
        "model_path": args.model_name_or_path,
        "retrieval_mode": args.retrieval_mode,
        "vis_mcq": vis_metrics,
        "retrieval": retrieval_metrics,
        "retrieval_macro_avg": macro_average(
            {
                subset: {
                    key: value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float)) and ("@" in key)
                }
                for subset, metrics in retrieval_metrics.items()
            }
        ),
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
