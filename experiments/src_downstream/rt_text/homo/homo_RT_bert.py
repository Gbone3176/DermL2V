import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from experiments.src_downstream.rt_text.homo.homo_RT_utils import (
    DEFAULT_DERMVARIANTS_DIR,
    DEFAULT_RETRIEVAL_SUBSETS,
    DEFAULT_VIS_DATASET,
    build_retrieval_dataset,
    build_results,
    build_vis_mcq_samples,
    evaluate_retrieval_metrics,
    load_jsonl,
    macro_average,
    resolve_output_file,
)


def encode_texts(tokenizer, model, texts, device, max_length, batch_size):
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
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
        all_embeddings.append(pooled.cpu())
    return torch.cat(all_embeddings, dim=0)


def evaluate_vis_mcq(tokenizer, model, device, vis_dataset_path, max_length, batch_size, max_samples):
    dataset = load_jsonl(vis_dataset_path)
    if max_samples > 0:
        dataset = dataset[:max_samples]

    queries, candidate_sets = build_vis_mcq_samples(dataset)
    if not queries:
        return {"dataset_path": vis_dataset_path, "count": 0, "accuracy": 0.0}

    q_embs = encode_texts(tokenizer, model, queries, device, max_length, batch_size)
    flat_candidates = [candidate for candidates in candidate_sets for candidate in candidates]
    cand_counts = [len(candidates) for candidates in candidate_sets]
    flat_c_embs = encode_texts(tokenizer, model, flat_candidates, device, max_length, batch_size)

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

    return {
        "dataset_path": vis_dataset_path,
        "count": total,
        "accuracy": (correct / total) if total > 0 else 0.0,
    }


def evaluate_retrieval_suite(tokenizer, model, device, dataset_dir, subsets, max_length, batch_size, max_samples):
    metrics_by_subset = {}

    for subset in subsets:
        dataset_path = os.path.join(dataset_dir, f"{subset}_test.jsonl")
        dataset = load_jsonl(dataset_path)
        if max_samples > 0:
            dataset = dataset[:max_samples]

        corpus, queries, relevant_docs = build_retrieval_dataset(dataset)
        if not corpus or not queries:
            metrics_by_subset[subset] = {
                "dataset_path": dataset_path,
                "query_count": 0,
                "corpus_count": 0,
            }
            continue

        query_ids = list(queries.keys())
        corpus_ids = list(corpus.keys())
        q_embs = encode_texts(tokenizer, model, [queries[qid] for qid in query_ids], device, max_length, batch_size)
        d_embs = encode_texts(tokenizer, model, [corpus[doc_id]["text"] for doc_id in corpus_ids], device, max_length, batch_size)
        results = build_results(q_embs, d_embs, query_ids, corpus_ids)
        metrics = evaluate_retrieval_metrics(relevant_docs, results, len(corpus_ids))
        metrics.update(
            {
                "dataset_path": dataset_path,
                "query_count": len(query_ids),
                "corpus_count": len(corpus_ids),
            }
        )
        metrics_by_subset[subset] = metrics

    return metrics_by_subset


def main():
    parser = argparse.ArgumentParser(description="Homogeneous RT benchmark using a BERT-style encoder")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--vis_dataset", type=str, default=DEFAULT_VIS_DATASET)
    parser.add_argument("--dermvariants_dir", type=str, default=DEFAULT_DERMVARIANTS_DIR)
    parser.add_argument("--retrieval_subsets", type=str, nargs="*", default=DEFAULT_RETRIEVAL_SUBSETS)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).to(device).eval()

    model_name = args.model_name or os.path.basename(args.model_path.rstrip("/"))
    output_path = resolve_output_file(args.output, model_name)
    if output_path and os.path.exists(output_path):
        print(f"Output already exists, skipping: {output_path}")
        return

    vis_metrics = evaluate_vis_mcq(
        tokenizer,
        model,
        device,
        args.vis_dataset,
        args.max_length,
        args.batch_size,
        args.max_samples,
    )
    retrieval_metrics = evaluate_retrieval_suite(
        tokenizer,
        model,
        device,
        args.dermvariants_dir,
        args.retrieval_subsets,
        args.max_length,
        args.batch_size,
        args.max_samples,
    )

    results = {
        "model_name": model_name,
        "model_path": args.model_path,
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
