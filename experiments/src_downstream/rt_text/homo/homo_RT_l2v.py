import argparse
import json
import os

import torch
import torch.nn.functional as F

from llm2vec import LLM2Vec

from experiments.src_downstream.rt_text.homo.homo_RT_utils import (
    DATASET_INSTRUCTIONS,
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


def encode_instruction_pairs(model, pairs, device, batch_size):
    return model.encode(
        pairs,
        batch_size=batch_size,
        convert_to_tensor=True,
        device=device,
    ).cpu()


def evaluate_vis_mcq(model, device, vis_dataset_path, batch_size, max_samples, instruction):
    dataset = load_jsonl(vis_dataset_path)
    if max_samples > 0:
        dataset = dataset[:max_samples]
    queries, candidate_sets = build_vis_mcq_samples(dataset)
    if not queries:
        return {"dataset_path": vis_dataset_path, "count": 0, "accuracy": 0.0}

    q_pairs = [[instruction, query] for query in queries]
    q_embs = encode_instruction_pairs(model, q_pairs, device, batch_size)
    flat_candidates = [candidate for candidates in candidate_sets for candidate in candidates]
    cand_counts = [len(candidates) for candidates in candidate_sets]
    c_pairs = [["", candidate] for candidate in flat_candidates]
    flat_c_embs = encode_instruction_pairs(model, c_pairs, device, batch_size)

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


def evaluate_retrieval_suite(model, device, dataset_dir, subsets, batch_size, max_samples):
    metrics_by_subset = {}
    for subset in subsets:
        dataset_path = os.path.join(dataset_dir, f"{subset}_test.jsonl")
        dataset = load_jsonl(dataset_path)
        if max_samples > 0:
            dataset = dataset[:max_samples]

        corpus, queries, relevant_docs = build_retrieval_dataset(dataset)
        if not corpus or not queries:
            metrics_by_subset[subset] = {"dataset_path": dataset_path, "query_count": 0, "corpus_count": 0}
            continue

        query_ids = list(queries.keys())
        corpus_ids = list(corpus.keys())
        instruction = DATASET_INSTRUCTIONS.get(subset, "")
        q_pairs = [[instruction, queries[qid]] for qid in query_ids]
        d_pairs = [["", corpus[doc_id]["text"]] for doc_id in corpus_ids]
        q_embs = encode_instruction_pairs(model, q_pairs, device, batch_size)
        d_embs = encode_instruction_pairs(model, d_pairs, device, batch_size)
        results = build_results(q_embs, d_embs, query_ids, corpus_ids)
        metrics = evaluate_retrieval_metrics(relevant_docs, results, len(corpus_ids))
        metrics.update({"dataset_path": dataset_path, "query_count": len(query_ids), "corpus_count": len(corpus_ids)})
        metrics_by_subset[subset] = metrics
    return metrics_by_subset


def main():
    parser = argparse.ArgumentParser(description="Homogeneous RT benchmark using LLM2Vec")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--base_model_name_or_path", type=str, required=True)
    parser.add_argument("--peft_model_name_or_path", type=str, default=None)
    parser.add_argument("--extra_model_name_or_path", type=str, nargs="*", default=None)
    parser.add_argument("--pooling_mode", type=str, default="mean")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--enable_bidirectional", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--vis_dataset", type=str, default=DEFAULT_VIS_DATASET)
    parser.add_argument("--dermvariants_dir", type=str, default=DEFAULT_DERMVARIANTS_DIR)
    parser.add_argument("--retrieval_subsets", type=str, nargs="*", default=DEFAULT_RETRIEVAL_SUBSETS)
    parser.add_argument("--vis_instruction", type=str, default=DATASET_INSTRUCTIONS["VisVariants"])
    parser.add_argument("--safetensors", action="store_true", default=False)
    args = parser.parse_args()

    extra_models = []
    if args.extra_model_name_or_path:
        for path in args.extra_model_name_or_path:
            clean_path = path.strip("[]\"' ")
            if clean_path:
                extra_models.append(clean_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=args.base_model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        extra_model_name_or_path=extra_models,
        pooling_mode=args.pooling_mode,
        max_length=args.max_length,
        enable_bidirectional=args.enable_bidirectional,
        torch_dtype=torch.bfloat16,
        use_safetensors=args.safetensors,
    ).to(device).eval()
    model.tokenizer.padding_side = "left"

    output_path = resolve_output_file(args.output, args.model_name)
    if output_path and os.path.exists(output_path):
        print(f"Output already exists, skipping: {output_path}")
        return

    vis_metrics = evaluate_vis_mcq(
        model,
        device,
        args.vis_dataset,
        args.batch_size,
        args.max_samples,
        args.vis_instruction,
    )
    retrieval_metrics = evaluate_retrieval_suite(
        model,
        device,
        args.dermvariants_dir,
        args.retrieval_subsets,
        args.batch_size,
        args.max_samples,
    )

    results = {
        "model_name": args.model_name,
        "base_model_name_or_path": args.base_model_name_or_path,
        "peft_model_name_or_path": args.peft_model_name_or_path,
        "extra_model_name_or_path": extra_models,
        "pooling_mode": args.pooling_mode,
        "enable_bidirectional": args.enable_bidirectional,
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
