import argparse
import json
import os

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from llm2vec import LLM2Vec

from experiments.src_downstream.rt_text.homo.homo_RT_utils import (
    DATASET_INSTRUCTIONS,
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


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _device_str(device: torch.device | str) -> str:
    return str(device)


def _looks_like_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error" in message


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _probe_batch(model, sentences, device_str: str) -> None:
    model._encode(sentences, device=device_str, convert_to_numpy=False, multiprocessing=False)


def determine_batch_size(model, pairs, device, requested_batch_size: int, max_batch_size: int, auto_batch_size: bool) -> int:
    if not pairs:
        return requested_batch_size

    batch_size = max(1, requested_batch_size)
    if not auto_batch_size:
        return batch_size

    sample_sentence = model._convert_to_str(*pairs[0])
    low = 1
    high = min(max_batch_size, len(pairs))
    best = 1
    device_str = _device_str(device)

    while low <= high:
        candidate = (low + high) // 2
        probe_sentences = [sample_sentence] * candidate
        try:
            _probe_batch(model, probe_sentences, device_str)
            best = candidate
            low = candidate + 1
        except RuntimeError as exc:
            if not _looks_like_oom(exc):
                raise
            _clear_cuda_cache()
            high = candidate - 1

    return min(max(1, best), max_batch_size)


def encode_instruction_pairs(model, pairs, device, batch_size, auto_batch_size=True, max_batch_size=64, show_progress_bar=True):
    if not pairs:
        hidden_size = getattr(model.model.config, "hidden_size", 0)
        return torch.empty((0, hidden_size), dtype=torch.float32)

    device_str = _device_str(device)
    model.eval()
    model.to(device_str)

    sentences = [model._convert_to_str(instruction, text) for instruction, text in pairs]
    length_sorted_idx = sorted(range(len(sentences)), key=lambda idx: -model._text_length(sentences[idx]))
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    sorted_pairs = [pairs[idx] for idx in length_sorted_idx]

    effective_batch_size = determine_batch_size(
        model,
        sorted_pairs,
        device,
        requested_batch_size=batch_size,
        max_batch_size=max_batch_size,
        auto_batch_size=auto_batch_size,
    )
    print(f"Using single-process batch_size={effective_batch_size} for {len(pairs)} pairs on {device_str}")

    all_embeddings = []
    start_index = 0
    progress_bar = tqdm(total=len(sentences_sorted), desc="Batches", disable=not show_progress_bar)
    while start_index < len(sentences_sorted):
        end_index = min(start_index + effective_batch_size, len(sentences_sorted))
        sentences_batch = sentences_sorted[start_index:end_index]
        current_batch_size = len(sentences_batch)
        while True:
            try:
                embeddings = model._encode(
                    sentences_batch[:current_batch_size],
                    device=device_str,
                    convert_to_numpy=False,
                    multiprocessing=False,
                )
                all_embeddings.append(embeddings)
                start_index += current_batch_size
                progress_bar.update(current_batch_size)
                break
            except RuntimeError as exc:
                if not _looks_like_oom(exc) or current_batch_size == 1:
                    progress_bar.close()
                    raise
                _clear_cuda_cache()
                current_batch_size = max(1, current_batch_size // 2)
                effective_batch_size = current_batch_size
    progress_bar.close()

    all_embeddings = torch.cat(all_embeddings, dim=0)
    inverse_idx = torch.argsort(torch.tensor(length_sorted_idx))
    return all_embeddings[inverse_idx].to(torch.float32)


def evaluate_vis_mcq(model, device, vis_dataset_path, batch_size, max_samples, instruction, auto_batch_size, max_batch_size):
    dataset = load_jsonl(vis_dataset_path)
    if max_samples > 0:
        dataset = dataset[:max_samples]
    queries, candidate_sets = build_vis_mcq_samples(dataset)
    if not queries:
        return {"dataset_path": vis_dataset_path, "count": 0, "accuracy": 0.0}

    q_pairs = [[instruction, query] for query in queries]
    q_embs = encode_instruction_pairs(model, q_pairs, device, batch_size, auto_batch_size=auto_batch_size, max_batch_size=max_batch_size)
    flat_candidates = [candidate for candidates in candidate_sets for candidate in candidates]
    cand_counts = [len(candidates) for candidates in candidate_sets]
    c_pairs = [["", candidate] for candidate in flat_candidates]
    flat_c_embs = encode_instruction_pairs(model, c_pairs, device, batch_size, auto_batch_size=auto_batch_size, max_batch_size=max_batch_size)

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


def evaluate_retrieval_suite(model, device, dataset_dir, subsets, batch_size, max_samples, retrieval_mode, auto_batch_size, max_batch_size):
    metrics_by_subset = {}
    dataset_paths, datasets_by_subset = load_retrieval_subset_datasets(dataset_dir, subsets, max_samples)

    if retrieval_mode == "mixed":
        corpus, queries_by_subset, relevant_docs_by_subset = build_mixed_retrieval_dataset(datasets_by_subset)
        corpus_ids = list(corpus.keys())
        if not corpus_ids:
            for subset in subsets:
                metrics_by_subset[subset] = {
                    "dataset_path": dataset_paths[subset],
                    "query_count": 0,
                    "corpus_count": 0,
                }
            return metrics_by_subset

        d_pairs = [["", corpus[doc_id]["text"]] for doc_id in corpus_ids]
        d_embs = encode_instruction_pairs(model, d_pairs, device, batch_size, auto_batch_size=auto_batch_size, max_batch_size=max_batch_size)
        for subset in subsets:
            queries = queries_by_subset[subset]
            relevant_docs = relevant_docs_by_subset[subset]
            query_ids = list(queries.keys())
            if not query_ids:
                metrics_by_subset[subset] = {
                    "dataset_path": dataset_paths[subset],
                    "query_count": 0,
                    "corpus_count": len(corpus_ids),
                }
                continue

            instruction = DATASET_INSTRUCTIONS.get(subset, "")
            q_pairs = [[instruction, queries[qid]] for qid in query_ids]
            q_embs = encode_instruction_pairs(model, q_pairs, device, batch_size, auto_batch_size=auto_batch_size, max_batch_size=max_batch_size)
            results = build_results(q_embs, d_embs, query_ids, corpus_ids)
            metrics = evaluate_retrieval_metrics(relevant_docs, results, len(corpus_ids))
            metrics.update(
                {
                    "dataset_path": dataset_paths[subset],
                    "query_count": len(query_ids),
                    "corpus_count": len(corpus_ids),
                }
            )
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
        instruction = DATASET_INSTRUCTIONS.get(subset, "")
        q_pairs = [[instruction, queries[qid]] for qid in query_ids]
        d_pairs = [["", corpus[doc_id]["text"]] for doc_id in corpus_ids]
        q_embs = encode_instruction_pairs(model, q_pairs, device, batch_size, auto_batch_size=auto_batch_size, max_batch_size=max_batch_size)
        d_embs = encode_instruction_pairs(model, d_pairs, device, batch_size, auto_batch_size=auto_batch_size, max_batch_size=max_batch_size)
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batch_size", type=int, default=128)
    parser.add_argument("--disable_auto_batch_size", action="store_true", default=False)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--vis_dataset", type=str, default=DEFAULT_VIS_DATASET)
    parser.add_argument("--dermvariants_dir", type=str, default=DEFAULT_DERMVARIANTS_DIR)
    parser.add_argument("--retrieval_subsets", type=str, nargs="*", default=DEFAULT_RETRIEVAL_SUBSETS)
    parser.add_argument("--retrieval_mode", type=str, default=DEFAULT_RETRIEVAL_MODE, choices=["mixed", "separate"])
    parser.add_argument("--vis_instruction", type=str, default=DATASET_INSTRUCTIONS["VisVariants"])
    parser.add_argument("--safetensors", action="store_true", default=False)
    args = parser.parse_args()

    extra_models = []
    if args.extra_model_name_or_path:
        for path in args.extra_model_name_or_path:
            clean_path = path.strip("[]\"' ")
            if clean_path:
                extra_models.append(clean_path)

    if torch.cuda.is_available():
        if args.cuda_device < 0 or args.cuda_device >= torch.cuda.device_count():
            raise ValueError(
                f"--cuda_device={args.cuda_device} is out of range for {torch.cuda.device_count()} visible CUDA devices."
            )
        torch.cuda.set_device(args.cuda_device)
        device = torch.device(f"cuda:{args.cuda_device}")
        model_load_kwargs = {
            "device_map": {"": args.cuda_device},
            "low_cpu_mem_usage": True,
        }
    else:
        device = torch.device("cpu")
        model_load_kwargs = {}

    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=args.base_model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        extra_model_name_or_path=extra_models,
        pooling_mode=args.pooling_mode,
        max_length=args.max_length,
        enable_bidirectional=args.enable_bidirectional,
        torch_dtype=torch.bfloat16,
        use_safetensors=args.safetensors,
        **model_load_kwargs,
    ).to(device).eval()
    model.tokenizer.padding_side = "left"

    output_path = resolve_output_file(args.output, args.model_name)
    if output_matches_retrieval_mode(output_path, args.retrieval_mode):
        print(f"Output already exists, skipping: {output_path}")
        return

    vis_metrics = evaluate_vis_mcq(
        model,
        device,
        args.vis_dataset,
        args.batch_size,
        args.max_samples,
        args.vis_instruction,
        not args.disable_auto_batch_size,
        args.max_batch_size,
    )
    retrieval_metrics = evaluate_retrieval_suite(
        model,
        device,
        args.dermvariants_dir,
        args.retrieval_subsets,
        args.batch_size,
        args.max_samples,
        args.retrieval_mode,
        not args.disable_auto_batch_size,
        args.max_batch_size,
    )

    results = {
        "model_name": args.model_name,
        "base_model_name_or_path": args.base_model_name_or_path,
        "peft_model_name_or_path": args.peft_model_name_or_path,
        "extra_model_name_or_path": extra_models,
        "pooling_mode": args.pooling_mode,
        "enable_bidirectional": args.enable_bidirectional,
        "batch_size": args.batch_size,
        "max_batch_size": args.max_batch_size,
        "auto_batch_size": (not args.disable_auto_batch_size),
        "cuda_device": args.cuda_device if torch.cuda.is_available() else None,
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
