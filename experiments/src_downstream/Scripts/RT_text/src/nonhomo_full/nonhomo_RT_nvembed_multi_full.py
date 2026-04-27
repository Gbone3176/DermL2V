import argparse
import json
import os

import torch
from sentence_transformers import SentenceTransformer

from experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_nvembed_full import (
    DEFAULT_RETRIEVAL_INSTRUCTION,
    add_eos,
    encode_batches,
    patch_nvembed_tokenizer_loading,
)
from experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_full_utils import (
    build_corpus_queries,
    build_output_file,
    build_results,
    evaluate_at_10,
    load_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset full retrieval evaluation using NVIDIA NV-Embed-style models")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query_instruction", default=DEFAULT_RETRIEVAL_INSTRUCTION)
    parser.add_argument("--attn_implementation", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    args = parser.parse_args()

    model_name = args.model_name or args.model_name_or_path.rstrip("/").split("/")[-1]
    pending = [path for path in dict.fromkeys(args.input) if not os.path.exists(build_output_file(args.output, path, model_name))]
    if not pending:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    patch_nvembed_tokenizer_loading()
    model = SentenceTransformer(
        args.model_name_or_path,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "right", "trust_remote_code": True},
        device=device,
        trust_remote_code=True,
    )
    model.max_seq_length = args.max_length

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
        query_texts = [queries[qid] for qid in query_ids]
        corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]
        query_prefix = f"Instruct: {args.query_instruction}\nQuery: "
        eos_token = getattr(model.tokenizer, "eos_token", None) or ""
        if eos_token:
            query_texts = add_eos(query_texts, eos_token)
            corpus_texts = add_eos(corpus_texts, eos_token)
        q_emb = encode_batches(model, query_texts, args.batch_size, prompt=query_prefix, desc="Encoding queries")
        d_emb = encode_batches(model, corpus_texts, args.batch_size, prompt=None, desc="Encoding documents")
        metrics = evaluate_at_10(relevant_docs, build_results(q_emb, d_emb, query_ids, corpus_ids), len(corpus_ids))
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
