import argparse
import json
import logging
import os
from typing import List, Union

import torch

from llm2vec.llm2vecV1 import LLM2Vec
from experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_full_utils import (
    build_flat_output_file,
    build_corpus_queries,
    evaluate_retrieval_metrics,
    load_jsonl,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_INSTRUCTION = "Given a dermatologic question, return the answer that most closely corresponds to the information being asked for."


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-dataset RT nonhomo full evaluation for LLM2Vec models with encode()")
    parser.add_argument("--dataset_file_path", action="append", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_model_name_or_path", required=True)
    parser.add_argument("--peft_model_name_or_path")
    parser.add_argument("--extra_model_name_or_path", nargs="*", default=[])
    parser.add_argument("--instruction", default=DEFAULT_RETRIEVAL_INSTRUCTION)
    parser.add_argument("--doc_add_instruction", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--enable_bidirectional", default="True")
    parser.add_argument(
        "--pooling_mode",
        default="mean",
        choices=["mean", "weighted_mean", "eos_token", "latent_pooling", "structured_selfattn", "structured_selfattn_fusion"],
    )
    parser.add_argument("--selfattn_attn_hidden_dim", type=int, default=512)
    parser.add_argument("--selfattn_num_hops", type=int, default=8)
    parser.add_argument("--selfattn_output_dropout", type=float, default=0.0)
    parser.add_argument("--selfattn_output_norm", default="layernorm", choices=["none", "layernorm", "l2", "rmsnorm"])
    return parser.parse_args()


def should_enable_flag(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def build_model_input(text: str, instruction: str) -> List[Union[str, int]]:
    return [instruction.strip() if instruction else "", text, 0]


def load_model(args: argparse.Namespace):
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=args.base_model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        extra_model_name_or_path=args.extra_model_name_or_path,
        enable_bidirectional=should_enable_flag(args.enable_bidirectional),
        merge_peft=True,
        pooling_mode=args.pooling_mode,
        max_length=args.max_length,
        selfattn_attn_hidden_dim=args.selfattn_attn_hidden_dim,
        selfattn_num_hops=args.selfattn_num_hops,
        selfattn_output_dropout=args.selfattn_output_dropout,
        selfattn_output_norm=args.selfattn_output_norm,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if compute_device.type == "cuda":
        model.to(compute_device)
    else:
        try:
            model.to(torch.float32)
        except Exception:
            pass
    return model, compute_device


def resolve_output_file(output_root: str, dataset_file_path: str) -> str:
    return build_flat_output_file(output_root, dataset_file_path)


def encode_dataset(
    model,
    compute_device: torch.device,
    dataset_file_path: str,
    output: str,
    instruction: str,
    doc_add_instruction: bool,
    batch_size: int,
) -> None:
    output_file = resolve_output_file(output, dataset_file_path)
    if os.path.exists(output_file):
        logger.info("Results already exist at %s, skipping...", output_file)
        return

    dataset = load_jsonl(dataset_file_path)
    corpus, queries, relevant_docs = build_corpus_queries(dataset)
    if not queries or not corpus or not relevant_docs:
        raise ValueError(
            f"No valid retrieval samples were built from {dataset_file_path}. "
            "Supported formats are {'question', 'right_choice', 'wrong_choices'} and {'query', 'doc'}."
        )

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

    query_inputs = [build_model_input(text=text, instruction=instruction) for text in query_texts]
    doc_instruction = instruction if doc_add_instruction else ""
    corpus_inputs = [build_model_input(text=text, instruction=doc_instruction) for text in corpus_texts]

    logger.info(
        "Encoding dataset %s with %d queries and %d docs via encode() (instruction=%s)",
        dataset_file_path,
        len(query_inputs),
        len(corpus_inputs),
        bool(instruction),
    )
    q_emb = model.encode(
        query_inputs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=compute_device,
    )
    d_emb = model.encode(
        corpus_inputs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=compute_device,
    )

    scores = cos_sim(q_emb, d_emb)
    scores[torch.isnan(scores)] = -1
    top_k = min(10, len(corpus_ids))
    top_vals, top_idx = torch.topk(scores, top_k, dim=1, largest=True, sorted=True)
    top_vals = top_vals.cpu().tolist()
    top_idx = top_idx.cpu().tolist()

    results = {}
    for i, qid in enumerate(query_ids):
        results[qid] = {}
        for rank, idx in enumerate(top_idx[i]):
            results[qid][corpus_ids[idx]] = top_vals[i][rank]

    metrics = evaluate_retrieval_metrics(relevant_docs, results, len(corpus_ids))
    logger.info(json.dumps(metrics, indent=4))
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def main() -> None:
    args = parse_args()
    dataset_files: List[str] = list(dict.fromkeys(args.dataset_file_path))

    pending = [
        path
        for path in dataset_files
        if not os.path.exists(resolve_output_file(args.output, path))
    ]
    if not pending:
        logger.info("All requested datasets already complete for %s", args.model_name)
        return

    model, compute_device = load_model(args)
    logger.info("Loaded model once for %d datasets: %s", len(pending), pending)

    for dataset_file_path in dataset_files:
        encode_dataset(
            model=model,
            compute_device=compute_device,
            dataset_file_path=dataset_file_path,
            output=args.output,
            instruction=args.instruction or "",
            doc_add_instruction=args.doc_add_instruction,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
