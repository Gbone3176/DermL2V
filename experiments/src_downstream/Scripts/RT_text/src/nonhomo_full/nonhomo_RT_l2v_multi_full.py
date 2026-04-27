import argparse
import json
import logging
import math
import os
from typing import List, Optional

import torch

from llm2vec.llm2vecV5 import LLM2Vec
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
REQUIRED_METRIC_KEYS = (
    "NDCG@3",
    "NDCG@5",
    "NDCG@10",
    "Recall@3",
    "Recall@5",
    "Recall@10",
)


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
    parser = argparse.ArgumentParser(description="Multi-dataset RT nonhomo full evaluation for LLM2Vec models")
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
    parser.add_argument("--selfattn_attn_hidden_dim", type=int, default=None)
    parser.add_argument("--selfattn_num_hops", type=int, default=None)
    parser.add_argument("--selfattn_output_dropout", type=float, default=None)
    parser.add_argument(
        "--selfattn_output_norm",
        default=None,
        choices=["none", "layernorm", "l2", "rmsnorm"],
    )
    parser.add_argument("--selfattn_gamma_init", type=float, default=None)
    parser.add_argument("--selfattn_gamma_learnable", default=None)
    parser.add_argument(
        "--selfattn_merge_mode",
        default=None,
        choices=["weighted_sum", "router"],
    )
    parser.add_argument("--selfattn_merge_temperature", type=float, default=None)
    parser.add_argument("--selfattn_merge_hidden_dim", type=int, default=None)
    parser.add_argument(
        "--selfattn_merge_input_norm",
        default=None,
        choices=["none", "layernorm", "l2"],
    )
    return parser.parse_args()


def should_enable_bidirectional(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def should_enable_flag(value: Optional[str], default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "y"}


def load_model(args: argparse.Namespace):
    model_kwargs = {
        "base_model_name_or_path": args.base_model_name_or_path,
        "peft_model_name_or_path": args.peft_model_name_or_path,
        "extra_model_name_or_path": args.extra_model_name_or_path,
        "enable_bidirectional": should_enable_bidirectional(args.enable_bidirectional),
        "merge_peft": True,
        "pooling_mode": args.pooling_mode,
        "max_length": args.max_length,
        "torch_dtype": torch.float16,
        "attn_implementation": "sdpa",
    }
    if args.selfattn_attn_hidden_dim is not None:
        model_kwargs["selfattn_attn_hidden_dim"] = args.selfattn_attn_hidden_dim
    if args.selfattn_num_hops is not None:
        model_kwargs["selfattn_num_hops"] = args.selfattn_num_hops
    if args.selfattn_output_dropout is not None:
        model_kwargs["selfattn_output_dropout"] = args.selfattn_output_dropout
    if args.selfattn_output_norm is not None:
        model_kwargs["selfattn_output_norm"] = args.selfattn_output_norm
    if args.selfattn_gamma_init is not None:
        model_kwargs["selfattn_gamma_init"] = args.selfattn_gamma_init
    if args.selfattn_gamma_learnable is not None:
        model_kwargs["selfattn_gamma_learnable"] = should_enable_flag(
            args.selfattn_gamma_learnable
        )
    if args.selfattn_merge_mode is not None:
        model_kwargs["selfattn_merge_mode"] = args.selfattn_merge_mode
    if args.selfattn_merge_temperature is not None:
        model_kwargs["selfattn_merge_temperature"] = args.selfattn_merge_temperature
    if args.selfattn_merge_hidden_dim is not None:
        model_kwargs["selfattn_merge_hidden_dim"] = args.selfattn_merge_hidden_dim
    if args.selfattn_merge_input_norm is not None:
        model_kwargs["selfattn_merge_input_norm"] = args.selfattn_merge_input_norm
    model = LLM2Vec.from_pretrained(**model_kwargs)
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


def is_complete_result_file(output_file: str) -> bool:
    if not os.path.exists(output_file):
        return False
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Result file is unreadable and will be recomputed: %s (%s)", output_file, exc)
        return False

    if not isinstance(metrics, dict):
        logger.warning("Result file is not a JSON object and will be recomputed: %s", output_file)
        return False

    missing = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
    if missing:
        logger.warning("Result file is missing metrics %s and will be recomputed: %s", missing, output_file)
        return False

    for key in REQUIRED_METRIC_KEYS:
        value = metrics[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            logger.warning("Result metric %s is invalid in %s and will be recomputed", key, output_file)
            return False
    return True


def write_json_atomic(output_file: str, metrics: dict) -> None:
    tmp_file = f"{output_file}.tmp.{os.getpid()}"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, output_file)
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


def encode_dataset(
    model,
    compute_device: torch.device,
    dataset_file_path: str,
    output: str,
    model_name: str,
    instruction: str,
    doc_add_instruction: bool,
    batch_size: int,
) -> None:
    output_file = resolve_output_file(output, dataset_file_path)
    if is_complete_result_file(output_file):
        logger.info("Complete results already exist at %s, skipping...", output_file)
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

    query_pairs = [[instruction, text] for text in query_texts]
    doc_instruction = instruction if doc_add_instruction else ""
    corpus_pairs = [[doc_instruction, text] for text in corpus_texts]

    logger.info("Encoding dataset %s with %d queries and %d docs", dataset_file_path, len(query_pairs), len(corpus_pairs))
    q_emb = model.encode(
        query_pairs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=compute_device,
    )
    d_emb = model.encode(
        corpus_pairs,
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
    write_json_atomic(output_file, metrics)


def main() -> None:
    args = parse_args()
    dataset_files: List[str] = list(dict.fromkeys(args.dataset_file_path))
    instruction = args.instruction or ""

    pending = [
        path
        for path in dataset_files
        if not is_complete_result_file(resolve_output_file(args.output, path))
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
            model_name=args.model_name,
            instruction=instruction,
            doc_add_instruction=args.doc_add_instruction,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
