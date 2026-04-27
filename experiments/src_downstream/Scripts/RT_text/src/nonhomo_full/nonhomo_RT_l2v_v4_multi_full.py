import argparse
import json
import logging
import os
from typing import List

import torch

from llm2vec.llm2vecV4 import LLM2Vec
from experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_full_utils import (
    build_corpus_queries,
    build_output_file,
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
    parser = argparse.ArgumentParser(description="Multi-dataset RT nonhomo full evaluation for LLM2Vec V4 models")
    parser.add_argument("--dataset_file_path", action="append", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_model_name_or_path", required=True)
    parser.add_argument("--peft_model_name_or_path")
    parser.add_argument("--extra_model_name_or_path", nargs="*", default=[])
    parser.add_argument("--instruction", default=DEFAULT_RETRIEVAL_INSTRUCTION)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--enable_bidirectional", default="True")
    parser.add_argument(
        "--pooling_mode",
        default="mean",
        choices=["mean", "weighted_mean", "eos_token", "latent_pooling", "res_mlp_pooling"],
    )
    parser.add_argument("--res_mlp_hidden_dim", type=int, default=None)
    parser.add_argument("--res_mlp_num_layers", type=int, default=4)
    parser.add_argument("--res_mlp_dropout", type=float, default=0.0)
    parser.add_argument("--res_mlp_gamma_init", type=float, default=1e-3)
    parser.add_argument("--res_mlp_gamma_learnable", default="True")
    parser.add_argument("--res_mlp_output_normalize", default="False")
    parser.add_argument("--res_mlp_output_layernorm", default="False")
    return parser.parse_args()


def as_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def main() -> None:
    args = parse_args()
    dataset_files: List[str] = list(dict.fromkeys(args.dataset_file_path))
    pending = [path for path in dataset_files if not os.path.exists(build_output_file(args.output, path, args.model_name))]
    if not pending:
        logger.info("All requested datasets already complete for %s", args.model_name)
        return

    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=args.base_model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        extra_model_name_or_path=args.extra_model_name_or_path,
        enable_bidirectional=as_bool(args.enable_bidirectional),
        merge_peft=True,
        pooling_mode=args.pooling_mode,
        max_length=args.max_length,
        res_mlp_hidden_dim=args.res_mlp_hidden_dim,
        res_mlp_num_layers=args.res_mlp_num_layers,
        res_mlp_dropout=args.res_mlp_dropout,
        res_mlp_gamma_init=args.res_mlp_gamma_init,
        res_mlp_gamma_learnable=as_bool(args.res_mlp_gamma_learnable),
        res_mlp_output_normalize=as_bool(args.res_mlp_output_normalize),
        res_mlp_output_layernorm=as_bool(args.res_mlp_output_layernorm),
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if compute_device.type == "cuda":
        model.to(compute_device)

    instruction = args.instruction or ""
    for dataset_file_path in dataset_files:
        output_file = build_output_file(args.output, dataset_file_path, args.model_name)
        if os.path.exists(output_file):
            logger.info("Results already exist at %s, skipping...", output_file)
            continue

        dataset = load_jsonl(dataset_file_path)
        corpus, queries, relevant_docs = build_corpus_queries(dataset)
        query_ids = list(queries.keys())
        corpus_ids = list(corpus.keys())
        query_pairs = [[instruction, queries[qid]] for qid in query_ids]
        corpus_pairs = [["", corpus[cid]["text"]] for cid in corpus_ids]

        q_emb = model.encode(query_pairs, batch_size=args.batch_size, show_progress_bar=True, convert_to_tensor=True, device=compute_device)
        d_emb = model.encode(corpus_pairs, batch_size=args.batch_size, show_progress_bar=True, convert_to_tensor=True, device=compute_device)

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
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
