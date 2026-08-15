#!/usr/bin/env python
"""Load DermL2V, encode text, or run the RT-full test sets."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from derml2v_llm2vec import LLM2Vec
from rt_full_utils import build_corpus_queries, build_results, evaluate_at_10, load_jsonl


LOGGER = logging.getLogger("derml2v_inference")

DATA_DIR = SCRIPT_DIR / "data"
WEIGHTS_DIR = SCRIPT_DIR / "weights"
LOCAL_BASE_MODEL = WEIGHTS_DIR / "Meta-Llama-3.1-8B-Instruct"
LOCAL_MNTP_ADAPTER = WEIGHTS_DIR / "llm2vec-mntp"
LOCAL_SUPERVISED_ADAPTER = WEIGHTS_DIR / "llm2vec-mntp-supervised"
LOCAL_CHECKPOINT_DIR = WEIGHTS_DIR / "DermL2V_adapter"

FALLBACK_BASE_MODEL = ""
FALLBACK_MNTP_ADAPTER = ""
FALLBACK_SUPERVISED_ADAPTER = ""
FALLBACK_CHECKPOINT_DIR = ""

DEFAULT_BASE_MODEL = str(LOCAL_BASE_MODEL if LOCAL_BASE_MODEL.is_dir() else FALLBACK_BASE_MODEL)
DEFAULT_MNTP_ADAPTER = str(LOCAL_MNTP_ADAPTER if LOCAL_MNTP_ADAPTER.is_dir() else FALLBACK_MNTP_ADAPTER)
DEFAULT_SUPERVISED_ADAPTER = str(LOCAL_SUPERVISED_ADAPTER if LOCAL_SUPERVISED_ADAPTER.is_dir() else FALLBACK_SUPERVISED_ADAPTER)
DEFAULT_CHECKPOINT_DIR = str(LOCAL_CHECKPOINT_DIR if LOCAL_CHECKPOINT_DIR.is_dir() else FALLBACK_CHECKPOINT_DIR)

DEFAULT_QUERY_INSTRUCTION = (
    "Given a dermatologic question, return the answer that most closely "
    "corresponds to the information being asked for."
)


def default_data_path(local_filename: str) -> str:
    return str(DATA_DIR / local_filename)


RT_FULL_DATASETS = {
    "DermSynth": {
        "display_name": "DermaSynth-E3",
        "source_path": default_data_path("DermaSynth-E3.jsonl"),
        "output_filename": "DermSynth_knowledgebase.json",
    },
    "MedMCQA": {
        "display_name": "MedMCQA",
        "source_path": default_data_path("MedMCQA.jsonl"),
        "output_filename": "MedMCQA_RT.json",
    },
    "MedQuAD": {
        "display_name": "MedQuAD",
        "source_path": default_data_path("MedQuAD.jsonl"),
        "output_filename": "MedQuAD_dermatology_qa_retrieval_doclt300.json",
    },
    "SCE": {
        "display_name": "SCE-Derma-SQ",
        "source_path": default_data_path("SCE-Derma-SQ.jsonl"),
        "output_filename": "sce_retrieval.json",
    },
}

DEFAULT_RT_OUTPUT_DIR = (
    SCRIPT_DIR
    / "results"
    / "rt_full"
    / "DermL2V_rt_full"
)


def first_existing_path(paths: Iterable[str]) -> str | None:
    for path in paths:
        if path and Path(path).expanduser().is_dir():
            return path
    return None


def default_checkpoint_dir() -> str:
    env_path = os.environ.get("DERML2V_CHECKPOINT_DIR")
    path = first_existing_path([env_path] if env_path else [])
    if path is not None:
        return path
    path = first_existing_path([DEFAULT_CHECKPOINT_DIR])
    return path or ""


def str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def existing_dir(path_value: str, label: str) -> str:
    if not path_value:
        raise FileNotFoundError(f"{label} was not provided.")
    path = Path(path_value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", action="append", help="Text to encode. Can be passed more than once.")
    parser.add_argument("--text_file", type=Path, help="UTF-8 text file. Each non-empty line is encoded separately.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path. Defaults to stdout.")
    parser.add_argument("--eval_rt_full", action="store_true", help="Run the four default nonhomogeneous RT test sets.")
    parser.add_argument("--rt_output_dir", type=Path, default=DEFAULT_RT_OUTPUT_DIR)
    parser.add_argument("--rt_dataset", action="append", choices=sorted(RT_FULL_DATASETS), help="Subset of RT datasets to run. Defaults to all.")
    parser.add_argument("--query_instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--score_batch_size", type=int, default=128)
    parser.add_argument("--eval_limit", type=int, help="Debug only: limit records per RT dataset before encoding.")

    parser.add_argument("--base_model_name_or_path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--peft_model_name_or_path", default=DEFAULT_MNTP_ADAPTER)
    parser.add_argument("--supervised_model_name_or_path", default=DEFAULT_SUPERVISED_ADAPTER)
    parser.add_argument("--checkpoint_dir", default=default_checkpoint_dir())
    parser.add_argument("--extra_model_name_or_path", action="append", default=[])

    parser.add_argument("--pooling_mode", choices=["mean", "weighted_mean", "eos_token", "last_token", "bos_token", "latent_pooling", "structured_selfattn", "structured_selfattn_fusion"], help="Optional override. By default the checkpoint llm2vec_config.json decides this.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--enable_bidirectional", default="true")
    parser.add_argument("--expected_dim", type=int, default=4096)
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings before output.")

    parser.add_argument("--selfattn_attn_hidden_dim", type=int)
    parser.add_argument("--selfattn_num_hops", type=int)
    parser.add_argument("--selfattn_output_dropout", type=float)
    parser.add_argument("--selfattn_output_norm", choices=["none", "layernorm", "l2", "rmsnorm"])
    parser.add_argument("--selfattn_gamma_init", type=float)
    parser.add_argument("--selfattn_gamma_learnable")
    parser.add_argument("--selfattn_merge_mode", choices=["weighted_sum", "router"])
    parser.add_argument("--selfattn_merge_temperature", type=float)
    parser.add_argument("--selfattn_merge_hidden_dim", type=int)
    parser.add_argument("--selfattn_merge_input_norm", choices=["none", "layernorm", "l2"])
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def read_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    if args.text:
        texts.extend(args.text)
    if args.text_file:
        with args.text_file.open("r", encoding="utf-8") as handle:
            texts.extend(line.strip() for line in handle if line.strip())
    if not texts:
        raise ValueError("Provide --text or --text_file.")
    return texts


def adapter_chain(args: argparse.Namespace, checkpoint_dir: str) -> list[str]:
    chain = []
    if args.supervised_model_name_or_path:
        chain.append(existing_dir(args.supervised_model_name_or_path, "supervised_model_name_or_path"))
    for extra_model in args.extra_model_name_or_path:
        chain.append(existing_dir(extra_model, "extra_model_name_or_path"))
    chain.append(existing_dir(checkpoint_dir, "checkpoint_dir"))
    return chain


def build_model_kwargs(args: argparse.Namespace, checkpoint_dir: str) -> dict:
    model_kwargs = {
        "base_model_name_or_path": existing_dir(args.base_model_name_or_path, "base_model_name_or_path"),
        "peft_model_name_or_path": existing_dir(args.peft_model_name_or_path, "peft_model_name_or_path"),
        "extra_model_name_or_path": adapter_chain(args, checkpoint_dir),
        "enable_bidirectional": str_to_bool(args.enable_bidirectional),
        "merge_peft": True,
        "max_length": args.max_length,
        "torch_dtype": torch.float16 if torch.cuda.is_available() and str(args.device).startswith("cuda") else torch.float32,
        "attn_implementation": args.attn_implementation,
    }
    optional_overrides = {
        "pooling_mode": args.pooling_mode,
        "selfattn_attn_hidden_dim": args.selfattn_attn_hidden_dim,
        "selfattn_num_hops": args.selfattn_num_hops,
        "selfattn_output_dropout": args.selfattn_output_dropout,
        "selfattn_output_norm": args.selfattn_output_norm,
        "selfattn_gamma_init": args.selfattn_gamma_init,
        "selfattn_gamma_learnable": str_to_bool(args.selfattn_gamma_learnable) if args.selfattn_gamma_learnable is not None else None,
    }
    if args.pooling_mode == "structured_selfattn_fusion":
        optional_overrides.update(
            {
                "selfattn_merge_mode": args.selfattn_merge_mode,
                "selfattn_merge_temperature": args.selfattn_merge_temperature,
                "selfattn_merge_hidden_dim": args.selfattn_merge_hidden_dim,
                "selfattn_merge_input_norm": args.selfattn_merge_input_norm,
            }
        )
    model_kwargs.update({key: value for key, value in optional_overrides.items() if value is not None})
    return model_kwargs


def load_model(args: argparse.Namespace):
    if not args.checkpoint_dir:
        raise ValueError(
            "No DermL2V checkpoint found. Pass --checkpoint_dir or set DERML2V_CHECKPOINT_DIR."
        )
    checkpoint_dir = existing_dir(args.checkpoint_dir, "checkpoint_dir")
    model = LLM2Vec.from_pretrained(**build_model_kwargs(args, checkpoint_dir))
    device = torch.device(args.device)
    if device.type == "cuda":
        model.to(device)
    else:
        model.to(torch.float32)
    model.eval()
    return model, device


def encode_pairs(
    model,
    pairs: Sequence[Sequence[str]],
    batch_size: int,
    device: torch.device,
    log_prefix: str,
) -> torch.Tensor:
    encoded_inputs = [model._convert_to_str(pair[0], pair[1]) for pair in pairs]
    all_embeddings = []
    total = len(encoded_inputs)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        if start == 0 or end == total or (start // batch_size) % 25 == 0:
            LOGGER.info("%s: encoding %s-%s/%s", log_prefix, start + 1, end, total)
        batch = encoded_inputs[start:end]
        embeddings = model._encode(batch, device=str(device), convert_to_numpy=False)
        all_embeddings.append(embeddings.detach().float().cpu())
    return torch.cat(all_embeddings, dim=0)


def encode_texts(model, texts: list[str], batch_size: int, device: torch.device) -> torch.Tensor:
    pairs = [["", text] for text in texts]
    return encode_pairs(model, pairs, batch_size, device, "text")


def serialize_embeddings(texts: list[str], embeddings: torch.Tensor) -> dict:
    vectors = embeddings.tolist()
    return {
        "dim": int(embeddings.shape[-1]),
        "count": len(texts),
        "items": [
            {
                "text": text,
                "embedding": vector,
            }
            for text, vector in zip(texts, vectors)
        ],
    }


def selected_rt_datasets(dataset_names: Sequence[str] | None) -> list[tuple[str, dict]]:
    names = list(dataset_names) if dataset_names else list(RT_FULL_DATASETS)
    return [(name, RT_FULL_DATASETS[name]) for name in names]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=4)
        handle.write("\n")


def write_rt_summary(
    results: dict[str, dict[str, float]],
    output_dir: Path,
    checkpoint_dir: str,
) -> None:
    rows = []
    for dataset_name, metrics in results.items():
        rows.append(
            (
                dataset_name,
                metrics["NDCG@10"] * 100.0,
                metrics["Recall@10"] * 100.0,
            )
        )
    avg_ndcg = sum(row[1] for row in rows) / len(rows)
    avg_recall = sum(row[2] for row in rows) / len(rows)
    avg = (avg_ndcg + avg_recall) / 2.0
    checkpoint_label = Path(checkpoint_dir).name.replace("checkpoint-", "cp")
    lines = [
        "# DermL2V RT-Full Summary at @10",
        "",
        f"Checkpoint: `{checkpoint_dir}`",
        "",
        "Metrics are reported as percentages.",
        "",
        "| CP | Dataset | NDCG@10 (%) | Recall@10 (%) |",
        "|---|---|---:|---:|",
    ]
    for dataset_name, ndcg, recall in rows:
        lines.append(f"| {checkpoint_label} | {dataset_name} | {ndcg:.2f} | {recall:.2f} |")
    lines.extend(
        [
            "",
            f"- Avg_NDCG@10: {avg_ndcg:.2f}%",
            f"- Avg_Recall@10: {avg_recall:.2f}%",
            f"- Avg: {avg:.2f}%",
        ]
    )
    summary_path = output_dir / "summary_at10.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_rt_full_eval(model, args: argparse.Namespace, checkpoint_dir: str, device: torch.device) -> None:
    output_dir = args.rt_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}

    for dataset_key, dataset_info in selected_rt_datasets(args.rt_dataset):
        source_path = Path(dataset_info["source_path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"RT dataset does not exist: {source_path}")
        LOGGER.info("Loading %s from %s", dataset_key, source_path)
        records = load_jsonl(str(source_path))
        if args.eval_limit is not None:
            records = records[: args.eval_limit]
        corpus, queries, relevant_docs = build_corpus_queries(records)
        query_ids = list(queries.keys())
        corpus_ids = list(corpus.keys())
        LOGGER.info(
            "%s: records=%s queries=%s corpus=%s",
            dataset_key,
            len(records),
            len(queries),
            len(corpus),
        )
        query_pairs = [[args.query_instruction, queries[query_id]] for query_id in query_ids]
        doc_pairs = [["", corpus[corpus_id]["text"]] for corpus_id in corpus_ids]

        query_embeddings = encode_pairs(
            model,
            query_pairs,
            args.batch_size,
            device,
            f"{dataset_key} queries",
        )
        doc_embeddings = encode_pairs(
            model,
            doc_pairs,
            args.batch_size,
            device,
            f"{dataset_key} docs",
        )
        retrieval_results = build_results(query_embeddings, doc_embeddings, query_ids, corpus_ids)
        metrics = evaluate_at_10(relevant_docs, retrieval_results, len(corpus_ids))
        display_name = dataset_info["display_name"]
        results[display_name] = metrics
        output_path = output_dir / dataset_info["output_filename"]
        write_json(output_path, metrics)
        LOGGER.info("Wrote %s: %s", output_path, metrics)

    write_rt_summary(results, output_dir, checkpoint_dir)
    LOGGER.info("Wrote %s", output_dir / "summary_at10.md")


def main() -> None:
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=args.log_level)
    model, device = load_model(args)
    if args.eval_rt_full:
        run_rt_full_eval(model, args, args.checkpoint_dir, device)
        return

    texts = read_texts(args)
    embeddings = encode_texts(model, texts, args.batch_size, device)
    if args.normalize:
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    if embeddings.shape[-1] != args.expected_dim:
        raise ValueError(f"Expected embedding dim {args.expected_dim}, got {embeddings.shape[-1]}")

    payload = serialize_embeddings(texts, embeddings)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")
        LOGGER.info("Wrote %s", args.output)
    else:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
