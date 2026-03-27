import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from transformers import HfArgumentParser, set_seed

from llm2vec.llm2vecV4 import LLM2Vec

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DATASET_NAME_MAPPING = {
    "eval3-text-benchmark_split_choices": "DermSynth_knowledgebase",
    "medmcqa_skin_retrieval_long_doc_test": "medmcqa_long",
    "medmcqa_skin_retrieval_short_doc_test": "medmcqa_short",
    "MedMCQA_RT_query_doc": "MedMCQA_RT",
    "MedQuAD_dermatology_qa_retrieval": "MedQuAD",
}

try:
    from accelerate import PartialState

    PartialState()
except Exception:
    pass


@dataclass
class ModelArguments:
    base_model_name_or_path: str = field(metadata={"help": "Path to pretrained base model"})
    peft_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to PEFT adapter"}
    )
    extra_model_name_or_path: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "Path to extra model"}
    )
    enable_bidirectional: bool = field(default=True)
    max_length: int = field(default=512)
    pooling_mode: str = field(
        default="mean",
        metadata={
            "choices": [
                "mean",
                "weighted_mean",
                "eos_token",
                "latent_pooling",
                "res_mlp_pooling",
            ]
        },
    )
    res_mlp_hidden_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Hidden dim of the residual MLP pooling head."},
    )
    res_mlp_num_layers: int = field(default=4)
    res_mlp_dropout: float = field(default=0.0)
    res_mlp_gamma_init: float = field(default=1e-3)
    res_mlp_gamma_learnable: bool = field(default=True)
    res_mlp_output_normalize: bool = field(default=False)
    res_mlp_output_layernorm: bool = field(default=False)


@dataclass
class DataArguments:
    dataset_file_path: str = field(metadata={"help": "Path to jsonl dataset file"})
    model_name: str = field(metadata={"help": "Name for output metrics file"})
    output: str = field(metadata={"help": "Output directory"})
    batch_size: int = field(default=16)
    instruction: Optional[str] = field(
        default=None,
        metadata={"help": "Instruction to prepend to all query texts"},
    )


def cos_sim(a: torch.Tensor, b: torch.Tensor):
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


def load_jsonl(file_path: str) -> List[dict]:
    data: List[dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_no} in {file_path}: {e}"
                ) from e
    return data


def sanitize_path_component(name: str) -> str:
    sanitized = name.strip().replace(os.sep, "_")
    if os.altsep:
        sanitized = sanitized.replace(os.altsep, "_")
    return sanitized or "unknown"


def build_output_file(output_root: str, dataset_file_path: str, model_name: str) -> str:
    dataset_stem = os.path.splitext(os.path.basename(dataset_file_path))[0]
    dataset_name = sanitize_path_component(
        DATASET_NAME_MAPPING.get(dataset_stem, dataset_stem)
    )
    model_file_name = f"{sanitize_path_component(model_name)}.json"
    dataset_output_dir = os.path.join(output_root, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    return os.path.join(dataset_output_dir, model_file_name)


def build_corpus_queries(
    dataset,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    relevant_docs: Dict[str, Dict[str, int]] = {}
    new_format_doc_ids: Dict[str, str] = {}

    def _normalize_text(value) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        return value or None

    for idx, sample in enumerate(dataset):
        pair_id = str(idx)

        if isinstance(sample, dict) and "question" in sample:
            question = _normalize_text(sample.get("question"))
            right_choice = _normalize_text(sample.get("right_choice"))
            wrong_choices = sample.get("wrong_choices") or []
            if isinstance(wrong_choices, str):
                wrong_choices = [wrong_choices]
            wrong_choices = [_normalize_text(wrong) for wrong in wrong_choices]
            wrong_choices = [wrong for wrong in wrong_choices if wrong]
            if not question or not right_choice:
                continue

            queries[pair_id] = question

            right_doc_id = f"{pair_id}_right"
            corpus[right_doc_id] = {"text": right_choice}
            relevant_docs[pair_id] = {right_doc_id: 1}

            for j, wrong in enumerate(wrong_choices):
                if not wrong:
                    continue
                wrong_doc_id = f"{pair_id}_wrong_{j}"
                corpus[wrong_doc_id] = {"text": wrong}
            continue

        if isinstance(sample, dict) and "query" in sample and "doc" in sample:
            query = _normalize_text(sample.get("query"))
            doc = _normalize_text(sample.get("doc"))
            if not query or not doc:
                continue

            query_id = _normalize_text(sample.get("id")) or pair_id
            if query_id in queries:
                query_id = f"{query_id}_{idx}"
            queries[query_id] = query

            doc_id = new_format_doc_ids.get(doc)
            if doc_id is None:
                base_doc_id = f"{query_id}_doc"
                doc_id = base_doc_id
                suffix = 1
                while doc_id in corpus:
                    doc_id = f"{base_doc_id}_{suffix}"
                    suffix += 1
                corpus[doc_id] = {"text": doc}
                new_format_doc_ids[doc] = doc_id

            relevant_docs[query_id] = {doc_id: 1}
            continue

        if not hasattr(sample, "texts") or len(sample.texts) < 2:
            continue
        queries[pair_id] = sample.texts[0]
        doc_id = f"{pair_id}_pos"
        corpus[doc_id] = {"text": sample.texts[1]}
        relevant_docs[pair_id] = {doc_id: 1}

    return corpus, queries, relevant_docs


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    set_seed(42)
    output_file = build_output_file(
        data_args.output, data_args.dataset_file_path, data_args.model_name
    )

    if os.path.exists(output_file):
        logger.info("Results already exist at %s, skipping...", output_file)
        return

    torch_dtype = torch.float16
    attn_implementation = "sdpa"
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.base_model_name_or_path,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        extra_model_name_or_path=model_args.extra_model_name_or_path,
        enable_bidirectional=model_args.enable_bidirectional,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_length,
        res_mlp_hidden_dim=model_args.res_mlp_hidden_dim,
        res_mlp_num_layers=model_args.res_mlp_num_layers,
        res_mlp_dropout=model_args.res_mlp_dropout,
        res_mlp_gamma_init=model_args.res_mlp_gamma_init,
        res_mlp_gamma_learnable=model_args.res_mlp_gamma_learnable,
        res_mlp_output_normalize=model_args.res_mlp_output_normalize,
        res_mlp_output_layernorm=model_args.res_mlp_output_layernorm,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )

    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if compute_device.type == "cuda":
        model.to(compute_device)
    else:
        if attn_implementation == "flash_attention_2":
            logging.warning(
                "CPU detected: downgrading attn_implementation from flash_attention_2 to sdpa to avoid runtime errors."
            )
            try:
                if hasattr(model, "model") and hasattr(model.model, "config"):
                    model.model.config._attn_implementation = "sdpa"
                if hasattr(model, "config"):
                    model.config._attn_implementation = "sdpa"
            except Exception:
                pass
        try:
            model.to(torch.float32)
        except Exception:
            pass

    dataset = load_jsonl(data_args.dataset_file_path)
    corpus, queries, relevant_docs = build_corpus_queries(dataset)
    if not queries or not corpus or not relevant_docs:
        raise ValueError(
            "No valid retrieval samples were built from the dataset. "
            "Supported formats are "
            "{'question', 'right_choice', 'wrong_choices'} and {'query', 'doc'}."
        )

    logger.info(
        "Built retrieval dataset with %d queries, %d corpus documents, and %d relevance mappings",
        len(queries),
        len(corpus),
        len(relevant_docs),
    )

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

    instruction = data_args.instruction if data_args.instruction else ""
    logger.info("Using query instruction: %r", instruction)

    query_pairs = [[instruction, text] for text in query_texts]
    corpus_pairs = [["", text] for text in corpus_texts]

    logger.info("Encoding %d queries using encode function", len(query_pairs))
    q_emb = model.encode(
        query_pairs,
        batch_size=data_args.batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=compute_device,
    )

    logger.info("Encoding %d documents using encode function", len(corpus_pairs))
    d_emb = model.encode(
        corpus_pairs,
        batch_size=data_args.batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=compute_device,
    )

    logger.info("Computing cosine similarity")
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
            doc_id = corpus_ids[idx]
            score = top_vals[i][rank]
            results[qid][doc_id] = score

    retriever = EvaluateRetrieval(model, score_function="cos_sim")
    eval_k = max(1, min(10, len(corpus_ids)))
    ndcg, _map, recall, precision = retriever.evaluate(
        relevant_docs,
        results,
        [eval_k],
        ignore_identical_ids=False,
    )

    ndcg_key = f"NDCG@{eval_k}"
    recall_key = f"Recall@{eval_k}"
    metrics = {
        "NDCG@10": ndcg[ndcg_key],
        "Recall@10": recall[recall_key],
    }
    logger.info(json.dumps(metrics, indent=4))
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
