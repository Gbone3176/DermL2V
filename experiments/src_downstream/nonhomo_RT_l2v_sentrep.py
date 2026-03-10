"""
Non-homogeneous Retrieval Test with Sentence Repetition (SentRep).

Based on nonhomo_RT_l2v_inst.py, this script embeds the sentence repetition trick:
each text is repeated r times (appended to itself), i.e. encoded as:

    "<text> <text> ... <text>"  (r+1 copies total)

With r=1 (default), each text appears twice, which may improve embedding quality
without any additional training (zero-shot).

Reference idea: Echo Embeddings / PromptEOL — appending the source text after the
instruction prompt so that the last (or mean) token representation has richer
access to the content.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from transformers import (
    HfArgumentParser,
    set_seed,
)

from llm2vec import LLM2Vec

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

try:
    from accelerate import PartialState
    PartialState()
except Exception:
    pass


@dataclass
class ModelArguments:
    base_model_name_or_path: str = field(metadata={"help": "Path to pretrained base model"})
    peft_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Path to PEFT adapter"})
    extra_model_name_or_path: Optional[List[str]] = field(default_factory=list, metadata={"help": "Path to extra model"})
    enable_bidirectional: bool = field(default=True)
    max_length: int = field(default=512)
    pooling_mode: str = field(default="mean", metadata={"choices": ["mean", "weighted_mean", "eos_token", "latent_pooling"]})


@dataclass
class DataArguments:
    dataset_file_path: str = field(metadata={"help": "Path to jsonl dataset file"})
    model_name: str = field(metadata={"help": "Name for output metrics file"})
    output: str = field(metadata={"help": "Output directory"})
    batch_size: int = field(default=16)
    instruction: Optional[str] = field(
        default=None,
        metadata={"help": "Instruction to prepend to all texts (queries and documents)"}
    )
    repeat_times: int = field(
        default=1,
        metadata={
            "help": (
                "Sentence repetition count r. "
                "Each text is encoded as the same text repeated (r+1) times, "
                "separated by a single space. "
                "r=0 means no repetition (original text); r=1 means text appears twice."
            )
        }
    )
    apply_repeat_to: str = field(
        default="both",
        metadata={
            "help": "Which side to apply sentence repetition to: 'query', 'doc', or 'both'.",
            "choices": ["query", "doc", "both"],
        }
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


def repeat_text(text: str, r: int) -> str:
    """Return text repeated (r+1) times, joined by a single space.
    r=0 → original text unchanged.
    r=1 → "text text"
    r=2 → "text text text"
    """
    if r <= 0:
        return text
    return " ".join([text] * (r + 1))


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
                raise ValueError(f"Invalid JSON on line {line_no} in {file_path}: {e}") from e
    return data


def build_corpus_queries(dataset) -> (Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]):
    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    relevant_docs: Dict[str, Dict[str, int]] = {}

    for idx, sample in enumerate(dataset):
        pair_id = str(idx)

        if isinstance(sample, dict) and "question" in sample:
            question = sample.get("question")
            right_choice = sample.get("right_choice")
            wrong_choices = sample.get("wrong_choices") or []
            if isinstance(wrong_choices, str):
                wrong_choices = [wrong_choices]
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
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    set_seed(42)
    os.makedirs(data_args.output, exist_ok=True)

    # Check if results already exist
    output_file = os.path.join(data_args.output, f"RT_{data_args.model_name}.json")
    if os.path.exists(output_file):
        logger.info(f"Results already exist at {output_file}, skipping...")
        return

    logger.info(
        f"Sentence repetition config: repeat_times={data_args.repeat_times}, "
        f"apply_repeat_to='{data_args.apply_repeat_to}'"
    )

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
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )

    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if compute_device.type == "cuda":
        model.to(compute_device)
    else:
        if attn_implementation == "flash_attention_2":
            logging.warning("CPU detected: downgrading attn_implementation from flash_attention_2 to sdpa.")
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

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

    instruction = data_args.instruction if data_args.instruction else ""
    r = data_args.repeat_times
    apply_to = data_args.apply_repeat_to

    logger.info(f"Using instruction: '{instruction}'")

    # Apply sentence repetition selectively
    if apply_to in ("query", "both"):
        query_texts_final = [repeat_text(t, r) for t in query_texts]
        logger.info(f"Applied sentence repetition (r={r}) to queries. Example: '{query_texts_final[0][:120]}'")
    else:
        query_texts_final = query_texts

    if apply_to in ("doc", "both"):
        corpus_texts_final = [repeat_text(t, r) for t in corpus_texts]
        logger.info(f"Applied sentence repetition (r={r}) to documents. Example: '{corpus_texts_final[0][:120]}'")
    else:
        corpus_texts_final = corpus_texts

    query_pairs = [[instruction, t] for t in query_texts_final]
    corpus_pairs = [[instruction, t] for t in corpus_texts_final]

    logger.info(f"Encoding {len(query_pairs)} queries")
    q_emb = model.encode(
        query_pairs,
        batch_size=data_args.batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=compute_device,
    )

    logger.info(f"Encoding {len(corpus_pairs)} documents")
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
    default_k_values = [1, 3, 5, 10, 100]
    k_values = [k for k in default_k_values if k <= top_k]
    if not k_values:
        k_values = [top_k]
    ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, k_values, ignore_identical_ids=False)

    metrics = {
        "repeat_times": r,
        "apply_repeat_to": apply_to,
        "pooling_mode": model_args.pooling_mode,
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
    }
    logger.info(json.dumps(metrics, indent=4))
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    main()
