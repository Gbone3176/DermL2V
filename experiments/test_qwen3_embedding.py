import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer
from transformers import HfArgumentParser, set_seed

from llm2vec.dataset.utils import load_dataset

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained Qwen3 embedding model"})
    attn_implementation: str = field(
        default="sdpa", metadata={"choices": ["eager", "sdpa", "flash_attention_2"]}
    )


@dataclass
class DataArguments:
    output_dir: str = field(metadata={"help": "Directory to store metrics and artifacts"})
    dataset_name: str = field(default="DermVariants", metadata={"help": "Dataset name registered in load_dataset"})
    dataset_file_path: Optional[str] = field(default=None, metadata={"help": "Dataset root or json file"})
    split: str = field(default="validation", metadata={"help": "Dataset split: train/validation/test"})
    batch_size: int = field(default=32)
    top_k: int = field(default=10)
    seed: int = field(default=42)
    dermqa_upsample_ratio: int = field(default=1)


def build_corpus_queries(dataset) -> (Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]):
    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    relevant_docs: Dict[str, Dict[str, int]] = {}

    for idx, sample in enumerate(dataset):
        if not hasattr(sample, "texts") or len(sample.texts) < 2:
            continue
        pair_id = str(idx)
        queries[pair_id] = sample.texts[0].split("!@#$%^&*()")[-1]
        doc_id = f"{pair_id}_pos"
        corpus[doc_id] = {"text": sample.texts[1].split("!@#$%^&*()")[-1]}
        relevant_docs[pair_id] = {doc_id: 1}

    return corpus, queries, relevant_docs


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    prompt_name: Optional[str] = None,
) -> torch.Tensor:
    encode_kwargs = {
        "sentences": texts,
        "batch_size": batch_size,
        "convert_to_tensor": True,
    }
    if prompt_name is not None:
        encode_kwargs["prompt_name"] = prompt_name
    return model.encode(**encode_kwargs)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    set_seed(data_args.seed)
    os.makedirs(data_args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"attn_implementation": model_args.attn_implementation}
    model = SentenceTransformer(
        model_args.model_name_or_path,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
        device=device,
    )

    dataset = load_dataset(
        data_args.dataset_name,
        split=data_args.split,
        file_path=data_args.dataset_file_path,
        effective_batch_size=data_args.batch_size,
        dermqa_upsample_ratio=data_args.dermqa_upsample_ratio,
    )

    corpus, queries, relevant_docs = build_corpus_queries(dataset)
    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

    logger.info(f"Encoding {len(query_texts)} queries with Qwen3 embedding model")
    q_emb = encode_texts(model, query_texts, batch_size=data_args.batch_size, prompt_name="query").to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Encoding {len(corpus_texts)} documents")
    d_emb = encode_texts(model, corpus_texts, batch_size=data_args.batch_size).to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Computing cosine similarity")
    scores = torch.nn.functional.normalize(q_emb, p=2, dim=1) @ torch.nn.functional.normalize(d_emb, p=2, dim=1).t()
    scores[torch.isnan(scores)] = -1
    top_k = min(data_args.top_k, len(corpus_ids))
    top_vals, top_idx = torch.topk(scores, top_k, dim=1, largest=True, sorted=True)

    results = {}
    for i, qid in enumerate(query_ids):
        results[qid] = {}
        for rank, idx in enumerate(top_idx[i].tolist()):
            doc_id = corpus_ids[idx]
            results[qid][doc_id] = top_vals[i][rank].item()

    retriever = EvaluateRetrieval(None, score_function="cos_sim")
    default_k_values = [1, 3, 5, 10, 100, 1000]
    k_values = [k for k in default_k_values if k <= data_args.top_k] or [data_args.top_k]
    ndcg, _map, recall, precision = retriever.evaluate(
        relevant_docs, results, k_values, ignore_identical_ids=False
    )

    metrics = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
    }
    logger.info(json.dumps(metrics, indent=4))

    metrics_path = os.path.join(data_args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
