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
    LlamaConfig,
    MistralConfig,
    GemmaConfig,
    Qwen2Config,
)
from tqdm import tqdm

from llm2vec import LLM2Vec
from llm2vec.dataset.utils import load_dataset

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Ensure Accelerate logging is initialized before dataset logs
try:
    from accelerate import PartialState
    PartialState()
except Exception:
    pass

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained base model"})
    peft_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Path to PEFT adapter"})
    bidirectional: bool = field(default=True)
    max_seq_length: int = field(default=512)
    torch_dtype: str = field(default="bfloat16", metadata={"choices": ["auto", "bfloat16", "float16", "float32"]})
    attn_implementation: str = field(default="flash_attention_2", metadata={"choices": ["eager", "sdpa", "flash_attention_2"]})
    pooling_mode: str = field(default="mean", metadata={"choices": ["mean", "weighted_mean", "eos_token"]})
    extra_model_name_or_path: Optional[List[str]] = field(default_factory=list, metadata={"help": "Path to extra model"})

@dataclass
class DataArguments:
    output_dir: str = field(metadata={"help": "Output directory"})
    dataset_name: str = field(default="DermVariants", metadata={"help": "Dataset name"})
    dataset_file_path: Optional[str] = field(default=None, metadata={"help": "Dataset root path"})
    split: str = field(default="validation", metadata={"help": "Dataset split"})
    batch_size: int = field(default=16)
    top_k: int = field(default=10)
    seed: int = field(default=42)
    dermqa_upsample_ratio: int=field(default=1)

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


def prepare_for_tokenization(model, text, pooling_mode="mean"):
    config_name = getattr(model.config, "_name_or_path", None)
    if config_name == "meta-llama/Meta-Llama-3-8B-Instruct" or isinstance(model.config, LlamaConfig):
        text = "<|start_header_id|>user<|end_header_id|>\n\n" + text.strip() + "<|eot_id|>"
        return text
    if config_name in [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
    ]:
        text = "[INST] " + text.strip() + " [/INST]"
    if config_name in [
        "google/gemma-2-9b-it",
    ]:
        text = "<bos><start_of_turn>user\n" + text.strip() + "<end_of_turn>"
    if config_name in [
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen3-8B-Embedding",
    ] or isinstance(model.config, Qwen2Config):
        text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"
    if pooling_mode == "eos_token":
        if config_name == "meta-llama/Meta-Llama-3-8B":
            text = text.strip() + "<|end_of_text|>"
        elif isinstance(model.config, LlamaConfig) or isinstance(model.config, MistralConfig):
            text = text.strip() + " </s>"
        elif isinstance(model.config, GemmaConfig):
            text = text.strip() + "<eos>"
        elif isinstance(model.config, Qwen2Config):
            text = text.strip() + "<|endoftext|>"
    return text

def encode_with_separator_batches(
    model: LLM2Vec,
    texts: List[str],
    batch_size: int,
    device: Optional[torch.device] = None,
    desc: Optional[str] = None,
) -> torch.Tensor:
    device = (
        device
        if device is not None
        else (model._infer_device() if hasattr(model, "_infer_device") else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    )
    all_embeddings = []
    iterator = range(0, len(texts), batch_size)
    if desc is not None:
        total_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = tqdm(iterator, desc=desc, total=total_batches)
    for i in iterator:
        batch = texts[i : i + batch_size]
        embs = model.encode_with_separator(batch, device=device)
        all_embeddings.append(embs.cpu())
    return torch.cat(all_embeddings, dim=0)


def build_corpus_queries(dataset) -> (Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]):
    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    relevant_docs: Dict[str, Dict[str, int]] = {}

    for idx, sample in enumerate(dataset):
        if not hasattr(sample, "texts") or len(sample.texts) < 2:
            continue
        pair_id = str(idx)
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

    set_seed(data_args.seed)
    os.makedirs(data_args.output_dir, exist_ok=True)

    torch_dtype = model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        extra_model_name_or_path=model_args.extra_model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    # Decide compute device and ensure model/device alignment
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if compute_device.type == "cuda":
        model.to(compute_device)
    else:
        # On CPU, flash-attn is unsupported; downgrade attention impl to sdpa if needed
        if getattr(model_args, "attn_implementation", "sdpa") == "flash_attention_2":
            logging.warning("CPU detected: downgrading attn_implementation from flash_attention_2 to sdpa to avoid runtime errors.")
            try:
                if hasattr(model, "model") and hasattr(model.model, "config"):
                    model.model.config._attn_implementation = "sdpa"
                if hasattr(model, "config"):
                    model.config._attn_implementation = "sdpa"
            except Exception:
                pass
        # Prefer float32 on CPU for stability
        try:
            model.to(torch.float32)
        except Exception:
            pass

    dataset = load_dataset(
        data_args.dataset_name,
        split=data_args.split,
        file_path=data_args.dataset_file_path,
        effective_batch_size=data_args.batch_size,

    )
    corpus, queries, relevant_docs = build_corpus_queries(dataset)

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

    # Apply model-specific prompt formatting before encoding
    query_texts = [
        prepare_for_tokenization(model, text, pooling_mode=model.pooling_mode)
        for text in query_texts
    ]
    corpus_texts = [
        prepare_for_tokenization(model, text, pooling_mode=model.pooling_mode)
        for text in corpus_texts
    ]

    logger.info(f"Encoding {len(query_texts)} queries")
    q_emb = encode_with_separator_batches(
        model, query_texts, data_args.batch_size, device=compute_device, desc="Encoding queries"
    )
    logger.info(f"Encoding {len(corpus_texts)} documents")
    d_emb = encode_with_separator_batches(
        model, corpus_texts, data_args.batch_size, device=compute_device, desc="Encoding documents"
    )

    logger.info("Computing cosine similarity")
    scores = cos_sim(q_emb, d_emb)
    scores[torch.isnan(scores)] = -1
    top_k = min(data_args.top_k, len(corpus_ids))
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
    default_k_values = [1, 3, 5, 10, 100, 1000]
    k_values = [k for k in default_k_values if k <= data_args.top_k]
    if not k_values:
        k_values = [data_args.top_k]
    ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, k_values, ignore_identical_ids=False)

    metrics = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
    }
    logger.info(json.dumps(metrics, indent=4))
    with open(os.path.join(data_args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
