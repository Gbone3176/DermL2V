import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from transformers import HfArgumentParser, set_seed
from llm2vec import LLM2Vec
# from llm2vec.llm2vec_wrapper import LLM2VecWrapper as LLM2Vec

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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
    data_file: str = field(metadata={"help": "dataset file with columns: prompt,response"})
    output_dir: str = field(metadata={"help": "Output directory"})
    batch_size: int = field(default=16)
    top_k: int = field(default=10)
    separator: str = field(default="!@#$%^&*()")
    instruction: str = field(default="Given a description of a dermatological condition, retrieve the description with the highest semantic relevance: ")
    seed: int = field(default=42)

def prepare_for_tokenization(model, text, pooling_mode="mean"):
    from transformers import LlamaConfig, MistralConfig, GemmaConfig
    # if getattr(model.config, "_name_or_path", None) == "meta-llama/Meta-Llama-3-8B-Instruct":
    if getattr(model.config, "_name_or_path", None) == "meta-llama/Meta-Llama-3-8B-Instruct" or isinstance(model.config, LlamaConfig):
        text = "<|start_header_id|>user<|end_header_id|>\n\n" + text.strip() + "<|eot_id|>"
        return text
    if pooling_mode == "eos_token":
        if getattr(model.config, "_name_or_path", None) == "meta-llama/Meta-Llama-3-8B":
            text = text.strip() + "<|end_of_text|>"
        elif isinstance(model.config, LlamaConfig) or isinstance(model.config, MistralConfig):
            text = text.strip() + " </s>"
        elif isinstance(model.config, GemmaConfig):
            text = text.strip() + "<eos>"
    return text

class SimVarDataset:
    def __init__(self, file_path: str, separator: str):
        self.separator = separator
        self.rows = self._read_jsonl(file_path)

    def _read_jsonl(self, file_path: str) -> List[Dict[str, str]]:
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "sentence1" not in record or "sentence2" not in record:
                    continue
                prompt = record.get("sentence1")
                variants = record.get("sentence2")
                if prompt is None or variants is None:
                    continue
                # Normalize variants into a single response string
                if isinstance(variants, list):
                    response = self.separator.join([v for v in variants if isinstance(v, str)])
                else:
                    response = str(variants)
                if response:
                    rows.append({"prompt": str(prompt), "response": response})
        return rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

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

def append_instruction(instruction: str, sentences: List[str]):
    new_sentences = []
    for s in sentences:
        new_sentences.append([instruction, s, 0])
    return new_sentences

def encode_texts(model: LLM2Vec, texts: List[str], instruction: str, batch_size: int):
    # texts = [prepare_for_tokenization(model, t, pooling_mode=model.pooling_mode) for t in texts]
    inputs = append_instruction(instruction, texts)
    return model.encode(inputs, batch_size=batch_size, convert_to_tensor=True)

def load_dataset(file_path: str, separator: str, instruction: str):
    dataset = SimVarDataset(file_path=file_path, separator=separator)
    corpus = {}
    queries = {}
    relevant_docs = {}
    for idx, row in enumerate(dataset.rows):
        pair_id = str(idx)
        qid = pair_id
        doc_id = pair_id
        # queries[qid] = instruction + separator + row["prompt"]
        # corpus[doc_id] = {"text": instruction + separator + row["response"]}
        queries[qid] = row["prompt"]
        corpus[doc_id] = {"text": row["response"]}
        relevant_docs[qid] = {doc_id: 1}
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

    corpus, queries, relevant_docs = load_dataset(data_args.data_file, data_args.separator, data_args.instruction)

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

    logger.info(f"Encoding {len(query_texts)} queries")
    q_emb = encode_texts(model, query_texts, data_args.instruction, data_args.batch_size)
    logger.info(f"Encoding {len(corpus_texts)} documents")
    d_emb = encode_texts(model, corpus_texts, data_args.instruction, data_args.batch_size)

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
