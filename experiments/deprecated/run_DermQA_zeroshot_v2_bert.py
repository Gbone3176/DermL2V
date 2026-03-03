import json
import logging
import os
import sys
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from beir.retrieval.evaluation import EvaluateRetrieval
from transformers import HfArgumentParser, set_seed, AutoTokenizer, AutoModel

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained BERT model"})
    max_seq_length: int = field(default=512)
    torch_dtype: str = field(default="float32", metadata={"choices": ["auto", "bfloat16", "float16", "float32"]})
    pooling_mode: str = field(default="mean", metadata={"choices": ["mean", "cls"]})

@dataclass
class DataArguments:
    data_file: str = field(metadata={"help": "CSV file with columns: prompt,response"})
    output_dir: str = field(metadata={"help": "Output directory"})
    batch_size: int = field(default=32)
    top_k: int = field(default=10)
    separator: str = field(default="!@#$%^&*()")
    instruction: str = field(default="Given a question related to skin diseases, retrieve the most relevant and appropriate answer to that question")
    seed: int = field(default=42)

class QACsvDataset:
    def __init__(self, file_path: str, separator: str):
        self.separator = separator
        self.rows = self._read_csv(file_path)
    def _read_csv(self, file_path: str) -> List[Dict[str, str]]:
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if "prompt" in r and "response" in r:
                    rows.append({"prompt": r["prompt"], "response": r["response"]})
        return rows

class BERTModel:
    def __init__(self, model_path, max_length=512, pooling="mean", device=None, dtype="float32"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.max_length = max_length
        self.pooling = pooling
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if dtype == "float16":
            self.model.half()
        elif dtype == "bfloat16":
            self.model.to(torch.bfloat16)
        self.model.eval()
    def encode(self, sentences: List[str], batch_size=32):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                if self.pooling == "mean":
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                elif self.pooling == "cls":
                    embeddings = outputs.last_hidden_state[:, 0, :]
                else:
                    raise ValueError(f"Unknown pooling mode: {self.pooling}")
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

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

def load_dataset(file_path: str, separator: str, instruction: str):
    dataset = QACsvDataset(file_path=file_path, separator=separator)
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
    dense_model = BERTModel(
        model_path=model_args.model_name_or_path,
        max_length=model_args.max_seq_length,
        pooling=model_args.pooling_mode,
        dtype=model_args.torch_dtype
    )
    corpus, queries, relevant_docs = load_dataset(data_args.data_file, data_args.separator, data_args.instruction)
    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]
    logger.info(f"Encoding {len(query_texts)} queries")
    q_emb = dense_model.encode(query_texts, batch_size=data_args.batch_size)
    logger.info(f"Encoding {len(corpus_texts)} documents")
    d_emb = dense_model.encode(corpus_texts, batch_size=data_args.batch_size)
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
    retriever = EvaluateRetrieval(dense_model, score_function="cos_sim")
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
