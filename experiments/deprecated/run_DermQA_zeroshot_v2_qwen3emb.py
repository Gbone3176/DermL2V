import json
import logging
import os
import sys
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from transformers import HfArgumentParser, set_seed
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained Qwen3-Embedding model"})
    attn_implementation: str = field(
        default="sdpa", metadata={"choices": ["eager", "sdpa", "flash_attention_2"]}
    )


@dataclass
class DataArguments:
    data_file: str = field(metadata={"help": "CSV file with columns: prompt,response"})
    output_dir: str = field(metadata={"help": "Output directory"})
    batch_size: int = field(default=16)
    top_k: int = field(default=10)
    seed: int = field(default=42)

class QACsvDataset:
    def __init__(self, file_path: str):
        self.rows = self._read_csv(file_path)
    def _read_csv(self, file_path: str) -> List[Dict[str, str]]:
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if "prompt" in r and "response" in r:
                    rows.append({"prompt": r["prompt"], "response": r["response"]})
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
    a = a.to("cpu")
    b = b.to("cpu")
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def encode_texts(model: SentenceTransformer, texts: List[str], is_query: bool, batch_size: int):
    if is_query:
        return model.encode(
            texts,
            prompt_name="query",
            batch_size=batch_size,
            convert_to_tensor=True,
        )
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
    )

def load_dataset(file_path: str):
    dataset = QACsvDataset(file_path=file_path)
    corpus = {}
    queries = {}
    relevant_docs = {}
    for idx, row in enumerate(dataset.rows):
        pair_id = str(idx)
        qid = pair_id
        doc_id = pair_id
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"attn_implementation": model_args.attn_implementation}

    model = SentenceTransformer(
        model_args.model_name_or_path,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
        device=device,
    )

    corpus, queries, relevant_docs = load_dataset(data_args.data_file)

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

    query_emb_path = os.path.join(data_args.output_dir, "query_embeddings.pt")
    corpus_emb_path = os.path.join(data_args.output_dir, "corpus_embeddings.pt")

    logger.info(f"Encoding {len(query_texts)} queries")
    q_emb = encode_texts(model, query_texts, is_query=True, batch_size=data_args.batch_size)
    q_emb = q_emb.to("cpu")
    torch.save({"ids": query_ids, "embeddings": q_emb}, query_emb_path)
    del q_emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Encoding {len(corpus_texts)} documents")
    d_emb = encode_texts(model, corpus_texts, is_query=False, batch_size=data_args.batch_size)
    d_emb = d_emb.to("cpu")
    torch.save({"ids": corpus_ids, "embeddings": d_emb}, corpus_emb_path)
    del d_emb
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    q_data = torch.load(query_emb_path, map_location="cpu")
    d_data = torch.load(corpus_emb_path, map_location="cpu")
    query_ids = q_data["ids"]
    corpus_ids = d_data["ids"]
    q_emb = q_data["embeddings"]
    d_emb = d_data["embeddings"]

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

    retriever = EvaluateRetrieval(None, score_function="cos_sim")
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
