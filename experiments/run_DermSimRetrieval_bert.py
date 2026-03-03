
import json
import logging
import os
import sys
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
from accelerate.logging import get_logger
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from transformers import HfArgumentParser, set_seed, AutoTokenizer, AutoModel
import torch.nn.functional as F

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    torch_dtype: str = field(
        default="float32", # BERT usually uses float32
        metadata={"help": "Torch dtype to use.", "choices": ["auto", "bfloat16", "float16", "float32"]},
    )
    pooling_mode: str = field(
        default="mean",
        metadata={"help": "Pooling mode to use.", "choices": ["mean", "cls"]},
    )

@dataclass
class DataArguments:
    data_file: str = field(
        metadata={"help": "Path to the json data file containing {'id':..., 'sentence1':..., 'sentence2':...} records."}
    )
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    batch_size: int = field(
        default=32, metadata={"help": "Batch size for encoding."}
    )
    top_k: int = field(
        default=10, metadata={"help": "Top K documents to retrieve."}
    )

def load_dataset(file_path: str):
    logger.info(f"Loading dataset from {file_path}")
    corpus = {}
    queries = {}
    relevant_docs = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]

    for item in data:
        # Use a consistent unique ID for each pair
        pair_id = str(item.get('id', hash(item.get('sentence1') + item.get('sentence2'))))
        
        qid = pair_id
        doc_id = pair_id

        queries[qid] = item.get('sentence1', '')
        corpus[doc_id] = {'text': item.get('sentence2', '')}
        relevant_docs[qid] = {doc_id: 1} # binary relevance

    return corpus, queries, relevant_docs

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

    def encode(self, sentences, batch_size=32, **kwargs):
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
                    # Mean pooling with attention mask
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
                
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())
                
        return torch.cat(all_embeddings, dim=0)

    def encode_queries(self, queries, batch_size, **kwargs):
        return self.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus, batch_size, **kwargs):
        sentences = [
            doc.get("text", "").strip()
            for doc in corpus
        ]
        return self.encode(sentences, batch_size=batch_size, **kwargs)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    os.makedirs(data_args.output_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading BERT model from {model_args.model_name_or_path}")
    dense_model = BERTModel(
        model_path=model_args.model_name_or_path,
        max_length=model_args.max_seq_length,
        pooling=model_args.pooling_mode,
        dtype=model_args.torch_dtype
    )
    # Filter k_values based on top_k
    default_k_values = [1, 3, 5, 10, 100, 1000]
    k_values = [k for k in default_k_values if k <= data_args.top_k]
    if not k_values:
        k_values = [data_args.top_k]
    # Use DRES for exact search
    model = DRES(dense_model, batch_size=data_args.batch_size)
    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    # Load data
    corpus, queries, relevant_docs = load_dataset(data_args.data_file)

    # Retrieve
    logger.info("Retrieving...")
    results = retriever.retrieve(corpus, queries)

    # Evaluate
    logger.info("Evaluating...")
    
    ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, k_values, ignore_identical_ids=False)
    
    # Save results
    metrics = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
    }
    
    logger.info("Results:")
    logger.info(json.dumps(metrics, indent=4))
    
    with open(os.path.join(data_args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Print top-k examples
    logger.info("Printing top-5 examples...")
    top_k_print = 5
    if len(results) > 0:
        query_id = list(results.keys())[0]
        scores_sorted = sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)
        logger.info(f"Query ({query_id}): {queries[query_id]}\n")
        
        for rank in range(min(top_k_print, len(scores_sorted))):
            doc_id = scores_sorted[rank][0]
            score = scores_sorted[rank][1]
            logger.info(f"Rank {rank + 1} (Score: {score:.4f}): {doc_id} - {corpus[doc_id].get('text')[:200]}...\n")

if __name__ == "__main__":
    main()
