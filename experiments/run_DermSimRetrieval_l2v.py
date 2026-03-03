
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate.logging import get_logger
from beir.retrieval.evaluation import EvaluateRetrieval
from peft import PeftModel
from transformers import HfArgumentParser, set_seed
from llm2vec import LLM2Vec

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
    peft_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the PEFT model."}
    )
    bidirectional: bool = field(
        default=True, metadata={"help": "Whether to use bidirectional attention."}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Torch dtype to use.", "choices": ["auto", "bfloat16", "float16", "float32"]},
    )
    pooling_mode: str = field(
        default="mean",
        metadata={"help": "Pooling mode to use.", "choices": ["mean", "weighted_mean", "eos_token"]},
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
        default=8, metadata={"help": "Batch size for encoding."}
    )
    top_k: int = field(
        default=10, metadata={"help": "Top K documents to retrieve."}
    )
    instruction: str = field(
        default="", metadata={"help": "Instruction to prepend to queries."}
    )

def prepare_for_tokenization(model, text, pooling_mode="mean"):
    from transformers import LlamaConfig, MistralConfig, GemmaConfig, Qwen2Config
    
    # Add instruction formatting for specific models if needed, similar to run_DermQA.py
    # Here we keep it simple or align with run_DermQA.py logic if necessary.
    # For now, we assume the instruction is handled by the append_instruction logic below
    # but we might need EOS tokens for some models.
    
    if pooling_mode == "eos_token":
        if getattr(model.config, "_name_or_path", None) == "meta-llama/Meta-Llama-3-8B":
            text = text.strip() + "<|end_of_text|>"
        elif isinstance(model.config, LlamaConfig) or isinstance(model.config, MistralConfig):
            text = text.strip() + " </s>"
        elif isinstance(model.config, GemmaConfig):
            text = text.strip() + "<eos>"
    return text

def append_instruction(instruction, sentences):
    # LLM2Vec expects [instruction, sentence, is_query] format for encoding if using instruction
    # If instruction is empty string, we can still use this format.
    # is_query=1 for query (sentence1), is_query=0 for document (sentence2) usually, 
    # but LLM2Vec.encode wrapper handles this if we pass list of strings or list of lists.
    # Let's follow the pattern in llm2vec examples.
    new_sentences = []
    for s in sentences:
        new_sentences.append([instruction, s, 0])
    return new_sentences

def load_dataset(file_path: str):
    logger.info(f"Loading dataset from {file_path}")
    corpus = {}
    queries = {}
    relevant_docs = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Check if it's a list of jsons or line-delimited json
        try:
            data = json.load(f)
            if not isinstance(data, list):
                # If it's a single dict (unlikely for a dataset) or other structure
                data = [data]
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]

    for item in data:
        # Assuming sentence1 is query, sentence2 is relevant document
        # And we assume unique IDs or we generate them. 
        # If 'id' is present, use it for query ID.
        # We need unique document IDs. Since this is a pair dataset, 
        # each sentence2 is a document. We can use id_doc as doc ID.
        
        # Use a consistent unique ID for each pair
        # Use the 'id' field if available, otherwise generate one
        pair_id = str(item.get('id', hash(item.get('sentence1') + item.get('sentence2'))))
        
        # We can now use the same ID because we set ignore_identical_ids=False in evaluate()
        qid = pair_id
        doc_id = pair_id

        queries[qid] = item.get('sentence1', '')
        corpus[doc_id] = {'text': item.get('sentence2', ''), 'title': ''}
        relevant_docs[qid] = {doc_id: 1} # binary relevance

    return corpus, queries, relevant_docs

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

def encode_queries(model, queries: List[str], batch_size: int, instruction: str, **kwargs):
    # Prepare queries with instruction
    new_sentences = append_instruction(instruction, queries)
    kwargs["show_progress_bar"] = True
    return model.encode(new_sentences, batch_size=batch_size, **kwargs)

def encode_corpus(model, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
    # Prepare corpus (no instruction usually, or empty)
    sentences = [
        (doc["title"] + " " + doc["text"]).strip() if "title" in doc else doc["text"].strip()
        for doc in corpus
    ]
    new_sentences = append_instruction("", sentences)
    kwargs["show_progress_bar"] = True
    return model.encode(new_sentences, batch_size=batch_size, **kwargs)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    os.makedirs(data_args.output_dir, exist_ok=True)
    
    # Load model
    # Initialize logger properly by just using standard logging if not using Accelerator explicitly for training loop,
    # or just use print for simple scripts. But here we can use standard logging to avoid Accelerator init requirement if not needed.
    # Or simply replace get_logger with logging.getLogger
    logger = logging.getLogger(__name__)
    logger.info("Loading model...")
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    
    model = LLM2Vec.from_pretrained(
        model_args.model_name_or_path,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    
    # Load data
    corpus, queries, relevant_docs = load_dataset(data_args.data_file)
    
    # Prepare lists for encoding
    query_ids = list(queries.keys())
    query_list = [queries[qid] for qid in query_ids]
    
    corpus_ids = list(corpus.keys())
    corpus_list = [corpus[cid] for cid in corpus_ids]

    logger.info(f"Encoding {len(query_list)} queries...")
    query_embeddings = encode_queries(
        model, query_list, batch_size=data_args.batch_size, instruction=data_args.instruction, convert_to_tensor=True
    )
    
    logger.info(f"Encoding {len(corpus_list)} documents...")
    corpus_embeddings = encode_corpus(
        model, corpus_list, batch_size=data_args.batch_size, convert_to_tensor=True
    )
    
    logger.info("Computing cosine similarity...")
    cos_scores = cos_sim(query_embeddings, corpus_embeddings)
    cos_scores[torch.isnan(cos_scores)] = -1

    # Get top-k values
    logger.info(f"Retrieving top-{data_args.top_k} results...")
    top_k = min(data_args.top_k, len(corpus_list))
    cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
        cos_scores, top_k, dim=1, largest=True, sorted=True
    )
    
    cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
    cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
    
    results = {}
    for i, qid in enumerate(query_ids):
        results[qid] = {}
        for rank, idx in enumerate(cos_scores_top_k_idx[i]):
            doc_id = corpus_ids[idx]
            # BEIR evaluation typically excludes the query itself if it appears in the corpus,
            # but here query and doc are distinct textually (sentence1 vs sentence2), even if they share ID.
            # However, if qid == doc_id, EvaluateRetrieval might ignore it depending on implementation?
            # Actually, standard BEIR logic: if query_id == doc_id, it is NOT ignored by default unless explicitly filtered.
            # BUT, in our code (copied from example), there was:
            # if corpus_id != query_id: results[query_id][corpus_id] = score
            # Since we set qid == doc_id for the correct pair, we MUST NOT filter it out!
            
            score = cos_scores_top_k_values[i][rank]
            results[qid][doc_id] = score

    # Evaluate using BEIR
    logger.info("Evaluating with BEIR...")
    # Filter k_values based on top_k
    # Standard BEIR k_values are [1, 3, 5, 10, 100, 1000]
    # We keep only those <= data_args.top_k
    retriever = EvaluateRetrieval(model, score_function="cos_sim") # model passed just to satisfy init, we provide results directly
    k_values = [k for k in retriever.k_values if k <= data_args.top_k]
    if not k_values: # Fallback if top_k is smaller than 1 (unlikely) or custom
        k_values = [data_args.top_k]
    # Evaluate
    logger.info("Evaluating...")
    # Set ignore_identical_ids=False to allow retrieving the document with the same ID as the query
    # This is crucial when the query and document are pairs from the same source with the same ID.
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

if __name__ == "__main__":
    main()
