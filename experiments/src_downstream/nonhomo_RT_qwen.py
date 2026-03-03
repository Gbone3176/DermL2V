import json
import logging
import os
import argparse
from typing import Dict, List, Optional

import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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


def encode_batches(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    prompt_name: Optional[str] = None,
    desc: Optional[str] = None,
) -> torch.Tensor:
    return model.encode(
        texts,
        prompt_name=prompt_name,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=desc is not None,
    )


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
    parser = argparse.ArgumentParser(description="RT: Retrieval evaluation using Qwen-Embedding")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if results already exist
    if args.output:
        model_name_str = os.path.basename(args.model_name_or_path.rstrip("/")).replace("/", "-")
        output_path = os.path.join(args.output, f"rt_metrics_{model_name_str}.json")
        if os.path.exists(output_path):
            logger.info(f"Results already exist at {output_path}, skipping...")
            return

    model_kwargs = {
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": True,
        "dtype": torch.float16,
    }
    tokenizer_kwargs = {"padding_side": "left", "trust_remote_code": True}
    try:
        model = SentenceTransformer(
            args.model_name_or_path,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            device=device,
            trust_remote_code=True,
        )
    except TypeError:
        model = SentenceTransformer(
            args.model_name_or_path,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            device=device,
        )
    model.max_seq_length = args.max_length

    dataset = load_jsonl(args.input)
    if args.max_samples and args.max_samples > 0:
        dataset = dataset[: args.max_samples]
    corpus, queries, relevant_docs = build_corpus_queries(dataset)

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

    logger.info(f"Encoding {len(query_texts)} queries")
    q_emb = encode_batches(model, query_texts, args.batch_size, prompt_name="query", desc="Encoding queries")
    logger.info(f"Encoding {len(corpus_texts)} documents")
    d_emb = encode_batches(model, corpus_texts, args.batch_size, prompt_name=None, desc="Encoding documents")

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

    retriever = EvaluateRetrieval(None, score_function="cos_sim")
    default_k_values = [1, 3, 5, 10, 100]
    k_values = [k for k in default_k_values if k <= top_k]
    if not k_values:
        k_values = [top_k]
    ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, k_values, ignore_identical_ids=False)

    metrics = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
    }
    logger.info(json.dumps(metrics, indent=4))
    if args.output:
        model_name_str = os.path.basename(args.model_name_or_path.rstrip("/")).replace("/", "-")
        output_path = os.path.join(args.output, f"rt_metrics_{model_name_str}.json")
        os.makedirs(args.output, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
