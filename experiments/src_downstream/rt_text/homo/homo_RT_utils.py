import json
import os
from typing import Dict, List, Optional, Tuple

import torch


DEFAULT_VIS_DATASET = "/mnt/nas1/disk06/bowenguo/datasets/image-text/Derm1M/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl"
DEFAULT_DERMVARIANTS_DIR = "/mnt/nas1/disk06/bowenguo/datasets/image-text/Derm1M/DermVariantsData"
DEFAULT_RETRIEVAL_SUBSETS = ["SemVariants", "DermQA", "SI1", "SI2"]

DATASET_INSTRUCTIONS = {
    "VisVariants": "Given a diagnosis-style dermatology text, retrieve the visual-description text that best matches it in meaning.",
    "SemVariants": "Read the provided dermatological condition description and return the candidate description that matches its meaning most closely.",
    "DermQA": "Given a dermatology-related question, select the answer that is most relevant to what the question is asking.",
    "SI1": "Retrieve the most appropriate answer for this dermatology question.",
    "SI2": "Given a dermatology question, retrieve the single most relevant and most correct answer passage that directly answers it.",
}


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


def sanitize_path_component(name: str) -> str:
    sanitized = name.strip().replace(os.sep, "_")
    if os.altsep:
        sanitized = sanitized.replace(os.altsep, "_")
    return sanitized or "unknown"


def resolve_output_file(output_root: Optional[str], model_name: str) -> Optional[str]:
    if not output_root:
        return None
    os.makedirs(output_root, exist_ok=True)
    return os.path.join(output_root, f"homo_RT_{sanitize_path_component(model_name)}.json")


def _normalize_text(value) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def build_vis_mcq_samples(dataset: List[dict]) -> Tuple[List[str], List[List[str]]]:
    queries: List[str] = []
    candidate_sets: List[List[str]] = []

    for item in dataset:
        if "question" in item:
            query = _normalize_text(item.get("question"))
            positive = _normalize_text(item.get("right_choice"))
            negatives = item.get("wrong_choices") or []
        else:
            query = _normalize_text(item.get("original"))
            positive = _normalize_text(item.get("positive_variant"))
            negatives = item.get("hard_negative_variants") or []

        if isinstance(negatives, str):
            negatives = [negatives]
        negatives = [_normalize_text(neg) for neg in negatives]
        negatives = [neg for neg in negatives if neg]

        candidates = [positive] + negatives if positive else negatives
        if not query or not positive or len(candidates) < 2:
            continue

        queries.append(query)
        candidate_sets.append(candidates)

    return queries, candidate_sets


def build_retrieval_dataset(dataset: List[dict]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    relevant_docs: Dict[str, Dict[str, int]] = {}
    doc_ids_by_text: Dict[str, str] = {}

    for idx, item in enumerate(dataset):
        query = _normalize_text(item.get("original"))
        positive = _normalize_text(item.get("positive_variant"))
        if not query or not positive:
            continue

        qid = str(item.get("id", idx))
        if qid in queries:
            qid = f"{qid}_{idx}"
        queries[qid] = query

        doc_id = doc_ids_by_text.get(positive)
        if doc_id is None:
            doc_id = f"d{len(doc_ids_by_text)}"
            doc_ids_by_text[positive] = doc_id
            corpus[doc_id] = {"text": positive}

        relevant_docs[qid] = {doc_id: 1}

    return corpus, queries, relevant_docs


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def build_results(
    q_emb: torch.Tensor,
    d_emb: torch.Tensor,
    query_ids: List[str],
    corpus_ids: List[str],
    top_k: int = 10,
) -> Dict[str, Dict[str, float]]:
    scores = cos_sim(q_emb, d_emb)
    scores[torch.isnan(scores)] = -1
    effective_top_k = max(1, min(top_k, len(corpus_ids)))
    top_vals, top_idx = torch.topk(scores, effective_top_k, dim=1, largest=True, sorted=True)
    top_vals = top_vals.cpu().tolist()
    top_idx = top_idx.cpu().tolist()

    results: Dict[str, Dict[str, float]] = {}
    for row_idx, query_id in enumerate(query_ids):
        results[query_id] = {}
        for score, doc_idx in zip(top_vals[row_idx], top_idx[row_idx]):
            results[query_id][corpus_ids[doc_idx]] = score
    return results


def evaluate_retrieval_metrics(
    relevant_docs: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    corpus_size: int,
) -> Dict[str, float]:
    eval_k = max(1, min(10, corpus_size))
    ndcg_total = 0.0
    map_total = 0.0
    recall_total = 0.0
    precision_total = 0.0
    mrr_total = 0.0
    query_count = 0

    for query_id, relevant in relevant_docs.items():
        if not relevant:
            continue
        ranked_docs = list(results.get(query_id, {}).keys())[:eval_k]
        relevant_set = {doc_id for doc_id, score in relevant.items() if score > 0}
        if not relevant_set:
            continue

        query_count += 1
        gains = [1 if doc_id in relevant_set else 0 for doc_id in ranked_docs]
        hits = sum(gains)
        recall_total += hits / len(relevant_set)
        precision_total += hits / eval_k

        dcg = 0.0
        ap = 0.0
        first_hit_rank = None
        hit_so_far = 0
        for rank, gain in enumerate(gains, start=1):
            if gain:
                hit_so_far += 1
                dcg += 1.0 / torch.log2(torch.tensor(rank + 1.0)).item()
                ap += hit_so_far / rank
                if first_hit_rank is None:
                    first_hit_rank = rank

        ideal_hits = min(len(relevant_set), eval_k)
        idcg = sum(1.0 / torch.log2(torch.tensor(rank + 1.0)).item() for rank in range(1, ideal_hits + 1))
        ndcg_total += (dcg / idcg) if idcg > 0 else 0.0
        map_total += (ap / len(relevant_set)) if relevant_set else 0.0
        mrr_total += (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0

    if query_count == 0:
        return {
            f"NDCG@{eval_k}": 0.0,
            f"MAP@{eval_k}": 0.0,
            f"Recall@{eval_k}": 0.0,
            f"P@{eval_k}": 0.0,
            f"MRR@{eval_k}": 0.0,
        }

    return {
        f"NDCG@{eval_k}": ndcg_total / query_count,
        f"MAP@{eval_k}": map_total / query_count,
        f"Recall@{eval_k}": recall_total / query_count,
        f"P@{eval_k}": precision_total / query_count,
        f"MRR@{eval_k}": mrr_total / query_count,
    }


def macro_average(metrics_by_subset: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not metrics_by_subset:
        return {}

    keys = sorted({key for metrics in metrics_by_subset.values() for key in metrics})
    averaged: Dict[str, float] = {}
    for key in keys:
        values = [metrics[key] for metrics in metrics_by_subset.values() if key in metrics]
        if values:
            averaged[key] = sum(values) / len(values)
    return averaged
