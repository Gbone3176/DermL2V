import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from beir.retrieval.evaluation import EvaluateRetrieval


DATASET_NAME_MAPPING = {
    "eval3-text-benchmark_split_choices": "DermSynth_knowledgebase",
    "medmcqa_skin_retrieval_long_doc_test": "medmcqa_long",
    "medmcqa_skin_retrieval_short_doc_test": "medmcqa_short",
    "MedMCQA_RT_query_doc": "MedMCQA_RT",
    "MedQuAD_dermatology_qa_retrieval": "MedQuAD",
    "MedQuAD_dermatology_qa_retrieval_doclt300": "MedQuAD_dermatology_qa_retrieval_doclt300",
}


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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


def build_output_file(output_root: str, dataset_file_path: str, model_name: str) -> str:
    dataset_stem = os.path.splitext(os.path.basename(dataset_file_path))[0]
    dataset_name = sanitize_path_component(DATASET_NAME_MAPPING.get(dataset_stem, dataset_stem))
    model_file_name = f"{sanitize_path_component(model_name)}.json"
    dataset_output_dir = os.path.join(output_root, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    return os.path.join(dataset_output_dir, model_file_name)


def _normalize_text(value) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def build_corpus_queries(
    dataset: List[dict],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    relevant_docs: Dict[str, Dict[str, int]] = {}
    new_format_doc_ids: Dict[str, str] = {}

    for idx, sample in enumerate(dataset):
        pair_id = str(idx)

        if isinstance(sample, dict) and "question" in sample:
            question = _normalize_text(sample.get("question"))
            right_choice = _normalize_text(sample.get("right_choice"))
            wrong_choices = sample.get("wrong_choices") or []
            if isinstance(wrong_choices, str):
                wrong_choices = [wrong_choices]
            wrong_choices = [_normalize_text(wrong) for wrong in wrong_choices]
            wrong_choices = [wrong for wrong in wrong_choices if wrong]
            if not question or not right_choice:
                continue

            queries[pair_id] = question

            right_doc_id = f"{pair_id}_right"
            corpus[right_doc_id] = {"text": right_choice}
            relevant_docs[pair_id] = {right_doc_id: 1}

            for j, wrong in enumerate(wrong_choices):
                wrong_doc_id = f"{pair_id}_wrong_{j}"
                corpus[wrong_doc_id] = {"text": wrong}
            continue

        if isinstance(sample, dict) and "query" in sample and "doc" in sample:
            query = _normalize_text(sample.get("query"))
            doc = _normalize_text(sample.get("doc"))
            if not query or not doc:
                continue

            query_id = _normalize_text(sample.get("id")) or pair_id
            if query_id in queries:
                query_id = f"{query_id}_{idx}"
            queries[query_id] = query

            doc_id = new_format_doc_ids.get(doc)
            if doc_id is None:
                base_doc_id = f"{query_id}_doc"
                doc_id = base_doc_id
                suffix = 1
                while doc_id in corpus:
                    doc_id = f"{base_doc_id}_{suffix}"
                    suffix += 1
                corpus[doc_id] = {"text": doc}
                new_format_doc_ids[doc] = doc_id

            relevant_docs[query_id] = {doc_id: 1}
            continue

        texts = getattr(sample, "texts", None)
        if texts is None or len(texts) < 2:
            continue
        queries[pair_id] = texts[0]
        doc_id = f"{pair_id}_pos"
        corpus[doc_id] = {"text": texts[1]}
        relevant_docs[pair_id] = {doc_id: 1}

    return corpus, queries, relevant_docs


def build_results(
    q_emb: torch.Tensor,
    d_emb: torch.Tensor,
    query_ids: List[str],
    corpus_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    scores = cos_sim(q_emb, d_emb)
    scores[torch.isnan(scores)] = -1
    top_k = min(10, len(corpus_ids))
    top_vals, top_idx = torch.topk(scores, top_k, dim=1, largest=True, sorted=True)
    top_vals = top_vals.cpu().tolist()
    top_idx = top_idx.cpu().tolist()

    results: Dict[str, Dict[str, float]] = {}
    for i, qid in enumerate(query_ids):
        results[qid] = {}
        for rank, idx in enumerate(top_idx[i]):
            doc_id = corpus_ids[idx]
            results[qid][doc_id] = top_vals[i][rank]
    return results


def evaluate_retrieval_metrics(
    relevant_docs: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    corpus_size: int,
) -> Dict[str, float]:
    retriever = EvaluateRetrieval(None, score_function="cos_sim")
    eval_ks = [k for k in (3, 5, 10) if k <= corpus_size]
    if not eval_ks:
        eval_ks = [1]
    ndcg, _, recall, _ = retriever.evaluate(
        relevant_docs,
        results,
        eval_ks,
        ignore_identical_ids=False,
    )
    metrics: Dict[str, float] = {}
    for k in (3, 5, 10):
        effective_k = min(k, corpus_size)
        metrics[f"NDCG@{k}"] = ndcg[f"NDCG@{effective_k}"]
        metrics[f"Recall@{k}"] = recall[f"Recall@{effective_k}"]
    return metrics


def evaluate_at_10(
    relevant_docs: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    corpus_size: int,
) -> Dict[str, float]:
    return evaluate_retrieval_metrics(relevant_docs, results, corpus_size)
