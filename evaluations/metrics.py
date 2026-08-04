"""
evaluations/metrics.py
----------------------
Implementation of classical retrieval metrics (Recall@K, Precision@K, Hit Rate@K, MRR, NDCG@K),
latency statistics calculation (Avg, Min, Max, Median per phase), and cache performance metrics.
"""

import os
import math
import statistics
from typing import List, Dict, Any, Union


def _is_doc_match(retrieved_doc: str, expected_doc: str) -> bool:
    """
    Normalizes document filenames to check matching relevance.
    Handles filename variations like LeavePolicy.pdf vs LeavePolicy(1).pdf.
    """
    if not retrieved_doc or not expected_doc:
        return False
    
    r_clean = os.path.basename(retrieved_doc).lower().replace(" ", "").replace("_", "").replace("-", "")
    e_clean = os.path.basename(expected_doc).lower().replace(" ", "").replace("_", "").replace("-", "")

    # Strip version numbers like (1)
    import re
    r_stem = re.sub(r"\(\d+\)", "", os.path.splitext(r_clean)[0])
    e_stem = re.sub(r"\(\d+\)", "", os.path.splitext(e_clean)[0])

    return r_stem == e_stem or r_stem in e_stem or e_stem in r_stem


def calculate_recall_at_k(retrieved_chunks: List[Dict[str, Any]], expected_doc: str, k: int) -> float:
    """
    Recall@K: 1.0 if expected document is found within top K retrieved chunks, else 0.0.
    """
    top_k_chunks = retrieved_chunks[:k]
    for chunk in top_k_chunks:
        doc = chunk.get("document", "") or chunk.get("payload", {}).get("source", "")
        if _is_doc_match(doc, expected_doc):
            return 1.0
    return 0.0


def calculate_precision_at_k(retrieved_chunks: List[Dict[str, Any]], expected_doc: str, k: int) -> float:
    """
    Precision@K: Fraction of top K retrieved chunks that match the expected document.
    """
    top_k_chunks = retrieved_chunks[:k]
    if not top_k_chunks:
        return 0.0

    matches = 0
    for chunk in top_k_chunks:
        doc = chunk.get("document", "") or chunk.get("payload", {}).get("source", "")
        if _is_doc_match(doc, expected_doc):
            matches += 1

    return matches / float(min(k, len(top_k_chunks)))


def calculate_hit_rate_at_k(retrieved_chunks: List[Dict[str, Any]], expected_doc: str, k: int) -> float:
    """
    Hit Rate@K: 1.0 if at least one matching document is present in top K, else 0.0.
    """
    return calculate_recall_at_k(retrieved_chunks, expected_doc, k)


def calculate_mrr(retrieved_chunks: List[Dict[str, Any]], expected_doc: str) -> float:
    """
    Mean Reciprocal Rank (MRR): 1 / rank of the first relevant retrieved document.
    """
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        doc = chunk.get("document", "") or chunk.get("payload", {}).get("source", "")
        if _is_doc_match(doc, expected_doc):
            return 1.0 / float(rank)
    return 0.0


def calculate_ndcg_at_k(retrieved_chunks, expected_doc, k):
    top_k = retrieved_chunks[:k]

    if not top_k:
        return 0.0

    dcg = 0.0
    for rank, chunk in enumerate(top_k, start=1):
        doc = chunk.get("document") or chunk.get("payload", {}).get("source", "")
        if _is_doc_match(doc, expected_doc):
            dcg += 1.0 / math.log2(rank + 1)

    # One relevant document
    return min(dcg, 1.0)


def _mean(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def _median(lst: List[float]) -> float:
    return statistics.median(lst) if lst else 0.0


def evaluate_retrieval_batch(
    retrieved_results_list: List[List[Dict[str, Any]]],
    expected_docs_list: List[str]
) -> Dict[str, float]:
    """
    Evaluates vector retrieval metrics across all queries.
    """
    n = len(expected_docs_list)
    if n == 0:
        return {
            "Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0,
            "Precision@1": 0.0, "Precision@3": 0.0, "Precision@5": 0.0,
            "Hit Rate": 0.0, "MRR": 0.0, "NDCG": 0.0
        }

    rec1, rec3, rec5 = [], [], []
    prec1, prec3, prec5 = [], [], []
    hit, mrr, ndcg = [], [], []

    for res, exp_doc in zip(retrieved_results_list, expected_docs_list):
        rec1.append(calculate_recall_at_k(res, exp_doc, 1))
        rec3.append(calculate_recall_at_k(res, exp_doc, 3))
        rec5.append(calculate_recall_at_k(res, exp_doc, 5))
        prec1.append(calculate_precision_at_k(res, exp_doc, 1))
        prec3.append(calculate_precision_at_k(res, exp_doc, 3))
        prec5.append(calculate_precision_at_k(res, exp_doc, 5))
        hit.append(calculate_hit_rate_at_k(res, exp_doc, 5))
        mrr.append(calculate_mrr(res, exp_doc))
        ndcg.append(calculate_ndcg_at_k(res, exp_doc, 5))

    return {
        "Recall@1": round(_mean(rec1), 4),
        "Recall@3": round(_mean(rec3), 4),
        "Recall@5": round(_mean(rec5), 4),
        "Precision@1": round(_mean(prec1), 4),
        "Precision@3": round(_mean(prec3), 4),
        "Precision@5": round(_mean(prec5), 4),
        "Hit Rate": round(_mean(hit), 4),
        "MRR": round(_mean(mrr), 4),
        "NDCG": round(_mean(ndcg), 4),
    }


def calculate_latency_stats(latency_records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Computes Average, Minimum, Maximum, and Median latency per pipeline phase.
    """
    phases = [
        "preprocessing",
        "embedding",
        "vector_retrieval",
        "llm",
        "total_response"
    ]
    summary = {}

    for phase in phases:
        values = [rec.get(phase, 0.0) for rec in latency_records if phase in rec]
        if not values:
            summary[phase] = {"Average": 0.0, "Minimum": 0.0, "Maximum": 0.0, "Median": 0.0}
        else:
            summary[phase] = {
                "Average": round(_mean(values), 4),
                "Minimum": round(min(values), 4),
                "Maximum": round(max(values), 4),
                "Median": round(_median(values), 4),
            }

    return summary


def calculate_cache_stats(cache_records: List[bool]) -> Dict[str, Union[int, float]]:
    """
    Calculates Cache Hits, Cache Misses, and Cache Hit Rate.
    """
    total = len(cache_records)
    hits = sum(1 for hit in cache_records if hit)
    misses = total - hits
    hit_rate = round(hits / float(total), 4) if total > 0 else 0.0

    return {
        "Cache Hits": hits,
        "Cache Misses": misses,
        "Total Queries": total,
        "Cache Hit Rate": hit_rate
    }
