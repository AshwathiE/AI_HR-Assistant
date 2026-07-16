# chat.py — RAG pipeline with hybrid search, deduplication, and reranking

import re
import time
from difflib import SequenceMatcher
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from logger import logger
from preprocessor import preprocess_query
from embeddings import generate_embedding
from vector_db import search_documents
from bm25 import bm25_search
from llm import generate_answer
from reranker import rerank_documents
from cache import query_cache

router = APIRouter()
response_times = [] 


class ChatRequest(BaseModel):
    question: str
    top_k: int = 3
    sources: Optional[List[str]] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[list] = None
    metadata: Optional[dict] = None


# -----------------------------------------------
# Reciprocal Rank Fusion
# -----------------------------------------------

def reciprocal_rank_fusion(
    vector_results,
    bm25_results,
    k=60
):
    """
    Merge results from vector search and BM25
    using Reciprocal Rank Fusion (RRF).

    Each result gets a combined score:
        score = 1/(k + rank_vector) + 1/(k + rank_bm25)

    Args:
        vector_results: Results from Qdrant vector search
        bm25_results: Results from BM25 keyword search
        k: Smoothing constant (default 60)

    Returns:
        Merged list of results sorted by RRF score
    """

    scores = {}
    result_map = {}

    # Score vector search results by rank
    for rank, result in enumerate(vector_results):

        key = result.get("id") or result["text"][:100]

        scores[key] = scores.get(key, 0) + (
            1.0 / (k + rank + 1)
        )

        result_map[key] = result

    # Score BM25 results by rank
    for rank, result in enumerate(bm25_results):

        key = result.get("id") or result["text"][:100]

        scores[key] = scores.get(key, 0) + (
            1.0 / (k + rank + 1)
        )

        # Keep whichever result has more metadata
        if key not in result_map:
            result_map[key] = result

    # Sort by combined RRF score
    ranked_keys = sorted(
        scores.keys(),
        key=lambda x: scores[x],
        reverse=True
    )

    merged = []

    for key in ranked_keys:

        result = result_map[key]
        result["rrf_score"] = round(scores[key], 6)
        merged.append(result)

    logger.info(
        f"RRF merged: {len(vector_results)} vector + "
        f"{len(bm25_results)} BM25 -> "
        f"{len(merged)} combined results"
    )

    return merged


# -----------------------------------------------
# Chunk Deduplication
# -----------------------------------------------

def deduplicate_chunks(results, similarity_threshold=0.90):
    """
    Remove duplicate and near-duplicate chunks.

    Phase 1: Deduplicate by point ID (exact same chunk
             from multiple retrievers).
    Phase 2: Deduplicate by text similarity using
             SequenceMatcher (catches near-duplicates
             differing by minor whitespace/punctuation).

    Returns:
        Deduplicated list keeping the higher-scored entry.
    """

    if not results:
        return results

    initial_count = len(results)

    # Phase 1: Deduplicate by point ID
    seen_ids = set()
    id_deduped = []

    for result in results:

        point_id = result.get("id")

        if point_id and point_id in seen_ids:
            continue

        if point_id:
            seen_ids.add(point_id)

        id_deduped.append(result)

    logger.info(
        f"Dedup phase 1 (ID): "
        f"{initial_count} -> {len(id_deduped)}"
    )

    # Phase 2: Deduplicate by text similarity
    unique = []

    for result in id_deduped:

        chunk_text_normalized = (
            result["text"].strip().lower()
        )

        is_duplicate = False

        for existing in unique:

            existing_text = (
                existing["text"].strip().lower()
            )

            # Quick length check: skip comparison if
            # lengths differ by more than 20%
            len_ratio = len(chunk_text_normalized) / max(
                len(existing_text), 1
            )

            if len_ratio < 0.8 or len_ratio > 1.2:
                continue

            similarity = SequenceMatcher(
                None,
                chunk_text_normalized,
                existing_text,
            ).ratio()

            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(result)

    logger.info(
        f"Dedup phase 2 (text similarity): "
        f"{len(id_deduped)} -> {len(unique)}"
    )

    logger.info(
        f"Total duplicates removed: "
        f"{initial_count - len(unique)}"
    )

    return unique


# -----------------------------------------------
# Chat Endpoint
# -----------------------------------------------

@router.post("/", response_model=ChatResponse, response_model_exclude_none=True)
def chat(request: ChatRequest):

    start_time = time.time()

    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    logger.info(f"New query received: {request.question}")

    if re.search(r"<[^>]+>", question):
        return ChatResponse(question=request.question,
                            answer="This kind of input cannot be processed.")

    if not re.search(r"[A-Za-z0-9]", question):
        return ChatResponse(
            question=request.question,
            answer="Your query appears to contain only special characters. Please enter a meaningful question.",
        )

    if not re.search(r"[A-Za-z]", question):
        return ChatResponse(
            question=request.question,
            answer="Currently, this AI HR Assistant supports English questions only. Please ask your question in English.",
        )

    # ---- Preprocessing ----
    t_pre = time.time()

    cleaned_query, keywords = preprocess_query(question)
    logger.info(f"Original Question: {question} -> Preprocessed Query: {cleaned_query}")

    search_text = " ".join(
        [cleaned_query] + keywords[:8]
    )

    logger.info(
        f"Preprocessing Time: {time.time() - t_pre:.3f} sec"
    )

    # ---- Cache Check ----
    cache_key = search_text.lower()

    if cache_key in query_cache:
        logger.info(f"Cache Hit: {cache_key}")
        return query_cache[cache_key]

    logger.info(f"Cache Miss: {cache_key}")

    # ---- Embedding ----
    t1 = time.time()
    query_embedding = generate_embedding(search_text)
    logger.info(f"Embedding Time: {time.time()-t1:.3f} sec")

    top_k = max(1, min(request.top_k or 3, 10))

    # ---- Vector Search ----
    t2 = time.time()
    vector_results = search_documents(
        query_embedding=query_embedding,
        top_k=20,
        selected_sources=request.sources,
    )
    logger.info(f"Vector Retrieval Time: {time.time()-t2:.3f} sec")
    logger.info(f"Vector results: {len(vector_results)} chunks")

    # ---- BM25 Keyword Search ----
    t_bm25 = time.time()
    bm25_results = bm25_search(
        query=cleaned_query,
        top_k=20,
        selected_sources=request.sources,
    )
    logger.info(f"BM25 Retrieval Time: {time.time()-t_bm25:.3f} sec")
    logger.info(f"BM25 results: {len(bm25_results)} chunks")

    # ---- Reciprocal Rank Fusion ----
    merged_results = reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results,
    )

    # ---- Similarity Threshold ----
    SIMILARITY_THRESHOLD = 0.25
    results = [
        r for r in merged_results
        if r.get("score", 0) >= SIMILARITY_THRESHOLD
        or r.get("rrf_score", 0) > 0
    ]
    logger.info(f"After similarity filtering: {len(results)} chunks")

    # ---- Chunk Deduplication ----
    results = deduplicate_chunks(results)

    # ---- Cross-Encoder Reranking ----
    t3 = time.time()
    results = rerank_documents(
        question=cleaned_query,
        results=results,
        top_k=top_k,
    )
    logger.info(f"Cross Encoder Time: {time.time()-t3:.3f} sec")
    logger.info(f"Final chunks after reranking: {len(results)}")

    # ---- Context Assembly ----
    context = ""
    for result in results:
        context += (
            f"[Source: {result['document']}, Chunk {result['chunk_number']}]\n"
            f"{result['text']}\n\n"
        )

    # ---- LLM Answer Generation ----
    t4 = time.time()
    llm_result = generate_answer(
        question=question,
        context=context,
        top_k=top_k,
    )
    logger.info(f"LLM Time: {time.time()-t4:.3f} sec")

    answer = llm_result.get(
        "answer",
        "I couldn't find this information in the uploaded company policies.",
    )

    # ---- Response Assembly ----
    sources = []
    for result in results:
        sources.append({
            "point_id": result["id"],
            "document": result["document"],
            "chunk_number": result["chunk_number"],
            "score": round(result.get("score", 0), 4),
            "text": result["text"],
        })

    response = ChatResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        metadata={
            "top_k": top_k,
            "retrieval_count": len(results),
            "search_text": search_text,
        },
    )
    query_cache[cache_key] = response

    logger.info(
        f"Stored response in cache: {cache_key}"
    )
   
    total_response_time = time.time() - start_time

    logger.info(
        f"Total Response Time: {total_response_time:.3f} sec"
    )

    response_times.append(total_response_time)

    average_response_time = (
        sum(response_times) / len(response_times)
    )

    logger.info(
        f"Average Response Time ({len(response_times)} requests): "
        f"{average_response_time:.3f} sec"
    )

    return response