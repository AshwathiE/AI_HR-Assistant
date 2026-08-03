# chat.py — RAG pipeline with hybrid search, deduplication, and reranking
import re
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.utils import deduplicate_chunks
from utils.logger import logger
from services.preprocessor import preprocess_query
from services.embeddings import generate_embedding
from services.vector_db import search_documents
from services.llm import generate_answer
from cache import query_cache

router = APIRouter()
response_times = []

class ChatRequest(BaseModel):
    question: str
    top_k: int = 3
    sources: Optional[List[str]] = None
    selected_document: Optional[str] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[list] = None
    metadata: Optional[dict] = None


@router.post("/", response_model=ChatResponse, response_model_exclude_none=True)
def chat(request: ChatRequest):

    start_time = time.time()

    logger.info(f"Top-K received from UI: {request.top_k}") 

    question = request.question.strip()
    selected_document = request.selected_document

    if selected_document in ["", "All Documents"]:
        selected_document = None

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    cleaned_query, _ = preprocess_query(question)
    search_text = cleaned_query

    if selected_document:
        cache_key = f"{search_text.lower()}:{selected_document}"
    else:
        cache_key = search_text.lower()

    if cache_key in query_cache:
        cached = query_cache[cache_key]
        meta = dict(cached.metadata or {})
        meta["cached"] = True
        meta["response_time"] = round(time.time() - start_time, 3)
        return ChatResponse(
            question=cached.question,
            answer=cached.answer,
            sources=cached.sources,
            metadata=meta,
        )

    query_embedding = generate_embedding(search_text)

    top_k = max(1, min(request.top_k or 3, 20))

    logger.info(f"Top-K used by backend: {top_k}")

    selected_sources = None
    if selected_document:
        selected_sources = [selected_document]
    elif request.sources:
        selected_sources = request.sources

    fallback_used = False

    vector_results = search_documents(
        query=question,
        query_embedding=query_embedding,
        top_k=20,
        selected_sources=selected_sources,
)

    results = vector_results

    if results:
        best_score = results[0]["score"]
        similarity_threshold = best_score * 0.75
    else:
        similarity_threshold = 0

    results = [
        r for r in results
        if r["score"] >= similarity_threshold
    ]

    results = deduplicate_chunks(results)

    results = results[:top_k]

    context = ""
    for result in results:
        context += (
            f"[Source: {result['document']}, Chunk {result['chunk_number']}]\n"
            f"{result['text']}\n\n"
        )

    logger.info(f"Results after filtering: {len(results)}")

    llm_result = generate_answer(
        question=question,
        context=context,
        top_k=top_k,
    )

    logger.info("========== CHUNKS SENT TO LLM ==========")

    for i, result in enumerate(results, start=1):
       logger.info(
        f"""
        Chunk {i}
        Document : {result['document']}
        Chunk No : {result['chunk_number']}
        Score    : {result['score']:.4f}

    {result['text']}
--------------------------------------------------------
 """
    )

    logger.info(f"Total chunks sent to LLM: {len(results)}")
    logger.info(f"Context length: {len(context)} characters")
    logger.info("========================================")

    answer = llm_result.get(
        "answer",
        "I couldn't find this information in the uploaded files.",
    )

    # Create payload for memory collection

    source_document = None
    if results:
        source_document = results[0]["document"]

    sources = []
    for result in results:
        sources.append({
            "point_id": result["id"],
            "document": result["document"],
            "chunk_number": result["chunk_number"],
            "score": round(result.get("score", 0), 4),
            "text": result["text"],
        })

    total_response_time = round(time.time() - start_time, 3)

    response = ChatResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        metadata={
            "top_k": top_k,
            "retrieval_count": len(results),
            "search_text": search_text,
            "response_time": total_response_time,
            "cached": False,
            "requested_document": selected_document,
            "source_document": source_document,
            "fallback_used": fallback_used,
        },
    )

    query_cache[cache_key] = response

    return response
