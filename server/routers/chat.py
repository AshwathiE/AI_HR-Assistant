# chat.py — RAG pipeline with hybrid search, deduplication, and reranking
import re
import time
from difflib import SequenceMatcher
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from whoosh_db import search_chunks
from utils import deduplicate_chunks
from logger import logger
from preprocessor import preprocess_query
from embeddings import generate_embedding
from vector_db import search_documents
from llm import generate_answer
from reranker import rerank_documents
from cache import query_cache
from utils import reciprocal_rank_fusion


from utils import deduplicate_chunks
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
    print("\n===== Vector Search Results =====")
    for i, result in enumerate(vector_results, start=1):
        print(f"\nChunk {i}:")
        print(result["text"])
    print("===== End of Vector Search Results =====\n")

    # ---- BM25 Keyword Search ----
    t_bm25 = time.time()
    bm25_results = search_chunks(
        cleaned_query,
        top_k=top_k,
        selected_sources=request.sources,
    )   
    logger.info(f"BM25 Retrieval Time: {time.time()-t_bm25:.3f} sec")
    logger.info(f"Type of bm25_results: {type(bm25_results)}")
    print("\n===== BM25 Search Results =====")
    for i, result in enumerate(bm25_results, start=1):
        print(f"\nChunk {i}:")
        print(result["text"])
    print("===== End of BM25 Search Results =====\n")
    logger.info(f"bm25_results: {len(bm25_results)}")

    # ---- Reciprocal Rank Fusion ----  
    merged_results = reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results,
    )

    # ---- Similarity Threshold ----
    SIMILARITY_THRESHOLD = 0.30
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

    print("Type:", type(results))
    print("Length:", len(results))

    if results:
        print("Keys:", results[0].keys())
        print("First Result:", results[0])

    logger.info(f"Cross Encoder Time: {time.time()-t3:.3f} sec")
    logger.info(f"Final chunks after reranking: {len(results)}")


    # ---- Context Assembly ----
    # ---- Context Assembly ----
    context = ""
    for result in results:
        context += (
            f"[Source: {result['document']}, Chunk {result['chunk_number']}]\n"
            f"{result['text']}\n\n"
        )

    logger.info(f"Context length sent to LLM: {len(context)} characters")
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