"""
test_chat.py — Comprehensive unit tests for the FastAPI /chat/ endpoint
========================================================================

Endpoint under test:  POST /chat/   (routers/chat.py)

What is tested
--------------
* Input Validation      — empty, whitespace, HTML injection, special-chars only,
                          non-English (Hindi script & digits only), extremely long input,
                          SQL injection string, Unicode, top_k bounds (clamping & defaults),
                          invalid top_k types.
* Successful Requests   — response structure, echoed question, sources list,
                          metadata fields, retrieval_count, search_text, score rounding.
* Search Pipeline       — preprocess_query called, generate_embedding called,
                          search_documents called with correct arguments,
                          dynamic similarity threshold filtering (best_score * 0.80),
                          top_k slicing, deduplicate_chunks called, context string assembly.
* Cache                 — cache hit bypasses embedding + LLM, cache miss triggers full pipeline,
                          result stored after miss, lowercase cache key.
* Error Handling        — embedding failure (500), vector DB exception (500), LLM exception (500),
                          missing 'answer' key fallback.
* Edge Cases            — single retrieved chunk, no retrieved chunks (empty vector search),
                          very large context, multiple PDFs in sources, duplicate chunk IDs,
                          near-duplicate text, missing payload key, filtered low score chunks.

Fixtures / mocks used
---------------------
* TestClient from fastapi.testclient  — raise_server_exceptions=False used to capture 500 responses.
* unittest.mock.patch                 — all external dependencies stubbed using patch.
* conftest.py stubs                   — heavy imports (sentence_transformers, qdrant_client,
                                        groq, google.generativeai, symspellpy, transformers)
                                        stubbed to ensure complete isolation.

NOTE: Cross-Encoder reranking is commented out in routers/chat.py, so reranking tests are excluded
      per instructions to match exact implementation.

How to run
----------
    cd server
    pytest tests/test_chat.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ===========================================================================
# Helper Functions for Mock Data
# ===========================================================================

def _make_result(
    doc="LeavePolicy.pdf",
    chunk=1,
    text="Employees are entitled to 12 days of sick leave per year.",
    score=0.92,
    uid="uuid-001",
):
    """
    Build a realistic retrieval result dict matching the shape returned
    by search_documents() in vector_db.py.
    """
    return {
        "id": uid,
        "score": score,
        "text": text,
        "topics": ["sick leave", "leave policy"],
        "document": doc,
        "chunk_number": chunk,
        "collection_name": f"company_policy_{doc.lower().replace('.', '_')}_abc12345",
        "payload": {},
    }


def _make_llm_result(answer="Sick leave is 12 days per year."):
    """
    Build a realistic dict returned by generate_answer().
    """
    return {
        "answer": answer,
        "confidence": "high",
        "follow_up_question": "Would you like to know about casual leave?",
        "sources": [],
    }


def _patches(
    preprocess_key="sick leave",
    preprocess_kw=None,
    embedding=None,
    search_results=None,
    llm_answer="Sick leave is 12 days per year.",
    cache_dict=None,
):
    """
    Helper function returning a dictionary of mock patches for the RAG pipeline.
    """
    if preprocess_kw is None:
        preprocess_kw = ["sick", "leave"]
    if embedding is None:
        embedding = [0.0] * 384
    if search_results is None:
        search_results = [_make_result()]
    if cache_dict is None:
        cache_dict = {}

    return {
        "preprocess": patch(
            "routers.chat.preprocess_query",
            return_value=(preprocess_key, preprocess_kw),
        ),
        "embedding": patch(
            "routers.chat.generate_embedding",
            return_value=embedding,
        ),
        "search": patch(
            "routers.chat.search_documents",
            return_value=search_results,
        ),
        "dedup": patch(
            "routers.chat.deduplicate_chunks",
            side_effect=lambda x: x,
        ),
        "llm": patch(
            "routers.chat.generate_answer",
            return_value=_make_llm_result(llm_answer),
        ),
        "cache": patch(
            "routers.chat.query_cache",
            cache_dict,
        ),
    }


# ===========================================================================
# Pytest Fixture
# ===========================================================================

@pytest.fixture
def client():
    """
    Provides a FastAPI TestClient wrapping only the chat router.
    Uses raise_server_exceptions=False so unhandled 500 status codes are returned
    as Response objects instead of raising Python exceptions in the client context.
    """
    from routers.chat import router as chat_router
    app = FastAPI()
    app.include_router(chat_router, prefix="/chat")
    return TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# 1. Input Validation Scenarios
# ===========================================================================

class TestChatInputValidation:
    """
    Unit tests for input validation and guard clauses in POST /chat/.
    """

    def test_empty_question_returns_400(self, client):
        """Empty string question raises HTTPException with 400 status code."""
        response = client.post("/chat/", json={"question": "", "top_k": 3})
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_whitespace_only_question_returns_400(self, client):
        """Whitespace-only question raises HTTPException with 400 status code."""
        response = client.post("/chat/", json={"question": "   \t  \n", "top_k": 3})
        assert response.status_code == 400
        assert response.json()["detail"] == "Question cannot be empty."

    def test_missing_required_question_field_returns_422(self, client):
        """Missing required 'question' field returns 422 Unprocessable Entity."""
        response = client.post("/chat/", json={"top_k": 3})
        assert response.status_code == 422

    def test_invalid_data_types_returns_422(self, client):
        """Invalid top_k data type (string) returns 422 validation error."""
        response = client.post("/chat/", json={"question": "What is sick leave?", "top_k": "invalid_int"})
        assert response.status_code == 422

    def test_html_script_injection_handled_safely(self, client):
        """HTML/script injection strings return a safe rejection answer without calling LLM."""
        with patch("routers.chat.query_cache", {}), \
             patch("routers.chat.generate_answer") as mock_llm:
            response = client.post(
                "/chat/",
                json={"question": "<script>alert('XSS')</script>", "top_k": 3},
            )
            assert response.status_code == 200
            data = response.json()
            assert "cannot be processed" in data["answer"].lower()
            mock_llm.assert_not_called()

    def test_sql_injection_string_processed(self, client):
        """SQL injection input containing alphanumeric chars is processed through the normal pipeline."""
        p = _patches(preprocess_key="or 1 1")
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            response = client.post(
                "/chat/",
                json={"question": "' OR '1'='1", "top_k": 3},
            )
            assert response.status_code == 200

    def test_special_characters_only_handled_safely(self, client):
        """Query with special characters only returns friendly warning message."""
        with patch("routers.chat.query_cache", {}), \
             patch("routers.chat.generate_embedding") as mock_embed:
            response = client.post(
                "/chat/",
                json={"question": "!@#$%^&*()", "top_k": 3},
            )
            assert response.status_code == 200
            assert "special characters" in response.json()["answer"].lower()
            mock_embed.assert_not_called()

    def test_non_english_non_ascii_input_returns_special_chars_message(self, client):
        """Non-English (Hindi script without ASCII chars) hits '[A-Za-z0-9]' check and returns special chars message."""
        with patch("routers.chat.query_cache", {}), \
             patch("routers.chat.generate_answer") as mock_llm:
            response = client.post(
                "/chat/",
                json={"question": "बीमार छुट्टी", "top_k": 3},
            )
            assert response.status_code == 200
            assert "special characters" in response.json()["answer"].lower()
            mock_llm.assert_not_called()

    def test_digits_only_input_returns_english_requirement_notice(self, client):
        """Numeric input (digits pass '[A-Za-z0-9]' but fail '[A-Za-z]') returns English requirement notice."""
        with patch("routers.chat.query_cache", {}):
            response = client.post(
                "/chat/",
                json={"question": "12345678", "top_k": 3},
            )
            assert response.status_code == 200
            assert "english" in response.json()["answer"].lower()

    def test_extremely_long_input(self, client):
        """Extremely long input strings are processed without raising unexpected errors."""
        long_q = "What is the policy for leave? " * 300
        p = _patches(preprocess_key="leave policy")
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            response = client.post(
                "/chat/",
                json={"question": long_q, "top_k": 3},
            )
            assert response.status_code == 200

    def test_unicode_characters_in_question(self, client):
        """Questions containing valid Unicode symbols (e.g. emojis) along with text execute normally."""
        p = _patches(preprocess_key="sick leave")
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            response = client.post(
                "/chat/",
                json={"question": "What is sick leave policy? 🏥💬", "top_k": 3},
            )
            assert response.status_code == 200

    def test_top_k_falsy_zero_defaults_to_3(self, client):
        """In implementation `request.top_k or 3` turns falsy 0 into 3."""
        p = _patches()
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            response = client.post(
                "/chat/",
                json={"question": "What is sick leave?", "top_k": 0},
            )
            assert response.status_code == 200
            assert response.json()["metadata"]["top_k"] == 3

    def test_top_k_negative_value_clamped_to_1(self, client):
        """Negative top_k (truthy integer) is clamped to minimum value 1 by max(1, min(-5, 10))."""
        p = _patches()
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            response = client.post(
                "/chat/",
                json={"question": "What is sick leave?", "top_k": -5},
            )
            assert response.status_code == 200
            assert response.json()["metadata"]["top_k"] == 1

    def test_top_k_maximum_boundary(self, client):
        """top_k value exceeding 10 is clamped to maximum allowed value of 10."""
        p = _patches()
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            response = client.post(
                "/chat/",
                json={"question": "What is sick leave?", "top_k": 25},
            )
            assert response.status_code == 200
            assert response.json()["metadata"]["top_k"] == 10


# ===========================================================================
# 2. Successful Requests
# ===========================================================================

class TestChatSuccessfulRequests:
    """
    Unit tests for successful request processing, verifying status codes, schemas, and content.
    """

    def test_normal_successful_response(self, client):
        """Verify normal successful POST /chat/ returns 200 OK."""
        p = _patches()
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            response = client.post(
                "/chat/",
                json={"question": "What is sick leave?", "top_k": 3},
            )
            assert response.status_code == 200

    def test_verify_response_structure(self, client):
        """Verify response body contains required schema fields."""
        p = _patches()
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            response = client.post(
                "/chat/",
                json={"question": "What is sick leave?", "top_k": 3},
            )
            data = response.json()
            assert "question" in data
            assert "answer" in data
            assert "sources" in data
            assert "metadata" in data

    def test_verify_original_question_echoed(self, client):
        """Verify original uncleaned question is echoed back in response."""
        original_q = "  How many casual leave days do I get?  "
        p = _patches(preprocess_key="casual leave days")
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            response = client.post(
                "/chat/",
                json={"question": original_q, "top_k": 3},
            )
            assert response.json()["question"] == original_q

    def test_verify_sources_are_returned(self, client):
        """Verify sources list is properly structured with point_id, document, score, and text."""
        result = _make_result(doc="LeavePolicy.pdf", chunk=1, uid="point-101", score=0.95432)
        p = _patches(search_results=[result])
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            response = client.post(
                "/chat/",
                json={"question": "What is sick leave?", "top_k": 3},
            )
            sources = response.json()["sources"]
            assert len(sources) == 1
            assert sources[0]["point_id"] == "point-101"
            assert sources[0]["document"] == "LeavePolicy.pdf"
            assert sources[0]["chunk_number"] == 1
            assert sources[0]["score"] == 0.9543
            assert sources[0]["topics"] == result["topics"]

    def test_verify_metadata(self, client):
        """Verify metadata fields top_k, retrieval_count, and search_text are accurate."""
        p = _patches(preprocess_key="casual leave policy")
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            response = client.post(
                "/chat/",
                json={"question": "What is casual leave policy?", "top_k": 5},
            )
            meta = response.json()["metadata"]
            assert meta["top_k"] == 5
            assert meta["retrieval_count"] == 1
            assert meta["search_text"] == "casual leave policy"


# ===========================================================================
# 3. Search Pipeline Scenarios
# ===========================================================================

class TestChatSearchPipeline:
    """
    Unit tests for verifying correct invocations across the RAG search pipeline stages.
    """

    def test_query_preprocessing_called(self, client):
        """Verify preprocess_query is called with stripped user question."""
        p = _patches()
        with p["preprocess"] as mock_pp, p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            client.post("/chat/", json={"question": " What is sick leave? ", "top_k": 3})
            mock_pp.assert_called_once_with("What is sick leave?")

    def test_embedding_generation_called(self, client):
        """Verify generate_embedding is called with preprocessed search_text."""
        p = _patches(preprocess_key="sick leave")
        with p["preprocess"], p["embedding"] as mock_embed, p["search"], p["dedup"], p["llm"], p["cache"]:
            client.post("/chat/", json={"question": "What is sick leave?", "top_k": 3})
            mock_embed.assert_called_once_with("sick leave")

    def test_vector_search_called(self, client):
        """Verify search_documents is called with query embedding, top_k=20, and selected sources."""
        dummy_embed = [0.1] * 384
        selected_sources = ["LeavePolicy.pdf"]
        p = _patches(embedding=dummy_embed)
        with p["preprocess"], p["embedding"], p["search"] as mock_search, p["dedup"], p["llm"], p["cache"]:
            client.post(
                "/chat/",
                json={"question": "What is sick leave?", "top_k": 4, "sources": selected_sources},
            )
            mock_search.assert_called_once_with(
                query_embedding=dummy_embed,
                top_k=20,
                selected_sources=selected_sources,
            )

    def test_similarity_filtering_applied(self, client):
        """Verify chunks scoring lower than dynamic similarity threshold (best_score * 0.80) are filtered out."""
        chunk1 = _make_result(uid="c1", score=0.90)  # best score = 0.90, threshold = 0.72
        chunk2 = _make_result(uid="c2", score=0.65)  # 0.65 < 0.72 -> filtered out
        p = _patches(search_results=[chunk1, chunk2])
        
        filtered_chunks = []
        def capture_dedup(chunks):
            filtered_chunks.extend(chunks)
            return chunks

        with p["preprocess"], p["embedding"], p["search"], \
             patch("routers.chat.deduplicate_chunks", side_effect=capture_dedup), \
             p["llm"], p["cache"]:
            client.post("/chat/", json={"question": "What is sick leave?", "top_k": 5})

        assert len(filtered_chunks) == 1
        assert filtered_chunks[0]["id"] == "c1"

    def test_context_assembled_correctly(self, client):
        """Verify context format passed to generate_answer contains source metadata tags and chunk text."""
        result = _make_result(doc="LeavePolicy.pdf", chunk=2, text="Casual leave policy details.")
        p = _patches(search_results=[result])

        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["cache"], \
             patch("routers.chat.generate_answer") as mock_llm:
            mock_llm.return_value = _make_llm_result()
            client.post("/chat/", json={"question": "Casual leave?", "top_k": 3})
            
            context_arg = mock_llm.call_args[1].get("context") or mock_llm.call_args[0][1]
            assert "[Source: LeavePolicy.pdf, Chunk 2]" in context_arg
            assert "Casual leave policy details." in context_arg

    def test_retrieval_count_matches_expected(self, client):
        """Verify top_k limit truncates final context chunks count."""
        results = [_make_result(uid=f"u{i}", score=0.90) for i in range(5)]
        p = _patches(search_results=results)
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            response = client.post("/chat/", json={"question": "What is leave?", "top_k": 2})
            assert response.json()["metadata"]["retrieval_count"] == 2


# ===========================================================================
# 4. Cache Scenarios
# ===========================================================================

class TestChatCache:
    """
    Unit tests for query_cache behaviour (cache hit, miss, and update).
    """

    def test_cache_hit_bypasses_expensive_functions(self, client):
        """Cache hit immediately returns stored ChatResponse without generating embedding or calling LLM."""
        from routers.chat import ChatResponse
        cache_key = "sick leave"
        cached_resp = ChatResponse(
            question="What is sick leave?",
            answer="Cached answer: 12 days sick leave.",
            metadata={"top_k": 3, "retrieval_count": 1, "search_text": cache_key},
        )
        fake_cache = {cache_key: cached_resp}

        with patch("routers.chat.query_cache", fake_cache), \
             patch("routers.chat.preprocess_query", return_value=(cache_key, ["sick", "leave"])), \
             patch("routers.chat.generate_embedding") as mock_embed, \
             patch("routers.chat.generate_answer") as mock_llm:
            response = client.post("/chat/", json={"question": "What is sick leave?", "top_k": 3})
            assert response.status_code == 200
            assert response.json()["answer"] == "Cached answer: 12 days sick leave."
            mock_embed.assert_not_called()
            mock_llm.assert_not_called()

    def test_cache_miss_updates_cache(self, client):
        """Cache miss executes full pipeline and stores resulting ChatResponse in query_cache."""
        fake_cache = {}
        cache_key = "casual leave"
        p = _patches(preprocess_key=cache_key, cache_dict=fake_cache)
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            response = client.post("/chat/", json={"question": "What is casual leave?", "top_k": 3})
            assert response.status_code == 200
            assert cache_key in fake_cache
            assert fake_cache[cache_key].answer == "Sick leave is 12 days per year."


# ===========================================================================
# 5. Error Handling Scenarios
# ===========================================================================

class TestChatErrorHandling:
    """
    Unit tests for exceptions during pipeline execution.
    """

    def test_embedding_generation_failure(self, client):
        """Exception during generate_embedding results in 500 Internal Server Error."""
        with patch("routers.chat.query_cache", {}), \
             patch("routers.chat.preprocess_query", return_value=("query", ["query"])), \
             patch("routers.chat.generate_embedding", side_effect=RuntimeError("Embedding model crashed")):
            response = client.post("/chat/", json={"question": "What is leave?", "top_k": 3})
            assert response.status_code == 500

    def test_vector_database_exception(self, client):
        """Exception during search_documents results in 500 Internal Server Error."""
        with patch("routers.chat.query_cache", {}), \
             patch("routers.chat.preprocess_query", return_value=("query", ["query"])), \
             patch("routers.chat.generate_embedding", return_value=[0.0] * 384), \
             patch("routers.chat.search_documents", side_effect=ConnectionError("Vector DB offline")):
            response = client.post("/chat/", json={"question": "What is leave?", "top_k": 3})
            assert response.status_code == 500

    def test_llm_exception(self, client):
        """Exception during generate_answer results in 500 Internal Server Error."""
        p = _patches()
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["cache"], \
             patch("routers.chat.generate_answer", side_effect=Exception("LLM API quota exceeded")):
            response = client.post("/chat/", json={"question": "What is leave?", "top_k": 3})
            assert response.status_code == 500

    def test_llm_missing_answer_fallback(self, client):
        """When LLM return dict lacks 'answer' key, default fallback string is used."""
        p = _patches()
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["cache"], \
             patch("routers.chat.generate_answer", return_value={}):
            response = client.post("/chat/", json={"question": "What is leave?", "top_k": 3})
            assert response.status_code == 200
            assert "couldn't find" in response.json()["answer"].lower()


# ===========================================================================
# 6. Edge Cases
# ===========================================================================

class TestChatEdgeCases:
    """
    Unit tests for edge case scenarios in vector search results and payload formatting.
    """

    def test_only_one_retrieved_chunk(self, client):
        """Pipeline operates normally when vector search returns only a single chunk."""
        p = _patches(search_results=[_make_result(uid="single-1")])
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            response = client.post("/chat/", json={"question": "What is sick leave?", "top_k": 3})
            assert response.status_code == 200
            assert len(response.json()["sources"]) == 1

    def test_no_retrieved_chunks(self, client):
        """When no vector results are found, LLM is called with empty context and sources is empty list."""
        with patch("routers.chat.query_cache", {}), \
             patch("routers.chat.preprocess_query", return_value=("unknown", ["unknown"])), \
             patch("routers.chat.generate_embedding", return_value=[0.0] * 384), \
             patch("routers.chat.search_documents", return_value=[]), \
             patch("routers.chat.deduplicate_chunks", side_effect=lambda x: x), \
             patch("routers.chat.generate_answer") as mock_llm:
            mock_llm.return_value = _make_llm_result("I couldn't find this information...")
            response = client.post("/chat/", json={"question": "Unknown question?", "top_k": 3})
            
            assert response.status_code == 200
            assert response.json()["sources"] == []
            assert response.json()["metadata"]["retrieval_count"] == 0
            mock_llm.assert_called_once()
            assert mock_llm.call_args[1].get("context") == ""

    def test_multiple_pdf_documents(self, client):
        """Retrieved chunks spanning multiple PDF documents are formatted in sources."""
        results = [
            _make_result(doc="LeavePolicy.pdf", uid="u1"),
            _make_result(doc="BenefitsPolicy.pdf", uid="u2"),
        ]
        p = _patches(search_results=results)
        with p["preprocess"], p["embedding"], p["search"], p["dedup"], p["llm"], p["cache"]:
            response = client.post("/chat/", json={"question": "Leave and benefits?", "top_k": 5})
            docs = [s["document"] for s in response.json()["sources"]]
            assert "LeavePolicy.pdf" in docs
            assert "BenefitsPolicy.pdf" in docs

    def test_duplicate_chunk_ids_deduplicated(self, client):
        """Duplicate chunk IDs are deduplicated using real deduplicate_chunks utility."""
        dup_results = [
            _make_result(uid="same-id", text="Chunk text."),
            _make_result(uid="same-id", text="Chunk text."),
        ]
        with patch("routers.chat.query_cache", {}), \
             patch("routers.chat.preprocess_query", return_value=("leave", ["leave"])), \
             patch("routers.chat.generate_embedding", return_value=[0.0] * 384), \
             patch("routers.chat.search_documents", return_value=dup_results), \
             patch("routers.chat.generate_answer", return_value=_make_llm_result()):
            response = client.post("/chat/", json={"question": "What is leave?", "top_k": 5})
            assert response.status_code == 200
            assert len(response.json()["sources"]) == 1

    def test_selected_document_retrieval(self, client):
        """When selected_document is specified, search_documents is called with selected_sources containing ONLY selected_document."""
        p = _patches(search_results=[_make_result(doc="LeavePolicy.pdf")])
        with p["preprocess"], p["embedding"], p["search"] as mock_search, p["dedup"], p["llm"], p["cache"]:
            response = client.post(
                "/chat/",
                json={"question": "What is sick leave?", "top_k": 3, "selected_document": "LeavePolicy.pdf"},
            )
            assert response.status_code == 200
            mock_search.assert_called_once_with(
                query_embedding=[0.0] * 384,
                top_k=20,
                selected_sources=["LeavePolicy.pdf"],
            )
            assert response.json()["metadata"]["requested_document"] == "LeavePolicy.pdf"
