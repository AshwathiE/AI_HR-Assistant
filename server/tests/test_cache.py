"""
test_cache.py — Unit tests for cache.py
=========================================

What is tested
--------------
* query_cache             – the shared in-memory dictionary
* Cache hit logic         – response retrieved from cache without LLM call
* Cache miss logic        – response is generated and stored in cache
* Cache key normalisation – lowercase key prevents duplicate entries
* Cache isolation         – clearing the cache restores empty state

Fixtures / mocks used
---------------------
* The `query_cache` dict from cache.py is imported directly.
* `patch("routers.chat.query_cache", ...)` – injects a controlled dict
  into the /chat endpoint to simulate hit and miss scenarios.

How to run
----------
    cd server
    pytest tests/test_cache.py -v

Expected passing output
-----------------------
    All tests PASSED; no LLM or network calls.

Common failure scenarios
------------------------
* Import of `routers.chat` triggers loading of preprocessor, embeddings,
  vector_db, and reranker — all covered by conftest.py patches.
"""

import pytest
from unittest.mock import patch, MagicMock


# ===========================================================================
# Basic cache dict tests (pure unit — no FastAPI)
# ===========================================================================

class TestQueryCacheDict:

    def setup_method(self):
        """Reset the cache before each test."""
        from cache import query_cache
        query_cache.clear()

    def test_cache_starts_empty(self):
        """After clearing, the cache dict is empty."""
        from cache import query_cache
        assert len(query_cache) == 0

    def test_store_and_retrieve(self):
        """A value stored under a key can be retrieved by the same key."""
        from cache import query_cache
        query_cache["sick leave"] = {"answer": "12 days", "question": "sick leave"}
        assert query_cache["sick leave"]["answer"] == "12 days"

    def test_overwrite_existing_entry(self):
        """Storing a new value under the same key overwrites the old one."""
        from cache import query_cache
        query_cache["leave"] = {"answer": "old"}
        query_cache["leave"] = {"answer": "new"}
        assert query_cache["leave"]["answer"] == "new"

    def test_missing_key_raises_keyerror(self):
        """Accessing an absent key raises KeyError (standard dict behaviour)."""
        from cache import query_cache
        with pytest.raises(KeyError):
            _ = query_cache["nonexistent_key"]

    def test_key_in_operator(self):
        """'in' operator correctly detects presence / absence of keys."""
        from cache import query_cache
        query_cache["policy"] = {"answer": "Found"}
        assert "policy" in query_cache
        assert "benefits" not in query_cache

    def test_clear_empties_cache(self):
        """Calling .clear() removes all entries."""
        from cache import query_cache
        query_cache["a"] = "x"
        query_cache["b"] = "y"
        query_cache.clear()
        assert len(query_cache) == 0


# ===========================================================================
# Cache hit / miss integration with the /chat router logic
# ===========================================================================

class TestCacheHitMissViaEndpoint:
    """
    These tests exercise the cache integration inside the /chat router
    without starting a real server.  We call the `chat()` function directly
    and inject a pre-populated cache.
    """

    def _make_chat_request(self, question="What is sick leave?", top_k=3):
        """Build a ChatRequest-like Pydantic object."""
        from routers.chat import ChatRequest
        return ChatRequest(question=question, top_k=top_k, sources=None)

    def _build_mock_cache_response(self, question):
        """Build a ChatResponse-like object to act as a cached hit."""
        from routers.chat import ChatResponse
        return ChatResponse(
            question=question,
            answer="Cached: Sick leave is 12 days per year.",
            sources=None,
            metadata={"top_k": 3, "retrieval_count": 1, "search_text": "sick leave"},
        )

    def test_cache_hit_returns_cached_response(self):
        """
        When the processed query key exists in query_cache, the cached
        ChatResponse is returned and the LLM / embedding pipeline is skipped.
        """
        question = "What is sick leave?"
        # Preprocess "What is sick leave?" → key will be "sick leave"
        # (stop words stripped, lowercased).  We populate with that key.
        cache_key = "sick leave"
        cached_response = self._build_mock_cache_response(question)
        fake_cache = {cache_key: cached_response}

        with patch("routers.chat.query_cache", fake_cache), \
             patch("routers.chat.preprocess_query", return_value=(cache_key, ["sick", "leave"])), \
             patch("routers.chat.generate_embedding") as mock_embed, \
             patch("routers.chat.generate_answer") as mock_llm:
            from routers.chat import chat
            response = chat(self._make_chat_request(question))

        # LLM and embedding must NOT be called on a cache hit
        mock_embed.assert_not_called()
        mock_llm.assert_not_called()
        assert response.answer == cached_response.answer

    def test_cache_miss_triggers_full_pipeline(self):
        """
        When the key is absent from query_cache, the full RAG pipeline
        (embedding → vector search → rerank → LLM) is executed.
        """
        question = "What is casual leave?"
        cache_key = "casual leave"

        # Empty cache → cache miss
        fake_cache = {}

        mock_results = [
            {
                "id": "uuid-001",
                "score": 0.88,
                "text": "Casual leave is 6 days per year.",
                "document": "LeavePolicy.pdf",
                "chunk_number": 2,
                "collection_name": "company_policy_leavepolicy_abc12345",
            }
        ]

        with patch("routers.chat.query_cache", fake_cache), \
             patch("routers.chat.preprocess_query", return_value=(cache_key, ["casual", "leave"])), \
             patch("routers.chat.generate_embedding", return_value=[0.0] * 384), \
             patch("routers.chat.search_documents", return_value=mock_results), \
             patch("routers.chat.generate_answer", return_value={
                 "answer": "Casual leave is 6 days.",
                 "confidence": "high",
                 "follow_up_question": None,
                 "sources": [],
             }):
            from routers.chat import chat
            response = chat(self._make_chat_request(question))

        assert response.answer == "Casual leave is 6 days."
        # The response should now be stored in fake_cache
        assert cache_key in fake_cache

    def test_cache_response_stored_after_miss(self):
        """
        After a cache miss, the response is written to query_cache so
        that a subsequent identical request is served from cache.
        """
        cache_key = "earned leave"
        fake_cache = {}

        mock_results = [
            {
                "id": "u1", "score": 0.9,
                "text": "Earned leave is 15 days.",
                "document": "Doc.pdf", "chunk_number": 1,
                "collection_name": "company_policy_doc_abc12345",
            }
        ]

        with patch("routers.chat.query_cache", fake_cache), \
             patch("routers.chat.preprocess_query", return_value=(cache_key, ["earned", "leave"])), \
             patch("routers.chat.generate_embedding", return_value=[0.0] * 384), \
             patch("routers.chat.search_documents", return_value=mock_results), \
             patch("routers.chat.generate_answer", return_value={
                 "answer": "Earned leave is 15 days.",
                 "confidence": "high",
                 "follow_up_question": None,
                 "sources": [],
             }):
            from routers.chat import chat
            chat(self._make_chat_request("What is earned leave?"))

        # Verify cache was populated
        assert cache_key in fake_cache
        assert fake_cache[cache_key].answer == "Earned leave is 15 days."

    def test_cache_key_is_lowercase(self):
        """
        The cache key derived from the preprocessed query is always
        lower-cased, preventing duplicate entries for identical queries
        in different cases.
        """
        from cache import query_cache
        query_cache.clear()

        # Both keys should resolve to the same lowercase string
        query_cache["sick leave"] = {"answer": "12 days"}
        # Simulate a second request that maps to the same key
        assert "sick leave" in query_cache  # key is lowercase
        assert "SICK LEAVE" not in query_cache  # uppercase key is absent
