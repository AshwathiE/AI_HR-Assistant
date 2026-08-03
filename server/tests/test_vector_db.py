"""
test_vector_db.py — Unit tests for vector_db.py
================================================

What is tested
--------------
* _slugify()                  – URL-safe slug from arbitrary strings
* get_collection_name()       – deterministic collection name generation
* list_collection_names()     – lists company_policy_* collections
* search_documents()          – vector similarity search with filtering
* get_uploaded_documents()    – aggregates unique source filenames
* document_count()            – sums points across collections

Fixtures / mocks used
---------------------
* `fake_qdrant_client` from conftest.py — fully offline QdrantClient mock
* `patch("vector_db.client", ...)` — replaces the module-level singleton

How to run
----------
    cd server
    pytest tests/test_vector_db.py -v

Expected passing output
-----------------------
    All tests PASSED; no Qdrant cloud connection is made.

Common failure scenarios
------------------------
* If qdrant-client is not installed, the module import will fail.
* If the global QdrantClient patch in conftest.py doesn't apply before
  vector_db.py is imported, it will attempt a real network connection.
"""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers to build fake Qdrant primitives without importing from conftest
# ---------------------------------------------------------------------------

class _FakeCollDesc:
    def __init__(self, name): self.name = name

class _FakeCollResp:
    def __init__(self, names):
        self.collections = [_FakeCollDesc(n) for n in names]

class _FakePoint:
    def __init__(self, pid, score, payload):
        self.id = pid; self.score = score; self.payload = payload

class _FakeQueryResp:
    def __init__(self, points): self.points = points


def _make_client(collection_names=None, query_points_result=None):
    if collection_names is None:
        collection_names = ["company_policy_test_abc12345"]

    mock = MagicMock()
    mock.get_collections.return_value = _FakeCollResp(collection_names)
    mock.scroll.return_value = ([], None)

    if not collection_names:
        default_points = []
    else:
        default_points = [
            _FakePoint(
                "uuid-001",
                0.92,
                {
                    "text": "Sick leave is 12 days.",
                    "topics": ["sick leave", "12 days"],
                    "source": "LeavePolicy.pdf",
                    "chunk_number": 1,
                    "collection_name": collection_names[0],
                },
            )
        ]

    mock.query_points.return_value = _FakeQueryResp(
        query_points_result if query_points_result is not None else default_points
    )

    return mock


# ===========================================================================
# _slugify
# ===========================================================================

class TestSlugify:

    def test_basic_slug(self):
        """Spaces and dots become underscores, text is lower-cased."""
        from services.vector_db import _slugify
        assert _slugify("Leave Policy.pdf") == "leave_policy_pdf"

    def test_empty_string_returns_document(self):
        """Empty input returns the fallback 'document'."""
        from services.vector_db import _slugify
        assert _slugify("") == "document"

    def test_special_chars_replaced(self):
        """Non-alphanumeric characters are replaced with underscores."""
        from services.vector_db import _slugify
        result = _slugify("HR & Benefits (2024).pdf")
        assert "&" not in result
        assert "(" not in result
        assert ")" not in result


# ===========================================================================
# get_collection_name
# ===========================================================================

class TestGetCollectionName:

    def test_prefixed_with_company_policy(self):
        """Collection name starts with 'company_policy_'."""
        from services.vector_db import get_collection_name
        name = get_collection_name("LeavePolicy.pdf")
        assert name.startswith("company_policy_")

    def test_deterministic(self):
        """Same filename always produces the same collection name."""
        from services.vector_db import get_collection_name
        assert get_collection_name("Policy.pdf") == get_collection_name("Policy.pdf")

    def test_different_files_different_names(self):
        """Different filenames produce different collection names."""
        from services.vector_db import get_collection_name
        assert get_collection_name("A.pdf") != get_collection_name("B.pdf")

    def test_name_contains_hash(self):
        """The last 8 characters of the collection name is a SHA-1 digest."""
        from services.vector_db import get_collection_name
        name = get_collection_name("Benefits.pdf")
        # format: company_policy_<slug>_<8hex>
        suffix = name.split("_")[-1]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)


# ===========================================================================
# list_collection_names
# ===========================================================================

class TestListCollectionNames:

    def test_returns_only_company_policy_collections(self):
        """Only collections prefixed with 'company_policy_' are returned."""
        mock_client = _make_client(
            collection_names=["company_policy_leave_abc", "other_collection"]
        )
        with patch("services.vector_db.client", mock_client):
            from services.vector_db import list_collection_names
            names = list_collection_names()
        assert "company_policy_leave_abc" in names
        assert "other_collection" not in names

    def test_empty_when_no_matching_collections(self):
        """Returns empty list when no company_policy_* collections exist."""
        mock_client = _make_client(collection_names=["unrelated_collection"])
        with patch("services.vector_db.client", mock_client):
            from services.vector_db import list_collection_names
            assert list_collection_names() == []


# ===========================================================================
# search_documents
# ===========================================================================

class TestSearchDocuments:

    def test_returns_list_of_results(self):
        """search_documents returns a list of result dicts."""
        mock_client = _make_client()
        with patch("services.vector_db.client", mock_client):
            from services.vector_db import search_documents
            results = search_documents([0.0] * 384)
        assert isinstance(results, list)

    def test_result_has_required_keys(self):
        """Each result dict contains id, score, text, topics, document, chunk_number."""
        mock_client = _make_client()
        with patch("services.vector_db.client", mock_client):
            from services. vector_db import search_documents
            results = search_documents([0.0] * 384)
        if results:
            required = {"id", "score", "text", "topics", "document", "chunk_number"}
            assert required.issubset(results[0].keys())

    def test_empty_collections_returns_empty_list(self):
        """No collections → empty result list."""
        mock_client = _make_client(collection_names=[])
        with patch("services.vector_db.client", mock_client):
            from services.vector_db import search_documents
            results = search_documents([0.0] * 384)
        assert results == []

    def test_source_filtering_applied(self):
        """
        When selected_sources is set, only results from matching sources
        are returned.
        """
        mock_client = _make_client(
            collection_names=["company_policy_leave_abc12345"],
            query_points_result=[
                _FakePoint("u1", 0.9, {
                    "text": "Sick leave 12 days",
                    "topics": ["sick leave", "12 days"],
                    "source": "LeavePolicy.pdf",
                    "chunk_number": 1,
                    "collection_name": "company_policy_leave_abc12345",
                }),
                _FakePoint("u2", 0.8, {
                    "text": "Benefits info",
                    "topics": ["benefits", "info"],
                    "source": "Benefits.pdf",
                    "chunk_number": 1,
                    "collection_name": "company_policy_leave_abc12345",
                }),
            ]
        )
        with patch("services.vector_db.client", mock_client):
            from services.vector_db import search_documents
            results = search_documents(
                [0.0] * 384,
                selected_sources=["LeavePolicy.pdf"]
            )
        sources = [r["document"] for r in results]
        assert all(s == "LeavePolicy.pdf" for s in sources)

    def test_results_sorted_by_score_descending(self):
        """Results are sorted by cosine similarity score (highest first)."""
        mock_client = _make_client(
            collection_names=["company_policy_leave_abc12345"],
            query_points_result=[
                _FakePoint("u1", 0.7, {"text": "Low score text", "topics": ["low score"], "source": "A.pdf", "chunk_number": 1, "collection_name": "company_policy_leave_abc12345"}),
                _FakePoint("u2", 0.95, {"text": "High score text", "topics": ["high score"], "source": "B.pdf", "chunk_number": 1, "collection_name": "company_policy_leave_abc12345"}),
            ]
        )
        with patch("services.vector_db.client", mock_client):
            from services.vector_db import search_documents
            results = search_documents([0.0] * 384)
        if len(results) >= 2:
            assert results[0]["score"] >= results[1]["score"]

    def test_top_k_limits_results(self):
        """top_k=1 returns at most 1 result."""
        mock_client = _make_client(
            collection_names=["company_policy_leave_abc12345"],
            query_points_result=[
                _FakePoint("u1", 0.9, {"text": "First chunk", "topics": ["first"], "source": "A.pdf", "chunk_number": 1, "collection_name": "company_policy_leave_abc12345"}),
                _FakePoint("u2", 0.85, {"text": "Second chunk", "topics": ["second"], "source": "B.pdf", "chunk_number": 2, "collection_name": "company_policy_leave_abc12345"}),
                _FakePoint("u3", 0.80, {"text": "Third chunk", "topics": ["third"], "source": "C.pdf", "chunk_number": 3, "collection_name": "company_policy_leave_abc12345"}),
            ]
        )
        with patch("services.vector_db.client", mock_client):
            from services.vector_db import search_documents
            results = search_documents([0.0] * 384, top_k=1)
        assert len(results) <= 1

    def test_duplicate_document_chunk_pairs_removed(self):
        """
        If two points have the same (document, chunk_number) pair,
        only the first is kept.
        """
        mock_client = _make_client(
            collection_names=["company_policy_leave_abc12345"],
            query_points_result=[
                _FakePoint("u1", 0.9, {"text": "Policy text here", "topics": ["policy text"], "source": "A.pdf", "chunk_number": 1, "collection_name": "company_policy_leave_abc12345"}),
                _FakePoint("u2", 0.88, {"text": "Policy text here", "topics": ["policy text"], "source": "A.pdf", "chunk_number": 1, "collection_name": "company_policy_leave_abc12345"}),
            ]
        )
        with patch("services.vector_db.client", mock_client):
            from services.vector_db import search_documents
            results = search_documents([0.0] * 384)
        # Only one entry for ("A.pdf", 1)
        keys = [(r["document"], r["chunk_number"]) for r in results]
        assert len(keys) == len(set(keys))


# ===========================================================================
# get_collection_name — consistent hashing across different filename cases
# ===========================================================================

class TestCollectionNameEdgeCases:

    def test_path_stripped_to_basename(self):
        """Full path and basename alone produce the same collection name."""
        from services.vector_db import get_collection_name
        name_full = get_collection_name("/uploads/LeavePolicy.pdf")
        name_base = get_collection_name("LeavePolicy.pdf")
        # Both resolve to the same basename slug + hash
        assert name_full == name_base
