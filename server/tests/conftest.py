"""
conftest.py — Shared pytest fixtures for the AI HR Assistant test suite.

All external dependencies (Qdrant, Gemini, Groq, SentenceTransformer,
CrossEncoder) are mocked here so that every unit test runs fully offline
without any network call, GPU usage, or model download.

Strategy
--------
* sentence_transformers, qdrant_client, groq, google.generativeai and
  google.api_core are injected into sys.modules BEFORE the server
  modules are imported.  This means the production files never import
  the real third-party libraries during testing.
* The SentenceTransformer / CrossEncoder stubs are registered as classes
  inside the fake `sentence_transformers` module so that instantiation
  calls like `SentenceTransformer("all-MiniLM-L6-v2")` work correctly.

Usage
-----
    cd server
    pytest tests/ -v                        # run the full suite
    pytest tests/test_preprocessor.py -v   # run a single module
"""

import sys
import os
import math
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# ① Add the server directory to sys.path so every `from X import Y` inside
#    the production modules resolves correctly when running from tests/.
# ---------------------------------------------------------------------------

SERVER_DIR = Path(__file__).resolve().parent.parent   # …/server/
sys.path.insert(0, str(SERVER_DIR))


# ===========================================================================
# ② Stub heavy third-party packages BEFORE any server module is imported.
#    Insert fake modules directly into sys.modules.
# ===========================================================================

# ---------------------------------------------------------------------------
# sentence_transformers stub
# ---------------------------------------------------------------------------

class _FakeSentenceTransformer:
    """Deterministic 384-dim zero vector; no model weights needed."""

    def __init__(self, *args, **kwargs):
        pass  # ignore model name

    def encode(self, text_or_list, **kwargs):
        if isinstance(text_or_list, list):
            return np.zeros((len(text_or_list), 384), dtype=np.float32)
        return np.zeros(384, dtype=np.float32)


class _FakeCrossEncoder:
    """Returns a fixed logit of 2.0 for every (question, passage) pair."""

    def __init__(self, *args, **kwargs):
        pass  # ignore model name

    def predict(self, pairs, **kwargs):
        return np.array([2.0] * len(pairs), dtype=np.float32)


_fake_st_module = ModuleType("sentence_transformers")
_fake_st_module.SentenceTransformer = _FakeSentenceTransformer
_fake_st_module.CrossEncoder = _FakeCrossEncoder

# Register the stub so any `from sentence_transformers import X` resolves
sys.modules.setdefault("sentence_transformers", _fake_st_module)
sys.modules.setdefault("sentence_transformers.cross_encoder", _fake_st_module)

# ---------------------------------------------------------------------------
# qdrant_client stub
# ---------------------------------------------------------------------------

class _FakeCollectionDescription:
    def __init__(self, name):
        self.name = name


class _FakeCollectionResponse:
    def __init__(self, names):
        self.collections = [_FakeCollectionDescription(n) for n in names]


class _FakeScoredPoint:
    """Simulates a Qdrant ScoredPoint / Record."""
    def __init__(self, point_id, score, payload):
        self.id = point_id
        self.score = score
        self.payload = payload
        self.vector = None   # for restore_document_points tests


class _FakeQueryResponse:
    def __init__(self, points):
        self.points = points


def _make_fake_qdrant_client(collection_names=None, query_points_result=None):
    """
    Build a MagicMock that behaves like QdrantClient.
    Both `collection_names` and `query_points_result` are configurable
    per-test by patching `vector_db.client`.
    """
    if collection_names is None:
        collection_names = ["company_policy_test_abc12345"]

    mock_client = MagicMock()
    mock_client.get_collections.return_value = _FakeCollectionResponse(
        collection_names
    )
    default_points = [
        _FakeScoredPoint(
            "uuid-001",
            0.92,
            {
                "text": "Employees are entitled to 12 days of sick leave per year.",
                "source": "LeavePolicy.pdf",
                "chunk_number": 1,
                "collection_name": collection_names[0],
            },
        )
    ]
    mock_client.query_points.return_value = _FakeQueryResponse(
        query_points_result if query_points_result is not None else default_points
    )
    mock_client.scroll.return_value = ([], None)
    return mock_client


# Build the fake qdrant_client module
_fake_qdrant_module = ModuleType("qdrant_client")
_fake_qdrant_models = ModuleType("qdrant_client.models")

# Provide the symbols imported by vector_db.py
_fake_qdrant_module.QdrantClient = MagicMock(
    return_value=_make_fake_qdrant_client()
)

for _cls_name in ("Distance", "PointIdsList", "PointStruct", "VectorParams"):
    setattr(_fake_qdrant_models, _cls_name, MagicMock())

_fake_qdrant_module.models = _fake_qdrant_models

sys.modules.setdefault("qdrant_client", _fake_qdrant_module)
sys.modules.setdefault("qdrant_client.models", _fake_qdrant_models)

# ---------------------------------------------------------------------------
# google.generativeai / google.api_core stub
# ---------------------------------------------------------------------------

_fake_google = ModuleType("google")
_fake_genai = ModuleType("google.generativeai")
_fake_genai.configure = MagicMock()
_fake_genai.GenerativeModel = MagicMock()

_fake_api_core = ModuleType("google.api_core")
_fake_api_core_exceptions = ModuleType("google.api_core.exceptions")

# Map ResourceExhausted to a real Python exception so it can be caught
class ResourceExhausted(Exception):
    pass

_fake_api_core_exceptions.ResourceExhausted = ResourceExhausted
_fake_api_core.exceptions = _fake_api_core_exceptions

sys.modules.setdefault("google", _fake_google)
sys.modules.setdefault("google.generativeai", _fake_genai)
sys.modules.setdefault("google.api_core", _fake_api_core)
sys.modules.setdefault("google.api_core.exceptions", _fake_api_core_exceptions)

# ---------------------------------------------------------------------------
# groq stub
# ---------------------------------------------------------------------------

_fake_groq_module = ModuleType("groq")
_fake_groq_module.Groq = MagicMock()
sys.modules.setdefault("groq", _fake_groq_module)

# ---------------------------------------------------------------------------
# transformers stub (used in routers/chat.py: "from transformers.models import gpt_sw3")
# ---------------------------------------------------------------------------

_fake_transformers = ModuleType("transformers")
_fake_transformers_models = ModuleType("transformers.models")
_fake_transformers_models.gpt_sw3 = MagicMock()
_fake_transformers.models = _fake_transformers_models

sys.modules.setdefault("transformers", _fake_transformers)
sys.modules.setdefault("transformers.models", _fake_transformers_models)
sys.modules.setdefault("transformers.models.gpt_sw3", MagicMock())

# ---------------------------------------------------------------------------
# symspellpy stub (used in preprocessor.py and utils.py)
# ---------------------------------------------------------------------------

class _FakeVerbosity:
    TOP = "TOP"

class _FakeSymSpell:
    def __init__(self, *args, **kwargs):
        self._dict = {}

    def create_dictionary_entry(self, word, count):
        self._dict[word] = count

    def lookup(self, word, verbosity, max_edit_distance=2):
        # Return empty by default; individual tests override via mock
        return []

_fake_symspell_module = ModuleType("symspellpy")
_fake_symspell_module.SymSpell = _FakeSymSpell
_fake_symspell_module.Verbosity = _FakeVerbosity

sys.modules.setdefault("symspellpy", _fake_symspell_module)



# ===========================================================================
# ③ Shared pytest fixtures
# ===========================================================================

@pytest.fixture
def sample_vocabulary():
    """A small in-memory vocabulary set for preprocessor/utils tests."""
    return {
        "leave", "sick", "casual", "earned", "paid",
        "work", "home", "policy", "employee", "training",
        "salary", "benefits", "hr", "human", "resources",
        "annual", "days", "entitlement", "medical", "maternity",
        "paternity", "notice", "period",
    }


@pytest.fixture
def sample_results():
    """
    Two distinct retrieval result dicts used across dedup, reranker,
    and LLM context assembly tests.
    """
    return [
        {
            "id": "uuid-001",
            "score": 0.92,
            "text": "Employees are entitled to 12 days of sick leave per year.",
            "document": "LeavePolicy.pdf",
            "chunk_number": 1,
            "collection_name": "company_policy_leavepolicy_abc12345",
            "payload": {},
        },
        {
            "id": "uuid-002",
            "score": 0.85,
            "text": "Casual leave is limited to 6 days annually.",
            "document": "LeavePolicy.pdf",
            "chunk_number": 2,
            "collection_name": "company_policy_leavepolicy_abc12345",
            "payload": {},
        },
    ]


@pytest.fixture
def duplicate_results():
    """Two results with identical text and chunk_number (both phases dedup)."""
    return [
        {
            "id": "uuid-001",
            "score": 0.92,
            "text": "Employees are entitled to 12 days of sick leave per year.",
            "document": "LeavePolicy.pdf",
            "chunk_number": 1,
            "collection_name": "company_policy_leavepolicy_abc12345",
        },
        {
            "id": "uuid-002",
            "score": 0.88,
            # Identical text → text-similarity dedup should catch this
            "text": "Employees are entitled to 12 days of sick leave per year.",
            "document": "LeavePolicy.pdf",
            "chunk_number": 1,
            "collection_name": "company_policy_leavepolicy_abc12345",
        },
    ]


@pytest.fixture
def fake_qdrant_client():
    """Return a fresh fake QdrantClient mock."""
    return _make_fake_qdrant_client()


@pytest.fixture
def fake_embedding():
    """384-dimensional zero vector as a plain Python list[float]."""
    return [0.0] * 384


@pytest.fixture
def fake_groq_response():
    """Simulates a successful Groq chat completion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = (
        '{"answer": "Sick leave is 12 days per year.", '
        '"confidence": "high", "follow_up_question": null, "sources": []}'
    )
    return resp


@pytest.fixture
def fake_gemini_response():
    """Simulates a successful Gemini generate_content() response."""
    resp = MagicMock()
    resp.text = (
        '{"answer": "Employees get 12 days sick leave.", '
        '"confidence": "high", "follow_up_question": null, "sources": []}'
    )
    return resp
