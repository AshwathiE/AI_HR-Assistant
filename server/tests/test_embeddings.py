"""
test_embeddings.py — Unit tests for embeddings.py
==================================================

What is tested
--------------
* generate_embedding()   – single text → 384-dim list[float]
* generate_embeddings()  – batch text list → list[list[float]]

Fixtures / mocks used
---------------------
* The SentenceTransformer model is globally replaced in conftest.py with
  `_FakeSentenceTransformer` which returns deterministic zero-vectors.
  No GPU or network access occurs.

How to run
----------
    cd server
    pytest tests/test_embeddings.py -v

Expected passing output
-----------------------
    All 10 tests PASSED; no network or model I/O.

Common failure scenarios
------------------------
* If sentence_transformers is not installed, all tests in this file will
  fail with ImportError — install via `pip install sentence-transformers`.
* If the global patch in conftest.py fails (e.g. import order issue),
  the real model will attempt to download from HuggingFace.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


# ===========================================================================
# generate_embedding — single text
# ===========================================================================

class TestGenerateEmbedding:

    def test_returns_list(self):
        """generate_embedding must return a plain Python list."""
        # pyrefly: ignore [missing-import]
        from services.embeddings import generate_embedding
        result = generate_embedding("sick leave policy")
        assert isinstance(result, list)

    def test_returns_384_dimensions(self):
        """The embedding vector must be 384-dimensional (all-MiniLM-L6-v2)."""
        from services.embeddings import generate_embedding
        result = generate_embedding("sick leave policy")
        assert len(result) == 384

    def test_elements_are_floats(self):
        """Every element in the embedding vector must be a float."""
        from services.embeddings import generate_embedding
        result = generate_embedding("sick leave")
        assert all(isinstance(v, float) for v in result)

    def test_empty_text_returns_empty_list(self):
        """
        Empty string (or whitespace-only) bypasses the model and returns [].
        This prevents encoding garbage embeddings.
        """
        from services.embeddings import generate_embedding
        result = generate_embedding("")
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        """Whitespace-only input is treated the same as empty string."""
        from services.embeddings import generate_embedding
        result = generate_embedding("   ")
        assert result == []

    def test_long_text_returns_fixed_length(self):
        """Very long text still produces a 384-dim vector (model truncates)."""
        from services.embeddings import generate_embedding
        long_text = "sick leave " * 500
        result = generate_embedding(long_text)
        assert len(result) == 384

    def test_special_characters_processed(self):
        """Text with special characters does not raise; returns a vector."""
        from services.embeddings import generate_embedding
        result = generate_embedding("!@#$% leave policy")
        assert isinstance(result, list)

    def test_non_english_text_processed(self):
        """Non-ASCII input does not raise; the model handles it gracefully."""
        from services.embeddings import generate_embedding
        result = generate_embedding("छुट्टी नीति")
        # The fake model still returns a 384-dim vector
        assert isinstance(result, list)


# ===========================================================================
# generate_embeddings — batch
# ===========================================================================

class TestGenerateEmbeddingsBatch:

    def test_batch_returns_list_of_lists(self):
        """Batch embedding returns list[list[float]]."""
        from services.embeddings import generate_embeddings
        chunks = ["sick leave", "casual leave"]
        result = generate_embeddings(chunks)
        assert isinstance(result, list)
        assert all(isinstance(v, list) for v in result)

    def test_batch_length_matches_input(self):
        """Number of returned embeddings equals number of input chunks."""
        from services.embeddings import generate_embeddings
        chunks = ["leave policy", "work from home", "maternity leave"]
        result = generate_embeddings(chunks)
        assert len(result) == 3

    def test_batch_each_is_384_dims(self):
        """Each embedding in the batch must be 384-dimensional."""
        from services.embeddings import generate_embeddings
        chunks = ["sick leave", "earned leave"]
        result = generate_embeddings(chunks)
        for emb in result:
            assert len(emb) == 384

    def test_empty_chunk_list_returns_empty(self):
        """Empty input list must return empty list without calling the model."""
        from services.embeddings import generate_embeddings
        result = generate_embeddings([])
        assert result == []

    def test_single_chunk_batch(self):
        """A list with one chunk must return a list containing one embedding."""
        from services.embeddings import generate_embeddings
        result = generate_embeddings(["leave policy"])
        assert len(result) == 1
        assert len(result[0]) == 384
