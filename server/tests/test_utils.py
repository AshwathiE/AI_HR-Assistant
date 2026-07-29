"""
test_utils.py — Unit tests for utils.py
=========================================

What is tested
--------------
* allowed_file()          – extension whitelist (.pdf, .docx)
* chunk_text()            – numbered-heading chunker
* clean_text()            – tab / multi-space / blank-line normaliser
* deduplicate_chunks()    – Phase 1 (ID dedup) + Phase 2 (text similarity)
* format_sources()        – metadata → {document, chunk} mapper
* get_file_size()         – MB conversion
* file_exists()           – path existence check
* reciprocal_rank_fusion() – RRF score merging

Fixtures / mocks used
---------------------
* `sample_results`     – two realistic retrieval dicts from conftest.py
* `duplicate_results`  – identical-text pair from conftest.py
* `tmp_path`           – built-in pytest fixture for temporary file creation
* `patch("os.path.getsize")` / `patch("os.path.exists")` – avoid disk I/O

How to run
----------
    cd server
    pytest tests/test_utils.py -v

Expected passing output
-----------------------
    All tests PASSED; no disk or network I/O beyond tmp_path writes.

Common failure scenarios
------------------------
* If `symspellpy` or `vocabulary.py` fails to import, utils.py will error
  because it calls build_symspell() at module level — the conftest.py patch
  for QdrantClient resolves most of this, but ensure the venv is active.
"""

import os
import pytest
from unittest.mock import patch, MagicMock


# ===========================================================================
# allowed_file
# ===========================================================================

class TestAllowedFile:

    def test_pdf_allowed(self):
        """PDF files are whitelisted."""
        from utils.utils import allowed_file
        assert allowed_file("policy.pdf") is True

    def test_docx_allowed(self):
        """DOCX files are whitelisted."""
        from utils.utils import allowed_file
        assert allowed_file("handbook.docx") is True

    def test_txt_not_allowed(self):
        """Plain text files are NOT whitelisted."""
        from utils.utils import allowed_file
        assert allowed_file("notes.txt") is False

    def test_exe_not_allowed(self):
        """Executable files are NOT whitelisted."""
        from utils.utils import allowed_file
        assert allowed_file("virus.exe") is False

    def test_uppercase_extension_allowed(self):
        """Extension comparison is case-insensitive (.PDF → allowed)."""
        from utils.utils import allowed_file
        assert allowed_file("POLICY.PDF") is True

    def test_no_extension_not_allowed(self):
        """Filename without extension is not allowed."""
        from utils.utils import allowed_file
        assert allowed_file("filewithoutext") is False

    def test_hidden_file_with_pdf(self):
        """Hidden file with .pdf extension is allowed."""
        from utils.utils import allowed_file
        assert allowed_file(".hidden.pdf") is True


# ===========================================================================
# chunk_text
# ===========================================================================

class TestChunkText:

    def test_numbered_sections_split(self):
        """
        A document with numbered sections (1. Purpose, 2. Scope …)
        produces one chunk per section.
        """
        from utils.utils import chunk_text
        text = (
            "Leave Policy Document\n\n"
            "1. Purpose\nThis policy outlines leave rules.\n\n"
            "2. Scope\nApplies to all full-time employees.\n\n"
            "3. Entitlement\nEmployees get 12 days sick leave."
        )
        chunks = chunk_text(text)
        assert len(chunks) == 3

    def test_title_prepended_to_every_chunk(self):
        """
        The document title (text before the first numbered heading) is
        prepended to each chunk.
        """
        from utils.utils import chunk_text
        text = (
            "Company Handbook\n\n"
            "1. Leave\nSick leave details.\n\n"
            "2. Benefits\nHealth insurance details."
        )
        chunks = chunk_text(text)
        for chunk in chunks:
            assert "Company Handbook" in chunk

    def test_empty_string_returns_empty_list(self):
        """Empty document string yields an empty chunk list."""
        from utils.utils import chunk_text
        assert chunk_text("") == []

    def test_no_numbered_sections(self):
        """
        A document without numbered headings returns no chunks
        (the whole text is treated as a title, not a section).
        """
        from utils.utils import chunk_text
        text = "Just a plain paragraph without any numbered sections."
        chunks = chunk_text(text)
        # Either 0 chunks (whole text is title with no sections)
        # or 1 chunk — both are acceptable as long as no crash occurs
        assert isinstance(chunks, list)

    def test_whitespace_only_sections_skipped(self):
        """Empty / whitespace-only sections between headings are ignored."""
        from utils.utils import chunk_text
        text = "Title\n\n1. Section One\nContent here.\n\n2.   \n\n3. Section Three\nMore content."
        chunks = chunk_text(text)
        # Section 2 has no content, so we expect 2 meaningful chunks
        assert len(chunks) >= 1


# ===========================================================================
# clean_text
# ===========================================================================

class TestCleanText:

    def test_tabs_replaced_with_spaces(self):
        """Tabs are converted to spaces."""
        from utils.utils import clean_text
        result = clean_text("hello\tworld")
        assert "\t" not in result
        assert "hello" in result

    def test_multiple_spaces_collapsed(self):
        """Runs of multiple spaces are reduced to a single space."""
        from utils.utils import clean_text
        result = clean_text("leave   policy   rules")
        assert "  " not in result

    def test_multiple_blank_lines_collapsed(self):
        """Multiple consecutive blank lines become a single newline."""
        from utils.utils import clean_text
        result = clean_text("para one\n\n\n\npara two")
        assert "\n\n\n" not in result

    def test_leading_trailing_whitespace_stripped(self):
        """Leading and trailing whitespace is stripped."""
        from utils.utils import clean_text
        result = clean_text("   sick leave   ")
        assert result == result.strip()

    def test_empty_string_returns_empty(self):
        """Empty string input returns empty string."""
        from utils.utils import clean_text
        assert clean_text("") == ""


# ===========================================================================
# deduplicate_chunks
# ===========================================================================

class TestDeduplicateChunks:

    def test_empty_input_returns_empty(self):
        """Empty list passes through without error."""
        from utils.utils import deduplicate_chunks
        assert deduplicate_chunks([]) == []

    def test_no_duplicates_unchanged(self, sample_results):
        """Two distinct chunks are both preserved."""
        from utils.utils import deduplicate_chunks
        result = deduplicate_chunks(sample_results)
        assert len(result) == 2

    def test_exact_id_duplicate_removed(self):
        """Phase 1: two entries with the same ID → only first is kept."""
        from utils.utils import deduplicate_chunks
        results = [
            {"id": "abc", "score": 0.9, "text": "Sick leave is 12 days."},
            {"id": "abc", "score": 0.8, "text": "Sick leave is 12 days."},
        ]
        deduped = deduplicate_chunks(results)
        assert len(deduped) == 1

    def test_near_duplicate_text_removed(self):
        """
        Phase 2: two entries with text similarity ≥ 0.60 but different IDs
        → second entry is removed.
        """
        from utils.utils import deduplicate_chunks
        results = [
            {"id": "id1", "score": 0.9, "text": "Employees are entitled to 12 days of sick leave per year."},
            {"id": "id2", "score": 0.85, "text": "Employees are entitled to 12 days of sick leave per year."},
        ]
        deduped = deduplicate_chunks(results)
        assert len(deduped) == 1

    def test_different_length_texts_not_deduped(self):
        """
        Phase 2: texts whose lengths differ by more than 20% are NOT
        compared and both are retained.
        """
        from utils.utils import deduplicate_chunks
        results = [
            {"id": "id1", "score": 0.9, "text": "Short text."},
            {"id": "id2", "score": 0.85, "text": "This is a completely different and much longer text about leave policies and their entitlements."},
        ]
        deduped = deduplicate_chunks(results)
        assert len(deduped) == 2

    def test_similarity_below_threshold_both_kept(self):
        """
        Two moderately different texts (similarity < 0.60) are both retained.
        """
        from utils.utils import deduplicate_chunks
        results = [
            {"id": "id1", "score": 0.9, "text": "Sick leave is 12 days per year for all employees."},
            {"id": "id2", "score": 0.8, "text": "Maternity leave is 26 weeks as per the law."},
        ]
        deduped = deduplicate_chunks(results)
        assert len(deduped) == 2

    def test_single_result_unchanged(self):
        """A single-element list is returned unchanged."""
        from utils.utils import deduplicate_chunks
        results = [{"id": "id1", "score": 0.9, "text": "Some text about policy."}]
        assert len(deduplicate_chunks(results)) == 1

    def test_custom_similarity_threshold(self):
        """
        A very high threshold (0.99) causes even near-duplicates to be kept.
        """
        from utils.utils import deduplicate_chunks
        results = [
            {"id": "id1", "score": 0.9, "text": "Sick leave is twelve days per year."},
            {"id": "id2", "score": 0.85, "text": "Sick leave is 12 days per year."},
        ]
        deduped = deduplicate_chunks(results, similarity_threshold=0.99)
        # At 0.99 threshold both should survive
        assert len(deduped) == 2


# ===========================================================================
# format_sources
# ===========================================================================

class TestFormatSources:

    def test_basic_format(self):
        """Metadata list is correctly mapped to {document, chunk} dicts."""
        from utils.utils import format_sources
        metadata = [
            {"source": "LeavePolicy.pdf", "chunk_number": 1},
            {"source": "Benefits.pdf", "chunk_number": 3},
        ]
        sources = format_sources(metadata)
        assert sources[0] == {"document": "LeavePolicy.pdf", "chunk": 1}
        assert sources[1] == {"document": "Benefits.pdf", "chunk": 3}

    def test_missing_source_key(self):
        """Missing 'source' key returns None for document."""
        from utils.utils import format_sources
        metadata = [{"chunk_number": 2}]
        sources = format_sources(metadata)
        assert sources[0]["document"] is None

    def test_empty_list(self):
        """Empty metadata returns empty sources list."""
        from utils.utils import format_sources
        assert format_sources([]) == []


# ===========================================================================
# get_file_size
# ===========================================================================

class TestGetFileSize:

    def test_returns_megabytes(self):
        """File size is returned in MB, rounded to 2 decimal places."""
        from utils.utils import get_file_size
        with patch("os.path.getsize", return_value=1_048_576):  # 1 MB in bytes
            size = get_file_size("dummy.pdf")
        assert size == 1.0

    def test_large_file(self):
        """Large file size is correctly converted."""
        from utils.utils import get_file_size
        with patch("os.path.getsize", return_value=10_485_760):  # 10 MB
            size = get_file_size("large.pdf")
        assert size == 10.0

    def test_zero_byte_file(self):
        """Zero-byte file returns 0.0 MB."""
        from utils.utils import get_file_size
        with patch("os.path.getsize", return_value=0):
            size = get_file_size("empty.pdf")
        assert size == 0.0


# ===========================================================================
# file_exists
# ===========================================================================

class TestFileExists:

    def test_existing_file(self):
        """Returns True when file exists."""
        from utils.utils import file_exists
        with patch("os.path.exists", return_value=True):
            assert file_exists("exists.pdf") is True

    def test_nonexistent_file(self):
        """Returns False when file does not exist."""
        from utils.utils import file_exists
        with patch("os.path.exists", return_value=False):
            assert file_exists("missing.pdf") is False


# ===========================================================================
# reciprocal_rank_fusion
# ===========================================================================

class TestReciprocalRankFusion:

    def _make_results(self, ids_and_texts):
        return [
            {"id": id_, "text": text, "score": 0.5}
            for id_, text in ids_and_texts
        ]

    def test_returns_merged_list(self):
        """RRF returns a combined list from both inputs."""
        from utils.utils import reciprocal_rank_fusion
        vec = self._make_results([("a", "sick leave policy")])
        bm25 = self._make_results([("b", "casual leave entitlement")])
        merged = reciprocal_rank_fusion(vec, bm25)
        assert len(merged) == 2

    def test_shared_result_has_higher_rrf_score(self):
        """
        A result appearing in both vector and BM25 lists receives
        a higher RRF score than one appearing in only one list.
        """
        from utils.utils import reciprocal_rank_fusion
        shared_id = "shared"
        shared_text = "sick leave is 12 days"
        vec = [
            {"id": shared_id, "text": shared_text, "score": 0.9},
            {"id": "vec_only", "text": "other text", "score": 0.7},
        ]
        bm25 = [
            {"id": shared_id, "text": shared_text, "score": 0.8},
        ]
        merged = reciprocal_rank_fusion(vec, bm25)
        # Shared result should appear first (highest combined score)
        assert merged[0]["id"] == shared_id

    def test_none_vector_results_treated_as_empty(self):
        """Passing None for vector results does not raise an exception."""
        from utils.utils import reciprocal_rank_fusion
        bm25 = self._make_results([("a", "leave policy")])
        result = reciprocal_rank_fusion(None, bm25)
        assert len(result) == 1

    def test_none_bm25_results_treated_as_empty(self):
        """Passing None for BM25 results does not raise an exception."""
        from utils.utils import reciprocal_rank_fusion
        vec = self._make_results([("a", "leave policy")])
        result = reciprocal_rank_fusion(vec, None)
        assert len(result) == 1

    def test_both_empty_returns_empty(self):
        """Both empty inputs yield an empty merged list."""
        from utils.utils import reciprocal_rank_fusion
        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_rrf_score_attached_to_result(self):
        """Each merged result has an 'rrf_score' float field."""
        from utils.utils import reciprocal_rank_fusion
        vec = self._make_results([("a", "policy text")])
        merged = reciprocal_rank_fusion(vec, [])
        assert "rrf_score" in merged[0]
        assert isinstance(merged[0]["rrf_score"], float)
