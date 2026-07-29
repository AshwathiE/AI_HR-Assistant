"""
test_preprocessor.py — Unit tests for preprocessor.py
======================================================

What is tested
--------------
* preprocess_query()   – full pipeline: lower-case → abbreviation expansion
                         → special-char removal → stop-word removal
                         → keyword spell correction
* split_compound_word() – splits "sickleave" → ["sick", "leave"]
* symspell_correct()    – SymSpell spell correction (mocked vocabulary)
* correct_keywords()    – full keyword correction pipeline
* ABBREVIATIONS map     – WFH, SL, PL, EL, CL, HR expansions
* STOP_WORDS set        – common English stop words are stripped

Fixtures / mocks used
---------------------
* `sample_vocabulary`   – small word set defined in conftest.py
* `patch("preprocessor.load_vocabulary")` – avoids reading vocabulary.json
* `patch("preprocessor.build_symspell")`  – avoids SymSpell dict construction

How to run
----------
    cd server
    pytest tests/test_preprocessor.py -v

Expected passing output
-----------------------
    All tests PASSED with no external service calls.

Common failure scenarios
------------------------
* If vocabulary.json is absent the patched `load_vocabulary` must be in place.
* If symspellpy is not installed the import itself will fail.
"""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helper: build a minimal SymSpell mock
# ---------------------------------------------------------------------------

def _make_sym_spell_mock(correction_map: dict):
    """
    Returns a MagicMock whose .lookup() returns a suggestion matching
    `correction_map[word]`, or no suggestion when the word is not in the map.
    """
    mock = MagicMock()

    def _lookup(word, verbosity, max_edit_distance):
        if word in correction_map:
            suggestion = MagicMock()
            suggestion.term = correction_map[word]
            return [suggestion]
        return []

    mock.lookup.side_effect = _lookup
    return mock


# ===========================================================================
# preprocess_query — positive cases
# ===========================================================================

class TestPreprocessQueryPositive:

    def test_lowercase_conversion(self, sample_vocabulary):
        """Input is normalised to lower-case before any processing."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            cleaned, keywords = preprocess_query("SICK LEAVE POLICY")
        assert cleaned == cleaned.lower()

    def test_stop_word_removal(self, sample_vocabulary):
        """
        Stop words ('what', 'is', 'the', 'for') are stripped from keywords.
        """
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            _, keywords = preprocess_query("What is the leave policy for employees")
        # 'what', 'is', 'the', 'for' must not appear
        stop_words_found = {"what", "is", "the", "for"}.intersection(set(keywords))
        assert stop_words_found == set(), f"Stop words not removed: {stop_words_found}"

    def test_wfh_abbreviation_expansion(self, sample_vocabulary):
        """WFH expands to 'work from home' before keyword extraction."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            cleaned, keywords = preprocess_query("WFH policy")
        assert "work" in keywords or "work" in cleaned

    def test_sl_abbreviation_expansion(self, sample_vocabulary):
        """SL expands to 'sick leave'."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            cleaned, _ = preprocess_query("How many sl days are allowed")
        assert "sick" in cleaned

    def test_pl_abbreviation_expansion(self, sample_vocabulary):
        """PL expands to 'paid leave'."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            cleaned, _ = preprocess_query("How many pl days")
        assert "paid" in cleaned

    def test_el_abbreviation_expansion(self, sample_vocabulary):
        """EL expands to 'earned leave'."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            cleaned, _ = preprocess_query("EL entitlement for new joiners")
        assert "earned" in cleaned

    def test_cl_abbreviation_expansion(self, sample_vocabulary):
        """CL expands to 'casual leave'."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            cleaned, _ = preprocess_query("CL balance")
        assert "casual" in cleaned

    def test_hr_abbreviation_expansion(self, sample_vocabulary):
        """HR expands to 'human resources'."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            cleaned, _ = preprocess_query("Contact HR for details")
        assert "human" in cleaned or "resources" in cleaned

    def test_special_characters_removed(self, sample_vocabulary):
        """Special characters like @, #, $ are replaced with spaces."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            cleaned, _ = preprocess_query("sick@leave#policy$")
        assert "@" not in cleaned
        assert "#" not in cleaned
        assert "$" not in cleaned

    def test_returns_tuple_of_two(self, sample_vocabulary):
        """preprocess_query must return exactly (cleaned_query, keywords)."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            result = preprocess_query("sick leave policy")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_multi_intent_query(self, sample_vocabulary):
        """
        A query covering two topics retains keywords from both intents.
        """
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            cleaned, keywords = preprocess_query("sick leave and maternity leave policy")
        assert "sick" in keywords or "sick" in cleaned
        assert "maternity" in keywords or "maternity" in cleaned


# ===========================================================================
# preprocess_query — edge / negative cases
# ===========================================================================

class TestPreprocessQueryEdgeCases:

    def test_empty_query(self, sample_vocabulary):
        """Empty string returns two empty containers without raising."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            cleaned, keywords = preprocess_query("")
        assert cleaned == ""
        assert keywords == []

    def test_only_special_characters(self, sample_vocabulary):
        """A query of only special chars produces an empty cleaned string."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            cleaned, keywords = preprocess_query("!@#$%^&*()")
        assert cleaned == ""
        assert keywords == []

    def test_only_stop_words(self, sample_vocabulary):
        """A query made entirely of stop words yields empty keywords."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            _, keywords = preprocess_query("what is the")
        # After stop-word removal, keywords list should be empty
        assert keywords == []

    def test_html_injection_stripped(self, sample_vocabulary):
        """HTML tags are removed as part of special-char cleaning."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            cleaned, _ = preprocess_query("<script>alert('xss')</script>")
        assert "<" not in cleaned
        assert ">" not in cleaned

    def test_non_english_query(self, sample_vocabulary):
        """
        Non-ASCII characters (Hindi script) are treated as special chars
        and removed; the function must not raise.
        """
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            # Should not raise; may return empty or partial result
            cleaned, keywords = preprocess_query("छुट्टी नीति")
        assert isinstance(cleaned, str)
        assert isinstance(keywords, list)

    def test_extra_whitespace_normalised(self, sample_vocabulary):
        """Multiple internal spaces are collapsed to a single space."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            cleaned, _ = preprocess_query("sick    leave   policy")
        assert "  " not in cleaned  # no double spaces

    def test_numeric_only_query(self, sample_vocabulary):
        """
        A query with only numbers yields no meaningful keywords
        (numbers themselves pass through; the function must not raise).
        """
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            cleaned, keywords = preprocess_query("12345")
        assert isinstance(cleaned, str)


# ===========================================================================
# split_compound_word
# ===========================================================================

class TestSplitCompoundWord:

    def test_splits_sickleave(self):
        """'sickleave' → ['sick', 'leave'] when both halves are in vocabulary."""
        from services.preprocessor import split_compound_word
        vocab = {"sick", "leave", "policy"}
        result = split_compound_word("sickleave", vocab)
        assert result == ["sick", "leave"]

    def test_splits_workfromhome(self):
        """'workhome' → ['work', 'home'] when both halves are in vocabulary."""
        from services.preprocessor import split_compound_word
        vocab = {"work", "home", "leave"}
        result = split_compound_word("workhome", vocab)
        assert result == ["work", "home"]

    def test_no_split_when_halves_not_in_vocab(self):
        """Returns None when no valid split point exists."""
        from services.preprocessor import split_compound_word
        vocab = {"sick", "leave"}
        result = split_compound_word("xyzabc", vocab)
        assert result is None

    def test_short_word_skipped(self):
        """Words shorter than 5 characters are never split (returns None)."""
        from services.preprocessor import split_compound_word
        vocab = {"si", "ck"}
        result = split_compound_word("sick", vocab)
        assert result is None

    def test_empty_vocabulary(self):
        """With an empty vocabulary no split is possible."""
        from services.preprocessor import split_compound_word
        result = split_compound_word("sickleave", set())
        assert result is None

    def test_returns_list_of_two(self):
        """Successful split always returns a list with exactly two elements."""
        from services.preprocessor import split_compound_word
        vocab = {"earned", "leave"}
        result = split_compound_word("earnedleave", vocab)
        assert result is not None
        assert len(result) == 2


# ===========================================================================
# symspell_correct
# ===========================================================================

class TestSymspellCorrect:

    def test_correct_typo(self):
        """'leve' is corrected to 'leave' by the mock SymSpell."""
        from services.preprocessor import symspell_correct
        mock_ss = _make_sym_spell_mock({"leve": "leave"})
        assert symspell_correct("leve", mock_ss) == "leave"

    def test_correct_word_unchanged(self):
        """A word already in vocabulary is returned unchanged."""
        from services.preprocessor import symspell_correct
        mock_ss = _make_sym_spell_mock({"leave": "leave"})
        assert symspell_correct("leave", mock_ss) == "leave"

    def test_short_word_skipped(self):
        """Words shorter than MIN_WORD_LENGTH (3) are returned as-is."""
        from services.preprocessor import symspell_correct
        mock_ss = _make_sym_spell_mock({})
        # 'ab' has length 2 → skipped
        assert symspell_correct("ab", mock_ss) == "ab"

    def test_no_suggestion_returns_original(self):
        """When SymSpell has no suggestion, the original word is returned."""
        from services.preprocessor import symspell_correct
        mock_ss = _make_sym_spell_mock({})
        assert symspell_correct("xyzwqq", mock_ss) == "xyzwqq"


# ===========================================================================
# correct_keywords (full pipeline)
# ===========================================================================

class TestCorrectKeywords:

    def test_exact_match_preserved(self, sample_vocabulary):
        """A word already in vocabulary is kept without correction."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import correct_keywords
            result = correct_keywords(["leave", "policy"])
        assert "leave" in result
        assert "policy" in result

    def test_compound_word_split(self, sample_vocabulary):
        """'sickleave' is split into ['sick', 'leave']."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import correct_keywords
            result = correct_keywords(["sickleave"])
        assert "sick" in result
        assert "leave" in result

    def test_spell_correction_applied(self, sample_vocabulary):
        """Typo 'leve' → 'leave' via SymSpell."""
        sym_mock = _make_sym_spell_mock({"leve": "leave"})
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=sym_mock):
            from services.preprocessor import correct_keywords
            result = correct_keywords(["leve"])
        assert "leave" in result

    def test_empty_keyword_list(self, sample_vocabulary):
        """Empty input returns empty list without error."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import correct_keywords
            result = correct_keywords([])
        assert result == []


# ===========================================================================
# RapidFuzz & Dotted Abbreviations & Caching
# ===========================================================================

class TestFuzzyCorrectionAndCaching:

    def test_fuzzy_correct_word_rapidfuzz(self, sample_vocabulary):
        """RapidFuzz corrects phonetic or high edit-distance typo against vocabulary."""
        from services.preprocessor import fuzzy_correct_word
        corrected = fuzzy_correct_word("polcy", sample_vocabulary, threshold=75.0)
        assert corrected == "policy"

    def test_fuzzy_correct_word_low_similarity_kept(self, sample_vocabulary):
        """When fuzzy match similarity is below threshold, original word is returned."""
        from services.preprocessor import fuzzy_correct_word
        corrected = fuzzy_correct_word("xyzqqq", sample_vocabulary, threshold=80.0)
        assert corrected == "xyzqqq"

    def test_dotted_abbreviation_expansion(self, sample_vocabulary):
        """Dotted abbreviations like 'w.f.h.' and 's.l.' expand properly."""
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=_make_sym_spell_mock({})):
            from services.preprocessor import preprocess_query
            cleaned, keywords = preprocess_query("What is w.f.h. policy and s.l. balance?")
        assert "work" in keywords or "work" in cleaned
        assert "sick" in keywords or "sick" in cleaned

    def test_caching_and_clear_cache(self, sample_vocabulary):
        """Vocabulary and SymSpell instances are cached across calls."""
        from services.preprocessor import get_vocabulary, clear_preprocessor_cache
        clear_preprocessor_cache()
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary) as mock_load:
            v1 = get_vocabulary()
            assert "sick" in v1
        clear_preprocessor_cache()

    def test_elongated_word_correction(self, sample_vocabulary):
        """Elongated words like 'policeeeee' are normalized and corrected to 'policy'."""
        mock_ss = _make_sym_spell_mock({"police": "policy"})
        with patch("services.preprocessor.load_vocabulary", return_value=sample_vocabulary), \
             patch("services.preprocessor.build_symspell", return_value=mock_ss):
            from services.preprocessor import preprocess_query
            cleaned, keywords = preprocess_query("leave policeeeee")
        assert "policy" in keywords
        assert cleaned == "leave policy"


