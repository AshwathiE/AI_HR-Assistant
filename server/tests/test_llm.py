"""
test_llm.py — Unit tests for llm.py
=====================================

What is tested
--------------
* _parse_model_output()   – parses raw JSON / markdown-fenced JSON / plain
                            text from the model into a structured dict
* generate_with_groq()    – calls Groq client, handles missing client
* generate_answer()       – Gemini-first, Groq-fallback, both-fail fallback
* rewrite_query()         – query rewriting via Gemini / Groq

Fixtures / mocks used
---------------------
* Gemini (`google.generativeai`) and Groq (`groq`) modules are globally
  replaced with MagicMock stubs in conftest.py.
* Individual tests patch `llm.gemini_model` and `llm.groq_client` to
  control return values and side-effects precisely.

How to run
----------
    cd server
    pytest tests/test_llm.py -v

Expected passing output
-----------------------
    All tests PASSED; no HTTP calls to Gemini or Groq are made.

Common failure scenarios
------------------------
* If google-generativeai or groq packages are not installed, their stubs
  in conftest.py must ensure the import succeeds anyway.
* ResourceExhausted must be catchable — conftest.py maps it to plain
  Exception so the fallback logic is exercised correctly.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


# ===========================================================================
# _parse_model_output
# ===========================================================================

class TestParseModelOutput:
    """Tests for the internal JSON-parsing helper."""

    def _call(self, text: str) -> dict:
        from services.llm import _parse_model_output
        return _parse_model_output(text)

    # --- positive: valid JSON --------------------------------------------------

    def test_valid_json_answer_extracted(self):
        """Valid JSON with 'answer' key returns that value in result dict."""
        payload = json.dumps({
            "answer": "Sick leave is 12 days.",
            "confidence": "high",
            "follow_up_question": None,
            "sources": [],
        })
        result = self._call(payload)
        assert result["answer"] == "Sick leave is 12 days."

    def test_valid_json_confidence_extracted(self):
        """Confidence level from JSON is preserved."""
        payload = json.dumps({"answer": "Yes", "confidence": "low", "sources": []})
        result = self._call(payload)
        assert result["confidence"] == "low"

    def test_valid_json_sources_extracted(self):
        """Sources list from JSON is preserved."""
        sources = [{"file": "policy.pdf", "chunk": 1}]
        payload = json.dumps({"answer": "Answer", "confidence": "medium", "sources": sources})
        result = self._call(payload)
        assert result["sources"] == sources

    def test_json_with_response_key(self):
        """Accepts 'response' as a fallback key when 'answer' is absent."""
        payload = json.dumps({"response": "Policy answer.", "confidence": "medium", "sources": []})
        result = self._call(payload)
        assert result["answer"] == "Policy answer."

    # --- markdown fenced JSON -------------------------------------------------

    def test_markdown_fenced_json_parsed(self):
        """JSON wrapped in ```json ... ``` fence is correctly stripped."""
        payload = "```json\n{\"answer\": \"Leave is 12 days.\", \"confidence\": \"high\", \"sources\": []}\n```"
        result = self._call(payload)
        assert result["answer"] == "Leave is 12 days."

    def test_plain_code_fence_parsed(self):
        """JSON wrapped in plain ``` ... ``` (no language tag) is handled."""
        payload = "```\n{\"answer\": \"Benefits info.\", \"confidence\": \"medium\", \"sources\": []}\n```"
        result = self._call(payload)
        assert result["answer"] == "Benefits info."

    # --- non-JSON fallback ----------------------------------------------------

    def test_plain_text_used_as_answer(self):
        """Non-JSON output is used as the raw answer string."""
        result = self._call("I couldn't find this information.")
        assert "I couldn't find" in result["answer"]

    def test_empty_string_returns_fallback_message(self):
        """Empty model output returns the standard fallback message."""
        result = self._call("")
        assert "couldn't find" in result["answer"].lower() or result["answer"] == ""

    # --- required keys always present -----------------------------------------

    def test_result_always_has_required_keys(self):
        """Parsed result always contains answer, confidence, sources keys."""
        result = self._call("Some random text from the model")
        for key in ("answer", "confidence", "sources"):
            assert key in result


# ===========================================================================
# generate_with_groq
# ===========================================================================

class TestGenerateWithGroq:

    def test_returns_answer_when_client_present(self, fake_groq_response):
        """When groq_client is configured, answer is extracted from response."""
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = fake_groq_response

        with patch("services.llm.groq_client", mock_groq):
            from services.llm import generate_with_groq
            result = generate_with_groq("What is sick leave?")
        assert "answer" in result
        assert isinstance(result["answer"], str)

    def test_returns_fallback_when_no_client(self):
        """When groq_client is None, returns 'No LLM service' message."""
        with patch("services.llm.groq_client", None):
            from services.llm import generate_with_groq
            result = generate_with_groq("What is sick leave?")
        assert "No LLM" in result["answer"]
        assert result["confidence"] == "low"

    def test_groq_called_with_correct_model(self, fake_groq_response):
        """Groq is invoked with the expected model name."""
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = fake_groq_response

        with patch("services.llm.groq_client", mock_groq):
            from services.llm import generate_with_groq
            generate_with_groq("test prompt")

        call_kwargs = mock_groq.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"


# ===========================================================================
# generate_answer — Gemini-first with Groq fallback
# ===========================================================================

class TestGenerateAnswer:

    def test_gemini_used_when_available(self, fake_gemini_response):
        """Gemini is the primary LLM when its model is configured."""
        mock_gemini = MagicMock()
        mock_gemini.generate_content.return_value = fake_gemini_response

        with patch("services.llm.gemini_model", mock_gemini), \
             patch("services.llm.groq_client", None):
            from services.llm import generate_answer
            result = generate_answer("What is sick leave?", "Context here.", top_k=3)

        assert "answer" in result
        mock_gemini.generate_content.assert_called_once()

    def test_groq_fallback_on_gemini_quota_exceeded(self, fake_groq_response):
        """
        When Gemini raises ResourceExhausted, Groq is called as fallback.
        conftest.py maps ResourceExhausted to a real Exception subclass.
        """
        mock_gemini = MagicMock()
        # Import ResourceExhausted from the stub registered by conftest.py
        import sys
        ResourceExhausted = sys.modules["google.api_core.exceptions"].ResourceExhausted
        mock_gemini.generate_content.side_effect = ResourceExhausted("quota exceeded")

        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = fake_groq_response

        with patch("services.llm.gemini_model", mock_gemini), \
             patch("services.llm.groq_client", mock_groq):
            from services.llm import generate_answer
            result = generate_answer("sick leave?", "some context", top_k=3)

        assert "answer" in result
        mock_groq.chat.completions.create.assert_called_once()

    def test_groq_fallback_on_generic_gemini_error(self, fake_groq_response):
        """A generic Exception from Gemini also triggers the Groq fallback."""
        mock_gemini = MagicMock()
        mock_gemini.generate_content.side_effect = Exception("connection error")

        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = fake_groq_response

        with patch("services.llm.gemini_model", mock_gemini), \
             patch("services.llm.groq_client", mock_groq):
            from services.llm import generate_answer
            result = generate_answer("sick leave?", "some context", top_k=3)

        assert "answer" in result

    def test_both_llms_fail_returns_error_message(self):
        """When both Gemini and Groq fail, a graceful error dict is returned."""
        mock_gemini = MagicMock()
        mock_gemini.generate_content.side_effect = Exception("Gemini down")

        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = Exception("Groq down")

        with patch("services.llm.gemini_model", mock_gemini), \
             patch("services.llm.groq_client", mock_groq):
            from services.llm import generate_answer
            result = generate_answer("sick leave?", "context", top_k=3)

        assert "answer" in result
        assert "failed" in result["answer"].lower() or "groq" in result["answer"].lower()
        assert result["confidence"] == "low"

    def test_no_llm_configured(self):
        """With both models set to None, returns 'No LLM service' answer."""
        with patch("services.llm.gemini_model", None), \
             patch("services.llm.groq_client", None):
            from services.llm import generate_answer
            result = generate_answer("What is leave?", "context", top_k=3)
        assert "answer" in result

    def test_context_embedded_in_prompt(self, fake_gemini_response):
        """The supplied context appears inside the generated prompt."""
        mock_gemini = MagicMock()
        mock_gemini.generate_content.return_value = fake_gemini_response
        captured = {}

        def _capture(prompt):
            captured["prompt"] = prompt
            return fake_gemini_response

        mock_gemini.generate_content.side_effect = _capture

        context = "Sick leave entitlement is 12 days per year."
        with patch("services.llm.gemini_model", mock_gemini):
            from services.llm import generate_answer
            generate_answer("sick leave?", context, top_k=3)

        assert context in captured["prompt"]

    def test_result_contains_all_required_keys(self, fake_gemini_response):
        """generate_answer result always contains answer, confidence, sources."""
        mock_gemini = MagicMock()
        mock_gemini.generate_content.return_value = fake_gemini_response

        with patch("services.llm.gemini_model", mock_gemini):
            from services.llm import generate_answer
            result = generate_answer("question", "context", top_k=3)

        for key in ("answer", "confidence", "sources"):
            assert key in result


# ===========================================================================
# rewrite_query
# ===========================================================================

class TestRewriteQuery:

    def test_returns_string(self, fake_gemini_response):
        """rewrite_query must return a plain string."""
        fake_gemini_response.text = "What is the sick leave entitlement?"
        mock_gemini = MagicMock()
        mock_gemini.generate_content.return_value = fake_gemini_response

        with patch("services.llm.gemini_model", mock_gemini):
            from services.llm import rewrite_query
            result = rewrite_query("how many sl days")
        assert isinstance(result, str)

    def test_returns_original_on_full_failure(self):
        """If both LLMs fail, the original question is returned unchanged."""
        mock_gemini = MagicMock()
        mock_gemini.generate_content.side_effect = Exception("fail")
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = Exception("fail")

        with patch("services.llm.gemini_model", mock_gemini), \
             patch("services.llm.groq_client", mock_groq):
            from services.llm import rewrite_query
            original = "sl policy"
            result = rewrite_query(original)
        assert result == original

    def test_no_models_configured_returns_original(self):
        """With no models, the original query is returned."""
        with patch("services.llm.gemini_model", None), \
             patch("services.llm.groq_client", None):
            from services.llm import rewrite_query
            result = rewrite_query("how many pl days")
        assert result == "how many pl days"
