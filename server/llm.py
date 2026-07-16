import json
import os
import re

from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
else:
    model = None


def _parse_model_output(output: str) -> dict:  ## converts gemini response into a python dictionary
    cleaned = output.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)

        if isinstance(payload, dict):
            return {
                "answer": payload.get("answer")
                or payload.get("response")
                or "I couldn't find this information in the uploaded company policies.",

                "confidence": payload.get("confidence", "medium"),

                "follow_up_question": payload.get("follow_up_question"),

                "sources": payload.get("sources", []),
            }

    except json.JSONDecodeError:
        pass

    return {
        "answer": cleaned
        or "I couldn't find this information in the uploaded company policies.",
        "confidence": "medium",
        "follow_up_question": None,
        "sources": [],
    }


def generate_answer(question: str, context: str, top_k: int) -> dict:

    if not model:
        return {
            "answer": "The AI service is not configured yet. Please add GEMINI_API_KEY.",
            "confidence": "low",
            "follow_up_question": None,
            "sources": [],
        }

    prompt = f"""
You are an AI Company Policy Assistant.

Answer ONLY from the retrieved context.

The retriever has selected the TOP {top_k} most relevant chunks.
Use information from ALL retrieved chunks whenever required.

If the answer is not available in the context, reply exactly:

"I couldn't find this information in the uploaded company policies."

Return ONLY valid JSON in this format:

{{
    "answer": "...",
    "confidence": "high | medium | low",
    "follow_up_question": null,
    "sources": []
}}

=========================
Retrieved Context
=========================

{context}

=========================
User Question
=========================

{question}

"""

    try:

        response = model.generate_content(prompt)

        text = getattr(response, "text", "")

        return _parse_model_output(text)

    except ResourceExhausted as e:

        print("\n========== GEMINI RESOURCE EXHAUSTED ==========")
        print(e)
        print("===============================================\n")

        return {
            "answer": str(e),
            "confidence": "low",
            "follow_up_question": None,
            "sources": [],
        }

    except Exception as e:

        print("\n========== GEMINI ERROR ==========")
        print(e)
        print("=================================\n")

        return {
            "answer": f"An error occurred while generating the response:\n{str(e)}",
            "confidence": "low",
            "follow_up_question": None,
            "sources": [],
        }

    logger.info(
    f"Context length sent to LLM: {len(context)} characters"
    )
    logger.info(
    "LLM response generated successfully"
    )

from logger import logger

def rewrite_query(question: str) -> str:
    """
    Rewrite the user query using LLM to:
    1. Correct any spelling mistakes.
    2. Expand HR-related abbreviations (e.g. SL -> sick leave, WFH -> work from home, CL -> casual leave).
    3. Capture contextual meaning by appending relevant synonyms or policy concept words.
    """
    if not model:
        logger.warning("LLM model is not configured. Skipping query rewriting.")
        return question

    prompt = f"""
You are a helpful HR Assistant. 
Analyze the user's question for a search query. Rewrite the question to:
1. Correct any misspelled words.
2. Expand all abbreviations (such as 'SL' or 'sl' to 'sick leave', 'WFH' or 'wfh' to 'work from home', 'CL' or 'cl' to 'casual leave', 'EL' or 'el' to 'earned leave', 'PL' or 'pl' to 'paid leave', 'HR' or 'hr' to 'human resources').
3. Keep the original query's core intent but append related contextual search terms, synonyms, or concept names (e.g. if searching for "taking time off for child birth", expand it with "maternity leave paternity leave adoption leave").
4. Return ONLY the rewritten, clean English search query without any explanation, preamble, quotes, or JSON formatting.

User Question: {question}
Rewritten Query:
"""
    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", "").strip()
        if text:
            # Clean up any formatting or markdown in case the model adds quotes
            text = text.replace('"', '').replace("'", "").strip()
            return text
    except Exception as e:
        logger.error(f"Failed to rewrite query using LLM: {e}")
    return question


