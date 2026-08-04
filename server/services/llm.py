# llm.py
import json
import os
import re

from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
import google.generativeai as genai
from groq import Groq

from utils.logger import logger

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
else:
    gemini_model = None

# Groq
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def _parse_model_output(output: str) -> dict:
    cleaned = output.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return {
                "answer": payload.get("answer") or payload.get("response") or
                "I couldn't find this information in the uploaded files.",
                "confidence": payload.get("confidence", "medium"),
                "follow_up_question": payload.get("follow_up_question"),
                "sources": payload.get("sources", []),
            }
    except json.JSONDecodeError:
        pass

    return {
        "answer": cleaned or "I couldn't find this information in the uploaded files.",
        "confidence": "medium",
        "follow_up_question": None,
        "sources": [],
    }


def generate_with_groq(prompt: str) -> dict:
    if not groq_client:
        return {
            "answer": "No LLM service is configured.",
            "confidence": "low",
            "follow_up_question": None,
            "sources": [],
        }

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return _parse_model_output(response.choices[0].message.content)


def generate_answer(question: str, context: str, top_k: int) -> dict:
    prompt = f"""
You are an AI Assistant.

Answer the user's question only using the provided context.

The retrieved context may come from one or more different documents.

Rules:

1. Read every retrieved section carefully before answering.
2. Treat each document as an independent source of information.
3. Never infer relationships between different documents unless the user explicitly asks to compare or relate them.
4. If multiple retrieved chunks belong to the same document, combine the information from those chunks into a complete answer.
5. If relevant information exists in multiple documents, present the information separately under the corresponding document or topic instead of merging them into a single statement.
6. Ignore retrieved sections that are unrelated to the user's question.
7. Do not generate information that is not present in the provided context.
8. If the required information is not found in any retrieved section, respond:
   "I couldn't find this information in the uploaded documents."

Response Guidelines:
- Answer directly and clearly.
- Keep information from different documents separate.
- Do not mix facts from unrelated documents.
- Use headings or bullet points when multiple documents contribute relevant information.

Retrieved Context:
{context}

User Question:
{question}
"""

    if gemini_model:
        try:
            response = gemini_model.generate_content(prompt)
            result = _parse_model_output(getattr(response, "text", ""))
            logger.info(f"Context length sent to LLM: {len(context)} characters")
            logger.info("LLM response generated successfully using Gemini")
            return result
        except ResourceExhausted:
            logger.warning("Gemini quota exceeded. Falling back to Groq.")
        except Exception as e:
            logger.error(f"Gemini failed: {e}. Falling back to Groq.")

    try:
        result = generate_with_groq(prompt)
        logger.info("LLM response generated successfully using Groq")
        return result
    except Exception as e:
        return {
            "answer": f"Both Gemini and Groq failed: {e}",
            "confidence": "low",
            "follow_up_question": None,
            "sources": [],
        }