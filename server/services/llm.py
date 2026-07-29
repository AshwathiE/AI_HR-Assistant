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
                "I couldn't find this information in the uploaded company policies.",
                "confidence": payload.get("confidence", "medium"),
                "follow_up_question": payload.get("follow_up_question"),
                "sources": payload.get("sources", []),
            }
    except json.JSONDecodeError:
        pass

    return {
        "answer": cleaned or "I couldn't find this information in the uploaded company policies.",
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
You are an AI Company Policy Assistant.

Answer ONLY from the retrieved context.

The retriever has selected the TOP {top_k} most relevant chunks.
Use information from ALL retrieved chunks whenever required.

If the answer is not available in the context, reply exactly:
"I couldn't find this information in the uploaded company policies."

Return ONLY valid JSON:
{{
 "answer":"",
 "confidence":"high | medium | low",
 "follow_up_question":null,
 "sources":[]
}}

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


def rewrite_query(question: str) -> str:
    prompt = f"""
Rewrite the following HR search query.
1. Correct spelling.
2. Expand abbreviations (SL=sick leave, CL=casual leave, EL=earned leave, PL=paid leave, WFH=work from home, HR=human resources).
3. Add useful HR policy synonyms.
Return only the rewritten query.

User Question:
{question}
"""

    if gemini_model:
        try:
            response = gemini_model.generate_content(prompt)
            text = getattr(response, "text", "").strip()
            if text:
                return text.replace('"', "").replace("'", "").strip()
        except Exception:
            logger.warning("Gemini rewrite failed. Using Groq.")

    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content.strip().replace('"', "").replace("'", "")
        except Exception as e:
            logger.error(f"Groq rewrite failed: {e}")

    return question
