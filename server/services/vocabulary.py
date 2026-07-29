import json
import re
from pathlib import Path

from utils.logger import logger

# Store vocabulary.json in the same folder as this file
###VOCAB_FILE = Path(__file__).parent / "vocabulary.json"

from pathlib import Path
VOCAB_FILE = Path("server/vocabulary.json")
logger.info(f"Vocabulary path: {VOCAB_FILE}")
logger.info(f"Vocabulary exists: {VOCAB_FILE.exists()}")

STOPWORDS = {
    "the", "is", "are", "was", "were",
    "to", "of", "for", "in", "on",
    "a", "an", "and", "or", "by",
    "with", "from", "at", "this",
    "that", "it"
}


def load_vocabulary():
    """
    Load vocabulary from vocabulary.json.
    Returns a set of words.
    """

    if not VOCAB_FILE.exists():
        logger.info("Vocabulary file not found. Starting with an empty vocabulary.")
        return set()

    try:
        with open(VOCAB_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            logger.info("Vocabulary file is empty. Starting with an empty vocabulary.")
            return set()

        vocabulary = set(json.loads(content))

        logger.info(
            f"Loaded vocabulary with {len(vocabulary)} words."
        )

        return vocabulary

    except Exception as e:
        logger.error(
            f"Failed to load vocabulary: {e}"
        )
        return set()


def save_vocabulary(vocabulary):
    """
    Save vocabulary into vocabulary.json.
    """

    try:
        VOCAB_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(VOCAB_FILE, "w", encoding="utf-8") as f:
            json.dump(
                sorted(vocabulary),
                f,
                indent=2
            )

        logger.info(
            f"Saved {len(vocabulary)} words to {VOCAB_FILE}"
        )

    except Exception as e:
        logger.error(
            f"Failed to save vocabulary: {e}"
        )


def update_vocabulary(chunks, rebuild=False):
    """
    Update vocabulary from document chunks.

    Supports:
    1. List[str]
    2. List[dict] where each dict contains a 'text' field.
    """

    logger.info(f"Updating vocabulary from {len(chunks)} chunks (rebuild={rebuild}).")

    if rebuild:
        vocabulary = set()
    else:
        vocabulary = load_vocabulary()

    new_words = 0

    for chunk in chunks:

        # Support metadata dictionaries
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
        elif isinstance(chunk, str):
            text = chunk
        else:
            logger.warning(
                f"Skipping unsupported chunk type: {type(chunk)}"
            )
            continue

        if not text:
            continue

        words = re.findall(
            r"\b[a-zA-Z]+\b",
            text.lower()
        )

        for word in words:

            if (
                len(word) > 2
                and word not in STOPWORDS
                and word not in vocabulary
            ):
                vocabulary.add(word)
                new_words += 1

    logger.info(f"New words added: {new_words}")
    logger.info(f"Vocabulary size before save: {len(vocabulary)}")

    save_vocabulary(vocabulary)

    logger.info(f"Vocabulary saved successfully.")
    logger.info(f"Current vocabulary size: {len(vocabulary)}")

    return vocabulary