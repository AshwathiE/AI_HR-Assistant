import re
from logger import logger
from vocabulary import load_vocabulary
from rapidfuzz import process, fuzz ## weighted fuzzy logic based on string weights similarity is made
STOP_WORDS = {
    "a", "an", "the",
    "is", "am", "are", "was", "were",
    "be", "been", "being",
    "do", "does", "did",
    "have", "has", "had",
    "can", "could", "will", "would",
    "shall", "should", "may", "might",
    "must",
    "i", "me", "my",
    "you", "your",
    "he", "him", "his",
    "she", "her",
    "it", "its",
    "we", "our",
    "they", "them",
    "what", "when",
    "where", "which",
    "who", "why", "how",
    "of", "to",
    "for", "in",
    "on", "at",
    "by", "with",
    "from",
    "and", "or",
    "but",
    "because",
    "this", "that",
    "these", "those",
    "there", "here",
    "please", "kindly"
}


# Fuzzy matching thresholds
HIGH_CONFIDENCE_THRESHOLD = 80   #similarity is 80 or above Direct replacement
MODERATE_CONFIDENCE_THRESHOLD = 65  # Include both original + corrected (query expansion)
MIN_WORD_LENGTH = 3  # Skip very short words to avoid false positives

ABBREVIATIONS = {
    "wfh": "work from home",
    "w.f.h.": "work from home",
    "sl": "sick leave",
    "s.l.": "sick leave",
    "cl": "casual leave",
    "c.l.": "casual leave",
    "el": "earned leave",
    "e.l.": "earned leave",
    "pl": "paid leave",
    "p.l.": "paid leave",
    "hr": "human resources",
    "h.r.": "human resources",
}

def split_compound_word(word, vocabulary):
    """
    Attempt to split a concatenated word into two known vocabulary words.
    e.g., "sickleave" -> ["sick", "leave"]
    """

    if len(word) < 5:
        return None

    for i in range(3, len(word) - 2): ## left word must contains atleast three words

        left = word[:i]
        right = word[i:]

        if left in vocabulary and right in vocabulary:

            logger.info(
                f"Compound split: {word} -> {left} + {right}"
            )

            return [left, right]

    return None

def fuzzy_correct_keywords(keywords):

    vocabulary = list(load_vocabulary())
    vocabulary_set = set(vocabulary)

    if not vocabulary:
        logger.warning("Vocabulary is empty.")
        return keywords

    corrected_words = []

    for word in keywords:

        if len(word) < MIN_WORD_LENGTH: ## less than  minimum word skips fuzzy matching
            corrected_words.append(word)
            continue

        if word in vocabulary_set:
            logger.info(f"Exact match: '{word}'")
            corrected_words.append(word)
            continue

        match = process.extractOne(  #t compares the user's word against every word in the vocabulary
            word,
            vocabulary,
            scorer=fuzz.WRatio
        )

        if match:

            matched_word, score, _ = match

            logger.info(
                f"Fuzzy candidate: '{word}' -> '{matched_word}' (score={score:.2f})"
            )

            if score >= HIGH_CONFIDENCE_THRESHOLD:

                logger.info(
                    f"✓ Fuzzy correction applied: '{word}' -> '{matched_word}'"
                )

                corrected_words.append(matched_word)

            elif score >= MODERATE_CONFIDENCE_THRESHOLD:

                logger.info(
                    f"✓ Query expansion applied: '{word}' -> ['{word}', '{matched_word}']"
                )

                corrected_words.append(word)
                corrected_words.append(matched_word)

            else:

                logger.info(
                    f"✗ No correction applied for '{word}' (score={score:.2f})"
                )

                corrected_words.append(word)

        else:

            logger.info(
                f"✗ No fuzzy match found for '{word}'"
            )

            corrected_words.append(word)

    logger.info(
        f"Final corrected keywords: {corrected_words}"
    )

    return corrected_words


def preprocess_query(query):
    """
    Clean and preprocess user query
    """

    # lowercase
    query = query.lower()

    # Expand abbreviations
    for abbr, expansion in ABBREVIATIONS.items():
        query = re.sub(rf"\b{abbr}\b", expansion, query)


    # remove special characters
    query = re.sub(
        r'[^a-zA-Z0-9\s]',
        ' ',
        query
    )


    # remove extra spaces
    query = re.sub(
        r'\s+',
        ' ',
        query
    ).strip()


    # split words
    words = query.split()


    # remove stopwords
    keywords = [
        word
        for word in words
        if word not in STOP_WORDS
    ]


    # fuzzy correction
    keywords = fuzzy_correct_keywords(
        keywords
    )

    vocabulary = list(load_vocabulary())

    logger.info(f"Vocabulary: {vocabulary}")

    logger.info(f"Vocabulary size: {len(vocabulary)}")
    logger.info(f"'policy' exists: {'policy' in vocabulary}")
    logger.info(f"'leave' exists: {'leave' in vocabulary}") 


    cleaned_query = " ".join(
        keywords
    )


    logger.info(
        f"Original Query: {query}"
    )

    logger.info(
        f"Processed Query: {cleaned_query}"
    )

    logger.info("Keywords: %s", keywords)


    return cleaned_query, keywords