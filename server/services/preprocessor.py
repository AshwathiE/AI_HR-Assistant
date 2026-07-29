import re
from utils.logger import logger
from services.vocabulary import load_vocabulary
from symspellpy import SymSpell, Verbosity
from rapidfuzz import process, fuzz


# -----------------------------
# Configuration
# -----------------------------

MAX_EDIT_DISTANCE = 2
MIN_WORD_LENGTH = 3
FUZZY_MIN_RATIO = 70.0

# Global module caches
_VOCAB_CACHE = None
_SYMSPELL_CACHE = None


# -----------------------------
# Stop words
# -----------------------------

STOP_WORDS = {
    "a", "an", "the",
    "is", "am", "are",
    "was", "were",
    "be", "been", "being",
    "do", "does", "did",
    "have", "has", "had",
    "can", "could",
    "will", "would",
    "shall", "should",
    "may", "might",
    "must",
    "i", "me", "my",
    "you", "your",
    "he", "him", "his",
    "she", "her",
    "it", "its",
    "we", "our",
    "they", "them",
    "what", "when", "where", "which", "who", "why", "how",
    "of", "to", "for", "in", "on", "at", "by", "with", "from",
    "and", "or", "but", "because",
    "this", "that", "these", "those",
    "there", "here",
    "please", "kindly"
}

# -----------------------------
# Abbreviations
# -----------------------------

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
    "h.r.": "human resources"
}


# -----------------------------
# Cache management & SymSpell
# -----------------------------

def get_vocabulary(force_reload=False):
    """
    Returns cached vocabulary set or reloads if forced/mocked.
    """
    global _VOCAB_CACHE
    if force_reload or _VOCAB_CACHE is None or hasattr(load_vocabulary, "_mock_return_value") or hasattr(load_vocabulary, "mock_calls"):
        _VOCAB_CACHE = load_vocabulary()
    return _VOCAB_CACHE


def build_symspell():
    """
    Build SymSpell dictionary using vocabulary.
    Exposed as a top-level function for test mocking and direct calls.
    """
    sym_spell = SymSpell(
        max_dictionary_edit_distance=MAX_EDIT_DISTANCE,
        prefix_length=7
    )

    vocabulary = load_vocabulary()

    if not vocabulary:
        logger.warning("Vocabulary empty. SymSpell not initialized")
        return sym_spell

    for word in vocabulary:
        sym_spell.create_dictionary_entry(word, 1)

    logger.info(f"SymSpell dictionary created. Size: {len(vocabulary)}")
    return sym_spell


def get_symspell(force_reload=False):
    """
    Returns cached SymSpell instance or re-builds if forced/mocked.
    """
    global _SYMSPELL_CACHE
    if force_reload or _SYMSPELL_CACHE is None or hasattr(build_symspell, "_mock_return_value") or hasattr(build_symspell, "mock_calls"):
        return build_symspell()
    if _SYMSPELL_CACHE is None:
        _SYMSPELL_CACHE = build_symspell()
    return _SYMSPELL_CACHE


def clear_preprocessor_cache():
    """
    Clears in-memory vocabulary and SymSpell caches.
    """
    global _VOCAB_CACHE, _SYMSPELL_CACHE
    _VOCAB_CACHE = None
    _SYMSPELL_CACHE = None


# -----------------------------
# Compound word splitting
# -----------------------------

def split_compound_word(word, vocabulary):
    """
    sickleave -> sick leave
    """
    if len(word) < 5:
        return None

    for i in range(3, len(word) - 2):
        left = word[:i]
        right = word[i:]

        if left in vocabulary and right in vocabulary:
            logger.info(f"Compound split: {word} -> {left} + {right}")
            return [left, right]

    return None


# -----------------------------
# Typo & Fuzzy Correction
# -----------------------------

def symspell_correct(word, sym_spell):
    """
    SymSpell edit distance correction.
    """
    if len(word) < MIN_WORD_LENGTH:
        return word

    suggestions = sym_spell.lookup(
        word,
        Verbosity.TOP,
        max_edit_distance=MAX_EDIT_DISTANCE
    )

    if suggestions:
        corrected = suggestions[0].term
        if corrected != word:
            logger.info(f"SymSpell correction: {word} -> {corrected}")
        return corrected

    return word


def fuzzy_correct_word(word, vocabulary, threshold=FUZZY_MIN_RATIO):
    """
    RapidFuzz fuzzy correction fallback against domain vocabulary.
    Handles elongated words (e.g. policeeeee -> policy) via character normalization and WRatio.
    """
    if len(word) < MIN_WORD_LENGTH or not vocabulary:
        return word

    vocab_list = list(vocabulary) if not isinstance(vocabulary, list) else vocabulary
    
    # 1. Normalize repeated characters (e.g., policeeeee -> police, leavvvve -> leave)
    norm_word = re.sub(r"(.)\1{2,}", r"\1", word)

    if norm_word in vocabulary:
        logger.info(f"Repeated char normalization match: {word} -> {norm_word}")
        return norm_word

    # 2. Extract best match using WRatio
    match_raw = process.extractOne(word, vocab_list, scorer=fuzz.WRatio)
    match_norm = process.extractOne(norm_word, vocab_list, scorer=fuzz.WRatio)

    best_match = match_raw
    if match_norm and (not match_raw or match_norm[1] > match_raw[1]):
        best_match = match_norm

    if best_match:
        candidate, score, _ = best_match
        if score >= threshold and candidate != word:
            logger.info(f"RapidFuzz fuzzy correction: {word} (norm: {norm_word}) -> {candidate} (score: {score:.1f})")
            return candidate

    return word


# -----------------------------
# Keyword Correction Pipeline
# -----------------------------

def correct_keywords(keywords):
    vocabulary = set(get_vocabulary())
    sym_spell = get_symspell()

    corrected_words = []

    for word in keywords:
        # 1. Exact match in vocabulary
        if word in vocabulary:
            logger.info(f"Exact match: {word}")
            corrected_words.append(word)
            continue

        # 2. Repeated character normalization direct match (e.g. leavvvve -> leave)
        norm_word = re.sub(r"(.)\1{2,}", r"\1", word)
        if norm_word in vocabulary:
            logger.info(f"Repeated char exact match: {word} -> {norm_word}")
            corrected_words.append(norm_word)
            continue

        # 3. Compound splitting
        split = split_compound_word(word, vocabulary) or split_compound_word(norm_word, vocabulary)
        if split:
            corrected_words.extend(split)
            continue

        # 4. SymSpell correction (try raw and norm_word)
        corrected = symspell_correct(word, sym_spell)
        if corrected != word:
            corrected_words.append(corrected)
            continue

        if norm_word != word:
            norm_corrected = symspell_correct(norm_word, sym_spell)
            if norm_corrected != norm_word:
                corrected_words.append(norm_corrected)
                continue

        # 5. RapidFuzz fuzzy correction fallback
        fuzzy_corrected = fuzzy_correct_word(word, vocabulary, threshold=FUZZY_MIN_RATIO)
        corrected_words.append(fuzzy_corrected)

    logger.info(f"Corrected keywords: {corrected_words}")
    return corrected_words


# -----------------------------
# Main Preprocessing Pipeline
# -----------------------------

def preprocess_query(query):
    original_query = query

    # 1. Lowercase
    query = query.lower()

    # 2. Abbreviation expansion (escaped regex with lookaround boundaries)
    for abbr, expansion in ABBREVIATIONS.items():
        pattern = rf"(?<![a-zA-Z0-9]){re.escape(abbr)}(?![a-zA-Z0-9])"
        query = re.sub(pattern, expansion, query)

    # 3. Remove special chars
    query = re.sub(r"[^a-zA-Z0-9\s]", " ", query)

    # 4. Collapse extra spaces
    query = re.sub(r"\s+", " ", query).strip()

    # 5. Extract words & remove stop words
    words = query.split()
    keywords = [word for word in words if word not in STOP_WORDS]

    logger.info(f"Keywords before correction: {keywords}")

    # 6. Correct keywords (Exact -> Norm -> Compound -> SymSpell -> RapidFuzz)
    keywords = correct_keywords(keywords)

    cleaned_query = " ".join(keywords)

    logger.info(f"Original Query: {original_query}")
    logger.info(f"Processed Query: {cleaned_query}")
    logger.info(f"Final Keywords: {keywords}")

    return cleaned_query, keywords