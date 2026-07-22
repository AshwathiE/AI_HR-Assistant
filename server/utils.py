import os
import re
from typing import List
from logger import logger
from difflib import SequenceMatcher

def allowed_file(filename: str) -> bool:
    """
    Check if the uploaded file has a supported extension.
    """

    allowed_extensions = {".pdf", ".docx"}

    extension = os.path.splitext(filename)[1].lower()

    return extension in allowed_extensions


def deduplicate_chunks(results, similarity_threshold=0.90):
    """
    Remove duplicate and near-duplicate chunks.

    Phase 1: Deduplicate by point ID (exact same chunk
             from multiple retrievers).
    Phase 2: Deduplicate by text similarity using
             SequenceMatcher (catches near-duplicates
             differing by minor whitespace/punctuation).

    Returns:
        Deduplicated list keeping the higher-scored entry.
    """

    if not results:
        return results

    initial_count = len(results)

    # Phase 1: Deduplicate by point ID
    seen_ids = set()
    id_deduped = []

    for result in results:

        point_id = result.get("id")

        if point_id and point_id in seen_ids:
            continue

        if point_id:
            seen_ids.add(point_id)

        id_deduped.append(result)

    logger.info(
        f"Dedup phase 1 (ID): "
        f"{initial_count} -> {len(id_deduped)}"
    )

    # Phase 2: Deduplicate by text similarity
    unique = []

    for result in id_deduped:

        chunk_text_normalized = (
            result["text"].strip().lower()
        )

        is_duplicate = False

        for existing in unique:

            existing_text = (
                existing["text"].strip().lower()
            )

            # Quick length check: skip comparison if
            # lengths differ by more than 20%
            len_ratio = len(chunk_text_normalized) / max(
                len(existing_text), 1
            )

            if len_ratio < 0.8 or len_ratio > 1.2:
                continue

            similarity = SequenceMatcher(
                None,
                chunk_text_normalized,
                existing_text,
            ).ratio()

            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(result)

    logger.info(
        f"Dedup phase 2 (text similarity): "
        f"{len(id_deduped)} -> {len(unique)}"
    )

    logger.info(
        f"Total duplicates removed: "
        f"{initial_count - len(unique)}"
    )

    return unique


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra spaces,
    tabs, and blank lines.
    """

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove multiple spaces
    text = re.sub(r" +", " ", text)

    # Remove multiple blank lines
    text = re.sub(r"\n\s*\n", "\n", text)

    return text.strip()

from typing import List

def chunk_text(text: str, chunk_size: int = 700) -> List[str]:  ###  nested function
    """
    Recursive text chunking.
    Splits text using paragraphs, lines, sentences,
    and words while maintaining context.
    """

    separators = [
        "\n\n",   # Paragraph level
        "\n",     # Line level
        ". ",     # Sentence level
        " "        # Word level
    ]



    def recursive_split(text, separators):

        text = text.strip()  ### removes unnesccessary spaces at the start and end of the text

        # If text already fits
        if len(text) <= chunk_size:  ### checking if the text fits in the chunk size , if it fits it is returned as a chunk no cutting is needed"
            return [text]

        # No separators left -> split by characters
        if not separators:   ### if the separators list is empty, it means we have reached the end of the list and the text is split by characters
            return [                                          ### if the doc has 15000 then it will be split into chunks of 700 characters each
                text[i:i + chunk_size]
                for i in range(0, len(text), chunk_size)
            ]

        separator = separators[0] ### takes the first separator from the list

        parts = text.split(separator) ### splits the text into parts based on the separator paragraph

        chunks = []
        current_chunk = ""

        for part in parts:   ## reads one para at a time to check if it fits in the chunksize

            part = part.strip()    ## removing the spaces

            if not part:  ## if the part is empty, it is skipped
                continue

            # Preserve the separator when rebuilding text
            if current_chunk:
                candidate = current_chunk + separator + part
            else:
                candidate = part

            if len(candidate) <= chunk_size:  ## if charcter =620 but limit=700 it accepts the characters
                current_chunk = candidate

            else:

                if current_chunk:   ## if it is large then goes to next seperators #3 deletes the seperator and goes to the next
                    chunks.extend(
                        recursive_split(
                            current_chunk,
                            separators[1:]
                        )
                    )

                current_chunk = part

        if current_chunk:
            chunks.extend(
                recursive_split(
                    current_chunk,
                    separators[1:]
                )
            )

        return chunks ## returns all chunks

    return recursive_split(text, separators)

def remove_uploaded_file(file_path: str): ## to remove the file after the response
    """
    Delete a file from uploads folder.
    """

    if os.path.exists(file_path): ## checks if the file exists in the uploads folder
        os.remove(file_path)  ## if it exists it removes it


def format_sources(metadata: list) -> list:
    """
    Convert ChromaDB metadata into
    a clean response.
    """

    sources = []

    for item in metadata:
        sources.append(
            {
                "document": item.get("source"),
                "chunk": item.get("chunk_number")
            }
        )

    return sources


def get_file_size(file_path: str) -> float:
    """
    Return file size in MB.
    """

    size = os.path.getsize(file_path)

    return round(size / (1024 * 1024), 2) ## rounds of the size of the file with 2 decimals


def file_exists(file_path: str) -> bool:
    """
    Check if file exists.
    """

    return os.path.exists(file_path)

# -----------------------------------------------
# Reciprocal Rank Fusion
# -----------------------------------------------

def reciprocal_rank_fusion(vector_results, bm25_results, k=60):

    if vector_results is None:
        vector_results = []

    if bm25_results is None:
        bm25_results = []
    """
    Merge results from vector search and BM25
    using Reciprocal Rank Fusion (RRF).

    Each result gets a combined score:
        score = 1/(k + rank_vector) + 1/(k + rank_bm25)

    Args:
        vector_results: Results from Qdrant vector search
        bm25_results: Results from BM25 keyword search
        k: Smoothing constant (default 60)

    Returns:
        Merged list of results sorted by RRF score
    """

    scores = {}
    result_map = {}

    ##Score vector search results by rank
    for rank, result in enumerate(vector_results):

        key = result.get("id") or result["text"][:100]

        scores[key] = scores.get(key, 0) + (
            1.0 / (k + rank + 1)
        )

        result_map[key] = result

    ##Score BM25 results by rank
    for rank, result in enumerate(bm25_results):

        key = result.get("id") or result["text"][:100]

        scores[key] = scores.get(key, 0) + (
            1.0 / (k + rank + 1)
        )

        # Keep whichever result has more metadata
        if key not in result_map:
            result_map[key] = result

    # Sort by combined RRF score
    ranked_keys = sorted(
        scores.keys(),
        key=lambda x: scores[x],
        reverse=True
    )

    merged = []

    for key in ranked_keys:

        result = result_map[key]
        result["rrf_score"] = round(scores[key], 6)
        merged.append(result)

    logger.info(
        f"RRF merged: {len(vector_results)} vector + "
        f"{len(bm25_results)} BM25 -> "
        f"{len(merged)} combined results"
    )

    return merged
