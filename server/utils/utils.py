import os
import re
from typing import List
from utils.logger import logger
from difflib import SequenceMatcher
from services.vocabulary import load_vocabulary
def allowed_file(filename: str) -> bool:
    """
    Check if the uploaded file has a supported extension.
    """

    allowed_extensions = {".pdf", ".docx"}

    extension = os.path.splitext(filename)[1].lower()

    return extension in allowed_extensions

from typing import List

##def chunk_text(text: str, chunk_size: int = 700) -> List[str]:
    ##"""
    ##Paragraph-based chunking.

    ##Each paragraph is treated as one unit.
    ##If a paragraph exceeds chunk_size,
    ##it is split into smaller chunks.
    ##"""

    # Split document into paragraphs
    ##  parts = text.split("\n\n")

    ##chunks = []
    ##current_chunk = ""

    ##for part in parts:

      ##  part = part.strip()

        ##if not part:
          ##  continue

        # Paragraph fits
        ##if len(part) <= chunk_size:

          ##  if current_chunk:

            ##    candidate = current_chunk + "\n\n" + part

              ##  if len(candidate) <= chunk_size:
                ##    current_chunk = candidate
               ## else:
                 ##   chunks.append(current_chunk)
                   ## current_chunk = part

            ##else:
              ##  current_chunk = part

        # Large paragraph
        ##else:

            # Save previous chunk
          ##  if current_chunk:
            ##    chunks.append(current_chunk)
              ##  current_chunk = ""

            # Split large paragraph
            ##for i in range(0, len(part), chunk_size):
             ##   chunks.append(part[i:i + chunk_size])

    ##if current_chunk:
      ##  chunks.append(current_chunk)

   ## return chunks

##def chunk_text(text: str, chunk_size: int = 700) -> List[str]:  ###  nested function
    ##"""
   ##Recursive text chunking.
    ##Splits text using paragraphs, lines, sentences,
    ##and words while maintaining context.
    ##"""

    ##separators = [
     ## "\n\n",   # Paragraph level
      ##"\n",     # Line level
    ##". ",     # Sentence level
      ##  " "        # Word level
    ##]


##text = text.strip()  ### removes unnesccessary spaces at the start and end of the text

    ##    # If text already fits
     ##if len(text) <= chunk_size:  ### checking if the text fits in the chunk size , if it fits it is returned as a chunk no cutting is needed"
        ##    return [text]

        # No separators left -> split by characters
        ##if not separators:   ### if the separators list is empty, it means we have reached the end of the list and the text is split by characters
          ####    text[i:i + chunk_size]
              ##  for i in range(0, len(text), chunk_size)
            ##]

        ##separator = separators[0] ### takes the first separator from the list

       ## parts = text.split(separator) ### splits the text into parts based on the separator paragraph

        ##chunks = []
       ## current_chunk = ""

       ## for part in parts:   ## reads one para at a time to check if it fits in the chunksize

          ##  part = part.strip()    ## removing the spaces

           ## if not part:  ## if the part is empty, it is skipped
            ##    continue

            # Preserve the separator when rebuilding text
           ## if current_chunk:
             ##   candidate = current_chunk + separator + part
            ##else:
              ##  candidate = part

            ##if len(candidate) <= chunk_size:  ## if charcter =620 but limit=700 it accepts the characters
              ##  current_chunk = candidate

            ##else:

              ##  if current_chunk:   ## if it is large then goes to next seperators #3 deletes the seperator and goes to the next
                ##    chunks.extend(
                  ##      recursive_split(
                  ##          current_chunk,
                  ##          separators[1:]
                  ##      )
                  ##  )

               ## current_chunk = part

        ##if current_chunk:
            ##chunks.extend(
                ##recursive_split(
                    ##current_chunk,
                    ##separators[1:]
                ##)
            ##)

        #return chunks ## returns all chunks

    #return recursive_split(text, separators)


##from typing import List

##def chunk_text(text: str, chunk_size: int = 700) -> List[str]:
   #"""
    #Paragraph-based chunking.
    #Splits text only at paragraph boundaries.
    #Paragraphs are never broken apart.
    #"""

   ##paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

   ##chunks = []
   ##current_chunk = ""

   ##for paragraph in paragraphs:

        # First paragraph in a chunk
        ##if not current_chunk:
          ##current_chunk = paragraph
          ##continue

        # Check if adding this paragraph exceeds the limit
        ##candidate = current_chunk + "\n\n" + paragraph

        ##if len(candidate) <= chunk_size:
          ##current_chunk = candidate
        ##else:
          ##chunks.append(current_chunk)
          ##current_chunk = paragraph

    # Add the last chunk
    #if current_chunk:
        #chunks.append(current_chunk)

        ##return chunks

import re
from typing import List

##def chunk_text(text: str, chunk_size: int = 700) -> List[str]:
  ##  """
    ##Markdown heading-based chunking.
    ##Splits text only at Markdown (##) headings.
    ##Each heading and its content form a chunk.
    ##"""

    ##sections = [
      ##  section.strip()
        ##for section in re.split(
            ##r"(?=^## )",
            ##text,
            ##flags=re.MULTILINE,
        ##)
        #if section.strip()
    #]

    ##chunks = []

    ##for section in sections:
      ##  chunks.append(section)

    ##return chunks


##def chunk_text(text: str, chunk_size: int = 300) -> List[str]:  ###  nested function
  ##  """
   ## Recursive text chunking.
   ## Splits text using paragraphs, lines, sentences,
   ## and words while maintaining context.
   ## """

   ## separators = [
     ##   "\n\n",   # Paragraph level
     ##   "\n",     # Line level
     ##   ". ",     # Sentence level
     ##   " "        # Word level
    ##]



    ##def recursive_split(text, separators):

       ## text = text.strip()  ### removes unnesccessary spaces at the start and end of the text

        # If text already fits
       ## if len(text) <= chunk_size:  ### checking if the text fits in the chunk size , if it fits it is returned as a chunk no cutting is needed"
       ##     return [text]

        # No separators left -> split by characters
        ##if not separators:   ### if the separators list is empty, it means we have reached the end of the list and the text is split by characters
        ##    return [                                          ### if the doc has 15000 then it will be split into chunks of 700 characters each
            ##    text[i:i + chunk_size]
            ##    for i in range(0, len(text), chunk_size)
            #]

        ##separator = separators[0] ### takes the first separator from the list

        ##parts = text.split(separator) ### splits the text into parts based on the separator paragraph

        ##chunks = []
        ##current_chunk = ""

        ##for part in parts:   ## reads one para at a time to check if it fits in the chunksize

            ##part = part.strip()    ## removing the spaces

            ##if not part:  ## if the part is empty, it is skipped
            ##    continue

            # Preserve the separator when rebuilding text
            ##if current_chunk:
            ##    candidate = current_chunk + separator + part
            ##else:
            ##    candidate = part

            ##if len(candidate) <= chunk_size:  ## if charcter =620 but limit=700 it accepts the characters
                ##current_chunk = candidate

           ## else:

             ##   if current_chunk:   ## if it is large then goes to next seperators #3 deletes the seperator and goes to the next
               ##     chunks.extend(
                 ##       recursive_split(
                   ##         current_chunk,
                     ##       separators[1:]
                       ## )
                    ##)

                ##current_chunk = part

       ## if current_chunk:
         ##   chunks.extend(
           ##     recursive_split(
             ##       current_chunk,
               ##     separators[1:]
                ##)
            ##)

        ##return chunks ## returns all chunks

   ## return recursive_split(text, separators)

MAX_CHUNK_SIZE = 300
MIN_HEADING_LENGTH = 40

def split_long_text(text: str, max_size: int = MAX_CHUNK_SIZE) -> List[str]:
    """
    Recursively split long text into smaller chunks using sentences.
    """

    text = text.strip()

    if not text:
        return []

    if len(text) <= max_size:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""

    for sentence in sentences:

        if len(current) + len(sentence) + 1 <= max_size:

            current += sentence + " "

        else:

            if current.strip():
                chunks.append(current.strip())

            current = sentence + " "

    if current.strip():
        chunks.append(current.strip())

    return chunks

def is_heading(line: str) -> bool:
    """
    Detect whether a line is likely a heading.
    """

    line = line.strip()

    if not line:
        return False

    # Numbered heading
    if re.match(r'^\d+\.\s+', line):
        return True

    # Too long → probably paragraph
    if len(line) > MIN_HEADING_LENGTH:
        return False

    # Ends with punctuation → paragraph
    if line.endswith("."):
        return False

    # Bullet point
    if line.startswith(("•", "-", "*")):
        return False

    words = line.split()

    if len(words) <= 8:

        capitals = sum(
            1
            for w in words
            if w[:1].isupper()
        )

        if capitals >= len(words) * 0.6:
            return True

    return False


def chunk_text(text: str) -> List[str]:
    """
    Production-ready hierarchical chunking.

    Priority:
    1. Numbered headings
    2. Plain headings
    3. Paragraphs
    4. Sentence splitting
    """

    # Normalize line endings
    text = re.sub(r'\r\n?', '\n', text)

    # Join single line breaks into spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Remove extra spaces
    text = re.sub(r' {2,}', ' ', text)

    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    text = text.strip()

    if not text:
        return []

    # ------------------------------------
    # Strategy 1
    # Numbered headings
    # ------------------------------------

    if re.search(r'^\d+\.\s+', text, flags=re.MULTILINE):

        sections = re.split(
            r'(?=^\d+\.\s+)',
            text,
            flags=re.MULTILINE
        )

        chunks = []

        title = ""

        # Extract document title
        if sections:

            first = sections[0].strip()

            if not re.match(r'^\d+\.\s+', first):

                title = first
                sections = sections[1:]


        # Attach title to every section
        for sec in sections:

            sec = sec.strip()

            if not sec:
                continue

            if title:

                chunk = f"{title}\n\n{sec}"

            else:

                chunk = sec


            # Split only if section is too large
            if len(chunk) > MAX_CHUNK_SIZE:

                chunks.extend(
                    split_long_text(chunk)
                )

            else:

                chunks.append(chunk)


        return chunks


    # ------------------------------------
    # Strategy 2
    # Plain headings
    # ------------------------------------

    lines = text.split("\n")

    chunks = []

    current_chunk = ""

    for line in lines:

        line = line.rstrip()

        if is_heading(line):

            if current_chunk.strip():

                chunks.extend(
                    split_long_text(current_chunk)
                )

            current_chunk = line + "\n"

        else:

            current_chunk += line + "\n"


    if current_chunk.strip():

        chunks.extend(
            split_long_text(current_chunk)
        )


    if len(chunks) > 1:

        return chunks


    # ------------------------------------
    # Strategy 3
    # Paragraphs
    # ------------------------------------

    paragraphs = re.split(
        r'\n\s*\n',
        text
    )

    chunks = []

    current = ""

    for para in paragraphs:

        para = para.strip()

        if not para:
            continue


        if len(current) + len(para) < MAX_CHUNK_SIZE:

            current += para + "\n\n"

        else:

            if current.strip():

                chunks.append(
                    current.strip()
                )

            current = para + "\n\n"


    if current.strip():

        chunks.append(
            current.strip()
        )


    # ------------------------------------
    # Strategy 4
    # Sentence split if still large
    # ------------------------------------

    final_chunks = []

    for chunk in chunks:

        if len(chunk) > MAX_CHUNK_SIZE:

            final_chunks.extend(
                split_long_text(chunk)
            )

        else:

            final_chunks.append(
                chunk.strip()
            )


    return [
        c for c in final_chunks
        if c
    ]

##def chunk_text(text: str) -> List[str]:
  ##  """
    ##Heading-based chunking.

    ##Each numbered section (1. Purpose, 2. Scope, ...)
    ##becomes one chunk.
    ##"""

    ##text = text.strip()

    # Split before numbered headings (1., 2., 3., ...)
    ##sections = re.split(
      ##  r'(?=^\d+\.\s)',
       ## text,
       ## flags=re.MULTILINE,
    ##)

    ##chunks = []

    ## Document title
    ##title = ""

    ##if sections:
      ##  first = sections[0].strip()

        # If first part is only the document title
        ##if not re.match(r'^\d+\.\s', first):
         ##   title = first
          ##  sections = sections[1:]

    # Attach title to every section
    ##for section in sections:

      ##  section = section.strip()

      ##  if not section:
        ##    continue

       ## if title:
        ##    chunk = f"{title}\n\n{section}"
       ## else:
         ##   chunk = section

      ##  chunks.append(chunk)

   ## return chunks


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

import re
from typing import List

import re
from typing import List


def deduplicate_chunks(results, similarity_threshold=0.60):
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



MAX_EDIT_DISTANCE = 2


from symspellpy import SymSpell, Verbosity

def build_symspell():

    sym_spell = SymSpell(
        max_dictionary_edit_distance=MAX_EDIT_DISTANCE,
        prefix_length=7
    )

    vocabulary = load_vocabulary()

    if not vocabulary:
        logger.warning(
            "Vocabulary is empty. SymSpell dictionary not created."
        )
        return sym_spell

    for word in vocabulary:
        sym_spell.create_dictionary_entry(
            word,
            1
        )

    logger.info(
        f"SymSpell dictionary loaded with {len(vocabulary)} words"
    )

    return sym_spell


sym_spell = build_symspell()

 # ---- BM25 Keyword Search ----
    ##t_bm25 = time.time()
    ##bm25_results = search_chunks(
        ##cleaned_query,
       ## top_k=top_k,
        ##selected_sources=request.sources,
    ##)   
    ##logger.info(f"BM25 Retrieval Time: {time.time()-t_bm25:.3f} sec")
    ##logger.info(f"Type of bm25_results: {type(bm25_results)}")
    #print("\n===== BM25 Search Results =====")
    ##for i, result in enumerate(bm25_results, start=1):
       ## print(f"\nChunk {i}:")
        ##print(result["text"])
    ##print("===== End of BM25 Search Results =====\n")
    ##logger.info(f"bm25_results: {len(bm25_results)}")

# ==========================================================
# Main Preprocessing Pipeline
# ==========================================================

def preprocess_query(query: str):
    """
    Preprocesses a user query for retrieval.

    Pipeline:
        1. Normalize text
        2. Expand abbreviations
        3. Correct spelling (SymSpell)
        4. Tokenize
        5. Remove stop words
        6. Remove duplicate words
        7. Return cleaned query and keywords

    Returns
    -------
    Tuple[str, List[str]]

    Example
    -------
    Input:
        "Can i take sickkleave??"

    Output:
        (
            "take sick leave",
            ["take", "sick", "leave"]
        )
    """

    original_query = query

    logger.info("=" * 60)
    logger.info("Incoming Query: %s", original_query)

    # ------------------------------------------------------
    # Step 1 : Normalize
    # ------------------------------------------------------

    query = normalize_text(query)

    logger.debug("Normalized Query: %s", query)

    # ------------------------------------------------------
    # Step 2 : Expand abbreviations
    # ------------------------------------------------------

    expanded_query = expand_abbreviations(query)

    if expanded_query != query:
        logger.info(
            "Expanded abbreviations: '%s' -> '%s'",
            query,
            expanded_query
        )

    query = expanded_query

    # ------------------------------------------------------
    # Step 3 : Spell correction
    # ------------------------------------------------------

    corrected_query = correct_spelling(query)

    query = corrected_query

    # ------------------------------------------------------
    # Step 4 : Tokenize
    # ------------------------------------------------------

    words = query.split()

    logger.debug("Tokenized Words: %s", words)

    # ------------------------------------------------------
    # Step 5 : Remove stop words
    # ------------------------------------------------------

    keywords = remove_stopwords(words)

    logger.debug("After Stopword Removal: %s", keywords)

    # ------------------------------------------------------
    # Step 6 : Remove duplicate words
    # ------------------------------------------------------

    keywords = remove_duplicate_words(keywords)

    cleaned_query = " ".join(keywords)

    # ------------------------------------------------------
    # Final Logging
    # ------------------------------------------------------

    logger.info("Processed Query : %s", cleaned_query)
    logger.info("Keywords        : %s", keywords)
    logger.info("=" * 60)

    return cleaned_query, keywords