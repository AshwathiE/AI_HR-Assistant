from pathlib import Path
from uuid import uuid4
from logger import logger
from whoosh.fields import Schema, TEXT, ID, NUMERIC
from whoosh.index import create_in, exists_in, open_dir
from whoosh.qparser import QueryParser
from whoosh.scoring import BM25F

# ---------------------------------------------------
# Index Location (Absolute Path)
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "whoosh_index"

# ---------------------------------------------------
# Schema
# ---------------------------------------------------

schema = Schema(
    id=ID(stored=True, unique=True),
    text=TEXT(stored=True),
    source=ID(stored=True),
    chunk_number=NUMERIC(stored=True),
)

# ---------------------------------------------------
# Create/Open Index
# ---------------------------------------------------

def get_index():
    """
    Opens the existing Whoosh index if present.
    Otherwise creates a new one.
    """

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if exists_in(str(INDEX_DIR)):
        print(f"Opening existing Whoosh index: {INDEX_DIR}")
        return open_dir(str(INDEX_DIR))

    print(f"Creating new Whoosh index: {INDEX_DIR}")
    return create_in(str(INDEX_DIR), schema)

# ---------------------------------------------------
# Index Chunks
# ---------------------------------------------------

##def index_chunks(chunks, source_file):
    """
    Index document chunks into Whoosh.
    Removes previous entries of the same document.
    """

    ##ix = get_index()

    ##writer = ix.writer()

    # Delete previous indexed version (if exists)
    ##writer.delete_by_term("source", source_file)

    ##for i, chunk in enumerate(chunks):

        ##writer.add_document(
        ##    id=str(uuid4()),
        ##    text=chunk,
        ##    source=source_file,
        ##    chunk_number=i + 1,
        ##)

    ##writer.commit()

    ##print(f"Indexed {len(chunks)} chunks from '{source_file}'.")

# ---------------------------------------------------
# Search
# ---------------------------------------------------


def search_chunks(query, top_k=10, selected_sources=None):

    ix = get_index()
    results = []

    with ix.searcher(weighting=BM25F()) as searcher:

        parser = QueryParser("text", schema=ix.schema)
        parsed_query = parser.parse(query)

        hits = searcher.search(parsed_query, limit=top_k)

        for rank, hit in enumerate(hits, start=1):

            result = {
                "id": hit["id"],
                "text": hit["text"],
                "document": hit["source"],
                "chunk_number": hit["chunk_number"],
                "score": hit.score,
            }

            results.append(result)

        logger.info(f"BM25 result count: {len(results)}")

    return results
# ---------------------------------------------------
# Debug Utility
# ---------------------------------------------------

def index_statistics():
    """
    Returns index statistics.
    """

    ix = get_index()

    with ix.searcher() as searcher:

        return {
            "documents": searcher.doc_count(),
            "index_path": str(INDEX_DIR),
        }