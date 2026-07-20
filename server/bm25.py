from rank_bm25 import BM25Okapi
from logger import logger

bm25_index = None
documents = [] ## stores every chunk along with meta data


def build_bm25(chunks):  ## calledwheenver new functions are uploaded 
    """
    Builds/Rebuilds BM25 index using all chunks.

    Args:
        chunks: Either a list of text strings (legacy)
                or a list of dicts with 'text', 'source',
                'chunk_number', 'collection_name', 'id' keys.
    """

    global bm25_index ## function is called globally so that upload is done globally and result is fetched
    global documents ## both index and documents are called globally
    # Normalize input: accept both plain strings
    # and metadata-enriched dicts
    ##is a built-in Python function used to check whether an object belongs to a particular class (or data type) or any of its subclasses
    if chunks and isinstance(chunks[0], str): ## if not in metadata format , the chunks are converetd to meta data format else itis stored as it is by appending default values
        documents = [
            {
                "text": chunk,
                "source": "Unknown",
                "chunk_number": 0,
                "collection_name": "",
                "id": "",
            }
            for chunk in chunks
        ]
    else:
        documents = chunks
    
    ## tokenizes the chunks for bm25 as it accpeta only tokenised documents 
    tokenized_docs = [
        doc["text"].lower().split()
        for doc in documents
    ]

    
    bm25_index = BM25Okapi(tokenized_docs)  ##initializing the bm25 index with tokenized chunks 

    logger.info(
        f"BM25 index built with {len(documents)} chunks"
    )

## keyword search
def bm25_search(query, top_k=10, selected_sources=None):  ##top_k: number of results to return , selected_sources: optional list of source filenames to restrict results to
    """
    BM25 keyword search with optional file-name filtering.

    Args:
        query: Search query string
        top_k: Number of results to return
        selected_sources: Optional list of source filenames
                          to restrict results to

    Returns:
        List of result dicts matching the shape used by
        vector search (text, bm25_score, document,
        chunk_number, collection_name, id, score)
    """

    global bm25_index
    global documents

    if bm25_index is None or not documents: ## if no documents indexed returns [] instead of crashing
        return []

    scores = bm25_index.get_scores(        ## compared and return 1 score per document
        query.lower().split()
    )

    ranked = sorted(    ## coverts values and indexes into sorted list of tuples
        enumerate(scores),
        key=lambda x: x[1],  ## based on the values of the tuple in descending order, reverse=True ensures that the highest scores are at the top of the list
        reverse=True
    )

    results = []

    for idx, score in ranked:

        if score <= 0:   ##BM25 gives zero (or negative) when there is effectively no keyword overlap. Such documents are skipped.
            continue

        doc = documents[idx] ## retreives the document using the index
        source = doc.get("source", "Unknown")## gets the source of the document

        ## file name filter: skip if not in selected sources
        if selected_sources and source not in selected_sources: ##if the source is not in the selected sources list, it is skipped
            continue

        results.append({
            "id": doc.get("id", ""),
            "text": doc["text"],
            "bm25_score": float(score),
            "score": float(score),
            "document": source,
            "chunk_number": doc.get("chunk_number", 0),
            "collection_name": doc.get(
                "collection_name", ""
            ),
        })

        if len(results) >= top_k: #After collecting top kresults, the loop exits
            break

    return results