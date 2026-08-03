from utils.logger import logger
import hashlib
import os
import re
from uuid import uuid4

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    VectorParams,
)

import spacy

nlp = spacy.load("en_core_web_sm")

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=500
)

COLLECTION_PREFIX = "company_policy"
VECTOR_SIZE = 768


def _slugify(text: str):##used to create a unique name for each document
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "document"


def get_collection_name(source_file: str): ## creates a collection name for each document by appending a hash of the filename to the collection prefix
    base_name = os.path.basename(source_file)
    slug = _slugify(base_name)
    digest = hashlib.sha1(base_name.encode("utf-8")).hexdigest()[:8]
    return f"{COLLECTION_PREFIX}_{slug}_{digest}"


def ensure_collection(collection_name: str):
    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

    return collection_name


def list_collection_names():
    return [
        c.name
        for c in client.get_collections().collections
        if c.name.startswith(COLLECTION_PREFIX)
    ]

def store_document(chunks, embeddings, source_file):
    collection_name = get_collection_name(source_file)
    ensure_collection(collection_name)
    BATCH_SIZE = 100

    points = []

    for chunk, embedding in zip(chunks, embeddings):

        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "section": chunk["section"],
                    "topics": chunk["topics"],
                    "source": source_file,
                    "chunk_number": chunk["chunk_number"],
                    "collection_name": collection_name,
            },
        )
    )

    total_points = len(points)

    logger.info(f"Uploading {total_points} chunks to Qdrant...")

    for start in range(0, total_points, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total_points)

        batch = points[start:end]

        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
        )

        logger.info(
            f"Uploaded batch {start//BATCH_SIZE + 1} "
            f"({start+1}-{end} of {total_points})"
        )

    logger.info("Document upload completed successfully.")

    return collection_name


def search_documents(query,query_embedding,top_k=20,selected_sources=None):
    """
    Retrieves the top_k most similar chunks across all collections.

    If fewer than top_k chunks are available, returns only the available chunks.
    Results are globally ranked by cosine similarity (highest first).
    """

    collections = list_collection_names()

    if not collections:
        return []

    results = []

    target_collections = set()
    if selected_sources:
        for s in selected_sources:
            target_collections.add(s)
            target_collections.add(get_collection_name(s))

    # Search each collection
    for collection_name in collections:
        # If the user selected specific documents,
        # search only those collections
        if selected_sources:
            matches = (
                collection_name in target_collections
                or any(
                    s in collection_name
                    or _slugify(s) in collection_name
                    or _slugify(os.path.splitext(s)[0]) in collection_name
                    for s in selected_sources
                )
            )
            if not matches:
                continue

        response = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,          # User-selected Top-K per collection
            with_payload=True,
            with_vectors=False,
        )

        for point in response.points:

            payload = point.payload or {}
            source = payload.get("source", "Unknown")

            if (
                selected_sources
                and source not in selected_sources
                and collection_name not in selected_sources
            ):
                continue

            results.append(
    {
        "id": str(point.id),
        "score": float(point.score),
        "payload": payload,
        "text": payload.get("text", ""),
        "section": payload.get("section", ""),
        "topics": payload.get("topics", []),
        "document": source,
        "chunk_number": payload.get("chunk_number", 0),
        "collection_name": payload.get(
            "collection_name",
            collection_name,
        ),
    }
)

    query_doc = nlp(query)

    for result in results:

        topics = result["payload"].get("topics", [])

        if not topics:
            continue

        topic_text = " ".join(topics)

        topic_doc = nlp(topic_text)

        similarity = query_doc.similarity(topic_doc)

        result["score"] += similarity * 0.10

    # Global ranking
    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # Remove duplicate chunks (optional)
    seen = set()
    unique_results = []

    for result in results:

        key = (
            result["document"],
            result["chunk_number"],
        )

        if key in seen:
            continue

        seen.add(key)
        unique_results.append(result)

    # If fewer than top_k exist, return only those
    final_results = unique_results[:top_k]

    logger.info("\n===== GLOBAL TOP RESULTS =====")

    for rank, result in enumerate(final_results, start=1):

        logger.info(
            f"Rank {rank} | "
            f"Cosine Similarity: {result['score']:.4f} | "
            f"Document: {result['document']} | "
            f"Chunk: {result['chunk_number']}"
        )

    logger.info(
        f"User selected Top-K : {top_k}"
    )

    logger.info(
        f"Chunks returned      : {len(final_results)}"
    )

    return final_results

def get_uploaded_documents():
    documents = set()

    for collection_name in list_collection_names():
        records, _ = client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            source = record.payload.get("source")
            if source:
                documents.add(source)

    return sorted(documents)


def delete_document(document_id):
    for collection_name in list_collection_names():
        client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=[document_id]),
        )
    return True


def document_count():
    total = 0
    for collection_name in list_collection_names():
        info = client.get_collection(collection_name=collection_name)
        total += info.points_count
    return total


def reset_database():
    for collection_name in list_collection_names(): ## all document chunks , all embeddings , all metadata , the entire collecion itself is deleted
        client.delete_collection(collection_name=collection_name)
    return True


def get_all_documents():
    records = []
    for collection_name in list_collection_names():
        collection_records, _ = client.scroll(  ##scroll() is Qdrant's method for retrieving points (records).
            collection_name=collection_name,
            limit=10000,  ## if collection contains 250 chunks no issues all are retrived , if the chunks containd 12000 then chunk size is limited to first 10000
            with_payload=True,
            with_vectors=False,  ## embedding vectors are skipped to increase the performance 
        )
        records.extend(collection_records)  ### adds each recoer indvidually  ##extend() gives you one flat list:
    return records   ###Record is the original Qdrant object,


def get_all_chunks():
    """
    Fetch all chunks from all collections.
    Used for rebuilding the BM25 index.
    """

    chunks = []

    collections = list_collection_names()

    print(f"Collections found: {collections}")

    for collection_name in collections:

        records, _ = client.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )

        print(
            f"{collection_name} -> {len(records)} records"
        )

        for record in records:

            payload = record.payload or {}

            chunks.append(
    {
        "id": str(record.id),
        "text": payload.get("text", ""),
        "section": payload.get("section", ""),
        "topics": payload.get("topics", []),
        "source": payload.get("source", ""),
        "chunk_number": payload.get("chunk_number", 0),
        "collection_name": payload.get(
            "collection_name",
            collection_name,
        ),
    }
)

    print(f"Total chunks fetched: {len(chunks)}")

    if chunks:
        print(chunks[0])

    return chunks


def delete_document_by_filename(source_file: str):
    """
    Deletes the Qdrant collection associated with the given source filename.
    """
    collection_name = get_collection_name(source_file)
    collections = client.get_collections().collections
    existing = [c.name for c in collections]
    if collection_name in existing:
        client.delete_collection(collection_name=collection_name)
        return True
    return False


def get_document_points(source_file: str):
    """
    Retrieves all points (including vectors and payloads) for the given source filename.
    """
    collection_name = get_collection_name(source_file)
    collections = client.get_collections().collections
    existing = [c.name for c in collections]
    if collection_name not in existing:
        return None
    records, _ = client.scroll(
        collection_name=collection_name,
        limit=10000,
        with_payload=True,
        with_vectors=True,
    )
    return records


def restore_document_points(source_file: str, records: list):
    """
    Restores Qdrant collection and points from a list of record objects.
    """
    collection_name = get_collection_name(source_file)
    collections = client.get_collections().collections
    existing = [c.name for c in collections]
    if collection_name in existing:
        client.delete_collection(collection_name=collection_name)
    
    if not records:
        return
        
    ensure_collection(collection_name)
    points = []
    for record in records:
        points.append(
            PointStruct(
                id=str(record.id),
                vector=record.vector,
                payload=record.payload,
            )
        )
    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=False,
    )


    print(f"Stored {len(points)} chunks in {collection_name}")