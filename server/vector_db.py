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

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_PREFIX = "company_policy"
VECTOR_SIZE = 384


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


def store_document(chunks, embeddings, source_file): ##stores the documents in vector db with there embeddings and metadata
    collection_name = get_collection_name(source_file)
    ensure_collection(collection_name)

    points = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": source_file,
                    "chunk_number": i + 1,
                    "collection_name": collection_name,
                },
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points,
    )

    return collection_name


def search_documents(query_embedding, top_k=3, selected_sources=None):
    collections = list_collection_names()

    if not collections:
        return []

    results = []

    for collection_name in collections:

        response = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        for point in response.points:

            payload = point.payload or {}
            source = payload.get("source", "Unknown")

            if selected_sources and source not in selected_sources:
                continue

            results.append(
                {
                    "id": str(point.id),
                    "score": float(point.score),
                    "payload": payload,
                    "text": payload.get("text", ""),
                    "document": source,
                    "chunk_number": payload.get("chunk_number", 0),
                    "collection_name": payload.get(
                        "collection_name",
                        collection_name,
                    ),
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


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
    Used for automatic BM25 index rebuilding.
    """
    chunks = []

    for collection_name in list_collection_names():  ##It returns all collection names stored in Qdrant.
        records, _ = client.scroll(  ##scroll() is a Qdrant method that retrieves stored points.
            collection_name=collection_name,
            limit=10000,  ## if collection contains 250 chunks no issues all are retrived , if the chunks containd 12000 then chunk size is limited to first 10000
            with_payload=True, ##every chunk stores the meatadata
            with_vectors=False,  ## every chunk has as embedddingd vectos so those vectors are skipped , this makes retrival faster
        )

        for record in records:        ### for processing every chunk indvidually 
            payload = record.payload or {}  ## Each record contains metadata if payload is missing then none to avoid errors

            chunks.append(
                {
                    "id": str(record.id), ## id is converted as string as stores bcz some can be string some can be UUId
                    "text": payload.get("text", ""),
                    "source": payload.get("source", ""),
                    "collection_name": collection_name,
                }
            )

    return chunks  ## all chunjs is stored together as one and returned


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
    )

