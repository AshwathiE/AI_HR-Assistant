import math
import os
import shutil

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from config import UPLOAD_FOLDER
from document_loader import load_document
from embeddings import generate_embeddings
from utils import chunk_text, clean_text
from vector_db import store_document, get_all_chunks
from logger import logger
from vocabulary import update_vocabulary
from bm25 import build_bm25
from cache import query_cache

router = APIRouter()

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported."
        )

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    if os.path.exists(file_path):
        raise HTTPException(
            status_code=400,
            detail="This document has already been uploaded."
        )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = load_document(file_path)
    cleaned_text = clean_text(text)

    if not cleaned_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text found in the document."
        )

    # Split into chunks
    chunks = chunk_text(cleaned_text, chunk_size=700)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Unable to split the document into chunks."
        )

    # Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Store in Qdrant
    collection_name = store_document(
        chunks=chunks,
        embeddings=embeddings,
        source_file=file.filename,
    )

    # Fetch all stored chunks (metadata + text)
    all_chunks = get_all_chunks()

    logger.info(f"Total chunks in Qdrant: {len(all_chunks)}")

    # Update vocabulary
    update_vocabulary(all_chunks)

    # Rebuild BM25
    build_bm25(all_chunks)

    # Clear cache
    query_cache.clear()
    logger.info("Query cache cleared after document upload.")

    return {
        "message": "Document uploaded successfully.",
        "file_name": file.filename,
        "total_chunks": len(chunks),
        "collection_name": collection_name,
    }


@router.get("/documents")  ## paginationn processs
def list_uploaded_documents(
    page: int = Query(1, ge=1),   # default page = 1 page numer is always a int
    page_size: int = Query(10, ge=1, le=100),  # default page size = 10  ##This controls how many documents appear on one page. le =maximum no of pages
):
    files = sorted(
        [
            file_name                                                       ##It loops through every item in the upload folder.
            for file_name in os.listdir(UPLOAD_FOLDER)
            if os.path.isfile(os.path.join(UPLOAD_FOLDER, file_name))
        ]
    )

    total_pages = max(1, math.ceil(len(files) / page_size)) if files else 1
    start = (page - 1) * page_size ##start=5
    end = start + page_size ##end=10

    return {
        "documents": files[start:end],   ## shows files for the current pagination request
        "count": len(files),   ## total no.of files
        "page": page,   ## current page no.
        "page_size": page_size,   ## no.of files per page
        "pages": total_pages,   ## total no.of pages
    }
