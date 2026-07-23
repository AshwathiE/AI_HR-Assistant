import math
import os
import shutil
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
##from whoosh_db import index_chunks

from config import UPLOAD_FOLDER
from document_loader import load_document
from embeddings import generate_embeddings
from utils import chunk_text, clean_text
from vector_db import (
    store_document,
    get_all_chunks,
    delete_document_by_filename,
    get_document_points,
    restore_document_points,
)
from logger import logger
from vocabulary import update_vocabulary
##from bm25 import build_bm25
from cache import query_cache

router = APIRouter()

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


@router.post("/")
async def upload_document(
    file: UploadFile = File(...),
    replace: bool = Query(False)
):
    if not file.filename or not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported."
        )

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    replaced = False

    # Check if a file with the same name already exists
    if os.path.exists(file_path):
        if not replace:
            # Return response indicating that the file already exists
            raise HTTPException(
                status_code=409,
                detail=f"The document '{file.filename}' has already been uploaded. Use replace=true to overwrite it."
            )
        else:
            replaced = True

    # Transaction variables
    old_file_backed_up = False
    backup_file_path = file_path + ".bak"
    old_points_backup = None

    if replaced:
        logger.info(f"Preparing to replace existing document: {file.filename}")
        # 1. Back up existing state
        try:
            if os.path.exists(file_path):
                # Back up file on disk
                shutil.move(file_path, backup_file_path)
                old_file_backed_up = True
                logger.info(f"Disk file backed up to {backup_file_path}")

            # Back up points from Qdrant
            old_points_backup = get_document_points(file.filename)
            if old_points_backup:
                logger.info(f"Backed up {len(old_points_backup)} points from Qdrant collection for {file.filename}")
            
            # Delete old collection in Qdrant
            delete_document_by_filename(file.filename)
            logger.info(f"Deleted old Qdrant collection for {file.filename}")

        except Exception as e:
            logger.error(f"Failed during backup stage for replacement: {e}")
            # If backup itself failed before write, clean up backup file if exists
            if old_file_backed_up and os.path.exists(backup_file_path):
                shutil.move(backup_file_path, file_path)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize replacement: {str(e)}"
            )

    try:
        # 2. Upload and process the new file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = load_document(file_path)
        cleaned_text = clean_text(text)

        if not cleaned_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No readable text found in the document."
            )

        # Split into chunks with size of 700 characters
        chunks = chunk_text(cleaned_text, chunk_size=700)

        print("\n========== GENERATED CHUNKS ==========")
        print("Number of chunks:", len(chunks))

        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}")
            print(chunk)
            print("-" * 80)

        print("========== END OF CHUNKS ==========\n")

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

        # 3. Clean up the backup file if replacement succeeded
        if replaced and old_file_backed_up and os.path.exists(backup_file_path):
            os.remove(backup_file_path)
            logger.info(f"Backup file {backup_file_path} deleted successfully.")

        # Fetch all stored chunks (metadata + text)
        all_chunks = get_all_chunks()
        logger.info(f"Total chunks in Qdrant: {len(all_chunks)}")

        # Update vocabulary
        update_vocabulary(all_chunks, rebuild=replaced)

        # Clear cache
        query_cache.clear()
        logger.info(f"Query cache cleared after document {'replacement' if replaced else 'upload'}.")

        return {
            "message": "Document replaced successfully." if replaced else "Document uploaded successfully.",
            "file_name": file.filename,
            "total_chunks": len(chunks),
            "collection_name": collection_name,
        }
    except Exception as e:
        logger.error(f"Error during document {'replacement' if replaced else 'upload'}: {e}. Initiating rollback.")
        # Rollback disk file
        if replaced:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as del_err:
                    logger.error(f"Could not remove failed file: {del_err}")
            
            if old_file_backed_up and os.path.exists(backup_file_path):
                try:
                    shutil.move(backup_file_path, file_path)
                    logger.info(f"Disk file rolled back from {backup_file_path} to {file_path}")
                except Exception as move_err:
                    logger.critical(f"Critical: Failed to restore backup file: {move_err}")

            # Rollback Qdrant collection
            try:
                if old_points_backup:
                    restore_document_points(file.filename, old_points_backup)
                    logger.info(f"Qdrant collection rolled back for {file.filename}")
                else:
                    delete_document_by_filename(file.filename)
            except Exception as db_roll_err:
                logger.critical(f"Critical: Failed to restore Qdrant points: {db_roll_err}")

        else:
            # For regular upload, clean up partial file if it was created
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as clean_err:
                    logger.error(f"Could not clean up failed upload file: {clean_err}")
            try:
                delete_document_by_filename(file.filename)
            except Exception as clean_db_err:
                logger.error(f"Could not clean up failed Qdrant collection: {clean_db_err}")

        # Re-raise HTTPException or wrap generic errors
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during {'replacement' if replaced else 'upload'}: {str(e)}"
        )

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
        "pages": total_pages,   ## total no.of pages ### here there is no previous or next , so pagination process is not applicable
    }


@router.delete("/{filename}")
def delete_document(filename: str):
    """
    Delete a document by its filename.

    This endpoint performs three operations in order, so that if any
    step fails the error is surfaced immediately before further damage:

    1. Delete the Qdrant collection that holds all embeddings and
       metadata for this document.
    2. Delete the physical file from the uploads folder on disk.
    3. Rebuild the BM25 keyword index and vocabulary from the remaining
       chunks, then clear the query cache so stale results are not served.

    Path parameter:
        filename: The exact filename as stored on disk (URL-encoded by
                  the browser automatically, e.g. "Company rulebook.pdf").

    Returns 200 with a summary on success.
    Raises 404 if the file does not exist anywhere.
    Raises 500 with detail if any step fails unexpectedly.
    """
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # Verify the file actually exists either on disk or in Qdrant before
    # attempting anything — prevents silent no-ops.
    file_exists_on_disk = os.path.isfile(file_path)
    qdrant_deleted = False

    if not file_exists_on_disk and not delete_document_by_filename(filename):
        logger.warning(f"Delete requested for unknown document: {filename}")
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found."
        )

    logger.info(f"Starting deletion of document: {filename}")

    # ── Step 1: Remove Qdrant collection ─────────────────────────────────────
    # Done first because it is the most critical data; if disk removal later
    # fails we have not stranded orphaned vectors in the database.
    try:
        qdrant_deleted = delete_document_by_filename(filename)
        if qdrant_deleted:
            logger.info(f"Qdrant collection deleted for: {filename}")
        else:
            # Collection was not found in Qdrant — still continue to clean disk.
            logger.warning(
                f"No Qdrant collection found for '{filename}'; "
                "proceeding to remove disk file."
            )
    except Exception as e:
        logger.error(f"Failed to delete Qdrant collection for '{filename}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove embeddings for '{filename}': {str(e)}"
        )

    # ── Step 2: Remove the physical file from disk ───────────────────────────
    if file_exists_on_disk:
        try:
            os.remove(file_path)
            logger.info(f"Disk file removed: {file_path}")
        except Exception as e:
            logger.error(f"Failed to remove disk file '{file_path}': {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Embeddings were removed but the disk file could not be deleted: {str(e)}"
            )

    # ── Step 3: Rebuild indexes and clear cache ───────────────────────────────
    # Fetch all remaining chunks (from every other document) and rebuild BM25
    # so the deleted file's text is no longer searchable.
    try:
        all_chunks = get_all_chunks()
        logger.info(
            f"Rebuilding BM25 and vocabulary from {len(all_chunks)} remaining chunks."
        )
        update_vocabulary(all_chunks, rebuild=True)
        ###build_bm25(all_chunks)
        query_cache.clear()
        logger.info("BM25, vocabulary, and query cache refreshed after deletion.")
    except Exception as e:
        # Index rebuild failure is non-fatal for data integrity (the file IS
        # gone), but we log it loudly so it is not missed.
        logger.error(
            f"Index rebuild failed after deleting '{filename}': {e}. "
            "Search quality may be degraded until next upload."
        )

    logger.info(f"Document '{filename}' deleted successfully.")
    return {
        "message": f"Document '{filename}' deleted successfully.",
        "filename": filename,
    }
@router.get("/health")
async def health_check():
    return {"status": "ok"} 

