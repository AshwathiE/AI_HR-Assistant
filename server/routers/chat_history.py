# routers/chat_history.py — Chat history CRUD API endpoints

import math
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

import chat_db
from logger import logger

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class RenameRequest(BaseModel):
    title: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str = Query("", description="Filter by title substring"),
):
    """
    Return paginated list of conversations, newest first.

    GET /chat/history?page=1&page_size=20&search=leave
    """
    result = chat_db.get_conversations(
        page=page,
        page_size=page_size,
        search=search,
    )
    return result


@router.get("/{conv_id}")
def get_conversation(conv_id: str):
    """
    Return a single conversation with all its messages.

    GET /chat/history/{id}
    """
    conv = chat_db.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail=f"Conversation '{conv_id}' not found.")

    messages = chat_db.get_messages(conv_id)
    return {**conv, "messages": messages}


@router.put("/{conv_id}")
def rename_conversation(conv_id: str, body: RenameRequest):
    """
    Rename a conversation title.

    PUT /chat/history/{id}  { "title": "New title" }
    """
    if not body.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty.")

    updated = chat_db.rename_conversation(conv_id, body.title.strip())
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Conversation '{conv_id}' not found.")

    logger.info("Renamed conversation %s to '%s'", conv_id, body.title)
    return updated


@router.delete("")
def clear_all_conversations():
    """
    Delete every conversation and its messages.

    DELETE /chat/history
    """
    count = chat_db.clear_all_conversations()
    return {"message": f"Deleted {count} conversation(s)."}


@router.delete("/{conv_id}")
def delete_conversation(conv_id: str):
    """
    Delete a single conversation (cascade deletes its messages).

    DELETE /chat/history/{id}
    """
    deleted = chat_db.delete_conversation(conv_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Conversation '{conv_id}' not found.")

    return {"message": f"Conversation '{conv_id}' deleted successfully."}
