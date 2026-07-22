import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from routers import chat, upload

from embeddings import generate_embedding
from vector_db import get_all_documents

# -----------------------------
# Project Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CLIENT_DIR = BASE_DIR / "client"
STATIC_DIR = CLIENT_DIR / "static"
TEMPLATE_DIR = CLIENT_DIR / "templates"

UPLOAD_FOLDER = BASE_DIR / "server" / "uploads"

# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI(
    title="AI Company Policy Assistant",
    description="RAG-based Company Policy Assistant",
    version="1.0.0",
)

# -----------------------------
# CORS
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Static Files
# -----------------------------
# Uploaded PDFs will be available at:
# http://localhost:8000/files/EmployeeBenefits.pdf
# -----------------------------
app.mount(
    "/static",
    StaticFiles(directory=CLIENT_DIR / "static"),
    name="static",
)
# -----------------------------
# Templates
# -----------------------------

templates = Jinja2Templates(
    directory=str(TEMPLATE_DIR)
)

# -----------------------------
# Routers
# -----------------------------

app.include_router(
    upload.router,
    prefix="/upload",
    tags=["Upload"],
)

app.include_router(
    chat.router,
    prefix="/chat",
    tags=["Chat"],
)

# -----------------------------
# Home Page
# -----------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    try:
        files = sorted(
            [
                f
                for f in os.listdir(UPLOAD_FOLDER)
                if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))
            ]
        )
    except Exception:
        files = []

    try:

        records = get_all_documents()

        grouped = {}

        for record in records:

            source = record.payload.get("source", "Unknown")
            chunk_num = record.payload.get("chunk_number", "?")
            text = record.payload.get("text", "")
            point_id = str(record.id)

            grouped.setdefault(source, []).append(
                {
                    "id": point_id,
                    "chunk_number": chunk_num,
                    "text_preview": text[:200]
                    + ("..." if len(text) > 200 else ""),
                    "full_text": text,
                    "char_count": len(text),
                }
            )

        for source in grouped:
            grouped[source].sort(
                key=lambda x: x["chunk_number"]
            )

        qdrant_groups = [
            {
                "source": source,
                "chunks": chunks,
                "total_chunks": len(chunks),
            }
            for source, chunks in sorted(grouped.items())
        ]

        total_points = sum(
            len(g["chunks"]) for g in qdrant_groups
        )

    except Exception:

        qdrant_groups = []
        total_points = 0

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "documents": files,
            "qdrant_groups": qdrant_groups,
            "total_points": total_points,
        },
    )

# -----------------------------
# Uploaded Documents
# -----------------------------

@app.get("/documents")
def list_documents():

    try:

        files = sorted(
            [
                f
                for f in os.listdir(UPLOAD_FOLDER)
                if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))
            ]
        )

        return {
            "documents": files
        }

    except Exception as e:

        return {
            "documents": [],
            "error": str(e),
        }

# -----------------------------
# View Database
# -----------------------------

@app.get("/view-db")
def view_db():
    return get_all_documents()

# -----------------------------
# Embedding Test
# -----------------------------

@app.get("/embedding")
def get_embedding_endpoint(text: str):

    embedding = generate_embedding(text)

    return {
        "text": text,
        "dimensions": len(embedding),
        "first_10_values": embedding[:10],
    }

# -----------------------------
# Favicon
# -----------------------------

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)