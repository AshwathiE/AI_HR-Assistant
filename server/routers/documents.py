from fastapi import APIRouter

from services.vector_db import get_all_documents

router = APIRouter()


@router.get("/")
def list_documents_from_db():     ## returns list of uploaded documents
    records = get_all_documents()

    documents = set()
    for record in records:         ## traverse through records
        source = record.payload.get("source", "")  ## get source from payload
        if source:
            documents.add(source)

    return {
        "documents": sorted(list(documents))
    }
