import re
from sentence_transformers import SentenceTransformer

# Load the embedding model only once.
# This model generates 768-dimensional embeddings.
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


def generate_embeddings(text_chunks):  ## to generate embediings for user querry
    """
    Generate embeddings for multiple text chunks (or a single string).

    Args:
        text_chunks (list | str): Text chunks to embed.

    Returns:
        list: List of embedding vectors.
    """
    if not text_chunks:
        return []

    embeddings = model.encode(text_chunks)
    return embeddings.tolist()


def build_embedding_text(topics: list) -> str: ## to genrate embeddings only from topics 
    """
    Creates embedding text from extracted topics ONLY.

    No chunk text, section title, document name, or chunk number is
    included — purely topics so that semantic retrieval is driven
    entirely by topic similarity.

    Args:
        topics: List of topic strings extracted from the chunk.

    Returns:
        A newline-joined string of topics ready for embedding.
    """
    if not topics:
        return ""

    # One topic per line for clean embedding input
    return "\n".join(topics)


def generate_document_embeddings(chunks: list, document_name: str) -> list:
    """
    Embeddings are generated ONLY from chunk["topics"].
    The original chunk text is never used here.

    Args:
        chunks:        List of dicts, each containing at least {"topics": [...]}.
        document_name: Name of the source document (kept for logging only).

    Returns:
        List of embedding vectors (one per chunk), as Python lists.
    """
    embedding_texts = [
        build_embedding_text(chunk["topics"])
        for chunk in chunks
    ]

    embeddings = model.encode(
        embedding_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.tolist()