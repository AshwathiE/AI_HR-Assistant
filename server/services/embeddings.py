from sentence_transformers import SentenceTransformer

# Load the embedding model only once
# This model generates 384-dimensional embeddings
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
def generate_embeddings(text_chunks): ## generates multiple embeddings 
    """
    Generate embeddings for multiple text chunks.

    Args:
        text_chunks (list): List of text chunks

    Returns:
        list: List of embedding vectors
    """

    if not text_chunks:
        return []

    embeddings = model.encode(text_chunks)

    return embeddings.tolist()
    logger.info(
    "Query embedding generated successfully"
)


import re


def build_embedding_text(
    chunk_text: str,
    document_name: str,
    chunk_number: int,
    section_title: str,
    topics: list,
):
    """
    Creates enriched text for embedding.
    Original chunk is NOT modified.
    """

    embedding_text = f"""
Document:
{document_name}

Section:
{section_title}

Topics:
{", ".join(topics)}

Content:
{chunk_text}
"""

    return embedding_text


def generate_document_embeddings(chunks, document_name):

    embedding_texts = []

    for chunk in chunks:

        embedding_texts.append(
            build_embedding_text(
                chunk_text=chunk["text"],
                document_name=document_name,
                chunk_number=chunk["chunk_number"],
                section_title=chunk["section"],
                topics=chunk["topics"],
            )
        )

    embeddings = model.encode(
        embedding_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.tolist()