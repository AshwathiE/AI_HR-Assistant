from sentence_transformers import SentenceTransformer

# Load the embedding model only once
# This model generates 384-dimensional embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embedding(text: str):
    """
    Generate embedding for a single text string.

    Args:
        text (str): Input text

    Returns:
        list: Embedding vector
    """
    if not text.strip():
        return []

    embedding = model.encode(text) #check the datatype of this ---string to  numpy ndimensional array with floating point numbers

    return embedding.tolist() ##converts the numpy ndimensional array to list of 384 floating point numbers
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
