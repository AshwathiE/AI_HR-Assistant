from sentence_transformers import CrossEncoder  ## crossencoders take both question and answer in a document together and output a single score, which is more accurate than using separate embeddings for question and answer
from logger import logger


# Load model once when server starts
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_documents(question, results, top_k=5): ## results : documents returened from the qdrant, top_k: documents to be returned after reranking

    if not results:
        logger.warning("No documents received for reranking")
        return []


    pairs = [] ## list holding question and document pairs for reranking

    for result in results: ## loops through retrived documents from results and creates a list of question and document pairs for reranking
        pairs.append(
            [
                question,       ## create a pair of question and document text for reranking and is appended to the pairs list
                result["text"]
            ]
        )


    # Generate relevance scores
    scores = reranker_model.predict(pairs) ## cross encoder reads each pair together and predict thier relevance score


    # Attach scores
    for result, score in zip(results, scores): ## loops through the results and scores and attaches the score to the result dictionary
        result["rerank_score"] = float(score) ## every retrieved document is assigned a rerank score based on the cross encoder prediction


    # Sort based on cross encoder score
    ranked_results = sorted(
        results,
        key=lambda x: x["rerank_score"], ##compare sorted results based on the rerank score assigned to each document
        reverse=True ## the order is computed in descending order so that the most relevant documents are at the top of the list
    )


    # Keep only required top_k results
    RERANK_THRESHOLD = 2  ## filteration threshold 

    filtered_results = [
        result
        for result in ranked_results
        if result["rerank_score"] > RERANK_THRESHOLD
    ]

    final_results = filtered_results[:top_k] ## only the top_k results are kept in the final_results list, which is returned to the user


    # ----------------------------
    # Logging Cross Encoder Output
    # ----------------------------
    logger.info(
        f"Cross Encoder reranked results: {len(final_results)}"
    )


    for result in final_results:
        logger.info(
            f"Rerank score: {result.get('rerank_score')} | "
            f"Document: {result.get('document')} | "
            f"Chunk: {result.get('chunk_number')}"
        )


    return final_results