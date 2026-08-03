##from sentence_transformers import CrossEncoder
##from logger import logger
##import math

#Load model once when server starts
##reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


##def rerank_documents(question, results, top_k=10):

    ##if not results:
        ##logger.warning("No documents received for reranking")
        ##return []

    # Create (question, document) pairs
    ##pairs = []

    ##for result in results:
        ##pairs.append([
            ##question,
            ##result["text"]
        ##])

    #Generate raw CrossEncoder scores (logits)
    ##scores = reranker_model.predict(pairs)

    # Convert logits to probabilities (0 to 1)
    ##for result, score in zip(results, scores):
        ##probability = 1 / (1 + math.exp(-score))
        ##result["rerank_score"] = probability

    # Sort by rerank score (highest first)
    ##ranked_results = sorted(
        ##results,
        ##key=lambda x: x["rerank_score"],
        ##reverse=True
    ##)

    # Keep only top_k results
    ##final_results = ranked_results[:top_k]

    ## Logging
    ##logger.info(
        ##f"Cross Encoder reranked results: {len(final_results)}"
    ##)

    ##for result in final_results:
        ##logger.info(
            ##f"Rerank score: {result.get('rerank_score'):.6f} | "
            ##f"Document: {result.get('document')} | "
            ##f"Chunk: {result.get('chunk_number')}"
        ##)

    ##return final_results