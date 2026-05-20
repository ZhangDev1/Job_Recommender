from app.retrieve import retrieve
from app.rank import rerank

def search(query: str, top_k: int = 5):
    """
    Take a search query, retrieve the top 50 or top_k * 10 candidates, then rerank the top_k candidates.
    
    Args:
        query: search query
        top_k: number of k final candidates
    Returns: 
        List of dictionaries of final reranked job results including all metadata.
    """

    # Set a candidate minimum to ensure we always have an ample number of reranking candidates later
    candidate_k = max(top_k * 10, 50)
    top_k_results = retrieve(query, top_k=candidate_k)
    reranked_results = rerank(query, top_k_results, top_k)
    return reranked_results


if __name__ == "__main__":
    test_query = "AI engineer building LLM agents and RAG pipelines with Python and vector search"
    print(search(test_query, 5)[:2])