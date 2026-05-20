from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_FIELDS = ["Job Title", "Role", "Job Description", "skills", "Responsibilities"]


def _job_text(job: dict) -> str:
    return " | ".join(job[f] for f in RERANK_FIELDS if job.get(f))


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank a list of candidate jobs against a query using a cross-encoder.

    Expects candidates to already have a 'similarity_score' from the retrieval stage.
    Returns top_k results with an added 'rerank_score' and updated 'rank'.
    """
    model = CrossEncoder(RERANK_MODEL)

    pairs = [(query, _job_text(job)) for job in candidates]
    scores = model.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    results = []
    for rank, (score, job) in enumerate(ranked[:top_k], start=1):
        job = dict(job)
        job["rerank_score"] = float(score)
        job["rank"] = rank
        results.append(job)

    return results
