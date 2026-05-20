import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

JOBS_PATH = DATA_DIR / "jobs_clean.json"
EMBEDDINGS_PATH = DATA_DIR / "job_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_artifacts():
    with open(JOBS_PATH, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)
    # Normalize rows so dot product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)
    return jobs, embeddings


def retrieve(query: str, top_k: int = 50) -> list[dict]:
    """
    Embed a query and return the top_k most similar jobs.

    Each result dict contains all original job fields plus:
      - "similarity_score": float in [-1, 1], higher is more relevant
      - "rank": 1-based rank
    """
    jobs, job_embeddings = load_artifacts()

    model = SentenceTransformer(MODEL_NAME)
    query_vec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

    # @ is python's built in matrix multiplcation operator
    scores = job_embeddings @ query_vec

    # Sort the jobs by similarity
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        job = dict(jobs[idx])
        job["similarity_score"] = float(scores[idx])
        job["rank"] = rank
        results.append(job)

    return results


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "AI engineer building LLM agents and RAG pipelines with Python and vector search"
    )

    print(f"Query: {query}\n")
    results = retrieve(query)

    for job in results:
        print(f"[{job['rank']}] {job['Job Title']} — {job['Company']}  (score: {job['similarity_score']:.4f})")
        print(f"     Role: {job['Role']}")
        print(f"     Skills: {job['skills'][:120]}...")
        print()
