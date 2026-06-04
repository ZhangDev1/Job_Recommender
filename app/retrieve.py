import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download


JOBS_FILENAME = "jobs_clean.json"
EMBEDDINGS_FILENAME = "job_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_artifacts():
    jobs_path = hf_hub_download(repo_id="devin-z/job-pipeline-data", filename=JOBS_FILENAME, repo_type="dataset")
    with open(jobs_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    embeddings = np.load(hf_hub_download(repo_id="devin-z/job-pipeline-data", filename=EMBEDDINGS_FILENAME, repo_type="dataset"))
    # Normalize rows so dot product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)
    return jobs, embeddings


def load_model():
    model = SentenceTransformer(MODEL_NAME)
    return model


def retrieve(query: str, model, jobs, job_embeddings, top_k: int = 50) -> list[dict]:
    """
    Embed a query and return the top_k most similar jobs.

    Each result dict contains all original job fields plus:
      - "similarity_score": float in [-1, 1], higher is more relevant
      - "rank": 1-based rank
    """

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
    jobs, job_embeddings = load_artifacts()
    model = load_model()
    results = retrieve(query, model, jobs, job_embeddings)

    for job in results:
        print(f"[{job['rank']}] {job['Job Title']} — {job['Company']}  (score: {job['similarity_score']:.4f})")
        print(f"     Role: {job['Role']}")
        print(f"     Skills: {job['skills'][:120]}...")
        print()
