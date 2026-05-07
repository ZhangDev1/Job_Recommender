import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

JOBS_PATH = DATA_DIR / "jobs_clean.json"
EMBEDDINGS_PATH = DATA_DIR / "job_embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"


def build_search_text(job: dict) -> str:
    """
    Convert structured job data into rich searchable text
    for embeddings / semantic search.
    """

    # Basic fields
    job_id = job.get("Job Id", "")
    title = job.get("Job Title", "")
    role = job.get("Role", "")
    company = job.get("Company", "")
    description = job.get("Job Description", "")
    responsibilities = job.get("Responsibilities", "")
    skills = job.get("skills", "")
    benefits = job.get("Benefits", "")

    # Metadata
    experience = job.get("Experience", "")
    qualifications = job.get("Qualifications", "")
    salary = job.get("Salary Range", "")
    work_type = job.get("Work Type", "")
    preference = job.get("Preference", "")

    # Location
    location = job.get("location", "")
    country = job.get("Country", "")
    latitude = job.get("latitude", "")
    longitude = job.get("longitude", "")

    # Company profile parsing
    company_profile_raw = job.get("Company Profile", "{}")

    try:
        company_profile = json.loads(company_profile_raw)
    except Exception:
        company_profile = {}

    sector = company_profile.get("Sector", "")
    industry = company_profile.get("Industry", "")
    city = company_profile.get("City", "")
    state = company_profile.get("State", "")
    website = company_profile.get("Website", "")
    ceo = company_profile.get("CEO", "")

    # Final embedding text
    text = f"""
    Job ID: {job_id}

    Job Title: {title}
    Role: {role}

    Company: {company}
    Sector: {sector}
    Industry: {industry}
    CEO: {ceo}
    Website: {website}

    Location: {location}, {country}
    Company HQ: {city}, {state}
    Coordinates: {latitude}, {longitude}

    Experience Required: {experience}
    Qualifications: {qualifications}
    Salary Range: {salary}
    Work Type: {work_type}
    Candidate Preference: {preference}

    Skills:
    {skills}

    Benefits:
    {benefits}

    Responsibilities:
    {responsibilities}

    Job Description:
    {description}
    """

    return text.strip()


def load_jobs():
    with open(JOBS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("Loading jobs...")
    jobs = load_jobs()

    print("Preparing searchable text...")
    documents = [build_search_text(job) for job in jobs]

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"Embedding shape: {embeddings.shape}")

    print("Saving embeddings...")
    np.save(EMBEDDINGS_PATH, embeddings)

    print("Done!")


if __name__ == "__main__":
    main()