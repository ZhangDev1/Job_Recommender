from pathlib import Path
import pandas as pd
import json


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "job_descriptions.csv"
OUTPUT_FILE = BASE_DIR / "data" / "jobs_clean.json"


def load_jobs(csv_path: Path, limit: int = 300) -> pd.DataFrame:
    """Load jobs from CSV into a DataFrame."""
    return pd.read_csv(csv_path, nrows=limit)


def clean_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning."""
    
    # Drop duplicate rows
    df = df.drop_duplicates()

    # Remove rows with missing job_description
    df = df.dropna(subset=["Job Description"])

    return df


def save_jobs_json(df: pd.DataFrame, output_path: Path):
    """Save DataFrame as formatted JSON."""

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = df.to_dict(orient="records")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(records)} jobs to {output_path}")


def main():
    df = load_jobs(INPUT_FILE)

    print(f"Loaded {len(df)} jobs")

    df = clean_jobs(df)

    print(f"After cleaning: {len(df)} jobs")

    save_jobs_json(df, OUTPUT_FILE)


if __name__ == "__main__":
    main()