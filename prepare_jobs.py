import pandas as pd
import json

# Load job descriptions into a dataframe
df = pd.read_csv("job_descriptions.csv", nrows=300)

# Convert DataFrame to JSON
json_data = df.to_json(orient="records", indent=4)

# Print JSON
print(json_data)

# Optional: Save JSON to a file
with open("data\jobs_clean.json", "w") as f:
    f.write(json_data)