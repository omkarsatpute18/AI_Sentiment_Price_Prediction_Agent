
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from apify_client import ApifyClient

# Load credentials
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

client = ApifyClient(os.getenv("APIFY_API_TOKEN"))

# Retrieve the existing successful run
run_id = "aNDSCoAcyFan74rDa"

run = client.run(run_id).get()

if not run:
    raise RuntimeError("Existing Apify run not found.")

print("Run status:", run.status)

# Get dataset using the typed Run object
dataset_id = run.default_dataset_id

results = list(
    client.dataset(dataset_id).iterate_items()
)

# Save output
output_file = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "social_posts"
    / "lihkg_threads_api.json"
)

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=4)

print("Records recovered:", len(results))
print("Saved to:", output_file)
