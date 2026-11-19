
import json
from pathlib import Path

file_path = Path(
    "data/raw/social_posts/lihkg_threads_api.json"
)

with open(file_path, "r", encoding="utf-8") as file:
    records = json.load(file)

print("Total records:", len(records))

print("\nAvailable fields:")

if records:
    for field in records[0].keys():
        print("-", field)

print("\nSample record:")

if records:
    print(json.dumps(
        records[0],
        indent=4,
        ensure_ascii=False
    ))
