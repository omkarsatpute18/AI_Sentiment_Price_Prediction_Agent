
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "social_posts"
    / "lihkg_search.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "social_posts"
    / "lihkg_search_filtered.json"
)

START_DATE = datetime(2026, 11, 10, tzinfo=timezone.utc)
END_DATE = datetime(2026, 11, 20, tzinfo=timezone.utc)

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    records = json.load(file)

filtered_records = []

for record in records:
    created_at = record.get("created_at")

    if not created_at:
        continue

    try:
        post_date = datetime.fromisoformat(
            created_at.replace("Z", "+00:00")
        )

        if START_DATE <= post_date < END_DATE:
            filtered_records.append(record)

    except ValueError:
        print("Invalid date:", created_at)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        filtered_records,
        file,
        ensure_ascii=False,
        indent=4
    )

print("Original records:", len(records))
print("Records within date range:", len(filtered_records))
print("Filtered file:", OUTPUT_FILE)

print("\nFiltered thread titles:")

for record in filtered_records:
    print(
        record.get("source_query"),
        "|",
        record.get("thread_id"),
        "|",
        record.get("title")
    )

