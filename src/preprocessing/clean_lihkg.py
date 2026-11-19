
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data/raw/social_posts/lihkg_threads_api.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data/processed/lihkg_cleaned.json"
)

START_DATE = datetime(2026, 11, 10, tzinfo=timezone.utc)
END_DATE = datetime(2026, 11, 20, tzinfo=timezone.utc)

STOCK_CODE = "0700"
STOCK_NAME = "Tencent"


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    cleaned = []
    seen = set()

    for record in records:
        content = record.get("own_text") or record.get("content")

        if not isinstance(content, str) or not content.strip():
            continue

        created_at = record.get("created_at")
        if not created_at:
            continue

        timestamp = datetime.fromisoformat(
            created_at.replace("Z", "+00:00")
        )

        if not START_DATE <= timestamp < END_DATE:
            continue

        thread_id = str(record.get("thread_id", ""))
        author = record.get("author")

        # Avoid dropping different comments with identical text.
        key = (thread_id, author, created_at, content.strip())

        if key in seen:
            continue

        seen.add(key)

        cleaned.append({
            "source": "LIHKG",
            "market": "HK",
            "stock_code": STOCK_CODE,
            "stock_name": STOCK_NAME,
            "thread_id": thread_id,
            "author": author,
            "title": record.get("title"),
            "text": content.strip(),
            "created_at": created_at,
            "thread_url": record.get("thread_url"),
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(cleaned, file, ensure_ascii=False, indent=2)

    print("Original records:", len(records))
    print("Cleaned records:", len(cleaned))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
