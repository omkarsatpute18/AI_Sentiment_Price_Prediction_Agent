
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data/raw/social_posts/naver_posts_api.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data/processed/naver_cleaned.json"
)

START_DATE = datetime.fromisoformat("2026-11-10T00:00:00+09:00")
END_DATE = datetime.fromisoformat("2026-11-20T00:00:00+09:00")


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    cleaned_records = []
    seen_ids = set()

    missing_text = 0
    duplicate_records = 0
    invalid_dates = 0
    outside_date_range = 0

    for record in records:

        post_id = record.get("postId")

        # Remove duplicate posts
        if not post_id or post_id in seen_ids:
            duplicate_records += 1
            continue

        seen_ids.add(post_id)

        # Validate and filter dates
        try:
            written_at = datetime.fromisoformat(
                record["writtenAt"]
            )
        except (ValueError, TypeError, KeyError):
            invalid_dates += 1
            continue

        if not START_DATE <= written_at < END_DATE:
            outside_date_range += 1
            continue

        title = (record.get("title") or "").strip()
        body = (record.get("text") or "").strip()

        combined_text = " ".join(
            part for part in [title, body] if part
        )

        if not combined_text:
            missing_text += 1
            continue

        cleaned_record = {
            "source": "Naver",
            "market": "South Korea",
            "exchange": record.get("market"),
            "stock_code": record.get("stockCode"),
            "stock_name": record.get("stockName"),
            "post_id": post_id,
            "url": record.get("url"),
            "title": title,
            "text": body,
            "combined_text": combined_text,
            "author": record.get("author"),
            "written_at": record.get("writtenAt"),
            "views": record.get("views", 0),
            "likes": record.get("likes", 0),
            "dislikes": record.get("dislikes", 0),
            "is_holder_verified": record.get("isHolderVerified"),
            "is_cleanbot_passed": record.get("isCleanbotPassed"),
            "post_type": record.get("postType")
        }

        cleaned_records.append(cleaned_record)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            cleaned_records,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nNaver preprocessing completed!")
    print("Original records:", len(records))
    print("Cleaned records:", len(cleaned_records))
    print("Duplicate records:", duplicate_records)
    print("Invalid dates:", invalid_dates)
    print("Outside date range:", outside_date_range)
    print("Missing text:", missing_text)

    print("\nStock-wise distribution:")

    stock_counts = {}

    for record in cleaned_records:
        code = record["stock_code"]
        stock_counts[code] = stock_counts.get(code, 0) + 1

    for code, count in stock_counts.items():
        print(code, ":", count)

    print("\nOutput:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
