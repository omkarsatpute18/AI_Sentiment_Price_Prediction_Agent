
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

FILE_PATH = Path(
    "data/raw/social_posts/naver_posts_api.json"
)

with open(FILE_PATH, "r", encoding="utf-8") as file:
    records = json.load(file)

print("=" * 55)
print("NAVER FINANCE - FINAL DATA VALIDATION")
print("=" * 55)

print("\nTotal records:", len(records))

# 1. Stock-wise distribution
print("\nSTOCK-WISE DISTRIBUTION")

stock_counts = Counter(
    record.get("stockCode")
    for record in records
)

for stock, count in stock_counts.items():
    print(f"{stock}: {count} posts")

# 2. Required field validation
required_fields = [
    "stockCode",
    "stockName",
    "title",
    "text",
    "author",
    "writtenAt",
    "views",
    "likes",
    "dislikes"
]

print("\nFIELD VALIDATION")

for field in required_fields:
    available = sum(
        record.get(field) is not None
        and record.get(field) != ""
        for record in records
    )

    print(f"{field}: {available}/{len(records)}")

# 3. Duplicate checking
post_ids = [
    record.get("postId")
    for record in records
    if record.get("postId")
]

print("\nDUPLICATE CHECK")
print("Unique post IDs:", len(set(post_ids)))
print("Duplicate post IDs:", len(post_ids) - len(set(post_ids)))

# 4. Date range
dates = []

for record in records:
    timestamp = record.get("writtenAt")

    if timestamp:
        dates.append(datetime.fromisoformat(timestamp))

if dates:
    print("\nDATE RANGE")
    print("Earliest:", min(dates))
    print("Latest:", max(dates))

# 5. Final status
print("\n" + "=" * 55)
print("VALIDATION COMPLETED")
print("=" * 55)
