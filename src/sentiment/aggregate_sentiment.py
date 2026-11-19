
import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data/processed/lihkg_sentiment.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data/processed/lihkg_aggregated_sentiment.json"
)


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    sentiment_groups = defaultdict(list)

    original_posts = 0
    skipped_records = 0

    for record in records:

        # Original posts have a title; replies do not.
        if record.get("title"):
            original_posts += 1
            continue

        sentiment_type = record.get("sentiment_type")
        sentiment_score = record.get("sentiment_score")

        if sentiment_type and sentiment_score is not None:
            sentiment_groups[sentiment_type].append(
                float(sentiment_score)
            )
        else:
            skipped_records += 1

    results = {}

    for category in ["financial", "business"]:

        scores = sentiment_groups[category]

        results[category] = {
            "total_replies": len(scores),
            "average_sentiment_score": (
                round(sum(scores) / len(scores), 4)
                if scores else None
            )
        }

    output = {
        "source": "LIHKG",
        "market": "Hong Kong",
        "stock": "Tencent",
        "original_posts_excluded": original_posts,
        "unclassified_replies_skipped": skipped_records,
        "sentiment_summary": results
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=4)

    print("\nSentiment aggregation completed!")
    print(json.dumps(output, ensure_ascii=False, indent=4))
    print("\nOutput:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
