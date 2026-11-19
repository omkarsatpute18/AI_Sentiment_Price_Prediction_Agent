
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data/processed/lihkg_relevance_review.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data/processed/lihkg_final_relevant.json"
)

# Manual decisions based on inspecting all nine records.
# Keys are author names and thread IDs, not text positions.
MANUAL_LABELS = {
    ("4180878", "李知恩"): "business_relevant",
    ("4180878", "精密周長大諗頭"): "business_relevant",
    ("4180878", "來自篤精的你"): "manual_review",
    ("4180878", "小粉紅祖宗"): "exclude",
    ("4181061", "九龍治水"): "business_relevant",
    ("4181061", "TACO"): "financially_relevant",
    ("4181061", "大角倔喜"): "business_relevant",
    ("4181061", "日本"): "manual_review",
    ("4181061", "初代超人"): "manual_review",
    ("4181061", "中島美雪"): "business_relevant",
}


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    final_records = []

    for record in records:

        key = (
            str(record.get("thread_id")),
            record.get("author")
        )

        label = MANUAL_LABELS.get(key, "manual_review")

        updated_record = {
            **record,
            "relevance_label": label,
            "financial_sentiment_eligible": (
                label == "financially_relevant"
            ),
            "business_sentiment_eligible": (
                label == "business_relevant"
            )
        }

        # Excluded records are retained for audit,
        # but will not enter sentiment analysis.
        final_records.append(updated_record)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            final_records,
            file,
            ensure_ascii=False,
            indent=4
        )

    counts = {}

    for record in final_records:
        label = record["relevance_label"]
        counts[label] = counts.get(label, 0) + 1

    print("\nManual relevance review completed!")
    print("Total records:", len(final_records))

    for label, count in counts.items():
        print(f"{label}: {count}")

    print(
        "Financial sentiment eligible:",
        sum(r["financial_sentiment_eligible"] for r in final_records)
    )

    print(
        "Business sentiment eligible:",
        sum(r["business_sentiment_eligible"] for r in final_records)
    )

    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
