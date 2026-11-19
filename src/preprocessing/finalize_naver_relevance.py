
import json
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data/processed/naver_relevance_review.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data/processed/naver_final_relevant.json"
)

# Human-reviewed classifications based on post IDs.
MANUAL_LABELS = {
    # Samsung Electronics
    "432512990": "irrelevant",
    "432512966": "irrelevant",
    "432512955": "manual_review",
    "432512954": "financially_relevant",
    "432512934": "manual_review",

    # SK Hynix
    "432513013": "manual_review",
    "432512956": "irrelevant",
    "432512951": "irrelevant",
    "432512945": "manual_review",
    "432512928": "financially_relevant",

    # Hyundai Motor
    "432512370": "irrelevant",
    "432511407": "business_relevant",
    "432511177": "manual_review",
    "432510983": "business_relevant",
    "432510429": "financially_relevant",

    # LG Energy Solution
    "432510749": "financially_relevant",
    "432509171": "financially_relevant",
    "432507982": "manual_review",
    "432507845": "irrelevant",
    "432506672": "financially_relevant",

    # Samsung Biologics
    "432472162": "financially_relevant",
    "432470980": "financially_relevant",
    "432447839": "business_relevant",
    "432440441": "business_relevant",
    "432433003": "irrelevant",
}


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    results = []
    matched_ids = set()

    for record in records:

        updated_record = record.copy()
        post_id = str(record.get("post_id"))

        if post_id in MANUAL_LABELS:
            updated_record["relevance_label"] = MANUAL_LABELS[post_id]
            updated_record["classification_method"] = "manual"
            matched_ids.add(post_id)
        else:
            updated_record["classification_method"] = "keyword_based"

        # Separate eligibility flags
        label = updated_record["relevance_label"]

        updated_record["financial_sentiment_eligible"] = (
            label in [
                "financially_relevant",
                "financial_and_business_relevant"
            ]
        )

        updated_record["business_sentiment_eligible"] = (
            label in [
                "business_relevant",
                "financial_and_business_relevant"
            ]
        )

        results.append(updated_record)

    missing_ids = set(MANUAL_LABELS) - matched_ids

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4
        )

    label_counts = Counter(
        record["relevance_label"] for record in results
    )

    stock_counts = defaultdict(Counter)

    for record in results:
        stock_counts[record["stock_name"]][
            record["relevance_label"]
        ] += 1

    print("\nNaver relevance finalization completed!")
    print("Total records:", len(results))
    print("Manual classifications applied:", len(matched_ids))
    print("Unmatched manual IDs:", len(missing_ids))

    if missing_ids:
        print("Missing IDs:", sorted(missing_ids))

    print("\nFinal relevance distribution:")

    for label, count in label_counts.items():
        print(f"{label}: {count}")

    print("\nStock-wise distribution:")

    for stock, counts in stock_counts.items():
        print(f"\n{stock}:")

        for label, count in counts.items():
            print(f"  {label}: {count}")

    print("\nOutput:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
