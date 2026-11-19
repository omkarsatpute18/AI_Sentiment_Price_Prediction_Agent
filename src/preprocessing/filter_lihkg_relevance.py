
import json
from pathlib import Path


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data/processed/lihkg_cleaned.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data/processed/lihkg_relevance_review.json"
)


# --------------------------------------------------
# 2. RELEVANCE KEYWORDS
# --------------------------------------------------

FINANCIAL_KEYWORDS = [
    "股價", "股市", "股票", "升幅", "跌幅",
    "股東", "投資", "買入", "沽出", "市值",
    "盈利", "收入", "財報", "業績",
    "share price", "stock", "shares",
    "investor", "investment", "market cap",
    "revenue", "profit", "earnings"
]

BUSINESS_KEYWORDS = [
    "業務", "公司", "競爭", "對手", "市場",
    "AI", "人工智能", "增長", "策略",
    "字節", "抖音", "阿里", "deepseek",
    "business", "competition", "competitor",
    "strategy", "growth", "technology"
]


# --------------------------------------------------
# 3. RELEVANCE CLASSIFICATION
# --------------------------------------------------

def classify_relevance(record):

    text = record.get("text", "")
    title = record.get("title") or ""

    combined_text = f"{title} {text}".lower()

    financial_matches = [
        keyword for keyword in FINANCIAL_KEYWORDS
        if keyword.lower() in combined_text
    ]

    business_matches = [
        keyword for keyword in BUSINESS_KEYWORDS
        if keyword.lower() in combined_text
    ]

    # Direct financial discussion
    if financial_matches:
        return {
            "relevance_label": "financially_relevant",
            "relevance_reason": "Contains direct financial or stock-market terms.",
            "matched_keywords": financial_matches
        }

    # Business strategy and competition discussion
    if business_matches:
        return {
            "relevance_label": "business_relevant",
            "relevance_reason": "Discusses company strategy, business or competition.",
            "matched_keywords": business_matches
        }

    # No clear financial or business signal
    return {
        "relevance_label": "manual_review",
        "relevance_reason": "No clear financial or business keywords detected.",
        "matched_keywords": []
    }


# --------------------------------------------------
# 4. PROCESS DATASET
# --------------------------------------------------

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    reviewed_records = []

    for record in records:

        classification = classify_relevance(record)

        reviewed_record = {
            **record,
            **classification
        }

        reviewed_records.append(reviewed_record)

    # Save every record, including manual-review cases
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            reviewed_records,
            file,
            ensure_ascii=False,
            indent=4
        )

    # Summary
    counts = {}

    for record in reviewed_records:
        label = record["relevance_label"]
        counts[label] = counts.get(label, 0) + 1

    print("\nLIHKG relevance filtering completed!")
    print("Total records:", len(reviewed_records))

    for label, count in counts.items():
        print(f"{label}: {count}")

    print("\nOutput:", OUTPUT_FILE)


if __name__ == "__main__":
    main()

