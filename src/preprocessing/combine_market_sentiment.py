
import json
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LIHKG_FILE = (
    PROJECT_ROOT / "data/processed/lihkg_sentiment.json"
)

NAVER_FILE = (
    PROJECT_ROOT / "data/processed/naver_sentiment.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data/processed/combined_social_sentiment.json"
)

SUMMARY_FILE = (
    PROJECT_ROOT / "data/processed/unified_market_summary.json"
)


def load_json(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_lihkg(records):

    normalized = []

    for index, record in enumerate(records, start=1):

        sentiment_type = record.get("sentiment_type")
        sentiment_score = record.get("sentiment_score")
        sentiment_label = record.get("sentiment_label")

        financial_eligible = (
            sentiment_type == "financial"
        )

        business_eligible = (
            sentiment_type == "business"
        )

        financial_score = (
            sentiment_score
            if financial_eligible and sentiment_score is not None
            else None
        )

        business_score = (
            sentiment_score
            if business_eligible and sentiment_score is not None
            else None
        )

        normalized.append({
            "source": "LIHKG",
            "market": "Hong Kong",
            "stock_code": record.get("stock_code"),
            "stock_name": record.get("stock_name"),
            "record_id": f"LIHKG_{record.get('thread_id')}_{index}",
            "post_id": None,
            "thread_id": record.get("thread_id"),
            "author": record.get("author"),
            "title": record.get("title"),
            "text": record.get("text"),
            "timestamp": record.get("created_at"),
            "url": record.get("thread_url"),

            "content_type": (
                "original_thread_post"
                if record.get("title")
                else "reply"
            ),

            "relevance_label": record.get("relevance_label"),

            "financial_sentiment_eligible": financial_eligible,
            "business_sentiment_eligible": business_eligible,

            "sentiment_label": sentiment_label,
            "sentiment_confidence": record.get(
                "sentiment_confidence"
            ),
            "sentiment_score": sentiment_score,

            "financial_sentiment_score": financial_score,
            "business_sentiment_score": business_score,

            "sentiment_model": record.get("sentiment_model")
        })

    return normalized


def normalize_naver(data):

    records = data["records"]
    normalized = []

    for record in records:

        sentiment_score = record.get("sentiment_score")

        financial_eligible = record.get(
            "financial_sentiment_eligible", False
        )

        business_eligible = record.get(
            "business_sentiment_eligible", False
        )

        normalized.append({
            "source": "Naver",
            "market": "South Korea",
            "stock_code": record.get("stock_code"),
            "stock_name": record.get("stock_name"),
            "record_id": f"NAVER_{record.get('post_id')}",
            "post_id": record.get("post_id"),
            "thread_id": None,
            "author": record.get("author"),
            "title": record.get("title"),
            "text": record.get("combined_text") or record.get("text"),
            "timestamp": record.get("written_at"),
            "url": record.get("url"),

            "content_type": "discussion_post",

            "relevance_label": record.get("relevance_label"),

            "financial_sentiment_eligible": financial_eligible,
            "business_sentiment_eligible": business_eligible,

            "sentiment_label": record.get("sentiment_label"),
            "sentiment_confidence": record.get(
                "sentiment_confidence"
            ),
            "sentiment_score": sentiment_score,

            "financial_sentiment_score": (
                sentiment_score
                if financial_eligible and sentiment_score is not None
                else None
            ),

            "business_sentiment_score": (
                sentiment_score
                if business_eligible and sentiment_score is not None
                else None
            ),

            "sentiment_model": record.get("sentiment_model")
        })

    return normalized


def calculate_category_summary(records, category):

    score_field = f"{category}_sentiment_score"

    # Only analysed records with an available category score.
    eligible_records = [
        record for record in records
        if record.get(score_field) is not None
        and (
            record["source"] != "LIHKG"
            or record["content_type"] == "reply"
        )
    ]

    scores = [
        record[score_field]
        for record in eligible_records
    ]

    sentiment_counts = Counter(
        record["sentiment_label"]
        for record in eligible_records
    )

    return {
        "analysed_posts": len(scores),

        "average_sentiment_score": (
            round(sum(scores) / len(scores), 4)
            if scores else None
        ),

        "positive_count": sentiment_counts["positive"],
        "negative_count": sentiment_counts["negative"],
        "neutral_count": sentiment_counts["neutral"]
    }


def build_summary(records):

    grouped_records = defaultdict(list)

    for record in records:
        key = (
            record["market"],
            record["stock_code"]
        )

        grouped_records[key].append(record)

    summary = {}

    for (market, stock_code), stock_records in grouped_records.items():

        stock_name = stock_records[0]["stock_name"]

        summary_key = f"{market}_{stock_code}"

        summary[summary_key] = {
            "market": market,
            "stock_code": stock_code,
            "stock_name": stock_name,

            "total_records": len(stock_records),

            "analysed_unique_posts": sum(
                record["sentiment_label"] is not None
                for record in stock_records
            ),

            "financial_sentiment": calculate_category_summary(
                stock_records, "financial"
            ),

            "business_sentiment": calculate_category_summary(
                stock_records, "business"
            )
        }

    return summary


def main():

    print("Loading Hong Kong and South Korean datasets...")

    lihkg_data = load_json(LIHKG_FILE)
    naver_data = load_json(NAVER_FILE)

    if not isinstance(lihkg_data, list):
        raise ValueError("Expected LIHKG sentiment data to be a list.")

    if not isinstance(naver_data, dict) or "records" not in naver_data:
        raise ValueError("Expected Naver sentiment data to contain a records list.")

    lihkg_records = normalize_lihkg(lihkg_data)
    naver_records = normalize_naver(naver_data)

    combined_records = lihkg_records + naver_records

    summary = build_summary(combined_records)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            combined_records,
            file,
            ensure_ascii=False,
            indent=4
        )

    with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nMarket data combination completed!")

    print("LIHKG records:", len(lihkg_records))
    print("Naver records:", len(naver_records))
    print("Combined records:", len(combined_records))

    analysed = sum(
        record["sentiment_label"] is not None
        for record in combined_records
    )

    print("Analysed records:", analysed)

    print("\nMarket-wise summary:")

    for key, stock in summary.items():

        print(
            f"\n{stock['stock_name']} "
            f"({stock['stock_code']}) - {stock['market']}"
        )

        print(
            "  Financial score:",
            stock["financial_sentiment"]["average_sentiment_score"]
        )

        print(
            "  Business score:",
            stock["business_sentiment"]["average_sentiment_score"]
        )

    print("\nCombined dataset:", OUTPUT_FILE)
    print("Market summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()
