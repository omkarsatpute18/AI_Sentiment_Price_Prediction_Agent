
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter

from transformers import pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data/processed/naver_final_relevant.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data/processed/naver_sentiment.json"
)

MODEL_NAME = "Horizon-Labs/multilingual-sentiment-small"

BATCH_SIZE = 8

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


def calculate_sentiment_score(label, confidence):
    """
    Convert sentiment prediction into a signed score.

    Positive = +confidence
    Negative = -confidence
    Neutral = 0
    """

    label = label.lower()

    if label == "positive":
        return confidence

    elif label == "negative":
        return -confidence

    elif label == "neutral":
        return 0.0

    raise ValueError(f"Unexpected sentiment label: {label}")


def load_model():

    logger.info("Loading sentiment model: %s", MODEL_NAME)

    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        device=-1
    )

    logger.info("Sentiment model loaded successfully.")

    return classifier


def analyse_records(classifier, records):

    eligible_records = [
        record for record in records
        if record.get("financial_sentiment_eligible")
        or record.get("business_sentiment_eligible")
    ]

    logger.info(
        "Total records: %s | Eligible records: %s",
        len(records),
        len(eligible_records)
    )

    texts = [
        record.get("combined_text") or record.get("text") or record.get("title", "")
        for record in eligible_records
    ]

    results = []

    for start in range(0, len(texts), BATCH_SIZE):

        batch_texts = texts[start:start + BATCH_SIZE]

        predictions = classifier(
            batch_texts,
            truncation=True,
            max_length=512,
            batch_size=BATCH_SIZE
        )

        for record, prediction in zip(
            eligible_records[start:start + BATCH_SIZE],
            predictions
        ):

            label = prediction["label"].lower()
            confidence = float(prediction["score"])

            updated_record = record.copy()

            updated_record.update({
                "sentiment_label": label,
                "sentiment_confidence": round(confidence, 4),
                "sentiment_score": round(
                    calculate_sentiment_score(label, confidence), 4
                ),
                "sentiment_model": MODEL_NAME
            })

            results.append(updated_record)

        processed = min(start + BATCH_SIZE, len(texts))

        logger.info(
            "Processed %s/%s eligible posts",
            processed,
            len(texts)
        )

    # Retain all original records, including ineligible ones.
    result_by_id = {
        str(record["post_id"]): record
        for record in results
    }

    final_records = []

    for record in records:

        post_id = str(record.get("post_id"))

        if post_id in result_by_id:
            final_records.append(result_by_id[post_id])

        else:
            skipped_record = record.copy()

            skipped_record.update({
                "sentiment_label": None,
                "sentiment_confidence": None,
                "sentiment_score": None,
                "sentiment_model": None
            })

            final_records.append(skipped_record)

    return final_records


def aggregate_sentiment(records):

    stock_data = defaultdict(
        lambda: {
            "financial": [],
            "business": []
        }
    )

    for record in records:

        score = record.get("sentiment_score")

        if score is None:
            continue

        stock_code = record["stock_code"]

        if record.get("financial_sentiment_eligible"):
            stock_data[stock_code]["financial"].append(record)

        if record.get("business_sentiment_eligible"):
            stock_data[stock_code]["business"].append(record)

    summary = {}

    for stock_code, categories in stock_data.items():

        stock_records = [
            record for record in records
            if record["stock_code"] == stock_code
        ]

        stock_name = stock_records[0]["stock_name"]

        summary[stock_code] = {
            "stock_name": stock_name
        }

        for category, category_records in categories.items():

            scores = [
                record["sentiment_score"]
                for record in category_records
            ]

            confidences = [
                record["sentiment_confidence"]
                for record in category_records
            ]

            sentiment_counts = Counter(
                record["sentiment_label"]
                for record in category_records
            )

            summary[stock_code][category] = {
                "total_posts": len(category_records),
                "average_sentiment_score": round(
                    sum(scores) / len(scores), 4
                ) if scores else None,
                "average_confidence": round(
                    sum(confidences) / len(confidences), 4
                ) if confidences else None,
                "positive_count": sentiment_counts["positive"],
                "negative_count": sentiment_counts["negative"],
                "neutral_count": sentiment_counts["neutral"]
            }

    return summary


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not records:
        raise ValueError("The input dataset is empty.")

    classifier = load_model()

    analysed_records = analyse_records(
        classifier,
        records
    )

    summary = aggregate_sentiment(analysed_records)

    analysed_count = sum(
        record["sentiment_label"] is not None
        for record in analysed_records
    )

    output = {
        "source": "Naver",
        "market": "South Korea",
        "sentiment_model": MODEL_NAME,
        "total_records": len(analysed_records),
        "analysed_unique_posts": analysed_count,
        "stock_summaries": summary,
        "records": analysed_records
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nNaver sentiment analysis completed!")
    print("Total records:", len(analysed_records))
    print("Analysed unique posts:", analysed_count)

    print("\nStock-wise sentiment summary:")

    for stock_code, data in summary.items():

        print(
            f"\n{data['stock_name']} ({stock_code})"
        )

        for category in ["financial", "business"]:

            result = data[category]

            print(f"  {category.capitalize()} sentiment:")
            print("    Posts:", result["total_posts"])
            print(
                "    Average score:",
                result["average_sentiment_score"]
            )
            print(
                "    Positive:",
                result["positive_count"],
                "| Negative:",
                result["negative_count"],
                "| Neutral:",
                result["neutral_count"]
            )

    print("\nOutput:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
