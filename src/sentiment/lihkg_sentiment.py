
import json
import logging
from pathlib import Path

import torch
from transformers import pipeline


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data/processed/lihkg_final_relevant.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data/processed/lihkg_sentiment.json"
)

MODEL_NAME = "Horizon-Labs/multilingual-sentiment-small"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. LOAD SENTIMENT MODEL
# --------------------------------------------------

def load_model():

    logger.info("Loading sentiment model: %s", MODEL_NAME)

    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        device=-1
    )

    logger.info("Sentiment model loaded successfully.")

    return classifier


# --------------------------------------------------
# 3. CALCULATE POLARITY SCORE
# --------------------------------------------------

def calculate_polarity(label, confidence):

    label = label.lower()

    if label == "positive":
        return confidence

    elif label == "negative":
        return -confidence

    return 0.0


# --------------------------------------------------
# 4. ANALYSE SENTIMENT
# --------------------------------------------------

def analyse_sentiment(classifier, record):

    text = record.get("text", "").strip()

    prediction = classifier(
        text,
        truncation=True,
        max_length=512
    )[0]

    label = prediction["label"].lower()
    confidence = float(prediction["score"])

    return {
        "sentiment_label": label,
        "sentiment_confidence": round(confidence, 4),
        "sentiment_score": round(
            calculate_polarity(label, confidence), 4
        ),
        "sentiment_model": MODEL_NAME
    }


# --------------------------------------------------
# 5. PROCESS DATASET
# --------------------------------------------------

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    classifier = load_model()

    results = []

    for index, record in enumerate(records, start=1):

        updated_record = record.copy()

        label = record.get("relevance_label")

        if label in [
            "financially_relevant",
            "business_relevant"
        ]:

            sentiment = analyse_sentiment(
                classifier,
                record
            )

            if label == "financially_relevant":
                sentiment_type = "financial"

            else:
                sentiment_type = "business"

            updated_record.update(sentiment)

            updated_record["sentiment_type"] = sentiment_type

            logger.info(
                "[%s/%s] %s | %s | Score: %s",
                index,
                len(records),
                sentiment_type,
                sentiment["sentiment_label"],
                sentiment["sentiment_score"]
            )

        else:

            updated_record.update({
                "sentiment_label": None,
                "sentiment_confidence": None,
                "sentiment_score": None,
                "sentiment_type": None,
                "sentiment_model": None
            })

            logger.info(
                "[%s/%s] Skipped: %s",
                index,
                len(records),
                label
            )

        results.append(updated_record)

    # --------------------------------------------------
    # 6. SAVE RESULTS
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 7. SUMMARY
    # --------------------------------------------------

    analysed = [
        record for record in results
        if record.get("sentiment_label") is not None
    ]

    print("\nLIHKG sentiment analysis completed!")

    print("Total records:", len(results))
    print("Analysed records:", len(analysed))

    print(
        "Financial records:",
        sum(r.get("sentiment_type") == "financial" for r in analysed)
    )

    print(
        "Business records:",
        sum(r.get("sentiment_type") == "business" for r in analysed)
    )

    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
