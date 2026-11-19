
import json
from pathlib import Path
from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data/processed/naver_final_relevant.json"
)

MODEL_NAME = "Horizon-Labs/multilingual-sentiment-small"


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    eligible = [
        record for record in records
        if record["financial_sentiment_eligible"]
        or record["business_sentiment_eligible"]
    ]

    # Test two examples per stock
    samples = []
    stock_counts = {}

    for record in eligible:
        code = record["stock_code"]

        if stock_counts.get(code, 0) < 2:
            samples.append(record)
            stock_counts[code] = stock_counts.get(code, 0) + 1

    print("Loading multilingual sentiment model...")

    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        device=-1
    )

    print("\nKorean sentiment test results:\n")

    for record in samples:

        text = record["combined_text"]

        prediction = classifier(
            text,
            truncation=True,
            max_length=512
        )[0]

        print("Company:", record["stock_name"])
        print("Title:", record["title"])
        print("Sentiment:", prediction["label"])
        print("Confidence:", round(prediction["score"], 4))
        print("-" * 60)


if __name__ == "__main__":
    main()
