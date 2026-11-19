
import os
import json
import logging
from datetime import datetime

import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "combined_social_sentiment.json"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "daily_sentiment.csv"
)

TIMEZONES = {
    "Hong Kong": "Asia/Hong_Kong",
    "South Korea": "Asia/Seoul",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def prepare_sentiment_data(records):
    """Parse timestamps and assign each record its local market date."""

    df = pd.DataFrame(records)

    # Parse mixed ISO timestamps safely
    df["parsed_timestamp"] = pd.to_datetime(
        df["timestamp"].apply(
            lambda x: datetime.fromisoformat(
                x.replace("Z", "+00:00")
            ) if isinstance(x, str) else None
        ),
        utc=True,
        errors="coerce"
    )

    if df["parsed_timestamp"].isna().any():
        logging.warning(
            "%d records have invalid timestamps",
            df["parsed_timestamp"].isna().sum()
        )

    # Convert timestamps to each market's local date
    def get_local_date(row):
        if pd.isna(row["parsed_timestamp"]):
            return None

        timezone = TIMEZONES.get(row["market"])

        if timezone is None:
            return None

        return (
            row["parsed_timestamp"]
            .tz_convert(timezone)
            .strftime("%Y-%m-%d")
        )

    df["date"] = df.apply(get_local_date, axis=1)

    # Preserve our LIHKG crowd-sentiment rule:
    # Exclude original thread posts; use replies only.
    df = df[
        (df["source"] != "LIHKG") |
        (df["content_type"] == "reply")
    ].copy()

    # Preserve stock codes as strings
    df["stock_code"] = df["stock_code"].astype(str)

    return df.dropna(subset=["date"])


def aggregate_category(df, score_column, eligible_column, prefix):
    """Aggregate one sentiment category by stock and local date."""

    eligible = df[
        (df[eligible_column] == True) &
        (df[score_column].notna())
    ].copy()

    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "market", "stock_code", "date",
                f"{prefix}_sentiment",
                f"{prefix}_post_count",
                f"{prefix}_positive_count",
                f"{prefix}_negative_count",
                f"{prefix}_neutral_count",
            ]
        )

    result = eligible.groupby(
        ["market", "stock_code", "date"]
    ).agg(
        **{
            f"{prefix}_sentiment": (score_column, "mean"),
            f"{prefix}_post_count": ("record_id", "nunique"),
            f"{prefix}_positive_count": (
                "sentiment_label",
                lambda x: (x == "positive").sum()
            ),
            f"{prefix}_negative_count": (
                "sentiment_label",
                lambda x: (x == "negative").sum()
            ),
            f"{prefix}_neutral_count": (
                "sentiment_label",
                lambda x: (x == "neutral").sum()
            ),
        }
    ).reset_index()

    return result


def main():
    logging.info("Starting daily sentiment aggregation")

    if not os.path.exists(INPUT_FILE):
        logging.error("Input file not found: %s", INPUT_FILE)
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    df = prepare_sentiment_data(records)

    # Base daily social-post counts
    daily = df.groupby(
        ["market", "stock_code", "stock_name", "date"]
    ).agg(
        total_posts=("record_id", "nunique"),
        source_count=("source", "nunique")
    ).reset_index()

    # Financial sentiment
    financial = aggregate_category(
        df,
        "financial_sentiment_score",
        "financial_sentiment_eligible",
        "financial"
    )

    # Business sentiment
    business = aggregate_category(
        df,
        "business_sentiment_score",
        "business_sentiment_eligible",
        "business"
    )

    # Merge category-level aggregates
    daily = daily.merge(
        financial,
        on=["market", "stock_code", "date"],
        how="left"
    )

    daily = daily.merge(
        business,
        on=["market", "stock_code", "date"],
        how="left"
    )

    # Number of unique posts with at least one sentiment score
    scored = df[
        df["financial_sentiment_score"].notna() |
        df["business_sentiment_score"].notna()
    ]

    scored_counts = scored.groupby(
        ["market", "stock_code", "date"]
    )["record_id"].nunique().reset_index(
        name="scored_post_count"
    )

    daily = daily.merge(
        scored_counts,
        on=["market", "stock_code", "date"],
        how="left"
    )

    daily["scored_post_count"] = (
        daily["scored_post_count"].fillna(0).astype(int)
    )

    daily = daily.sort_values(
        ["market", "stock_code", "date"]
    ).reset_index(drop=True)

    daily.to_csv(OUTPUT_FILE, index=False)

    print("\nDaily sentiment aggregation completed!")
    print("Total stock-date records:", len(daily))
    print("Stocks:", daily["stock_code"].nunique())
    print("Date range:", daily["date"].min(), "to", daily["date"].max())

    print("\nRecords by stock:")
    print(
        daily.groupby(
            ["stock_name", "stock_code"]
        ).size()
    )

    print("\nDaily sentiment data:")
    print(daily.to_string(index=False))

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
