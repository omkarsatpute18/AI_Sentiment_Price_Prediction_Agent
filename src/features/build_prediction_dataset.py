
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PRICE_FILE = ROOT / "data/processed/stock_features.csv"
SENTIMENT_FILE = ROOT / "data/processed/daily_sentiment.csv"
OUTPUT_FILE = ROOT / "data/processed/time_aware_dataset.csv"

# Load datasets
prices = pd.read_csv(PRICE_FILE, dtype={"stock_code": str})
sentiment = pd.read_csv(SENTIMENT_FILE, dtype={"stock_code": str})

prices["date"] = pd.to_datetime(prices["date"])
sentiment["date"] = pd.to_datetime(sentiment["date"])

# Validate merge keys
if prices.duplicated(["stock_code", "date"]).any():
    raise ValueError("Duplicate stock/date rows in price features.")

if sentiment.duplicated(["stock_code", "date"]).any():
    raise ValueError("Duplicate stock/date rows in sentiment data.")

# Sort records
prices = prices.sort_values(["stock_code", "date"])
sentiment = sentiment.sort_values(["stock_code", "date"])

# Keep sentiment columns separate from price columns
sentiment_columns = [
    "stock_code",
    "date",
    "total_posts",
    "source_count",
    "financial_sentiment",
    "financial_post_count",
    "business_sentiment",
    "business_post_count",
    "scored_post_count",
]

sentiment = sentiment[sentiment_columns].copy()
sentiment = sentiment.rename(columns={"date": "sentiment_date"})

# Align sentiment separately for each stock
integrated_parts = []

for stock_code, stock_prices in prices.groupby("stock_code"):
    stock_sentiment = sentiment[
        sentiment["stock_code"] == stock_code
    ].drop(columns=["stock_code"])

    stock_prices = stock_prices.sort_values("date")

    if stock_sentiment.empty:
        result = stock_prices.copy()
        for column in sentiment_columns[2:]:
            result[column] = pd.NA
        result["sentiment_date"] = pd.NaT
    else:
        result = pd.merge_asof(
            stock_prices,
            stock_sentiment.sort_values("sentiment_date"),
            left_on="date",
            right_on="sentiment_date",
            direction="backward",
            tolerance=pd.Timedelta(days=7),
        )

    integrated_parts.append(result)

# Combine all stocks
integrated = pd.concat(integrated_parts, ignore_index=True)
integrated = integrated.sort_values(["stock_code", "date"])

# Sentiment availability and age
integrated["sentiment_available"] = (
    integrated["sentiment_date"].notna().astype(int)
)

integrated["sentiment_age_days"] = (
    integrated["date"] - integrated["sentiment_date"]
).dt.days

# Create next-trading-session prediction target
integrated["next_trading_date"] = (
    integrated.groupby("stock_code")["date"].shift(-1)
)

integrated["next_close"] = (
    integrated.groupby("stock_code")["close"].shift(-1)
)

integrated["target_next_day_return"] = (
    integrated["next_close"] / integrated["close"] - 1
)

integrated["target_next_day_direction"] = (
    integrated["target_next_day_return"] > 0
).astype("Int64")

# Last available row per stock has no future target
integrated.loc[
    integrated["next_close"].isna(),
    "target_next_day_direction"
] = pd.NA

# Save output
integrated.to_csv(OUTPUT_FILE, index=False)

# Validation
print("TIME-AWARE DATASET CREATED")
print("--------------------------")
print("Total records:", len(integrated))
print("Unique stocks:", integrated["stock_code"].nunique())
print(
    "Records with sentiment:",
    integrated["sentiment_available"].sum()
)
print(
    "Records without sentiment:",
    (integrated["sentiment_available"] == 0).sum()
)
print(
    "Records with prediction target:",
    integrated["target_next_day_return"].notna().sum()
)
print(
    "Duplicate stock/date rows:",
    integrated.duplicated(["stock_code", "date"]).sum()
)
print(
    "Sentiment date after price date:",
    (integrated["sentiment_date"] > integrated["date"]).sum()
)
print("Output:", OUTPUT_FILE)
