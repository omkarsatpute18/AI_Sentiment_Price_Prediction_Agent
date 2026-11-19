
import pandas as pd
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[2]

PRICE_FILE = ROOT / "data/processed/stock_features.csv"
SENTIMENT_FILE = ROOT / "data/processed/daily_sentiment.csv"
OUTPUT_FILE = ROOT / "data/processed/integrated_dataset.csv"

# Load datasets
prices = pd.read_csv(
    PRICE_FILE,
    dtype={"stock_code": str}
)

sentiment = pd.read_csv(
    SENTIMENT_FILE,
    dtype={"stock_code": str}
)

# Standardize dates
prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")
sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.strftime("%Y-%m-%d")

# Check merge keys
if prices.duplicated(["stock_code", "date"]).any():
    raise ValueError("Duplicate stock/date rows in price features.")

if sentiment.duplicated(["stock_code", "date"]).any():
    raise ValueError("Duplicate stock/date rows in sentiment data.")

# Keep every price record
integrated = prices.merge(
    sentiment.drop(columns=["market", "stock_name"], errors="ignore"),
    on=["stock_code", "date"],
    how="left",
    validate="one_to_one"
)

# Flag dates with sentiment observations
integrated["sentiment_available"] = (
    integrated["total_posts"].notna().astype(int)
)

# Save integrated dataset
integrated.to_csv(OUTPUT_FILE, index=False)

# Validation report
print("INTEGRATION COMPLETED")
print("---------------------")
print("Price records:", len(prices))
print("Sentiment records:", len(sentiment))
print("Integrated records:", len(integrated))
print("Stocks:", integrated["stock_code"].nunique())
print(
    "Dates with sentiment:",
    integrated["sentiment_available"].sum()
)
print(
    "Dates without sentiment:",
    (integrated["sentiment_available"] == 0).sum()
)
print(
    "Duplicate stock/date rows:",
    integrated.duplicated(["stock_code", "date"]).sum()
)
print("Output:", OUTPUT_FILE)
