
import os
import time
import logging
from datetime import date

import pandas as pd
import yfinance as yf


# Project paths
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "stock_prices")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# Historical date range
START_DATE = "2024-11-01"
END_DATE = (date.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


# Stock configuration
STOCKS = {
    "0700.HK": {
        "stock_code": "0700",
        "stock_name": "Tencent",
        "market": "Hong Kong",
        "currency": "HKD",
    },
    "005930.KS": {
        "stock_code": "005930",
        "stock_name": "Samsung Electronics",
        "market": "South Korea",
        "currency": "KRW",
    },
    "000660.KS": {
        "stock_code": "000660",
        "stock_name": "SK Hynix",
        "market": "South Korea",
        "currency": "KRW",
    },
    "005380.KS": {
        "stock_code": "005380",
        "stock_name": "Hyundai Motor",
        "market": "South Korea",
        "currency": "KRW",
    },
    "373220.KS": {
        "stock_code": "373220",
        "stock_name": "LG Energy Solution",
        "market": "South Korea",
        "currency": "KRW",
    },
    "207940.KS": {
        "stock_code": "207940",
        "stock_name": "Samsung Biologics",
        "market": "South Korea",
        "currency": "KRW",
    },
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def download_stock_data(ticker, details):
    """Download daily historical stock data for one ticker."""

    logging.info("Downloading %s (%s)", details["stock_name"], ticker)

    try:
        data = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if data.empty:
            logging.warning("No data returned for %s", ticker)
            return None

        # Handle possible MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        # Standardize date column
        date_column = "Date" if "Date" in data.columns else "Datetime"
        data["date"] = pd.to_datetime(data[date_column]).dt.strftime(
            "%Y-%m-%d"
        )

        data = data.rename(columns={
            "Adj Close": "adj_close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        required_columns = [
            "date", "open", "high", "low",
            "close", "adj_close", "volume"
        ]

        data = data[required_columns].copy()

        # Add stock metadata
        data["ticker"] = ticker
        data["stock_code"] = details["stock_code"]
        data["stock_name"] = details["stock_name"]
        data["market"] = details["market"]
        data["currency"] = details["currency"]

        data = data.dropna(subset=["date", "close"])
        data = data.sort_values("date").reset_index(drop=True)

        # Save individual raw CSV
        output_path = os.path.join(
            RAW_DIR,
            f"{details['stock_code']}_daily.csv"
        )

        data.to_csv(output_path, index=False)

        logging.info(
            "Saved %s: %d trading-day records",
            ticker,
            len(data),
        )

        return data

    except Exception as error:
        logging.error("Failed to download %s: %s", ticker, error)
        return None


def main():
    all_data = []

    logging.info("Starting historical stock price collection")
    logging.info("Date range: %s to %s", START_DATE, END_DATE)

    for ticker, details in STOCKS.items():
        result = download_stock_data(ticker, details)

        if result is not None:
            all_data.append(result)

        # Small delay between requests
        time.sleep(2)

    if not all_data:
        logging.error("No stock price data was downloaded.")
        return

    combined_data = pd.concat(all_data, ignore_index=True)

    combined_data = combined_data.sort_values(
        ["stock_code", "date"]
    ).reset_index(drop=True)

    output_path = os.path.join(
        PROCESSED_DIR,
        "combined_stock_prices.csv"
    )

    combined_data.to_csv(output_path, index=False)

    print("\nStock price collection completed!")
    print("Total records:", len(combined_data))
    print("Stocks downloaded:", combined_data["stock_code"].nunique())
    print("\nRecords per stock:")
    print(combined_data.groupby(
        ["stock_name", "stock_code"]
    ).size())

    print("\nCombined data saved to:", output_path)


if __name__ == "__main__":
    main()
