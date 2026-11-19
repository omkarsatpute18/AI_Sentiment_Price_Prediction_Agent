
import os
import logging

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT, "data", "processed", "combined_stock_prices.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT, "data", "processed", "stock_features.csv"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index."""

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    # Handle uninterrupted gains and losses
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)

    return rsi


def calculate_macd(series):
    """Calculate MACD, signal line and histogram."""

    ema_12 = series.ewm(
        span=12, adjust=False, min_periods=12
    ).mean()

    ema_26 = series.ewm(
        span=26, adjust=False, min_periods=26
    ).mean()

    macd = ema_12 - ema_26

    signal = macd.ewm(
        span=9, adjust=False, min_periods=9
    ).mean()

    histogram = macd - signal

    return macd, signal, histogram


def engineer_features(stock_data):
    """Calculate technical indicators for one stock."""

    stock_data = stock_data.copy()

    stock_data["date"] = pd.to_datetime(stock_data["date"])
    stock_data = stock_data.sort_values("date").reset_index(drop=True)

    close = stock_data["close"]
    high = stock_data["high"]
    low = stock_data["low"]
    volume = stock_data["volume"]

    # Daily and multi-day returns
    stock_data["daily_return"] = close.pct_change()
    stock_data["return_5d"] = close.pct_change(periods=5)
    stock_data["return_10d"] = close.pct_change(periods=10)
    stock_data["return_20d"] = close.pct_change(periods=20)

    # Moving averages
    stock_data["sma_5"] = close.rolling(window=5).mean()
    stock_data["sma_10"] = close.rolling(window=10).mean()
    stock_data["sma_20"] = close.rolling(window=20).mean()
    stock_data["sma_50"] = close.rolling(window=50).mean()

    # Exponential moving averages
    stock_data["ema_12"] = close.ewm(
        span=12, adjust=False, min_periods=12
    ).mean()

    stock_data["ema_26"] = close.ewm(
        span=26, adjust=False, min_periods=26
    ).mean()

    # Relative Strength Index
    stock_data["rsi_14"] = calculate_rsi(close, period=14)

    # MACD
    (
        stock_data["macd"],
        stock_data["macd_signal"],
        stock_data["macd_histogram"],
    ) = calculate_macd(close)

    # Rolling volatility
    stock_data["volatility_5d"] = (
        stock_data["daily_return"].rolling(5).std()
    )

    stock_data["volatility_20d"] = (
        stock_data["daily_return"].rolling(20).std()
    )

    # Volume features
    stock_data["volume_sma_20"] = volume.rolling(20).mean()

    stock_data["volume_change"] = volume.pct_change()

    # Price position relative to moving averages
    stock_data["price_to_sma_20"] = close / stock_data["sma_20"] - 1
    stock_data["price_to_sma_50"] = close / stock_data["sma_50"] - 1

    # High-low price range
    stock_data["high_low_range"] = (high - low) / close

    # Remove rows where indicators are unavailable
    stock_data = stock_data.replace(
        [np.inf, -np.inf], np.nan
    )

    stock_data = stock_data.dropna().reset_index(drop=True)

    return stock_data


def main():
    logging.info("Starting stock feature engineering")

    if not os.path.exists(INPUT_FILE):
        logging.error("Input file not found: %s", INPUT_FILE)
        return

    # Read stock codes as strings to preserve leading zeros
    data = pd.read_csv(
        INPUT_FILE,
        dtype={"stock_code": str}
    )

    all_features = []

    for stock_code, stock_data in data.groupby("stock_code"):

        stock_name = stock_data["stock_name"].iloc[0]

        logging.info(
            "Calculating features for %s (%s)",
            stock_name,
            stock_code
        )

        features = engineer_features(stock_data)

        logging.info(
            "%s: %d rows retained after feature engineering",
            stock_name,
            len(features)
        )

        all_features.append(features)

    if not all_features:
        logging.error("No features were generated.")
        return

    combined_features = pd.concat(
        all_features,
        ignore_index=True
    )

    combined_features["date"] = combined_features[
        "date"
    ].dt.strftime("%Y-%m-%d")

    combined_features = combined_features.sort_values(
        ["stock_code", "date"]
    ).reset_index(drop=True)

    combined_features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature engineering completed!")
    print("Total records:", len(combined_features))
    print("Number of stocks:", combined_features["stock_code"].nunique())

    print("\nRecords per stock:")
    print(
        combined_features.groupby(
            ["stock_name", "stock_code"]
        ).size()
    )

    print("\nGenerated features:")
    print(combined_features.columns.tolist())

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
