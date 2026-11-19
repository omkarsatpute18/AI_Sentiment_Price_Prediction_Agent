
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data/processed/time_aware_dataset.csv"
OUTPUT_FILE = ROOT / "reports/walk_forward_results.csv"

df = pd.read_csv(DATA_FILE, dtype={"stock_code": str})
df["date"] = pd.to_datetime(df["date"])

df = df.dropna(subset=["target_next_day_direction"]).copy()
df["target_next_day_direction"] = (
    df["target_next_day_direction"].astype(int)
)

features = [
    "daily_return", "return_5d", "return_10d", "return_20d",
    "rsi_14", "volatility_5d", "volatility_20d",
    "volume_change", "price_to_sma_20",
    "price_to_sma_50", "high_low_range"
]

X = pd.concat(
    [
        df[features],
        pd.get_dummies(df["stock_code"], prefix="stock", dtype=int)
    ],
    axis=1
)

y = df["target_next_day_direction"]

dates = np.array(sorted(df["date"].unique()))

# Four expanding-window test periods
windows = [
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.00),
]

results = []

for number, (train_end, test_end) in enumerate(windows, start=1):

    train_end_index = int(len(dates) * train_end)
    test_end_index = int(len(dates) * test_end)

    train_dates = dates[:train_end_index]
    test_dates = dates[train_end_index:test_end_index]

    test_start_date = pd.Timestamp(test_dates[0])

    train_mask = (
    (df["date"] < test_start_date) &
    (pd.to_datetime(df["next_trading_date"]) < test_start_date)
)

    test_mask = df["date"].isin(test_dates)

    train = df.loc[train_mask].copy()
    test = df.loc[test_mask].copy()

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    # Logistic Regression
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)
    logistic_predictions = model.predict(X_test)

    # Per-stock majority baseline
    majority = train.groupby("stock_code")[
        "target_next_day_direction"
    ].agg(lambda values: values.mode().iloc[0])

    majority_predictions = test["stock_code"].map(majority)

    # Momentum baseline
    momentum_predictions = (
        test["daily_return"] > 0
    ).astype(int)

    predictions = {
        "Logistic Regression": logistic_predictions,
        "Per-stock Majority": majority_predictions,
        "Momentum": momentum_predictions,
    }

    for strategy, predicted in predictions.items():

        results.append({
            "window": number,
            "strategy": strategy,
            "train_end": str(train_dates[-1])[:10],
            "test_start": str(test_dates[0])[:10],
            "test_end": str(test_dates[-1])[:10],
            "training_records": len(train),
            "testing_records": len(test),
            "accuracy": accuracy_score(y_test, predicted),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predicted
            ),
            "f1_score": f1_score(
                y_test, predicted, zero_division=0
            )
        })

results_df = pd.DataFrame(results)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_FILE, index=False)

print("WALK-FORWARD STRATEGY COMPARISON")
print("--------------------------------")
print(results_df.to_string(index=False))

print("\nAVERAGE PERFORMANCE BY STRATEGY")
print("--------------------------------")

summary = results_df.groupby("strategy")[
    ["accuracy", "balanced_accuracy", "f1_score"]
].mean()

print(summary.to_string())

print("\nSaved to:", OUTPUT_FILE)
