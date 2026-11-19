
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data/processed/time_aware_dataset.csv"
OUTPUT_FILE = ROOT / "reports/advanced_model_results.csv"

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

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        )
    }

    for name, estimator in models.items():

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator)
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "window": number,
            "model": name,
            "test_start": str(test_dates[0])[:10],
            "test_end": str(test_dates[-1])[:10],
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            )
        })

results_df = pd.DataFrame(results)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_FILE, index=False)

print("ADVANCED MODEL COMPARISON")
print("-------------------------")
print(results_df.to_string(index=False))

print("\nAVERAGE PERFORMANCE")
print("-------------------")

summary = results_df.groupby("model")[
    ["accuracy", "balanced_accuracy", "f1_score"]
].mean()

print(summary.to_string())

print("\nSaved to:", OUTPUT_FILE)
