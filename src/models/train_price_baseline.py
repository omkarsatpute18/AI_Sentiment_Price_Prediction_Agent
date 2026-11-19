
import pandas as pd
import joblib

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data/processed/time_aware_dataset.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE, dtype={"stock_code": str})
df["date"] = pd.to_datetime(df["date"])

# Technical features only: no sentiment or future information
features = [
    "daily_return",
    "return_5d",
    "return_10d",
    "return_20d",
    "rsi_14",
    "volatility_5d",
    "volatility_20d",
    "volume_change",
    "price_to_sma_20",
    "price_to_sma_50",
    "high_low_range",
]

# Remove rows without a target
df = df.dropna(subset=["target_next_day_direction"]).copy()
df["target_next_day_direction"] = (
    df["target_next_day_direction"].astype(int)
)

# Add stock identity as categorical model input
X = pd.concat(
    [
        df[features],
        pd.get_dummies(df["stock_code"], prefix="stock", dtype=int),
    ],
    axis=1,
)

y = df["target_next_day_direction"]

# Chronological split: first 80% of dates for training,
# remaining 20% for testing.
dates = sorted(df["date"].unique())
split_index = int(len(dates) * 0.8)
split_date = dates[split_index]

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]

print("CHRONOLOGICAL DATA SPLIT")
print("------------------------")
print("Training cutoff:", split_date)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training date range:",
      df.loc[train_mask, "date"].min(),
      "to", df.loc[train_mask, "date"].max())
print("Testing date range:",
      df.loc[test_mask, "date"].min(),
      "to", df.loc[test_mask, "date"].max())

# Define baseline and ML models
models = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000)),
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )),
    ]),
}

results = []
trained_models = {}

# Train and evaluate
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        row["roc_auc"] = roc_auc_score(y_test, probabilities)
    else:
        row["roc_auc"] = None

    results.append(row)
    trained_models[name] = model

    print(classification_report(
        y_test, predictions, zero_division=0
    ))

# Save evaluation results
results_df = pd.DataFrame(results)
results_file = OUTPUT_DIR / "price_baseline_results.csv"
results_df.to_csv(results_file, index=False)

# Save trained models and their feature order
joblib.dump(
    {
        "models": trained_models,
        "feature_columns": list(X.columns),
        "training_cutoff": str(split_date),
    },
    MODEL_DIR / "price_baseline_models.joblib",
)

print("\nMODEL COMPARISON")
print("----------------")
print(results_df.to_string(index=False))

print("\nResults saved to:", results_file)
print("Models saved to:", MODEL_DIR / "price_baseline_models.joblib")
