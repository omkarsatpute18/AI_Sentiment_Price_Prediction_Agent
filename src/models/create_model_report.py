
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

WALK_FILE = ROOT / "reports/walk_forward_results.csv"
ADVANCED_FILE = ROOT / "reports/advanced_model_results.csv"
OUTPUT_FILE = ROOT / "reports/final_model_comparison.csv"

walk = pd.read_csv(WALK_FILE)
advanced = pd.read_csv(ADVANCED_FILE)

# Summarize walk-forward strategies
walk_summary = (
    walk.groupby("strategy")[
        ["accuracy", "balanced_accuracy", "f1_score"]
    ]
    .mean()
    .reset_index()
    .rename(columns={"strategy": "model"})
)

# Summarize advanced models
advanced_summary = (
    advanced.groupby("model")[
        ["accuracy", "balanced_accuracy", "f1_score"]
    ]
    .mean()
    .reset_index()
)

# Combine results
final = pd.concat(
    [walk_summary, advanced_summary],
    ignore_index=True
)

final = final.sort_values(
    "balanced_accuracy",
    ascending=False
).reset_index(drop=True)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
final.to_csv(OUTPUT_FILE, index=False)

print("FINAL MODEL COMPARISON")
print("----------------------")
print(final.to_string(index=False))

print("\nSaved to:", OUTPUT_FILE)
