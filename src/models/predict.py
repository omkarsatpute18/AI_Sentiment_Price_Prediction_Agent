
import joblib
import pandas as pd

from pathlib import Path


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = ROOT / "data" / "processed" / "stock_features.csv"
MODEL_FILE = ROOT / "models" / "price_baseline_models.joblib"
OUTPUT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# TECHNICAL FEATURES
# --------------------------------------------------

TECHNICAL_FEATURES = [
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


# --------------------------------------------------
# LOAD SAVED MODELS
# --------------------------------------------------

def load_models():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    bundle = joblib.load(MODEL_FILE)

    return (
        bundle["models"],
        bundle["feature_columns"],
        bundle["training_cutoff"],
    )


# --------------------------------------------------
# PREPARE LATEST STOCK FEATURES
# --------------------------------------------------

def prepare_latest_features(stock_row, feature_columns):

    X = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns,
        dtype=float
    )

    # Add technical indicator values

    for feature in TECHNICAL_FEATURES:

        if feature not in feature_columns:
            raise ValueError(
                f"Missing model feature: {feature}"
            )

        X.loc[0, feature] = stock_row[feature]

    # Add stock identity using one-hot encoding

    stock_dummy = f"stock_{stock_row['stock_code']}"

    if stock_dummy not in feature_columns:
        raise ValueError(
            f"Unknown stock code: {stock_row['stock_code']}"
        )

    X.loc[0, stock_dummy] = 1

    return X


# --------------------------------------------------
# GENERATE PREDICTIONS
# --------------------------------------------------

def generate_predictions():

    # Load price features

    features = pd.read_csv(
        FEATURE_FILE,
        dtype={"stock_code": str}
    )

    features["date"] = pd.to_datetime(
        features["date"]
    )

    models, feature_columns, training_cutoff = load_models()

    results = []

    # Process each stock separately

    stock_codes = sorted(
        features["stock_code"].unique()
    )

    for stock_code in stock_codes:

        stock_data = features[
            features["stock_code"] == stock_code
        ].sort_values("date")

        if stock_data.empty:
            continue

        # Latest available feature record

        latest = stock_data.iloc[-1]

        # Prepare model input

        X_latest = prepare_latest_features(
            latest,
            feature_columns
        )

        # Generate predictions from every model

        for model_name, model in models.items():

            prediction = int(
                model.predict(X_latest)[0]
            )

            probabilities = model.predict_proba(
                X_latest
            )[0]

            classes = list(model.classes_)

            if 1 in classes:
                probability_up = float(
                    probabilities[classes.index(1)]
                )
            else:
                probability_up = 0.0

            probability_down = 1 - probability_up

            results.append({

                "stock_code": stock_code,
                "stock_name": latest["stock_name"],
                "market": latest["market"],
                "currency": latest["currency"],

                "prediction_date": latest["date"],

                "model": model_name,

                "predicted_direction": (
                    "UP" if prediction == 1 else "DOWN"
                ),

                "probability_up": probability_up,
                "probability_down": probability_down,

                "training_cutoff": training_cutoff,

            })

    results_df = pd.DataFrame(results)

    # Save prediction results

    output_file = OUTPUT_DIR / "latest_predictions.csv"

    results_df.to_csv(
        output_file,
        index=False
    )

    return results_df, output_file


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    predictions, output_file = generate_predictions()

    print("\nLATEST STOCK PREDICTIONS")
    print("=" * 70)

    print(
        predictions[
            [
                "stock_name",
                "model",
                "predicted_direction",
                "probability_up",
                "probability_down",
                "prediction_date",
            ]
        ].to_string(index=False)
    )

    print("\nTraining cutoff:", predictions["training_cutoff"].iloc[0])

    print("\nPredictions saved to:")
    print(output_file)
