
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Stock Sentiment & Prediction Agent",
    page_icon="📈",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    prices = pd.read_csv(
        DATA_DIR / "combined_stock_prices.csv",
        dtype={"stock_code": str}
    )

    features = pd.read_csv(
        DATA_DIR / "stock_features.csv",
        dtype={"stock_code": str}
    )

    sentiment = pd.read_csv(
        DATA_DIR / "daily_sentiment.csv",
        dtype={"stock_code": str}
    )

    prices["date"] = pd.to_datetime(prices["date"])
    features["date"] = pd.to_datetime(features["date"])
    sentiment["date"] = pd.to_datetime(sentiment["date"])

    return prices, features, sentiment


try:
    prices, features, sentiment = load_data()

except Exception as error:
    st.error(f"Unable to load project datasets: {error}")
    st.stop()


# ==================================================
# HEADER
# ==================================================

st.title("📈 AI Stock Sentiment & Prediction Agent")

st.markdown(
    "Analyze stock market movements, technical indicators, "
    "and social media sentiment across Hong Kong and South Korea."
)

st.divider()


# ==================================================
# STOCK SELECTION
# ==================================================

stock_info = (
    prices[
        ["stock_code", "stock_name", "market", "currency", "ticker"]
    ]
    .drop_duplicates()
    .sort_values("stock_name")
)

stock_names = stock_info["stock_name"].tolist()

selected_stock = st.sidebar.selectbox(
    "Select a Stock",
    stock_names
)

selected_info = stock_info[
    stock_info["stock_name"] == selected_stock
].iloc[0]

stock_code = selected_info["stock_code"]
currency = selected_info["currency"]
market = selected_info["market"]


# ==================================================
# FILTER SELECTED STOCK DATA
# ==================================================

stock_prices = prices[
    prices["stock_code"] == stock_code
].sort_values("date")

stock_features = features[
    features["stock_code"] == stock_code
].sort_values("date")

stock_sentiment = sentiment[
    sentiment["stock_code"] == stock_code
].sort_values("date")


if stock_prices.empty:
    st.warning("No price data available for this stock.")
    st.stop()


# ==================================================
# LATEST STOCK INFORMATION
# ==================================================

latest = stock_prices.iloc[-1]

previous = stock_prices.iloc[-2] if len(stock_prices) > 1 else None

latest_price = latest["close"]

if previous is not None and previous["close"] != 0:

    daily_change = (
        (latest_price - previous["close"])
        / previous["close"]
    ) * 100

else:
    daily_change = 0


# ==================================================
# STOCK OVERVIEW
# ==================================================

st.subheader(selected_stock)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest Closing Price",
    f"{currency} {latest_price:,.2f}",
    f"{daily_change:+.2f}%"
)

col2.metric(
    "Market",
    market
)

col3.metric(
    "Latest Price Date",
    latest["date"].strftime("%d %b %Y")
)

col4.metric(
    "Total Trading Records",
    f"{len(stock_prices):,}"
)

st.divider()


# ==================================================
# STOCK PRICE HISTORY
# ==================================================

st.header("Stock Price History")

price_chart = px.line(
    stock_prices,
    x="date",
    y="close",
    title=f"{selected_stock} - Historical Closing Price",
    labels={
        "date": "Date",
        "close": f"Closing Price ({currency})"
    }
)

price_chart.update_layout(
    hovermode="x unified",
    height=450
)

st.plotly_chart(
    price_chart,
    use_container_width=True
)


# ==================================================
# TECHNICAL ANALYSIS
# ==================================================

st.divider()
st.header("Technical Analysis")

if not stock_features.empty:

    # ----------------------------------------------
    # MOVING AVERAGES
    # ----------------------------------------------

    st.subheader("Moving Averages")

    ma_chart = px.line(
        stock_features,
        x="date",
        y=["close", "sma_20", "sma_50"],
        title=f"{selected_stock} - Moving Average Analysis",
        labels={
            "date": "Date",
            "value": f"Price ({currency})",
            "variable": "Indicator"
        }
    )

    ma_chart.update_layout(
        height=450,
        hovermode="x unified"
    )

    st.plotly_chart(
        ma_chart,
        use_container_width=True
    )

    # ----------------------------------------------
    # RSI
    # ----------------------------------------------

    st.subheader("Relative Strength Index (RSI)")

    rsi_chart = px.line(
        stock_features,
        x="date",
        y="rsi_14",
        title="RSI (14)",
        labels={
            "date": "Date",
            "rsi_14": "RSI Value"
        }
    )

    rsi_chart.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="Overbought (70)"
    )

    rsi_chart.add_hline(
        y=30,
        line_dash="dash",
        annotation_text="Oversold (30)"
    )

    rsi_chart.update_yaxes(range=[0, 100])

    rsi_chart.update_layout(height=350)

    st.plotly_chart(
        rsi_chart,
        use_container_width=True
    )

    # ----------------------------------------------
    # MACD
    # ----------------------------------------------

    st.subheader("MACD Analysis")

    macd_chart = px.line(
        stock_features,
        x="date",
        y=["macd", "macd_signal"],
        title="MACD and Signal Line",
        labels={
            "date": "Date",
            "value": "MACD Value",
            "variable": "Indicator"
        }
    )

    macd_chart.add_trace(
        go.Bar(
            x=stock_features["date"],
            y=stock_features["macd_histogram"],
            name="MACD Histogram",
            opacity=0.35
        )
    )

    macd_chart.update_layout(
        height=400,
        hovermode="x unified"
    )

    st.plotly_chart(
        macd_chart,
        use_container_width=True
    )

    # ----------------------------------------------
    # TRADING VOLUME
    # ----------------------------------------------

    st.subheader("Trading Volume")

    volume_chart = px.bar(
        stock_features,
        x="date",
        y="volume",
        title=f"{selected_stock} - Daily Trading Volume",
        labels={
            "date": "Date",
            "volume": "Trading Volume"
        }
    )

    volume_chart.update_layout(height=350)

    st.plotly_chart(
        volume_chart,
        use_container_width=True
    )

else:

    st.warning(
        "Technical indicators are not available for this stock."
    )


# ==================================================
# SOCIAL MEDIA SENTIMENT
# ==================================================

st.divider()
st.header("Social Media Sentiment")

if not stock_sentiment.empty:

    latest_sentiment = stock_sentiment.iloc[-1]

    st.caption(
        "Latest available sentiment date: "
        + latest_sentiment["date"].strftime("%d %b %Y")
    )

    # ----------------------------------------------
    # SENTIMENT METRICS
    # ----------------------------------------------

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Total Posts",
        int(latest_sentiment["total_posts"])
    )

    s2.metric(
        "Financial Sentiment",
        (
            f"{latest_sentiment['financial_sentiment']:.3f}"
            if pd.notna(latest_sentiment["financial_sentiment"])
            else "N/A"
        )
    )

    s3.metric(
        "Business Sentiment",
        (
            f"{latest_sentiment['business_sentiment']:.3f}"
            if pd.notna(latest_sentiment["business_sentiment"])
            else "N/A"
        )
    )

    s4.metric(
        "Scored Posts",
        int(latest_sentiment["scored_post_count"])
    )

    # ----------------------------------------------
    # SENTIMENT CHART
    # ----------------------------------------------

    sentiment_chart = px.line(
        stock_sentiment,
        x="date",
        y=[
            "financial_sentiment",
            "business_sentiment"
        ],
        markers=True,
        title="Daily Social Sentiment",
        labels={
            "date": "Date",
            "value": "Sentiment Score",
            "variable": "Sentiment Type"
        }
    )

    sentiment_chart.update_layout(
        height=400,
        hovermode="x unified",
        legend_title_text="Sentiment Type"
    )

    sentiment_chart.update_xaxes(
        type="date",
        tickformat="%d %b %Y"
    )

    sentiment_chart.update_yaxes(
        range=[-1, 1],
        title="Sentiment Score"
    )

    sentiment_chart.update_traces(
        marker=dict(size=9),
        connectgaps=False
    )

    st.plotly_chart(
        sentiment_chart,
        use_container_width=True
    )

else:

    st.info(
        "No social sentiment data is available for this stock yet."
    )


# ==================================================
# AI STOCK PREDICTION
# ==================================================

st.divider()
st.header("🤖 AI Stock Prediction")

PREDICTION_FILE = REPORT_DIR / "latest_predictions.csv"

PERFORMANCE_FILE = REPORT_DIR / "final_model_comparison.csv"


if PREDICTION_FILE.exists():

    predictions = pd.read_csv(
        PREDICTION_FILE,
        dtype={"stock_code": str}
    )

    selected_predictions = predictions[
        predictions["stock_code"] == stock_code
    ].copy()

    st.caption(
        "Experimental next-trading-session predictions "
        "based on technical indicators."
    )

    if not selected_predictions.empty:

        # ------------------------------------------
        # MAIN MODEL PREDICTIONS
        # ------------------------------------------

        main_models = selected_predictions[
            selected_predictions["model"].isin(
                ["Logistic Regression", "Random Forest"]
            )
        ]

        prediction_columns = st.columns(2)

        for column, (_, row) in zip(
            prediction_columns,
            main_models.iterrows()
        ):

            with column:

                st.subheader(row["model"])

                st.metric(
                    "Predicted Direction",
                    row["predicted_direction"]
                )

                st.metric(
                    "UP Score",
                    f"{row['probability_up'] * 100:.2f}%"
                )

                st.progress(
                    min(
                        max(float(row["probability_up"]), 0.0),
                        1.0
                    )
                )

                st.caption(
                    f"DOWN Score: "
                    f"{row['probability_down'] * 100:.2f}%"
                )

        # ------------------------------------------
        # MODEL COMPARISON
        # ------------------------------------------

        st.subheader("Prediction Comparison")

        display_predictions = selected_predictions[
            [
                "model",
                "predicted_direction",
                "probability_up",
                "probability_down"
            ]
        ].copy()

        display_predictions["probability_up"] *= 100

        display_predictions["probability_down"] *= 100

        display_predictions.columns = [
            "Model",
            "Predicted Direction",
            "UP Score (%)",
            "DOWN Score (%)"
        ]

        st.dataframe(
            display_predictions.round(2),
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------
        # PREDICTION INFORMATION
        # ------------------------------------------

        latest_prediction_date = pd.to_datetime(
            selected_predictions["prediction_date"]
        ).max()

        training_cutoff = pd.to_datetime(
            selected_predictions["training_cutoff"].iloc[0]
        )

        st.info(
            f"Latest input data date: "
            f"{latest_prediction_date.strftime('%d %b %Y')} | "
            f"Model training cutoff: "
            f"{training_cutoff.strftime('%d %b %Y')}"
        )

        # ------------------------------------------
        # HISTORICAL MODEL PERFORMANCE
        # ------------------------------------------

        st.subheader("Historical Model Performance")

        if PERFORMANCE_FILE.exists():

            performance = pd.read_csv(PERFORMANCE_FILE)

            st.caption(
                "Average walk-forward validation results. "
                "These metrics describe historical performance, "
                "not the confidence of today's prediction."
            )

            st.dataframe(
                performance[
                    [
                        "model",
                        "accuracy",
                        "balanced_accuracy",
                        "f1_score"
                    ]
                ].style.format({
                    "accuracy": "{:.2%}",
                    "balanced_accuracy": "{:.2%}",
                    "f1_score": "{:.2%}"
                }),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Historical model performance report is unavailable."
            )

        # ------------------------------------------
        # MODEL LIMITATIONS
        # ------------------------------------------

        st.warning(
            "These predictions are experimental and are not financial "
            "advice. The models have shown limited historical predictive "
            "performance. They currently use technical indicators only "
            "and do not include social sentiment. Displayed model scores "
            "are not calibrated probabilities."
        )

    else:

        st.warning(
            "No predictions are available for the selected stock."
        )

else:

    st.warning(
        "Prediction file not found. Run src/models/predict.py first."
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Academic Project | Data Sources: Yahoo Finance, Naver and LIHKG"
)

st.caption(
    "Sentiment coverage is currently limited. "
    "Sentiment scores should not be interpreted as guaranteed "
    "indicators of future stock performance."
)
