
import os
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from apify_client import ApifyClient


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "social_posts"

OUTPUT_FILE = DATA_DIR / "naver_posts_api.json"

load_dotenv(PROJECT_ROOT / ".env")

APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")
NAVER_ACTOR_ID = os.getenv("NAVER_ACTOR_ID")


# --------------------------------------------------
# 2. TEST CONFIGURATION
# --------------------------------------------------


KOREAN_STOCKS = [
    "005930",  # Samsung Electronics
    "000660",  # SK Hynix
    "005380",  # Hyundai Motor
    "373220",  # LG Energy Solution
    "207940"   # Samsung Biologics
]

DATE_FROM = "2026-11-10"

MAX_RESULTS = 200



# --------------------------------------------------
# 3. LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 4. SCRAPE NAVER STOCK DISCUSSIONS
# --------------------------------------------------

def scrape_naver(client):

    run_input = {
        "dateFrom": DATE_FROM,
        "maxResults": MAX_RESULTS,
        "stocks": KOREAN_STOCKS
    }

    logger.info("Starting Naver Finance scraper...")
    logger.info("Stocks: %s", KOREAN_STOCKS)
    logger.info("Date from: %s", DATE_FROM)
    logger.info("Maximum results: %s", MAX_RESULTS)

    run = client.actor(NAVER_ACTOR_ID).call(
        run_input=run_input
    )

    if not run:
        raise RuntimeError("Naver actor did not return a run.")

    logger.info("Actor status: %s", run.status)

    if run.status != "SUCCEEDED":
        raise RuntimeError(
            f"Naver actor finished with status: {run.status}"
        )

    dataset_id = run.default_dataset_id

    if not dataset_id:
        raise RuntimeError("Naver dataset ID not found.")

    logger.info("Dataset ID: %s", dataset_id)

    dataset = client.dataset(dataset_id)

    results = list(dataset.iterate_items())

    logger.info("Records retrieved: %s", len(results))

    return results


# --------------------------------------------------
# 5. SAVE DATASET
# --------------------------------------------------

def save_results(results):

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4
        )

    logger.info("Naver dataset saved successfully.")
    logger.info("Output: %s", OUTPUT_FILE)


# --------------------------------------------------
# 6. MAIN EXECUTION
# --------------------------------------------------

def main():

    if not APIFY_TOKEN:
        raise ValueError("APIFY_API_TOKEN missing in .env")

    if not NAVER_ACTOR_ID:
        raise ValueError("NAVER_ACTOR_ID missing in .env")

    client = ApifyClient(APIFY_TOKEN)

    results = scrape_naver(client)

    if not results:
        logger.warning("No records were returned.")
        return

    save_results(results)

    print("\nNaver scraping completed successfully!")
    print("Stocks tested:", len(KOREAN_STOCKS))
    print("Records retrieved:", len(results))
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
