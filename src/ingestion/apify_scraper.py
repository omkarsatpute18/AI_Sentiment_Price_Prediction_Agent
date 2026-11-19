
import json
import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from apify_client import ApifyClient


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "social_posts"

SEARCH_FILE = DATA_DIR / "lihkg_search.json"

OUTPUT_FILE = DATA_DIR / "lihkg_threads_api.json"

load_dotenv(PROJECT_ROOT / ".env")

APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")
LIHKG_ACTOR_ID = os.getenv("LIHKG_ACTOR_ID")

# Start with two threads to control scraping costs.
# Selected relevant threads from the expanded LIHKG search.
SELECTED_THREAD_IDS = [
    "4181061",
    "4180878",
    "4179443",
    "4182148",
    "4179515",
    "4181817"
]


# --------------------------------------------------
# 2. LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. LOAD SEARCH RESULTS
# --------------------------------------------------

def load_search_results():

    if not SEARCH_FILE.exists():
        raise FileNotFoundError(
            f"Search dataset not found: {SEARCH_FILE}"
        )

    with open(SEARCH_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    if isinstance(records, dict):
        records = records.get("items", [])

    if not isinstance(records, list):
        raise ValueError("Unexpected search dataset format.")

    logger.info("Loaded %s search records.", len(records))

    return records


# --------------------------------------------------
# 4. EXTRACT UNIQUE THREAD IDS
# --------------------------------------------------

def extract_thread_ids():

    thread_ids = SELECTED_THREAD_IDS

    if not thread_ids:
        raise ValueError("No thread IDs selected.")

    logger.info("Selected %s threads for extraction.", len(thread_ids))

    return thread_ids

# --------------------------------------------------
# 5. RUN LIHKG THREAD ACTOR
# --------------------------------------------------


def scrape_threads(client, thread_ids):

    run_input = {
        "mode": "thread",
        "threadIds": thread_ids,
        "maxItems": len(thread_ids)
    }

    logger.info("Starting LIHKG thread extraction...")

    run = client.actor(LIHKG_ACTOR_ID).call(
        run_input=run_input
    )

    if not run:
        raise RuntimeError("Apify actor did not return a run.")

    dataset_id = run.default_dataset_id

    if not dataset_id:
        raise RuntimeError("Apify dataset ID was not returned.")

    logger.info("Actor run status: %s", run.status)
    logger.info("Dataset ID: %s", dataset_id)

    dataset = client.dataset(dataset_id)
    results = list(dataset.iterate_items())

    logger.info("Retrieved %s records.", len(results))

    return results


# --------------------------------------------------
# 6. SAVE RESULTS
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

    logger.info("Results saved successfully!")
    logger.info("Output file: %s", OUTPUT_FILE)


# --------------------------------------------------
# 7. MAIN EXECUTION
# --------------------------------------------------

def main():

    if not APIFY_TOKEN:
        raise ValueError("APIFY_API_TOKEN missing in .env")

    if not LIHKG_ACTOR_ID:
        raise ValueError("LIHKG_ACTOR_ID missing in .env")

    client = ApifyClient(APIFY_TOKEN)

    thread_ids = extract_thread_ids()

    results = scrape_threads(client, thread_ids)

    if not results:
        raise RuntimeError("No thread records were returned.")

    save_results(results)

    print("\nLIHKG scraping completed successfully!")
    print("Threads requested:", len(thread_ids))
    print("Records retrieved:", len(results))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
