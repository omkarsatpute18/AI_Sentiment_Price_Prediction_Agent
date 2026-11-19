
import json
import os
import logging
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from apify_client import ApifyClient


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "social_posts"

OUTPUT_FILE = DATA_DIR / "lihkg_search_expanded.json"

load_dotenv(PROJECT_ROOT / ".env")

APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")
LIHKG_ACTOR_ID = os.getenv("LIHKG_ACTOR_ID")

# Company names and Hong Kong stock codes
SEARCH_TERMS = [
    "騰訊",
    "0700",
    "阿里巴巴",
    "9988",
    "小米",
    "1810",
    "美團",
    "3690",
    "比亞迪",
    "1211"
]

MAX_ITEMS_PER_TERM = 20

START_DATE = datetime(2026, 11, 10, tzinfo=timezone.utc)
END_DATE = datetime(2026, 11, 20, tzinfo=timezone.utc)

# LIHKG actor date format
ACTOR_DATE_FROM = "11/10/26"


# --------------------------------------------------
# 2. LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. SEARCH LIHKG
# --------------------------------------------------

def search_lihkg(client):

    all_records = []

    for term in SEARCH_TERMS:

        logger.info("Searching LIHKG for: %s", term)

        run_input = {
            "mode": "search",
            "keywords": [term],
            "maxItems": MAX_ITEMS_PER_TERM,
            "order": "score",
            "dateFrom": ACTOR_DATE_FROM
        }

        run = client.actor(LIHKG_ACTOR_ID).call(
            run_input=run_input
        )

        if not run:
            logger.warning("No run returned for %s", term)
            continue

        logger.info("Actor status: %s", run.status)

        if run.status != "SUCCEEDED":
            logger.warning("Search failed for %s", term)
            continue

        dataset_id = run.default_dataset_id

        if not dataset_id:
            logger.warning("No dataset returned for %s", term)
            continue

        dataset = client.dataset(dataset_id)
        records = list(dataset.iterate_items())

        logger.info(
            "Retrieved %s records for %s",
            len(records),
            term
        )

        # Preserve the exact search term used
        for record in records:
            record["search_keyword"] = term

        all_records.extend(records)

    return all_records


# --------------------------------------------------
# 4. FILTER DATE RANGE
# --------------------------------------------------

def filter_by_date(records):

    filtered = []

    for record in records:

        created_at = record.get("created_at")

        if not created_at:
            continue

        try:
            post_date = datetime.fromisoformat(
                created_at.replace("Z", "+00:00")
            )

            if START_DATE <= post_date < END_DATE:
                filtered.append(record)

        except (ValueError, TypeError):
            logger.warning(
                "Invalid date for thread %s",
                record.get("thread_id")
            )

    return filtered


# --------------------------------------------------
# 5. REMOVE DUPLICATES
# --------------------------------------------------

def remove_duplicates(records):

    unique_records = {}

    for record in records:

        thread_id = record.get("thread_id")

        if not thread_id:
            continue

        # Keep the first occurrence of each thread
        if thread_id not in unique_records:
            unique_records[thread_id] = record

    return list(unique_records.values())


# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

def save_results(records):

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=4
        )

    logger.info("Saved %s unique threads.", len(records))
    logger.info("Output: %s", OUTPUT_FILE)


# --------------------------------------------------
# 7. MAIN
# --------------------------------------------------

def main():

    if not APIFY_TOKEN:
        raise ValueError("APIFY_API_TOKEN missing in .env")

    if not LIHKG_ACTOR_ID:
        raise ValueError("LIHKG_ACTOR_ID missing in .env")

    client = ApifyClient(APIFY_TOKEN)

    raw_records = search_lihkg(client)

    if not raw_records:
        raise RuntimeError("No LIHKG search records returned.")

    date_filtered = filter_by_date(raw_records)

    unique_records = remove_duplicates(date_filtered)

    save_results(unique_records)

    print("\nLIHKG EXPANDED SEARCH COMPLETED")
    print("--------------------------------")
    print("Search terms:", len(SEARCH_TERMS))
    print("Raw records:", len(raw_records))
    print("After date filtering:", len(date_filtered))
    print("Unique threads:", len(unique_records))
    print("Date range: November 10–19, 2026")
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
