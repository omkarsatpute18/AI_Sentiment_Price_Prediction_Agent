
import json
import re
import html
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data/processed/naver_cleaned.json"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/naver_relevance_review.json"


FINANCIAL_KEYWORDS = [
    "주가", "주식", "시세", "상한가", "하한가",
    "급등", "급락", "상승", "하락", "매수", "매도",
    "공매도", "거래량", "시가총액", "시총",
    "실적", "매출", "영업이익", "순이익",
    "적자", "흑자", "배당", "목표가", "적정가",
    "저평가", "고평가", "분기보고서", "투자",
    "수익률", "손실", "주주", "외국인", "기관"
]

BUSINESS_KEYWORDS = [
    "사업", "경쟁", "경쟁력", "시장점유율",
    "기술", "신제품", "출시", "생산", "공급",
    "수요", "수주", "계약", "공장", "설비",
    "반도체", "메모리", "HBM", "파운드리",
    "AI", "인공지능", "배터리", "2차전지",
    "전고체", "LFP", "전기차", "자동차",
    "리콜", "품질", "특허", "연구개발",
    "R&D", "중국", "수출", "실적발표"
]


def find_matches(text, keywords):
    text_lower = text.lower()

    return [
        keyword for keyword in keywords
        if keyword.lower() in text_lower
    ]


def classify_post(record):

    title = record.get("title", "")
    body = record.get("combined_text", "")

    text = html.unescape(f"{title} {body}")
    text = re.sub(r"\s+", " ", text)

    financial_matches = find_matches(
        text, FINANCIAL_KEYWORDS
    )

    business_matches = find_matches(
        text, BUSINESS_KEYWORDS
    )

    if financial_matches and business_matches:
        label = "financial_and_business_relevant"

    elif financial_matches:
        label = "financially_relevant"

    elif business_matches:
        label = "business_relevant"

    else:
        label = "manual_review"

    return {
        "relevance_label": label,
        "financial_keywords_found": financial_matches,
        "business_keywords_found": business_matches
    }


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    results = []

    for record in records:

        updated_record = record.copy()
        updated_record.update(classify_post(record))

        results.append(updated_record)

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4
        )

    label_counts = Counter(
        record["relevance_label"]
        for record in results
    )

    print("\nNaver relevance filtering completed!")
    print("Total records:", len(results))

    print("\nRelevance distribution:")

    for label, count in label_counts.items():
        print(f"{label}: {count}")

    print("\nStock-wise relevance:")

    stock_counts = {}

    for record in results:
        stock = record["stock_name"]
        label = record["relevance_label"]

        stock_counts.setdefault(stock, Counter())
        stock_counts[stock][label] += 1

    for stock, counts in stock_counts.items():
        print(f"\n{stock}:")

        for label, count in counts.items():
            print(f"  {label}: {count}")

    print("\nOutput:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
