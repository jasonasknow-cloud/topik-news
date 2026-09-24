import os
import json
import html
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

from google import genai
from google.genai import types


# ============================================================
# GEMINI SETUP
# ============================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY secret is missing!")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.8-flash"


# ============================================================
# RSS FEEDS
# These are the current Yonhap News TV RSS addresses.
# ============================================================

RSS_FEEDS = {
    "정치": "https://www.yonhapnewstv.co.kr/category/news/politics/feed/",
    "경제": "https://www.yonhapnewstv.co.kr/category/news/economy/feed/",
    "사회": "https://www.yonhapnewstv.co.kr/category/news/society/feed/",
    "세계": "https://www.yonhapnewstv.co.kr/category/news/international/feed/"
}


# ============================================================
# CLEAN HTML FROM RSS DESCRIPTIONS
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# GET XML FROM RSS FEED
# ============================================================

def fetch_feed(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/rss+xml, application/xml, text/xml"
        }
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


# ============================================================
# GET FIRST NEWS STORY FROM EACH CATEGORY
# ============================================================

def fetch_rss_articles():

    articles = []

    content_namespace = "{http://purl.org/rss/1.0/modules/content/}encoded"

    for category, url in RSS_FEEDS.items():

        print(f"Fetching {category} news...")

        try:
            xml_data = fetch_feed(url)
            root = ET.fromstring(xml_data)

            items = root.findall(".//item")

            if not items:
                raise RuntimeError(f"No RSS items found for {category}")

            item = items[0]

            title_element = item.find("title")
            description_element = item.find("description")
            content_element = item.find(content_namespace)

            title = (
                title_element.text.strip()
                if title_element is not None and title_element.text
                else ""
            )

            # Prefer content:encoded if available.
            if content_element is not None and content_element.text:
                description = content_element.text
            elif description_element is not None and description_element.text:
                description = description_element.text
            else:
                description = ""

            title = clean_text(title)
            description = clean_text(description)

            if not title:
                raise RuntimeError(f"RSS item has no title for {category}")

            print(f"  Found: {title}")

            articles.append({
                "original_title": title,
                "original_text": description,
                "category": category
            })

        except Exception as e:
            print(f"ERROR fetching {category}: {e}")

    return articles


# ============================================================
# GEMINI JSON SCHEMA
# ============================================================

NEWS_SCHEMA = {
    "type": "object",
    "properties": {
        "title_ko": {
            "type": "string"
        },
        "body_ko": {
            "type": "string"
        },
        "translation_en": {
            "type": "string"
        },
        "vocab": {
            "type": "string"
        }
    },
    "required": [
        "title_ko",
        "body_ko",
        "translation_en",
        "vocab"
    ]
}


# ============================================================
# REWRITE NEWS FOR TOPIK STUDENTS
# ============================================================

def rewrite_for_topik(article):

    prompt = f"""
You are an expert Korean language instructor creating reading
material for advanced TOPIK learners at Levels 4-6.

Rewrite the following Korean news article for TOPIK learners.

Requirements:

1. Use standard Korean news writing.
2. Use the descriptive plain style (해라체):
   - ㄴ다 / 는다
   - 었다 / 했다
   - 다
3. Keep the Korean natural and authentic.
4. Make the language appropriate for TOPIK Levels 4-6.
5. Avoid slang and unnecessary sensational language.
6. Keep important facts from the original article.
7. Do not invent facts that are not in the original.
8. Write approximately 4-7 Korean sentences.
9. Provide an accurate English translation.
10. Provide 3-5 useful advanced vocabulary items with English meanings.
11. Return ONLY the requested JSON fields.

Original category:
{article["category"]}

Original headline:
{article["original_title"]}

Original article:
{article["original_text"]}
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                response_mime_type="application/json",
                response_schema=NEWS_SCHEMA
            )
        )

        text = response.text.strip()

        if not text:
            raise RuntimeError("Gemini returned an empty response")

        result = json.loads(text)

        # Make sure all required fields actually contain something.
        required_fields = [
            "title_ko",
            "body_ko",
            "translation_en",
            "vocab"
        ]

        for field in required_fields:
            if not result.get(field):
                raise RuntimeError(
                    f"Gemini response is missing field: {field}"
                )

        return result

    except Exception as e:

        print(
            f"ERROR generating article for "
            f"{article['category']}: {e}"
        )

        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("Starting daily TOPIK news automation")
    print("=" * 60)

    raw_articles = fetch_rss_articles()

    print()
    print(f"Successfully fetched {len(raw_articles)} news articles.")

    # IMPORTANT:
    # If RSS feeds fail, stop the workflow instead of creating
    # an empty news_data.json.
    if not raw_articles:
        raise RuntimeError(
            "No news articles were fetched from any RSS feed."
        )

    processed_articles = []

    # Use Korea time for the date shown on the website.
    today = datetime.now(
        ZoneInfo("Asia/Seoul")
    ).strftime("%Y-%m-%d")

    for article in raw_articles:

        print()
        print(f"Processing {article['category']}...")

        rewritten = rewrite_for_topik(article)

        if rewritten:

            processed_articles.append({
                "category": article["category"],
                "date": today,
                "title": rewritten["title_ko"],
                "body": rewritten["body_ko"],
                "translation": rewritten["translation_en"],
                "vocab": rewritten["vocab"]
            })

    print()
    print(
        f"Successfully generated "
        f"{len(processed_articles)} TOPIK articles."
    )

    # IMPORTANT:
    # Do not silently create an empty file.
    if not processed_articles:
        raise RuntimeError(
            "Gemini failed to generate any articles. "
            "news_data.json will NOT be treated as successful."
        )

    # Save the JSON file that index.html loads.
    with open(
        "news_data.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            processed_articles,
            f,
            ensure_ascii=False,
            indent=4
        )

    print()
    print("news_data.json successfully created.")
    print(f"Number of articles: {len(processed_articles)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
