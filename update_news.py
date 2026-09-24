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
# YONHAP NEWS TV - LATEST RSS FEED
# ============================================================

RSS_URLS = [
    "http://www.yonhapnewstv.co.kr/browse/feed/",
    "https://www.yonhapnewstv.co.kr/browse/feed/"
]


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# FETCH RSS FEED
# ============================================================

def fetch_feed():

    last_error = None

    for url in RSS_URLS:

        print(f"Trying RSS feed: {url}")

        try:

            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/rss+xml, application/xml, text/xml"
                }
            )

            with urllib.request.urlopen(
                request,
                timeout=30
            ) as response:

                xml_data = response.read()

                print(
                    f"RSS download successful: "
                    f"{len(xml_data)} bytes"
                )

                return xml_data

        except Exception as e:

            print(f"RSS attempt failed: {e}")
            last_error = e

    raise RuntimeError(
        f"Could not download Yonhap RSS feed: {last_error}"
    )


# ============================================================
# GET THE LATEST 12 STORIES
# ============================================================

def fetch_latest_articles():

    xml_data = fetch_feed()

    try:
        root = ET.fromstring(xml_data)

    except ET.ParseError as e:

        preview = xml_data[:500].decode(
            "utf-8",
            errors="replace"
        )

        raise RuntimeError(
            f"Yonhap returned something that is not valid RSS XML.\n"
            f"First 500 characters:\n{preview}\n"
            f"XML error: {e}"
        )

    items = root.findall(".//item")

    if not items:

        raise RuntimeError(
            "Yonhap RSS feed was downloaded, "
            "but no news items were found."
        )

    articles = []

    content_namespace = (
        "{http://purl.org/rss/1.0/modules/content/}encoded"
    )

    for item in items[:12]:

        title_element = item.find("title")
        description_element = item.find("description")
        content_element = item.find(content_namespace)
        link_element = item.find("link")

        title = (
            title_element.text.strip()
            if title_element is not None and title_element.text
            else ""
        )

        if content_element is not None and content_element.text:
            description = content_element.text

        elif (
            description_element is not None
            and description_element.text
        ):
            description = description_element.text

        else:
            description = ""

        link = (
            link_element.text.strip()
            if link_element is not None and link_element.text
            else ""
        )

        categories = []

        for category_element in item.findall("category"):

            if category_element.text:

                categories.append(
                    category_element.text.strip()
                )

        title = clean_text(title)
        description = clean_text(description)

        if not title:
            continue

        articles.append({
            "title": title,
            "description": description,
            "link": link,
            "rss_categories": categories
        })

    if not articles:

        raise RuntimeError(
            "The Yonhap RSS feed contained items, "
            "but no usable article titles were found."
        )

    print(
        f"Successfully collected "
        f"{len(articles)} latest news stories."
    )

    for number, article in enumerate(
        articles,
        start=1
    ):

        print(
            f"{number}. {article['title']}"
        )

    return articles


# ============================================================
# GEMINI OUTPUT SCHEMA
# ============================================================

NEWS_SCHEMA = {
    "type": "object",
    "properties": {

        "articles": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,

            "items": {

                "type": "object",

                "properties": {

                    "category": {
                        "type": "string",
                        "enum": [
                            "정치",
                            "경제",
                            "사회",
                            "세계"
                        ]
                    },

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
                    "category",
                    "title_ko",
                    "body_ko",
                    "translation_en",
                    "vocab"
                ]
            }
        }
    },

    "required": [
        "articles"
    ]
}


# ============================================================
# ASK GEMINI TO SELECT AND REWRITE THE NEWS
# ============================================================

def generate_topik_news(raw_articles):

    article_text = ""

    for index, article in enumerate(
        raw_articles,
        start=1
    ):

        article_text += f"""
ARTICLE {index}

Headline:
{article["title"]}

Description:
{article["description"]}

RSS categories:
{", ".join(article["rss_categories"])}

----------------------------------------
"""

    prompt = f"""
You are an expert Korean language instructor creating
daily reading material for advanced TOPIK learners
(Levels 4-6).

Below are the latest news stories from Yonhap News TV.

Your job is to select EXACTLY FOUR stories:

1. One 정치 (Politics) story
2. One 경제 (Economy) story
3. One 사회 (Society) story
4. One 세계 (World) story

Choose the most useful and newsworthy story available
for each category.

IMPORTANT:

- Use only information contained in the supplied articles.
- Do NOT invent facts.
- Do NOT combine unrelated articles.
- Do NOT create fictional information.
- Each selected story must genuinely fit its assigned category.
- Use each original article only once.
- Avoid entertainment, sports, weather, advertisements,
  and trivial stories when choosing the four stories.

FOR EACH STORY:

Write a natural Korean headline.

Then write a substantial Korean reading passage.

The article should contain approximately 8-12 Korean sentences.

The body should be approximately 500-800 Korean characters.

Do not make the article artificially long.
Use the available facts to provide useful background,
details, reasons, reactions, developments, and consequences
when those details are present in the original article.

The passage should feel like a substantial TOPIK
reading passage rather than a short news summary.

Use standard Korean news writing.

Use descriptive plain style (해라체):
- ㄴ다 / 는다
- 었다 / 했다
- 다

Make the Korean appropriate for TOPIK Levels 4-6.

Avoid slang and unnecessary sensational language.

Keep all important facts from the original article.

Do NOT invent facts or add information that is not
supported by the supplied article.

The natural structure should generally be:

1. Introduce the main event.
2. Explain the important facts and background.
3. Describe relevant reasons, reactions, or developments.
4. Explain consequences or significance when supported
   by the original article.

Also provide:

- An accurate English translation of the Korean article.
- 3-5 useful advanced vocabulary items with English meanings.

Return EXACTLY FOUR stories:

정치
경제
사회
세계
"""


    prompt += "\n\nARTICLE LIST:\n" + article_text

    print()
    print(
        "Asking Gemini to select and rewrite "
        "four TOPIK stories..."
    )

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,

            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=NEWS_SCHEMA
            )
        )

        text = response.text.strip()

        if not text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        result = json.loads(text)

        if "articles" not in result:
            raise RuntimeError(
                "Gemini response did not contain an 'articles' field."
            )

        articles = result["articles"]

        if len(articles) != 4:
            raise RuntimeError(
                f"Gemini returned {len(articles)} articles "
                f"instead of exactly 4."
            )

        expected_categories = {
            "정치",
            "경제",
            "사회",
            "세계"
        }

        actual_categories = {
            article.get("category")
            for article in articles
        }

        if actual_categories != expected_categories:

            raise RuntimeError(
                "Gemini did not return exactly one article "
                "for each category.\n"
                f"Returned categories: {actual_categories}"
            )

        required_fields = [
            "category",
            "title_ko",
            "body_ko",
            "translation_en",
            "vocab"
        ]

        for article in articles:

            for field in required_fields:

                if not article.get(field):

                    raise RuntimeError(
                        f"Gemini article is missing: {field}"
                    )

        return articles

    except Exception as e:

        raise RuntimeError(
            f"Gemini news generation failed: {e}"
        )


# ============================================================
# SAVE NEWS_DATA.JSON
# ============================================================

def save_news_data(articles):

    today = datetime.now(
        ZoneInfo("Asia/Seoul")
    ).strftime("%Y-%m-%d")

    processed_articles = []

    category_order = [
        "정치",
        "경제",
        "사회",
        "세계"
    ]

    articles_by_category = {
        article["category"]: article
        for article in articles
    }

    for category in category_order:

        article = articles_by_category[category]

        processed_articles.append({
            "category": category,
            "date": today,
            "title": article["title_ko"],
            "body": article["body_ko"],
            "translation": article["translation_en"],
            "vocab": article["vocab"]
        })

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
    print(
        f"news_data.json created successfully "
        f"with {len(processed_articles)} articles."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("Starting Daily TOPIK News Automation")
    print("=" * 60)

    raw_articles = fetch_latest_articles()

    processed_articles = generate_topik_news(
        raw_articles
    )

    save_news_data(
        processed_articles
    )

    print()
    print(
        "Daily TOPIK news generation completed successfully."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
