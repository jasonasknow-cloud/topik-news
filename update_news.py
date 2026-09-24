import os
import json
import html
import re
import time
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

# We try the newest model first, then fall back to older
# stable Flash models if Google is temporarily unavailable.
MODEL_NAMES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash"
]


# ============================================================
# YONHAP NEWS TV - LATEST RSS FEED
# ============================================================

RSS_URLS = [
    "https://www.yonhapnewstv.co.kr/browse/feed/",
    "http://www.yonhapnewstv.co.kr/browse/feed/"
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
                    "Accept": (
                        "application/rss+xml, "
                        "application/xml, "
                        "text/xml"
                    )
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
# GET LATEST NEWS STORIES
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
            "Yonhap returned something that is not valid RSS XML.\n"
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

    # Collect the latest 12 stories.
    for item in items[:12]:

        title_element = item.find("title")
        description_element = item.find("description")
        content_element = item.find(content_namespace)
        link_element = item.find("link")

        title = (
            title_element.text.strip()
            if (
                title_element is not None
                and title_element.text
            )
            else ""
        )

        if (
            content_element is not None
            and content_element.text
        ):
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
            if (
                link_element is not None
                and link_element.text
            )
            else ""
        )

        rss_categories = []

        for category_element in item.findall("category"):

            if category_element.text:

                rss_categories.append(
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
            "rss_categories": rss_categories
        })

    if not articles:

        raise RuntimeError(
            "The Yonhap RSS feed contained items, "
            "but no usable article titles were found."
        )

    print()
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
# BUILD GEMINI PROMPT
# ============================================================

def build_prompt(raw_articles):

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

IMPORTANT RULES:

- Use only information contained in the supplied articles.
- Do NOT invent facts.
- Do NOT combine unrelated articles.
- Do NOT create fictional information.
- Each selected story must genuinely fit its assigned category.
- Use each original article only once.
- Avoid entertainment, sports, weather, advertisements,
  and trivial stories.
- Prioritize stories that are useful for Korean language learners.
- Keep important factual details from the original material.

FOR EACH STORY:

Write a natural Korean headline.

Then write a substantial Korean reading passage.

LENGTH:

- Approximately 8-12 Korean sentences.
- Approximately 500-800 Korean characters.
- The article should feel like a real TOPIK reading passage,
  not a two- or three-sentence news summary.
- Do not add meaningless filler just to make the article longer.

CONTENT STRUCTURE:

When the original article contains enough information,
organize the passage naturally:

1. Introduce the main event.
2. Explain important facts and background.
3. Describe relevant causes, reactions, or developments.
4. Explain consequences or significance when supported
   by the original article.

Do not invent background information.

LANGUAGE:

- Use standard Korean news writing.
- Use descriptive plain style (해라체).
- Use endings such as ㄴ다 / 는다 / 었다 / 했다 / 다.
- Make the Korean appropriate for TOPIK Levels 4-6.
- Avoid slang.
- Avoid unnecessary sensational language.
- Use natural vocabulary and grammar appropriate for
  advanced Korean learners.
- Do not make the language artificially difficult.

ALSO PROVIDE:

- An accurate English translation of the Korean article.
- 3-5 useful advanced vocabulary items with English meanings.

OUTPUT:

Return EXACTLY FOUR stories:

정치
경제
사회
세계
"""

    prompt += "\n\nARTICLE LIST:\n" + article_text

    return prompt


# ============================================================
# VALIDATE GEMINI RESULT
# ============================================================

def validate_articles(result):

    if not isinstance(result, dict):

        raise RuntimeError(
            "Gemini response was not a JSON object."
        )

    if "articles" not in result:

        raise RuntimeError(
            "Gemini response did not contain an 'articles' field."
        )

    articles = result["articles"]

    if not isinstance(articles, list):

        raise RuntimeError(
            "Gemini 'articles' field was not a list."
        )

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


# ============================================================
# ASK GEMINI TO CREATE THE FOUR STORIES
# ============================================================

def generate_topik_news(raw_articles):

    prompt = build_prompt(raw_articles)

    print()
    print(
        "Asking Gemini to select and rewrite "
        "four TOPIK stories..."
    )

    last_error = None

    # Try each available model.
    for model_name in MODEL_NAMES:

        # Try each model up to 3 times.
        for attempt in range(1, 4):

            print(
                f"Trying {model_name} "
                f"(attempt {attempt}/3)..."
            )

            try:

                response = client.models.generate_content(
                    model=model_name,
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

                articles = validate_articles(result)

                print()
                print(
                    f"SUCCESS: {model_name} generated "
                    f"all four articles."
                )

                return articles

            except Exception as e:

                last_error = e

                error_text = str(e)

                print(
                    f"{model_name} failed:"
                )
                print(error_text)

                # 503 means Google's server is temporarily
                # unavailable. Retry rather than immediately failing.
                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                ):

                    if attempt < 3:

                        wait_seconds = attempt * 10

                        print(
                            f"Temporary Gemini server problem. "
                            f"Waiting {wait_seconds} seconds..."
                        )

                        time.sleep(wait_seconds)

                        continue

                    print()
                    print(
                        f"{model_name} failed three times. "
                        "Trying the next Gemini model..."
                    )

                    break

                # For other errors, there is probably something
                # genuinely wrong with the request, API key,
                # schema, etc. Do not hide those errors.
                raise RuntimeError(
                    f"Gemini news generation failed: {e}"
                )

    raise RuntimeError(
        "All Gemini models failed.\n"
        f"Last error: {last_error}"
    )


# ============================================================
# SAVE NEWS_DATA.JSON
# ============================================================

def save_news_data(articles):

    today = datetime.now(
        ZoneInfo("Asia/Seoul")
    ).strftime("%Y-%m-%d")

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

    processed_articles = []

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

    # Final safety check.
    if len(processed_articles) != 4:

        raise RuntimeError(
            "Final news_data.json does not contain exactly "
            "four articles."
        )

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
        "news_data.json created successfully "
        f"with {len(processed_articles)} articles."
    )

    for article in processed_articles:

        print(
            f"- {article['category']}: "
            f"{article['title']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("Starting Daily TOPIK News Automation")
    print("=" * 60)

    # 1. Download current Yonhap news.
    raw_articles = fetch_latest_articles()

    # 2. Select and rewrite the four stories.
    processed_articles = generate_topik_news(
        raw_articles
    )

    # 3. Save the final JSON used by the website.
    save_news_data(
        processed_articles
    )

    print()
    print("=" * 60)
    print("Daily TOPIK news generation completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
