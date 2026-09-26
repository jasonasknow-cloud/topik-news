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

MODEL_NAMES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
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
            "Yonhap returned invalid RSS XML.\n"
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
# LOAD EXISTING ARCHIVE
# ============================================================

def load_existing_archive():

    filename = "news_data.json"

    if not os.path.exists(filename):

        print()
        print(
            "No existing news_data.json found. "
            "Starting a new archive."
        )

        return []

    try:

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if not isinstance(data, list):

            raise RuntimeError(
                "news_data.json does not contain a list."
            )

        print()
        print(
            f"Loaded {len(data)} existing articles "
            "from the archive."
        )

        return data

    except json.JSONDecodeError as e:

        raise RuntimeError(
            f"Could not read news_data.json as JSON: {e}"
        )


# ============================================================
# GET RECENT HEADLINES FROM ARCHIVE
# ============================================================

def get_recent_headlines(archive, limit=40):

    headlines = []

    for article in archive[:limit]:

        title = article.get("title")

        if title:

            headlines.append(title)

    return headlines


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

def build_prompt(
    raw_articles,
    recent_headlines
):

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

    previous_text = ""

    if recent_headlines:

        previous_text = """

RECENTLY PUBLISHED HEADLINES

These stories have already appeared on the website.
Do NOT use them again unless there is absolutely no
other suitable story available.

"""

        for title in recent_headlines:

            previous_text += f"- {title}\n"

    prompt = f"""
You are an expert Korean language instructor creating
daily reading material for advanced TOPIK learners
(Levels 4-6).

Below are the latest news stories from Yonhap News TV.

Select EXACTLY FOUR NEW stories:

1. 정치 (Politics)
2. 경제 (Economy)
3. 사회 (Society)
4. 세계 (World)

Choose exactly one story for each category.

IMPORTANT:

- Use only information contained in the supplied articles.
- Do NOT invent facts.
- Do NOT combine unrelated stories.
- Do NOT create fictional information.
- Each selected story must genuinely fit its category.
- Use each original article only once.
- Avoid entertainment, sports, weather, advertising,
  and trivial stories.
- Prefer substantial stories that are useful for
  Korean language learners.
- Avoid headlines that have already appeared in the
  recent archive whenever possible.

FOR EACH STORY:

Write a natural Korean headline.

Then write a substantial Korean reading passage.

LENGTH:

- Approximately 8-12 Korean sentences.
- Approximately 500-800 Korean characters.
- The article should feel like a substantial TOPIK
  reading passage rather than a short news summary.
- Do not add meaningless filler just to increase length.

CONTENT STRUCTURE:

When supported by the original article, naturally include:

1. The main event.
2. Important facts and background.
3. Relevant causes, reactions, or developments.
4. Consequences or significance.

Do NOT invent background information.

LANGUAGE:

- Use standard Korean news writing.
- Use descriptive plain style (해라체).
- Use endings such as ㄴ다 / 는다 / 었다 / 했다 / 다.
- Make the Korean appropriate for TOPIK Levels 4-6.
- Avoid slang.
- Avoid unnecessary sensational language.
- Use useful vocabulary for advanced Korean learners.
- Do not make the language artificially difficult.

ALSO PROVIDE:

- An accurate English translation.
- Exactly 10 useful advanced vocabulary items or expressions from the article, with English meanings.
- Choose words that are useful for TOPIK learners. Avoid trivial or overly basic words.
- Format the vocabulary as a numbered list from 1 to 10, with one item per line.
RETURN EXACTLY FOUR STORIES:

정치
경제
사회
세계

{previous_text}

ARTICLE LIST:

{article_text}
"""

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

def generate_topik_news(
    raw_articles,
    recent_headlines
):

    prompt = build_prompt(
        raw_articles,
        recent_headlines
    )

    print()
    print(
        "Asking Gemini to select and rewrite "
        "four new TOPIK stories..."
    )

    last_error = None

    for model_name in MODEL_NAMES:

        for attempt in range(1, 4):

            print()
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
                    f"four new articles."
                )

                return articles

            except Exception as e:

                last_error = e
                error_text = str(e)

                print()
                print(
                    f"{model_name} failed:"
                )
                print(error_text)

                # If this model has reached its quota,
                # skip it and immediately try the next model.
                if "429" in error_text:

                    print()
                    print(
                        f"{model_name} has reached its quota. "
                        "Moving to the next model..."
                    )

                    break

                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                ):

                    if attempt < 3:

                        wait_seconds = 30 * attempt

                        print(
                            f"Gemini is temporarily unavailable. "
                            f"Waiting {wait_seconds} seconds..."
                        )

                        time.sleep(wait_seconds)

                        continue

                    print()
                    print(
                        f"{model_name} failed three times. "
                        "Moving to the next model..."
                    )

                    break

                raise RuntimeError(
                    f"Gemini news generation failed: {e}"
                )

    raise RuntimeError(
        "All Gemini models failed.\n"
        f"Last error: {last_error}"
    )


# ============================================================
# ADD NEW STORIES TO THE ARCHIVE
# ============================================================

def add_to_archive(
    existing_archive,
    new_articles
):

    now = datetime.now(
        ZoneInfo("Asia/Seoul")
    )

    published_at = now.strftime(
        "%Y-%m-%d %H:%M"
    )

    today = now.strftime(
        "%Y-%m-%d"
    )

    new_entries = []

    # Use existing headlines to prevent duplicates.
    existing_headlines = {
        clean_text(
            article.get("title", "")
        )
        for article in existing_archive
        if article.get("title")
    }

    for article in new_articles:

        title = clean_text(
            article["title_ko"]
        )

        # Don't add an exact duplicate.
        if title in existing_headlines:

            print(
                f"Skipping duplicate article: {title}"
            )

            continue

        new_entry = {
            "category": article["category"],
            "date": today,
            "published_at": published_at,
            "title": article["title_ko"],
            "body": article["body_ko"],
            "translation": article["translation_en"],
            "vocab": article["vocab"]
        }

        new_entries.append(new_entry)

        existing_headlines.add(title)

    if not new_entries:

        raise RuntimeError(
            "All generated stories were duplicates. "
            "Nothing was added to the archive."
        )

    # Newest stories go first.
    final_archive = (
        new_entries + existing_archive
    )

    with open(
        "news_data.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_archive,
            f,
            ensure_ascii=False,
            indent=4
        )

    print()
    print(
        f"Added {len(new_entries)} new stories."
    )

    print(
        f"Archive now contains "
        f"{len(final_archive)} total stories."
    )

    for article in new_entries:

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

    # 1. Load all previous stories.
    existing_archive = load_existing_archive()

    # 2. Get the latest Yonhap stories.
    raw_articles = fetch_latest_articles()

    # 3. Give Gemini recent headlines so it can avoid
    #    repeating stories that are already on the site.
    recent_headlines = get_recent_headlines(
        existing_archive,
        limit=40
    )

    # 4. Generate four new stories.
    new_articles = generate_topik_news(
        raw_articles,
        recent_headlines
    )

    # 5. ADD the new stories to the existing archive.
    #    Do NOT replace the old stories.
    add_to_archive(
        existing_archive,
        new_articles
    )

    print()
    print("=" * 60)
    print(
        "Daily TOPIK news generation completed successfully."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
