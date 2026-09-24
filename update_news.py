import os
import json
from datetime import datetime
import urllib.request
import xml.etree.ElementTree as ET
import google.generativeai as genai

# Configure Gemini API
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY secret is missing!")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# RSS Feeds for major Korean news categories
RSS_FEEDS = {
    "정치": "https://www.yonhapnewstv.co.kr/browse/feed/v1/0002",
    "경제": "https://www.yonhapnewstv.co.kr/browse/feed/v1/0003",
    "사회": "https://www.yonhapnewstv.co.kr/browse/feed/v1/0004",
    "세계": "https://www.yonhapnewstv.co.kr/browse/feed/v1/0005"
}

def fetch_rss_articles():
    articles = []
    for category, url in RSS_FEEDS.items():
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                xml_data = response.read()
                root = ET.fromstring(xml_data)
                item = root.find('.//item')
                if item is not None:
                    title = item.find('title').text if item.find('title') is not None else ""
                    description = item.find('description').text if item.find('description') is not None else ""
                    articles.append({
                        "original_title": title,
                        "original_text": description,
                        "category": category
                    })
        except Exception as e:
            print(f"Error fetching {category} RSS: {e}")
    return articles

def rewrite_for_topik(article):
    prompt = f"""
You are an expert Korean language instructor designing reading material for advanced TOPIK learners (Levels 4-6).
Take this news item and rewrite it in standard Korean news style using 헤라체 (the descriptive plain style ending in -ㄴ다/는다, -었다/했다, and -다), which is standard for TOPIK reading passages and journalism, suitable for a TOPIK Level 4-6 reading level. Avoid hyper-localized slang or overly dense media idioms, but keep it authentic.
Also provide a clear English translation of the rewritten text, and list 3-5 key advanced vocabulary terms with their English meanings.

Original Headline: {article['original_title']}
Original Content: {article['original_text']}

Output your response STRICTLY as a valid JSON object with the following keys, with no markdown formatting around it:
{{
  "title_ko": "Rewritten headline in Korean",
  "body_ko": "Rewritten body text in Korean",
  "translation_en": "English translation of the body text",
  "vocab": "Key Vocab 1: definition<br>Key Vocab 2: definition"
}}
"""
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        return json.loads(text)
    except Exception as e:
        print(f"AI generation error: {e}")
        return None

def main():
    print("Fetching daily news RSS feeds...")
    raw_articles = fetch_rss_articles()
    print(f"Fetched {len(raw_articles)} articles from feeds.")
    
    processed_articles = []
    today = datetime.now().strftime("%Y-%m-%d")

    for article in raw_articles:
        print(f"Processing category: {article['category']}...")
        rewritten = rewrite_for_topik(article)
        if rewritten:
            processed_articles.append({
                "category": article['category'],
                "date": today,
                "title": rewritten.get("title_ko"),
                "body": rewritten.get("body_ko"),
                "translation": rewritten.get("translation_en"),
                "vocab": rewritten.get("vocab")
            })

    # Save to a JSON file that index.html can load
    with open("news_data.json", "w", encoding="utf-8") as f:
        json.dump(processed_articles, f, ensure_ascii=False, indent=4)
    print(f"Successfully generated news_data.json with {len(processed_articles)} articles!")

if __name__ == "__main__":
    main()
