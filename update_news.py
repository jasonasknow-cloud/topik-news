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
