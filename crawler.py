"""Bounded crawler for official RNSIT HTML pages (PDFs are skipped)."""

from collections import deque
import json
import os
import re
import time
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()


START_URL = "https://www.rnsit.ac.in/"
ALLOWED_HOSTS = {"rnsit.ac.in", "www.rnsit.ac.in"}
MAX_PAGES = max(1, int(os.getenv("RNSIT_MAX_PAGES", "300")))
REQUEST_TIMEOUT = (5, 20)
REQUEST_DELAY_SECONDS = max(0.0, float(os.getenv("RNSIT_CRAWL_DELAY", "0.35")))
MIN_CONTENT_CHARS = 160
SKIP_SUFFIXES = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".zip", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".mp4", ".mp3")
TRACKING_PARAMS = {"fbclid", "gclid", "dclid", "mc_cid", "mc_eid"}
DROP_ELEMENTS = "script, style, nav, footer, header, form, aside, .header_sec_top, .header_sec_bottom, .header_sec_links, .intr-mnu, .breadcrub_sec"


def canonicalize_url(url):
    """Normalize URL noise while preserving meaningful query parameters."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname.lower() if parsed.hostname else ""
        port = parsed.port
    except ValueError:
        return None
    if host not in ALLOWED_HOSTS or parsed.scheme.lower() not in {"http", "https"}:
        return None
    if parsed.username or parsed.password or port not in {None, 80, 443}:
        return None
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/" and "." not in path.rsplit("/", 1)[-1]:
        path = path.rstrip("/") + "/"
    query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
             if key.lower() not in TRACKING_PARAMS and not key.lower().startswith("utm_")]
    # The apex and www hosts serve the same official site; use one canonical host.
    return urlunsplit(("https", "www.rnsit.ac.in", path, urlencode(query, doseq=True), ""))


def extract_content(html, page_url):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    canonical = soup.find("link", rel="canonical")
    canonical_url = canonicalize_url(urljoin(page_url, canonical.get("href", ""))) if canonical else page_url
    if not canonical_url:
        canonical_url = page_url
    content = soup.find("div", class_="main_wrapper") or soup.find("main") or soup.find("article")
    if content is None:
        return title, canonical_url, "", soup
    for element in content.select(DROP_ELEMENTS):
        element.decompose()
    text = re.sub(r"\s+", " ", content.get_text(" ", strip=True)).strip()
    return title, canonical_url, text, soup


def crawl():
    start = canonicalize_url(START_URL)
    queue = deque([start])
    queued = {start}
    attempted = set()
    pages = []
    counts = {"discovered": 1, "processed": 0, "skipped": 0, "failed": 0, "duplicates": 0}
    session = requests.Session()
    session.headers.update({"User-Agent": "RNSITKnowledgeAssistant/1.0 (polite educational crawler)"})

    while queue and len(attempted) < MAX_PAGES:
        current_url = queue.popleft()
        if current_url in attempted:
            counts["duplicates"] += 1
            continue
        attempted.add(current_url)
        print("Crawling:", current_url)
        if REQUEST_DELAY_SECONDS:
            time.sleep(REQUEST_DELAY_SECONDS)
        try:
            response = session.get(current_url, timeout=REQUEST_TIMEOUT, allow_redirects=False)
        except requests.RequestException as error:
            counts["failed"] += 1
            print("Request failed:", current_url, type(error).__name__)
            continue
        if response.status_code in {301, 302, 303, 307, 308}:
            redirect_url = canonicalize_url(urljoin(current_url, response.headers.get("Location", "")))
            if not redirect_url:
                counts["failed"] += 1
                print("Redirect left the allowed RNSIT hosts:", current_url)
            elif redirect_url not in queued and redirect_url not in attempted:
                if len(queued) < MAX_PAGES:
                    queue.append(redirect_url)
                    queued.add(redirect_url)
                    counts["discovered"] += 1
            else:
                counts["duplicates"] += 1
            continue
        if response.status_code != 200:
            counts["failed"] += 1
            print("HTTP failure:", current_url, response.status_code)
            continue
        if "text/html" not in response.headers.get("Content-Type", "").lower():
            counts["skipped"] += 1
            continue

        title, canonical_url, text, soup = extract_content(response.text, current_url)
        if not title or not canonical_url or len(text) < MIN_CONTENT_CHARS:
            counts["skipped"] += 1
        else:
            pages.append({
                "url": canonical_url,
                "title": title,
                "content": text,
                "crawl_status": "processed",
            })
            counts["processed"] += 1

        for link in soup.find_all("a", href=True):
            candidate = canonicalize_url(urljoin(current_url, link["href"]))
            if not candidate:
                continue
            path = urlsplit(candidate).path.lower()
            if path.endswith(SKIP_SUFFIXES):
                counts["skipped"] += 1
                continue
            if candidate in queued or candidate in attempted:
                counts["duplicates"] += 1
                continue
            if len(queued) >= MAX_PAGES:
                continue
            queue.append(candidate)
            queued.add(candidate)
            counts["discovered"] += 1

    counts["skipped"] += len(queue)
    with open("pages.json", "w", encoding="utf-8") as file:
        json.dump(pages, file, ensure_ascii=False, indent=2)
    report = {**counts, "limit": MAX_PAGES, "pages_file": "pages.json", "pdf_ingestion": "excluded; linked PDFs are skipped"}
    with open("crawl_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print("\nCrawl report:", json.dumps(report, indent=2))


if __name__ == "__main__":
    crawl()
