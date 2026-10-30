
import json
import re
from pathlib import Path

import pysbd


INPUT_FILE = Path("pages.json")
OUTPUT_FILE = Path("chunks.json")

TARGET_CHUNK_SIZE = 1000
OVERLAP_SENTENCES = 2

segmenter = pysbd.Segmenter(
    language="en",
    clean=False
)


def split_long_sentence(sentence, max_chars):
    """Split an oversized sentence at word boundaries."""
    words = sentence.split()
    pieces = []
    current = ""

    for word in words:
        # Handle an unusually long word safely.
        if len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""

            for i in range(0, len(word), max_chars):
                pieces.append(word[i:i + max_chars])

            continue

        candidate = f"{current} {word}".strip()

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = word

    if current:
        pieces.append(current)

    return pieces


def prepare_sentences(text):
    """Normalize whitespace and split text into sentence units."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = segmenter.segment(text)
    units = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) <= TARGET_CHUNK_SIZE:
            units.append(sentence)
        else:
            units.extend(
                split_long_sentence(sentence, TARGET_CHUNK_SIZE)
            )

    return units


def create_chunks(text):
    """Build chunks with sentence-aware overlap."""
    units = prepare_sentences(text)
    chunks = []
    start = 0

    while start < len(units):
        end = start
        current_text = ""

        # Add units until the target size would be exceeded.
        while end < len(units):
            candidate = f"{current_text} {units[end]}".strip()

            if len(candidate) > TARGET_CHUNK_SIZE and current_text:
                break

            current_text = candidate
            end += 1

        if current_text:
            chunks.append(current_text)

        if end >= len(units):
            break

        # Retain up to two preceding units for context.
        start = max(start + 1, end - OVERLAP_SENTENCES)

    return chunks


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        pages = json.load(file)

    if not isinstance(pages, list):
        raise ValueError("pages.json must contain a JSON list.")

    all_chunks = []
    seen_urls = set()
    seen_chunk_texts = set()
    empty_chunks = 0
    duplicate_chunk_content = 0
    skipped_same_page_duplicates = 0
    repeated_content_across_pages = 0
    skipped_status_pages = 0

    for page_number, page in enumerate(pages):
        if not isinstance(page, dict):
            print(f"Skipping invalid page at index {page_number}")
            continue

        if page.get("crawl_status") in ("failed", "skipped"):
            skipped_status_pages += 1
            continue

        url_value = page.get("url")
        title_value = page.get("title")
        content_value = page.get("content")
        url = url_value.strip() if isinstance(url_value, str) else ""
        title = title_value.strip() if isinstance(title_value, str) else ""
        content = content_value.strip() if isinstance(content_value, str) else ""

        if not url or not title:
            print(f"Skipping page {page_number}: missing URL or title")
            continue

        if url in seen_urls:
            print(f"Skipping duplicate URL: {url}")
            continue

        seen_urls.add(url)

        if not content:
            print(f"Skipping empty page: {url}")
            continue

        page_chunks = create_chunks(content)
        page_seen_chunk_texts = set()

        for chunk_index, chunk_text in enumerate(page_chunks):
            if not chunk_text.strip():
                empty_chunks += 1
                continue
            if chunk_text in seen_chunk_texts:
                duplicate_chunk_content += 1
                if url in seen_urls and chunk_text in page_seen_chunk_texts:
                    skipped_same_page_duplicates += 1
                    continue
                repeated_content_across_pages += 1
            page_seen_chunk_texts.add(chunk_text)
            seen_chunk_texts.add(chunk_text)
            all_chunks.append({
                "chunk_id": len(all_chunks),
                "page_chunk_index": chunk_index,
                "title": title,
                "url": url,
                "text": chunk_text
            })

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    # Basic validation of the saved result.
    ids = [chunk["chunk_id"] for chunk in all_chunks]

    assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"
    assert all(chunk["text"].strip() for chunk in all_chunks)
    assert all(len(chunk["text"]) <= TARGET_CHUNK_SIZE
               for chunk in all_chunks)

    print("Pages loaded:", len(pages))
    print("Unique URLs processed:", len(seen_urls))
    print("Pages marked failed/skipped:", skipped_status_pages)
    print("Chunks created:", len(all_chunks))
    print("Empty chunks:", empty_chunks)
    print("Duplicate exact chunk content:", duplicate_chunk_content)
    print("Same-page exact duplicates skipped:", skipped_same_page_duplicates)
    print("Repeated exact content across pages retained:", repeated_content_across_pages)
    print("Maximum chunk length:",
          max((len(c["text"]) for c in all_chunks), default=0))
    print("Saved to:", OUTPUT_FILE.resolve())

    if all_chunks:
        print("\nFirst chunk:")
        print(json.dumps(all_chunks[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
