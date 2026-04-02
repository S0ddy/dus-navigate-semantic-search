"""
Full ingestion pipeline: scrape → parse → embed → index

Usage:
    python pipeline.py                        # scrape all sections from overview HTML
    python pipeline.py --url <url>            # scrape a single URL
    python pipeline.py --save chunks.json     # also save chunks to disk
"""

import re
import sys
import json
import argparse
from urllib.parse import urljoin

from bs4 import BeautifulSoup

import config
from parse_chunks import process_html_to_chunks, generate_document_id

OVERVIEW_HTML = "web-site/Overview _ Fannie Mae Multifamily Guide.html"
BASE_URL      = "https://mfguide.fanniemae.com"


def extract_section_urls(local_html_path: str) -> list:
    with open(local_html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    seen, urls = set(), []
    for a in soup.find_all("a", href=True):
        m = re.search(r"(/node/\d+)", a["href"])
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            urls.append(urljoin(BASE_URL, m.group(1)))

    return sorted(urls, key=lambda u: int(u.rsplit("/", 1)[-1]))


def run(urls: list = None, save_path: str = None):
    if urls is None:
        urls = extract_section_urls(OVERVIEW_HTML)

    print(f"Pages to process: {len(urls)}")
    config.STORE.create_index()

    all_chunks = []
    total = len(urls)

    for i, url in enumerate(urls, 1):
        print(f"[{i:3}/{total}] {url}")
        try:
            html   = config.SCRAPER.fetch(url)
            chunks = process_html_to_chunks(html, url)

            texts   = [c["text"] for c in chunks]
            vectors = config.EMBEDDER.embed(texts)

            docs = [
                {"text": c["text"], "embedding": v, **c["metadata"]}
                for c, v in zip(chunks, vectors)
            ]

            config.STORE.index_documents(docs)
            all_chunks.extend(chunks)
            print(f"       → {len(chunks)} chunks indexed")

        except Exception as e:
            print(f"       [ERROR] {e}", file=sys.stderr)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"Chunks saved to {save_path}")

    print(f"\nDone. Total chunks indexed: {len(all_chunks)}")


def update_page(url: str):
    document_id = generate_document_id(url)
    print(f"Deleting existing chunks for {url} (document_id={document_id})...")
    config.STORE.delete_page_chunks(document_id)

    html    = config.SCRAPER.fetch(url)
    chunks  = process_html_to_chunks(html, url)
    texts   = [c["text"] for c in chunks]
    vectors = config.EMBEDDER.embed(texts)
    docs    = [{"text": c["text"], "embedding": v, **c["metadata"]} for c, v in zip(chunks, vectors)]
    config.STORE.index_documents(docs)
    print(f"Done. {len(docs)} chunks indexed for {url}")


def main():
    parser = argparse.ArgumentParser(description="Scrape, parse, embed, and index MF Guide sections.")
    parser.add_argument("--url",    help="Process a single URL instead of all sections")
    parser.add_argument("--update", metavar="URL", help="Delete and re-index a single page")
    parser.add_argument("--save",   metavar="FILE", help="Also save chunks to a JSON file")
    args = parser.parse_args()

    if args.update:
        update_page(args.update)
    else:
        urls = [args.url] if args.url else None
        run(urls=urls, save_path=args.save)


if __name__ == "__main__":
    main()
