"""
HTML Content Parser and Chunking Pipeline
Parses HTML pages and produces token-aware, metadata-enriched chunks
ready for embedding and OpenSearch indexing.

Usage:
    python parse_chunks.py --input "web-site/Overview _ Fannie Mae Multifamily Guide.html" \
                           --url "https://mfguide.fanniemae.com/node/10711" \
                           --output chunks.json
"""

import re
import sys
import json
import hashlib
import argparse
from datetime import datetime, timezone
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not installed. Falling back to character-based token estimation.", file=sys.stderr)

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 50
MIN_CHUNK_SIZE = 100
ENCODING_NAME = "cl100k_base"

ALLOWLIST = {'h1', 'h2', 'h3', 'h4', 'p', 'li', 'blockquote', 'td', 'th', 'dt', 'dd', 'pre', 'code'}

HEADING_TAGS = {'h1', 'h2', 'h3', 'h4'}

# Regex for ad/tracking/noise class and id patterns
NOISE_PATTERN = re.compile(
    r'\b(ad|advertisement|promo|tracking|cookie|onetrust|popup|modal|overlay|newsletter)\b',
    re.IGNORECASE
)

# ── Phase 1: Noise Removal ─────────────────────────────────────────────────────

def phase1_remove_noise(soup: BeautifulSoup) -> BeautifulSoup:
    # L1 — Analytics & Tracking
    for tag in soup.find_all(['script', 'noscript']):
        tag.decompose()

    # L2 — Styles
    for tag in soup.find_all(['style']):
        tag.decompose()
    for tag in soup.find_all('link', rel=lambda r: r and 'stylesheet' in r):
        tag.decompose()
    for tag in soup.find_all(True):
        if tag.attrs is not None and 'style' in tag.attrs:
            del tag['style']

    # L3 — Layout Chrome
    for tag in soup.find_all(['nav', 'header', 'footer', 'aside']):
        tag.decompose()

    # L4 — Interactive Elements
    for tag in soup.find_all(['form', 'button', 'input', 'select', 'textarea', 'label']):
        tag.decompose()

    # L5 — Ads & Tracking Widgets (match class or id)
    for tag in soup.find_all(True):
        if tag.attrs is None:
            continue
        classes = ' '.join(tag.get('class', []))
        tag_id = tag.get('id', '')
        if NOISE_PATTERN.search(classes) or NOISE_PATTERN.search(tag_id):
            tag.decompose()

    # L6 — Embedded Media
    for tag in soup.find_all(['iframe', 'svg', 'canvas']):
        tag.decompose()

    # L7 — HTML Comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    return soup


# ── Phase 2: Semantic Content Extraction ──────────────────────────────────────

def _find_main_content(soup: BeautifulSoup):
    """Return the main content region using a priority chain."""
    candidates = [
        lambda s: s.find('main'),
        lambda s: s.find(attrs={'role': 'main'}),
        lambda s: s.find('div', id='content-area'),
        lambda s: s.find('section', id='content'),
        lambda s: s.find('div', id='content'),
        lambda s: s.find('div', class_=re.compile(r'\bregion-content\b')),
        lambda s: s.find('article'),
        lambda s: s.find('body'),
    ]
    for fn in candidates:
        result = fn(soup)
        if result:
            return result
    return soup


def phase2_semantic_extract(soup: BeautifulSoup) -> list:
    main_content = _find_main_content(soup)
    content_blocks = []
    current_heading = ''

    for element in main_content.descendants:
        if not hasattr(element, 'name') or element.name not in ALLOWLIST:
            continue

        text = element.get_text(strip=True)
        if not text:
            continue

        if element.name in HEADING_TAGS:
            current_heading = text

        content_blocks.append({
            'text': text,
            'section_heading': current_heading,
            'element_type': element.name,
        })

    return content_blocks


# ── Phase 3: Section Grouping ──────────────────────────────────────────────────

def group_into_sections(content_blocks: list) -> list:
    sections = []
    current_section = {'heading': '', 'text': ''}

    for block in content_blocks:
        if block['element_type'].startswith('h'):
            if current_section['text']:
                sections.append(current_section)
            current_section = {
                'heading': block['section_heading'],
                'text': block['text'] + '\n',
            }
        else:
            if not current_section['heading']:
                current_section['heading'] = block['section_heading']
            current_section['text'] += block['text'] + '\n'

    if current_section['text']:
        sections.append(current_section)

    return sections


# ── Phase 4: Token-Aware Chunking ─────────────────────────────────────────────

def count_tokens(text: str) -> int:
    if _TIKTOKEN_AVAILABLE:
        enc = tiktoken.get_encoding(ENCODING_NAME)
        return len(enc.encode(text))
    return len(text) // 4  # fallback: ~4 chars per token


def create_overlapping_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list:
    total_tokens = count_tokens(text)

    if total_tokens <= MIN_CHUNK_SIZE or total_tokens <= chunk_size:
        return [text.strip()]

    if _TIKTOKEN_AVAILABLE:
        enc = tiktoken.get_encoding(ENCODING_NAME)
        tokens = enc.encode(text)
    else:
        # Character-based fallback
        char_chunk = chunk_size * 4
        char_overlap = overlap * 4
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + char_chunk, len(text))
            chunk_text = text[start:end]
            if end < len(text):
                search_start = int(len(chunk_text) * 0.8)
                best_break = -1
                for ending in ['.', '!', '?', '\n']:
                    pos = chunk_text.rfind(ending, search_start)
                    if pos > best_break:
                        best_break = pos
                if best_break > 0:
                    chunk_text = chunk_text[:best_break + 1]
                    end = start + len(chunk_text)
            chunks.append(chunk_text.strip())
            if end >= len(text):
                break
            start = end - char_overlap
            if start >= end:
                start = end
        return [c for c in chunks if c]

    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = enc.decode(chunk_tokens)

        if end_idx < len(tokens):
            search_start = int(len(chunk_text) * 0.8)
            best_break = -1
            for ending in ['.', '!', '?', '\n']:
                pos = chunk_text.rfind(ending, search_start)
                if pos > best_break:
                    best_break = pos
            if best_break > 0:
                chunk_text = chunk_text[:best_break + 1].strip()
                actual_tokens = enc.encode(chunk_text)
                end_idx = start_idx + len(actual_tokens)

        chunks.append(chunk_text.strip())

        if end_idx >= len(tokens):
            break

        start_idx = end_idx - overlap
        if start_idx >= end_idx:
            start_idx = end_idx

    return [c for c in chunks if c]


# ── Phase 5: ID Generation ─────────────────────────────────────────────────────

def generate_document_id(url: str) -> str:
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".lower().rstrip('/')
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}-{chunk_index:04d}"


# ── Phase 6: Metadata Enrichment ──────────────────────────────────────────────

def create_chunk_metadata(
    document_id, chunk_id, source_url, page_title,
    section_heading, chunk_index, total_chunks, language
) -> dict:
    return {
        'document_id': document_id,
        'chunk_id': chunk_id,
        'source_url': source_url,
        'page_title': page_title,
        'section_heading': section_heading,
        'chunk_index': chunk_index,
        'total_chunks': total_chunks,
        'language': language,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def process_html_to_chunks(html: str, url: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list:
    soup = BeautifulSoup(html, 'html.parser')

    title_tag = soup.find('title')
    page_title = title_tag.get_text(strip=True) if title_tag else ''

    html_tag = soup.find('html')
    language = html_tag.get('lang', 'en') if html_tag else 'en'

    soup = phase1_remove_noise(soup)
    content_blocks = phase2_semantic_extract(soup)

    if not content_blocks:
        return []

    sections = group_into_sections(content_blocks)
    document_id = generate_document_id(url)

    all_chunks = []
    global_chunk_index = 0

    for section in sections:
        section_text = section['text'].strip()
        section_heading = section['heading']

        if not section_text:
            continue

        text_chunks = create_overlapping_chunks(section_text, chunk_size=chunk_size, overlap=overlap)

        for chunk_text in text_chunks:
            chunk_id = generate_chunk_id(document_id, global_chunk_index)
            chunk_payload = {
                'text': chunk_text,
                'metadata': create_chunk_metadata(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    source_url=url,
                    page_title=page_title,
                    section_heading=section_heading,
                    chunk_index=global_chunk_index,
                    total_chunks=0,  # updated below
                    language=language,
                ),
            }
            all_chunks.append(chunk_payload)
            global_chunk_index += 1

    total_chunks = len(all_chunks)
    for chunk in all_chunks:
        chunk['metadata']['total_chunks'] = total_chunks

    return all_chunks


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Parse HTML and produce semantic chunks for OpenSearch indexing.')
    parser.add_argument('--input', required=True, help='Path to the HTML file')
    parser.add_argument('--url', required=True, help='Source URL for metadata')
    parser.add_argument('--output', default='chunks.json', help='Output JSON file (default: chunks.json)')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE, help=f'Token chunk size (default: {DEFAULT_CHUNK_SIZE})')
    parser.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP, help=f'Token overlap (default: {DEFAULT_OVERLAP})')
    args = parser.parse_args()

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            html = f.read()
    except Exception as e:
        print(f"Failed to read input file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing: {args.input}")
    chunks = process_html_to_chunks(html, args.url, chunk_size=args.chunk_size, overlap=args.overlap)

    if not chunks:
        print("No chunks produced — check that the HTML has semantic content.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. {len(chunks)} chunks written to {args.output}")
    if chunks:
        sample = chunks[0]
        print(f"Sample chunk 0: [{sample['metadata']['section_heading']}] {sample['text'][:80]}...")


if __name__ == '__main__':
    main()
