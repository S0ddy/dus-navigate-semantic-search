# AI Agent Prompt: HTML Content Parser and Chunking Pipeline

## Context

You need to implement an HTML content parsing and chunking pipeline for semantic search. This pipeline will:

1. **Parse and clean HTML content** using a two-phase sanitization approach
2. **Extract semantic content** while preserving document structure
3. **Create token-aware chunks** with overlapping windows
4. **Generate metadata-enriched chunk payloads** ready for embedding and search

**Important:** The embedding and OpenSearch integration already exist in your environment with mocked JSON chunks. You need to replace those mocked chunks with real chunks generated from HTML parsing.

---

## Architecture Overview

```
HTML Input → Phase 1: Remove Noise → Phase 2: Extract Semantic Content →
Group by Sections → Token-Based Chunking → Add Metadata → JSON Chunks
```

**Do NOT implement:** Embedding generation or OpenSearch indexing (already exists in your environment)

**DO implement:** Everything from HTML parsing through chunk creation with metadata

---

## Phase 1: HTML Noise Removal

### Objective
Remove all non-semantic elements while preserving the DOM structure for Phase 2.

### Elements to Remove (Blacklist Approach)

**L1 — Analytics & Tracking**
- Tags: `<script>`, `<noscript>`

**L2 — Styles**
- Tags: `<style>`, `<link rel="stylesheet">`
- Attributes: All `style=""` attributes

**L3 — Layout Chrome**
- Tags: `<nav>`, `<header>`, `<footer>`, `<aside>`

**L4 — Interactive Elements**
- Tags: `<form>`, `<button>`, `<input>`, `<select>`, `<textarea>`, `<label>`

**L5 — Ads & Tracking Widgets**
- Elements with class/id containing these patterns (case-insensitive regex):
  - `ad`, `advertisement`, `promo`, `tracking`
  - `cookie`, `onetrust`, `popup`, `modal`, `overlay`, `newsletter`

**L6 — Embedded Media**
- Tags: `<iframe>`, `<svg>`, `<canvas>`

**L7 — HTML Comments**
- All `<!-- comments -->`

### Implementation Notes
- Use BeautifulSoup4 or similar HTML parser
- Parse HTML: `soup = BeautifulSoup(html, 'html.parser')`
- Before Phase 1 processing, extract page metadata:
  - `page_title` from `<title>` tag
  - `language` from `<html lang="...">` attribute (default: "en")
- Use `.decompose()` to remove elements entirely (not just hide)
- Process in-place to modify the soup object

---

## Phase 2: Semantic Content Extraction

### Objective
Extract only meaningful content using an allowlist approach, preserving structural context.

### Content Scope Selection

Find the main content region using this priority chain (first match wins):

1. `<main>` tag
2. `[role="main"]` attribute
3. `<div id="content-area">`
4. `<section id="content">`
5. `<div id="content">`
6. `<div class="region-content">` (regex: `\bregion-content\b`)
7. `<article>` tag
8. `<body>` tag (fallback)

### Semantic Element Allowlist

Only extract text from these element types:

**Headings:** `h1`, `h2`, `h3`, `h4`  
**Body Text:** `p`, `li`, `blockquote`  
**Tables:** `td`, `th`  
**Definitions:** `dt`, `dd`  
**Code:** `pre`, `code`

### Content Block Structure

For each allowed element, create a content block:

```python
{
    'text': 'extracted text content',
    'section_heading': 'current heading context',
    'element_type': 'p'  # or h1, h2, etc.
}
```

**Important:** Track the current section heading as you traverse the DOM. When you encounter a heading element (`h1`-`h4`), update the current heading context. All subsequent elements inherit this heading until a new heading is found.

### Algorithm

```python
content_blocks = []
current_heading = ''

# Iterate through descendants of main_content
for element in main_content.descendants:
    if element.name not in ALLOWLIST:
        continue
   
    text = element.get_text(strip=True)
    if not text:
        continue
   
    # Update heading context when we hit a heading
    if element.name in ['h1', 'h2', 'h3', 'h4']:
        current_heading = text
   
    content_blocks.append({
        'text': text,
        'section_heading': current_heading,
        'element_type': element.name
    })
```

---

## Phase 3: Section Grouping

### Objective
Group consecutive content blocks under the same heading into logical sections for chunking.

### Algorithm

```python
sections = []
current_section = {'heading': '', 'text': ''}

for block in content_blocks:
    # Start new section when we hit a heading element
    if block['element_type'].startswith('h'):
        if current_section['text']:  # Save previous section
            sections.append(current_section)
        current_section = {
            'heading': block['section_heading'],
            'text': block['text'] + '\n'
        }
    else:
        # Add to current section
        if not current_section['heading']:
            current_section['heading'] = block['section_heading']
        current_section['text'] += block['text'] + '\n'

# Don't forget the last section
if current_section['text']:
    sections.append(current_section)
```

---

## Phase 4: Token-Aware Chunking

### Configuration

```python
DEFAULT_CHUNK_SIZE = 512    # tokens
DEFAULT_OVERLAP = 50        # tokens (~10%)
MIN_CHUNK_SIZE = 100        # tokens
ENCODING_NAME = "cl100k_base"  # tiktoken encoding
```

### Token Counting

Use `tiktoken` library for accurate token counting:

```python
import tiktoken

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))
```

**Fallback** (if tiktoken unavailable): Estimate 1 token ≈ 4 characters

### Chunking Strategy

**Why overlap?** A concept split across chunks would be incomplete in each chunk. Overlap ensures the concept appears fully in at least one chunk.

**Why 512 tokens?** Safe for all AWS Bedrock embedding models:
- Amazon Titan: supports up to 8192 tokens
- Cohere Embed: supports up to 512 tokens
- 512 is the conservative default for broad compatibility

### Chunking Algorithm

For each section:

1. **Check if chunking needed:**
   - If total tokens < MIN_CHUNK_SIZE: return as single chunk
   - If total tokens ≤ CHUNK_SIZE: return as single chunk
   - Otherwise, proceed with chunking

2. **Token-based splitting:**
   ```python
   encoding = tiktoken.get_encoding(encoding_name)
   tokens = encoding.encode(section_text)
   
   chunks = []
   start_idx = 0
   
   while start_idx < len(tokens):
       end_idx = min(start_idx + chunk_size, len(tokens))
       chunk_tokens = tokens[start_idx:end_idx]
       chunk_text = encoding.decode(chunk_tokens)
       
       # Try to break at sentence boundaries (if not at end)
       if end_idx < len(tokens):
           search_start = int(len(chunk_text) * 0.8)  # Last 20% of chunk
           sentence_endings = ['.', '!', '?', '\n']
           
           best_break = -1
           for ending in sentence_endings:
               pos = chunk_text.rfind(ending, search_start)
               if pos > best_break:
                   best_break = pos
           
           if best_break > 0:
               chunk_text = chunk_text[:best_break + 1].strip()
               actual_tokens = encoding.encode(chunk_text)
               end_idx = start_idx + len(actual_tokens)
       
       chunks.append(chunk_text.strip())
       
       if end_idx >= len(tokens):
           break
       
       # Create overlap by moving start position
       start_idx = end_idx - overlap
       
       # Prevent infinite loop
       if start_idx >= end_idx:
           start_idx = end_idx
   ```

---

## Phase 5: ID Generation

### Document ID

Generate deterministic, collision-resistant IDs from URLs:

```python
import hashlib
from urllib.parse import urlparse

def generate_document_id(url: str) -> str:
    # Normalize URL
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".lower().rstrip('/')
   
    # SHA-256 hash, first 16 hex characters
    hash_obj = hashlib.sha256(normalized.encode('utf-8'))
    return hash_obj.hexdigest()[:16]
```

**Properties:**
- Stable: Same URL → Same ID
- Unique: Different URLs → Different IDs
- 16 hex chars = 64 bits = collision-resistant

### Chunk ID

```python
def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}-{chunk_index:04d}"
```

**Format:** `{document_id}-{chunk_index}`  
**Example:** `f8d0d8ad3fd4a506-0000`, `f8d0d8ad3fd4a506-0001`, etc.

---

## Phase 6: Metadata Enrichment

### Chunk Metadata Structure

Each chunk must include comprehensive metadata:

```python
from datetime import datetime, timezone

def create_chunk_metadata(
    document_id: str,
    chunk_id: str,
    source_url: str,
    page_title: str,
    section_heading: str,
    chunk_index: int,
    total_chunks: int,
    language: str
) -> dict:
    return {
        'document_id': document_id,       # For re-indexing and updates
        'chunk_id': chunk_id,             # Unique identifier
        'source_url': source_url,         # Provenance
        'page_title': page_title,         # From <title> tag
        'section_heading': section_heading,  # Contextual heading
        'chunk_index': chunk_index,       # 0-based position
        'total_chunks': total_chunks,     # Total chunks in document
        'language': language,             # From <html lang="...">
        'created_at': datetime.now(timezone.utc).isoformat()  # ISO 8601 timestamp
    }
```

---

## Final Output Format

### Complete Chunk Structure

Each chunk in the output array should have this exact structure:

```json
{
  "text": "actual chunk content text here...",
  "metadata": {
    "document_id": "f8d0d8ad3fd4a506",
    "chunk_id": "f8d0d8ad3fd4a506-0000",
    "source_url": "https://mfguide.fanniemae.com/node/10711",
    "page_title": "Overview | Fannie Mae Multifamily Guide",
    "section_heading": "Overview",
    "chunk_index": 0,
    "total_chunks": 22,
    "language": "en",
    "created_at": "2026-04-01T13:50:08.913149+00:00"
  }
}
```

### Example Output

Based on a real example from the system:

```json
[
  {
    "text": "Overview\nSection 101\nUsing the Guide",
    "metadata": {
      "document_id": "f8d0d8ad3fd4a506",
      "chunk_id": "f8d0d8ad3fd4a506-0000",
      "source_url": "https://mfguide.fanniemae.com/node/10711",
      "page_title": "Overview | Fannie Mae Multifamily Guide",
      "section_heading": "Overview",
      "chunk_index": 0,
      "total_chunks": 22,
      "language": "en",
      "created_at": "2026-04-01T13:50:08.913149+00:00"
    }
  },
  {
    "text": "Using the Guide\n101.01\nOrganization",
    "metadata": {
      "document_id": "f8d0d8ad3fd4a506",
      "chunk_id": "f8d0d8ad3fd4a506-0001",
      "source_url": "https://mfguide.fanniemae.com/node/10711",
      "page_title": "Overview | Fannie Mae Multifamily Guide",
      "section_heading": "Using the Guide",
      "chunk_index": 1,
      "total_chunks": 22,
      "language": "en",
      "created_at": "2026-04-01T13:50:08.913904+00:00"
    }
  }
]
```

---

## Main Pipeline Function

### Complete Implementation

```python
def process_html_to_chunks(
    html: str,
    url: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> list:
    """
    Complete pipeline: HTML → chunks with metadata.
   
    Args:
        html: Raw HTML content
        url: Source URL
        chunk_size: Target chunk size in tokens (default: 512)
        overlap: Overlap size in tokens (default: 50)
   
    Returns:
        List of chunk dictionaries ready for embedding
    """
    # Stage 1: Parse HTML and extract metadata
    soup = BeautifulSoup(html, 'html.parser')
   
    # Extract page metadata before cleaning
    title_tag = soup.find('title')
    page_title = title_tag.get_text(strip=True) if title_tag else ''
   
    html_tag = soup.find('html')
    language = html_tag.get('lang', 'en') if html_tag else 'en'
   
    # Phase 1: Remove noise
    soup = phase1_remove_noise(soup)
   
    # Stage 2: Phase 2 - Extract structured content
    content_blocks = phase2_semantic_extract(soup)
   
    if not content_blocks:
        return []
   
    # Stage 3: Group by sections
    sections = group_into_sections(content_blocks)
   
    # Stage 4: Generate document ID
    document_id = generate_document_id(url)
   
    # Stage 5: Chunk each section
    all_chunks = []
    global_chunk_index = 0
   
    for section in sections:
        section_text = section['text'].strip()
        section_heading = section['heading']
       
        if not section_text:
            continue
       
        # Create overlapping chunks
        text_chunks = create_overlapping_chunks(
            section_text,
            chunk_size=chunk_size,
            overlap=overlap
        )
       
        # Create payload for each chunk
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
                    total_chunks=0,  # Will update after processing
                    language=language
                )
            }
           
            all_chunks.append(chunk_payload)
            global_chunk_index += 1
   
    # Stage 6: Update total_chunks in all metadata
    total_chunks = len(all_chunks)
    for chunk in all_chunks:
        chunk['metadata']['total_chunks'] = total_chunks
   
    return all_chunks
```

---

## Dependencies

### Required Python Libraries

```python
from bs4 import BeautifulSoup, Comment, NavigableString
import tiktoken
import hashlib
import re
from datetime import datetime, timezone
from urllib.parse import urlparse
```

### Installation

```bash
pip install beautifulsoup4 tiktoken
```

---

## Testing & Validation

### Test HTML Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Test Document</title>
    <script>console.log('tracking');</script>
</head>
<body>
    <nav><a href="/">Home</a></nav>
    <main>
        <h1>Main Heading</h1>
        <p>First paragraph of content.</p>
        <h2>Subheading</h2>
        <p>Second paragraph under subheading.</p>
    </main>
    <footer>Copyright 2026</footer>
</body>
</html>
```

### Expected Output

Should produce 2-4 chunks depending on token count, with:
- No nav or footer content
- No script content
- Proper section heading tracking ("Main Heading" → "Subheading")
- Sequential chunk IDs (0000, 0001, etc.)
- All metadata fields populated

### Validation Checklist

✅ All non-semantic elements removed (scripts, nav, footer, etc.)  
✅ Only allowlisted elements extracted (h1-h4, p, li, etc.)  
✅ Section headings properly tracked and assigned  
✅ Chunks are ~512 tokens with 50 token overlap  
✅ Sentence boundaries respected when possible  
✅ Document ID is deterministic (same URL → same ID)  
✅ Chunk IDs are sequential and unique  
✅ All metadata fields present and correctly formatted  
✅ Timestamps in ISO 8601 format with UTC timezone  
✅ Output is valid JSON that matches example structure

---

## Integration Points

### Input
Your system should accept:
- `html`: Raw HTML string
- `url`: Source URL string

### Output
Your system should produce:
- List of chunk dictionaries (JSON-serializable)
- Each chunk has `text` and `metadata` fields
- Metadata matches the exact structure shown above

### Next Steps (Already in Your Environment)
1. **Embedding Generation:** Extract `chunk['text']` and send to embedding model
2. **OpenSearch Indexing:** Combine chunk with embedding vector and index
3. **Semantic Search:** Use existing search infrastructure with real chunks

---

## Performance Considerations

### For Large Documents

- Process documents in batches if handling many URLs
- Consider parallel processing for multiple documents
- Token counting can be expensive - cache results when possible
- BeautifulSoup parsing is CPU-intensive for very large HTML

### Memory Management

- Use generators for large chunk lists if memory constrained
- Clear soup object after processing: `soup.decompose()`
- Don't keep all chunks in memory if processing thousands of documents

### Optimization Tips

- Compile regex patterns once and reuse
- Use tiktoken's batch encoding if available
- Profile token counting vs chunking to find bottlenecks

---

## Common Edge Cases

### Empty Content
- If no semantic content found after Phase 2, return empty list `[]`
- Don't create chunks with empty text

### No Headings
- If content has no headings, use empty string `''` for section_heading
- Still create chunks from paragraph content

### Very Short Documents
- If total tokens < MIN_CHUNK_SIZE (100), return as single chunk
- Set chunk_index to 0, total_chunks to 1

### Very Long Sections
- A single section might produce many chunks (10+)
- Ensure overlap doesn't cause infinite loops
- All chunks from same section share same section_heading

### Special Characters
- Preserve UTF-8 characters in text
- URLs should be properly encoded
- Timestamps must be ISO 8601 with timezone

---

## Task Summary

**Implement a Python function** `process_html_to_chunks(html, url, chunk_size=512, overlap=50)` that:

1. Parses HTML and removes all non-semantic content
2. Extracts semantic content with structural context
3. Groups content into logical sections
4. Creates token-aware overlapping chunks
5. Generates deterministic IDs
6. Enriches each chunk with comprehensive metadata
7. Returns JSON-serializable list of chunks

The output chunks will replace mocked JSON in your existing embedding and search pipeline.

**Do NOT implement:** Embedding generation, OpenSearch client, or indexing logic.

**Test your implementation** with the provided example HTML and validate against the checklist above.