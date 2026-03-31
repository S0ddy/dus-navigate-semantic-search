# Semantic Search Prototype for Fannie Mae MF Guide

## Context

The goal of this project is to build a **semantic search prototype** for content from the Fannie Mae Multifamily Guide page:

`https://mfguide.fanniemae.com/node/10711`

The current working assumptions are:

- OpenSearch is running locally on macOS.
- OpenSearch Dashboards is already available locally.
- The initial dataset will be **manually created chunks** rather than a full crawler/parser pipeline.
- The system should support:
  - storing each chunk as a document,
  - generating an embedding for each chunk,
  - saving the embedding into OpenSearch as a vector,
  - running semantic similarity search against those stored vectors.

**Embeddings are generated locally using the free Python library `sentence-transformers` (model: all-MiniLM-L6-v2, dimension 384). No OpenAI or paid APIs are used.**

There are two possible implementation approaches:

1. **Python-managed embeddings (current approach)**
   - Python reads chunk text.
   - Python generates embeddings locally using `sentence-transformers`.
   - Python writes the text plus vector into OpenSearch.

2. **OpenSearch-managed embedding pipelines (not used)**
   - Python sends plain text documents to OpenSearch.
   - OpenSearch generates embeddings using its configured model pipeline.
   - OpenSearch stores the generated vector automatically.

Because the current request is to describe the **context and implementation plan for Python scripts**, this document focuses on the Python-side workflow and how the scripts should be organized. Even if OpenSearch-managed pipelines are later adopted, Python will still be useful for setup, data preparation, ingestion, and testing.

---

## Project Objective

Build a minimal but extensible semantic search workflow with Python that can:

1. accept manually defined text chunks,
2. prepare them for indexing,
3. generate or attach embeddings (using local Python model),
4. index them into OpenSearch,
5. run semantic search queries,
6. inspect and validate results.

This prototype should be easy to run locally and easy to evolve later into a larger ingestion pipeline.

---

## Recommended Scope for the First Version

The first version should stay small and predictable.

### In scope
- A local OpenSearch index for chunk documents
- A few manually authored sample chunks
- Python scripts for:
  - index creation
  - ingestion
  - semantic search
  - optional inspection / cleanup
- Environment-variable-based configuration

### Out of scope for now
- Full website scraping
- Automatic chunking from raw HTML
- Re-ranking
- Hybrid search tuning
- UI / frontend integration
- Production deployment concerns

---

## Proposed Python Script Structure

A clean first version can be implemented with the following files:

```text
project/
├── .env
├── requirements.txt
├── chunks.json
├── create_index.py
├── ingest_chunks.py
├── search_chunks.py
├── verify_index.py
└── README.md
```

### 1. `.env`
Stores local configuration values.

Example:
```env
OPENSEARCH_URL=https://localhost:9200
OPENSEARCH_USER=admin
OPENSEARCH_INITIAL_ADMIN_PASSWORD=yourpassword
OPENSEARCH_INDEX=mfguide_chunks
```

### 2. `requirements.txt`
Lists Python dependencies.

```txt
opensearch-py
python-dotenv
sentence-transformers
```

### 3. `chunks.json`
Contains manually defined chunks for the prototype.

Example:
```json
[
  {
    "chunk_id": "c1",
    "title": "Eligibility Requirements",
    "text": "The borrower must satisfy the eligibility requirements for multifamily financing."
  },
  {
    "chunk_id": "c2",
    "title": "Property Standards",
    "text": "Eligible properties must meet occupancy, condition, and compliance requirements."
  }
]
```

---

## Implementation Plan

## Phase 1 — Configuration and Connectivity

### Goal
Confirm that Python can connect successfully to local OpenSearch.

### Script
`verify_index.py` or a simple connectivity script

### Responsibilities
- Load environment variables
- Create an OpenSearch client
- Verify the cluster is reachable
- Print cluster info or health status

### Why this comes first
This isolates connection issues before index creation or ingestion logic is added.

---

## Phase 2 — Create the Vector Index

### Goal
Create the OpenSearch index with the correct schema for semantic search.

### Script
`create_index.py`

### Responsibilities
- Read settings from `.env`
- Define index settings and mappings
- Create fields such as:
  - `chunk_id`
  - `title`
  - `text`
  - `embedding` (vector field, dimension 384)
- Set `index.knn` if required by the chosen OpenSearch vector configuration
- Recreate or skip the index depending on development needs

### Important design decision
The `embedding` field dimension must match the embedding model output exactly (384 for all-MiniLM-L6-v2).

### Suggested behavior
- Optionally delete and recreate the index during development
- Print the resulting mapping after creation

---

## Phase 3 — Ingest Manual Chunks

### Goal
Read chunk documents from JSON and prepare them for indexing.

### Script
`ingest_chunks.py`

### Responsibilities
- Load `chunks.json`
- Validate required fields
- Normalize whitespace if needed
- Generate embeddings using the local Python model
- Send documents into the configured index

### Ingestion mode
- Python reads chunk text, generates embedding with `sentence-transformers`, and writes `{chunk_id, title, text, embedding}` into OpenSearch.

---

## Phase 4 — Run Semantic Search

### Goal
Search stored chunks using a natural-language query.

### Script
`search_chunks.py`

### Responsibilities
- Accept a search query from the command line or hardcoded input
- Generate a query embedding using the same local Python model
- Run vector similarity search
- Return the top matching chunks
- Print:
  - score
  - chunk ID
  - title
  - text snippet

---

## Phase 5 — Validation and Debugging

### Goal
Make it easy to confirm that data was indexed correctly.

### Script
`verify_index.py`

### Responsibilities
- Count indexed documents
- Fetch a small sample of documents
- Confirm the embedding field exists
- Optionally print vector length for one sample document
- Help diagnose dimension mismatches or missing fields

---

## Suggested Script Responsibilities in More Detail

## `create_index.py`
This script should:
- connect to OpenSearch,
- build the index mapping,
- create the index if it does not exist,
- optionally replace the index in development.

Suggested function breakdown:
- `load_config()`
- `build_client()`
- `build_index_mapping()`
- `create_index()`

---

## `ingest_chunks.py`
This script should:
- load chunk data from JSON,
- validate records,
- prepare documents,
- index them one by one or in bulk.

Suggested function breakdown:
- `load_chunks(path)`
- `validate_chunk(chunk)`
- `get_embedding(text)` or `prepare_for_pipeline(chunk)`
- `index_document(doc)`
- `bulk_index(docs)`

Possible later improvement:
- use OpenSearch bulk indexing for better performance

---

## `search_chunks.py`
This script should:
- read a query,
- transform it into the same embedding space as indexed documents,
- submit a k-NN or vector query,
- format readable results.

Suggested function breakdown:
- `get_query_embedding(query)`
- `build_search_body(vector, k)`
- `semantic_search(query, k=5)`
- `print_results(hits)`

---

## Configuration Strategy

The scripts should not hardcode local values. Use environment variables instead.

Recommended config items:
- OpenSearch URL
- username
- password
- index name
- embedding model name (for future flexibility)

---

## Error Handling Expectations

Each script should fail clearly and early.

### Examples
- missing `.env` variables,
- OpenSearch connection failures,
- invalid chunk JSON,
- embedding API failures,
- index mapping dimension mismatches,
- empty search results.

### Recommended behavior
- print helpful error messages,
- return non-zero exit codes when appropriate,
- avoid silent failures.

---

## Logging and Output

For the prototype, simple console logging is enough.

Recommended output style:
- clearly label each step,
- show which index is being used,
- show how many chunks were loaded,
- confirm successful indexing per chunk or in batches,
- print top search matches with scores.

This will make local debugging much easier.

---

## Development Order

The scripts should be implemented in this order:

1. `verify_index.py`
   - prove OpenSearch connectivity

2. `create_index.py`
   - create the vector index and mapping

3. `ingest_chunks.py`
   - read manual chunks and index them

4. `search_chunks.py`
   - test end-to-end semantic retrieval

5. improve with:
   - bulk indexing
   - pipeline support
   - search tuning
   - better result formatting

This order reduces confusion and makes debugging incremental.

---

## Future Evolution

Once the prototype works, the Python scripts can expand to support:

- HTML extraction from the target webpage
- automatic chunk generation
- metadata fields such as section headers or source URL
- hybrid search with BM25 + vector search
- re-ranking of top results
- a small API or CLI interface
- migration from manual chunks to automatically processed source content

---

## Recommended First Deliverable

The first deliverable should be a working local demo with:

- 3 to 10 manually created chunks,
- one vector index in OpenSearch,
- one ingestion script,
- one search script,
- repeatable configuration using `.env`.

That is enough to validate the semantic search architecture before investing in scraping, chunking strategy, or ranking improvements.

---

## Summary

This project is a local semantic search prototype built around manually created chunks from the Fannie Mae Multifamily Guide content and stored in OpenSearch for vector-based retrieval.

The Python implementation should be organized into small, clear scripts that handle:

- configuration,
- OpenSearch connectivity,
- index creation,
- chunk ingestion,
- semantic search,
- result verification.

The best first milestone is a minimal end-to-end workflow that proves the core loop:

```text
manual chunk -> embedding -> OpenSearch vector index -> semantic query -> ranked results
```

Even if OpenSearch-managed model pipelines are later adopted, the Python scripts remain valuable as the orchestration layer for setup, ingestion, testing, and debugging.
