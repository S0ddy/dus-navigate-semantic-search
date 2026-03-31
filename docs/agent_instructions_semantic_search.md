# Semantic Search Agent Instructions (Free + Simple Setup)

## Objective

Build a **fully local, free, and simple semantic search pipeline** using:

- Python
- sentence-transformers (local embeddings)
- OpenSearch (vector storage + search)

No external APIs. No paid services. No OpenSearch ML pipelines.

---

## Architecture Overview

```text
chunks.json
   ↓
Python script
   ↓
Generate embeddings (sentence-transformers)
   ↓
Store in OpenSearch (knn_vector)
   ↓
Semantic search using vector similarity
```

---

## Dependencies

Install required packages:

```bash
pip install sentence-transformers opensearch-py python-dotenv
```

---

## Embedding Model

Use:

```python
all-MiniLM-L6-v2
```

Key details:
- Dimension: **384**
- Runs locally
- No API key required

---

## OpenSearch Index Setup

### Requirements

- `knn` must be enabled
- vector field must match dimension = 384

### Example Mapping

```json
PUT mfguide_chunks
{
  "settings": {
    "index": {
      "knn": true
    }
  },
  "mappings": {
    "properties": {
      "chunk_id": { "type": "keyword" },
      "title": { "type": "text" },
      "text": { "type": "text" },
      "embedding": {
        "type": "knn_vector",
        "dimension": 384
      }
    }
  }
}
```

---

## Data Format

Create a `chunks.json` file:

```json
[
  {
    "chunk_id": "c1",
    "title": "Eligibility",
    "text": "Borrowers must meet eligibility requirements."
  }
]
```

---

## Embedding Logic

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return model.encode(text).tolist()
```

---

## Ingestion Logic

For each chunk:

1. Generate embedding
2. Store document in OpenSearch

Document structure:

```json
{
  "chunk_id": "...",
  "title": "...",
  "text": "...",
  "embedding": [vector]
}
```

---

## Search Logic

1. Convert query → embedding
2. Run k-NN search

### Example Query

```json
GET mfguide_chunks/_search
{
  "size": 3,
  "query": {
    "knn": {
      "embedding": {
        "vector": [query_vector],
        "k": 3
      }
    }
  }
}
```

---

## Critical Rules

- MUST use the same model for:
  - indexing
  - querying

- Embedding dimension MUST match index (384)

- Always convert embeddings:
```python
.tolist()
```

---

## Common Mistakes

- Wrong vector dimension
- Mixing embedding models
- Forgetting knn=true
- Sending numpy arrays instead of lists

---

## Minimal Workflow

```text
1. Create index
2. Load chunks.json
3. Generate embeddings locally
4. Index documents into OpenSearch
5. Convert query to embedding
6. Run vector search
7. Return top results
```

---

## Goal for Agent

Implement scripts that:

- Create index
- Ingest chunks
- Perform semantic search

Keep everything:
- local
- free
- simple
