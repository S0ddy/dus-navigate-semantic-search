# Fannie Mae MF Guide — Semantic Search

Local semantic search over the [Fannie Mae Multifamily Guide](https://mfguide.fanniemae.com/node/10711) using OpenSearch and sentence-transformers.

## Prerequisites

- Python 3.9+
- Docker (for running OpenSearch locally)

## Setup

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```

**2. Configure environment:**

```bash
cp .env.example .env
```

Edit `.env` and set your OpenSearch password:

```env
OPENSEARCH_URL=https://localhost:9200
OPENSEARCH_USER=admin
OPENSEARCH_INITIAL_ADMIN_PASSWORD=your_password_here
OPENSEARCH_INDEX=mfguide_chunks
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**3. Start OpenSearch:**

```bash
docker compose -f docker/docker-compose.yml up -d
```

> OpenSearch takes ~30 seconds to be ready on first start.

## Run

**Option A — Streamlit UI:**

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). Use the Admin panel to run the ingestion pipeline, then use the search field.

**Option B — CLI:**

```bash
# Ingest all sections
python pipeline.py

# Search
python search.py "delegating underwriting authority"
```

## Project Structure

```
app.py             # Streamlit UI
config.py          # settings and provider wiring (swap providers here)
pipeline.py        # scrape → parse → embed → index
search.py          # query → embed → search → print results
parse_chunks.py    # HTML → token-aware chunks with metadata

providers/
  scraper/         # how pages are fetched   (requests, playwright, ...)
  embedder/        # how text is embedded    (sentence-transformers, openai, ...)
  store/           # where vectors are stored (opensearch, pinecone, ...)

docker/            # docker-compose for local OpenSearch
docs/              # planning and architecture notes
```

## Swapping a Provider

Change two lines in `config.py` — nothing else needs to change:

```python
# Example: switch to OpenAI embeddings
from providers.embedder.openai_embedder import OpenAIEmbedder
EMBEDDER = OpenAIEmbedder()
```
