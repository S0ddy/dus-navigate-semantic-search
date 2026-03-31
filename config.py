import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ── Settings ──────────────────────────────────────────────────────────────────

OPENSEARCH_URL      = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
OPENSEARCH_USER     = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_INITIAL_ADMIN_PASSWORD")
OPENSEARCH_INDEX    = os.getenv("OPENSEARCH_INDEX", "mfguide_chunks")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM   = 384

if not OPENSEARCH_PASSWORD:
    print("Missing OPENSEARCH_INITIAL_ADMIN_PASSWORD in .env", file=sys.stderr)
    sys.exit(1)

# ── Provider wiring — swap these lines to change implementations ──────────────

from providers.embedder.sentence_transformer import SentenceTransformerEmbedder
from providers.store.opensearch_store import OpenSearchStore
from providers.scraper.requests_scraper import RequestsScraper

EMBEDDER = SentenceTransformerEmbedder(EMBEDDING_MODEL, EMBEDDING_DIM)
STORE    = OpenSearchStore(OPENSEARCH_URL, OPENSEARCH_USER, OPENSEARCH_PASSWORD, OPENSEARCH_INDEX, EMBEDDING_DIM)
SCRAPER  = RequestsScraper()
