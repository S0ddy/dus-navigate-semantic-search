import sys
from opensearchpy import OpenSearch

from .base import BaseStore


class OpenSearchStore(BaseStore):
    def __init__(self, url: str, user: str, password: str, index: str, dim: int = 384):
        self.index = index
        self.dim = dim
        self.client = OpenSearch(
            hosts=[url],
            http_auth=(user, password),
            use_ssl=url.startswith("https"),
            verify_certs=False,
        )

    def create_index(self):
        if self.client.indices.exists(index=self.index):
            print(f"Deleting existing index '{self.index}'...")
            self.client.indices.delete(index=self.index)
        self.client.indices.create(index=self.index, body=self._mapping())
        print(f"Index '{self.index}' created.")

    def index_documents(self, docs: list):
        for doc in docs:
            try:
                self.client.index(index=self.index, body=doc)
            except Exception as e:
                print(f"  [ERROR] Failed to index {doc.get('chunk_id')}: {e}", file=sys.stderr)

    def search(self, vector: list, k: int = 5) -> list:
        body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {"vector": vector, "k": k}
                }
            },
        }
        res = self.client.search(index=self.index, body=body)
        return res["hits"]["hits"]

    def _mapping(self) -> dict:
        return {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "text":            {"type": "text"},
                    "embedding":       {"type": "knn_vector", "dimension": self.dim},
                    "document_id":     {"type": "keyword"},
                    "chunk_id":        {"type": "keyword"},
                    "source_url":      {"type": "keyword"},
                    "page_title":      {"type": "text"},
                    "section_heading": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "chunk_index":     {"type": "integer"},
                    "total_chunks":    {"type": "integer"},
                    "language":        {"type": "keyword"},
                    "created_at":      {"type": "date"},
                }
            },
        }
