from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.model = SentenceTransformer(model_name)
        self.dim = dim

    def embed(self, texts: list) -> list:
        return self.model.encode(texts).tolist()
