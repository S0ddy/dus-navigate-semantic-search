class BaseEmbedder:
    def embed(self, texts: list) -> list:
        """Embed a list of strings and return a list of float vectors."""
        raise NotImplementedError
