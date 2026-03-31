class BaseStore:
    def create_index(self):
        """Create (or recreate) the vector index."""
        raise NotImplementedError

    def index_documents(self, docs: list):
        """Index a list of documents (each must include an 'embedding' field)."""
        raise NotImplementedError

    def search(self, vector: list, k: int = 5) -> list:
        """Run a kNN search and return the top-k hits."""
        raise NotImplementedError
