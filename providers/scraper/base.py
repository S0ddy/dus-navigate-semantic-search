class BaseScraper:
    def fetch(self, url: str) -> str:
        """Fetch a URL and return raw HTML."""
        raise NotImplementedError
