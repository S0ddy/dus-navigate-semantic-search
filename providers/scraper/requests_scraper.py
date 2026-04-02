import time
import requests
from pathlib import Path

from .base import BaseScraper

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class RequestsScraper(BaseScraper):
    def __init__(self, delay: float = 0.5, cache_dir: str = ".page_cache"):
        self.delay = delay
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update(HEADERS)

    def fetch(self, url: str, force_refresh: bool = False) -> str:
        node_id = url.rsplit("/", 1)[-1]
        cache_file = self.cache_dir / f"{node_id}.html"

        if cache_file.exists() and not force_refresh:
            return cache_file.read_text(encoding="utf-8")

        resp = self._session.get(url, timeout=15)
        resp.raise_for_status()
        html = resp.text
        cache_file.write_text(html, encoding="utf-8")
        time.sleep(self.delay)
        return html
