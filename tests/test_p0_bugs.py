"""
TDD tests for P0 bugs.

P0-1  update_page silently re-indexes stale cached content because
      RequestsScraper.fetch() always returns the cache file when it exists.

P0-2  delete_page_chunks() starts a delete_by_query without waiting for it to
      finish, so index_documents() can run while old chunks are still present,
      producing duplicates.

Run:
    pytest tests/test_p0_bugs.py -v
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Make project root importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── P0-1: Cache bypass ──────────────────────────────────────────────────────────

class TestFetchCacheBypass:
    """
    Bug: RequestsScraper.fetch() returns the cached file unconditionally.
    Fix: introduce a force_refresh parameter that skips the cache.
    """

    def _make_scraper(self, tmp_path):
        from providers.scraper.requests_scraper import RequestsScraper
        return RequestsScraper(cache_dir=str(tmp_path))

    def _stale_cache(self, tmp_path):
        f = tmp_path / "1234.html"
        f.write_text("<html>stale</html>", encoding="utf-8")
        return f

    def _fresh_response(self, text="<html>fresh</html>"):
        mock_resp = MagicMock()
        mock_resp.text = text
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    # ── baseline (must always pass) ────────────────────────────────────────────

    def test_fetch_uses_cache_when_no_force_refresh(self, tmp_path):
        """Normal fetch() must still serve from cache — we must not break this."""
        scraper = self._make_scraper(tmp_path)
        self._stale_cache(tmp_path)

        with patch.object(scraper._session, "get") as mock_get:
            result = scraper.fetch("https://mfguide.fanniemae.com/node/1234")

        mock_get.assert_not_called()
        assert result == "<html>stale</html>"

    # ── failing tests (red before fix) ────────────────────────────────────────

    def test_fetch_bypasses_cache_when_force_refresh_true(self, tmp_path):
        """
        FAILS before fix.
        fetch(url, force_refresh=True) must make an HTTP request even if a
        cache file already exists, and must return the live response.
        """
        scraper = self._make_scraper(tmp_path)
        self._stale_cache(tmp_path)

        with patch.object(scraper._session, "get", return_value=self._fresh_response()):
            result = scraper.fetch(
                "https://mfguide.fanniemae.com/node/1234",
                force_refresh=True,
            )

        assert result == "<html>fresh</html>", (
            "force_refresh=True must return live content, not the cached file"
        )

    def test_force_refresh_overwrites_cache_file(self, tmp_path):
        """
        FAILS before fix.
        After force_refresh=True the cache file must contain the fresh content
        so the next normal fetch() also gets the updated version.
        """
        scraper = self._make_scraper(tmp_path)
        cache_file = self._stale_cache(tmp_path)

        with patch.object(scraper._session, "get", return_value=self._fresh_response()):
            scraper.fetch(
                "https://mfguide.fanniemae.com/node/1234",
                force_refresh=True,
            )

        assert cache_file.read_text(encoding="utf-8") == "<html>fresh</html>", (
            "Cache file must be overwritten so subsequent normal fetches are fresh"
        )

    def test_update_page_calls_fetch_with_force_refresh(self):
        """
        FAILS before fix.
        pipeline.update_page() must pass force_refresh=True to the scraper so
        it always ingests the current live page, not a stale cache entry.

        Uses a mock config injected via sys.modules so we never touch real
        OpenSearch or load a SentenceTransformer model.
        """
        mock_config = MagicMock()
        mock_config.SCRAPER.fetch.return_value = (
            "<html><head><title>T</title></head>"
            "<body><main><p>content</p></main></body></html>"
        )
        mock_config.EMBEDDER.embed.return_value = [[0.0] * 384]

        # Remove cached pipeline so it re-imports with mock_config
        sys.modules.pop("pipeline", None)
        with patch.dict("sys.modules", {"config": mock_config}):
            import pipeline
            pipeline.update_page("https://mfguide.fanniemae.com/node/1234")
        sys.modules.pop("pipeline", None)  # restore for other tests

        mock_config.SCRAPER.fetch.assert_called_once_with(
            "https://mfguide.fanniemae.com/node/1234",
            force_refresh=True,
        )


# ── P0-2: delete_by_query race condition ───────────────────────────────────────

class TestDeleteWaitsForCompletion:
    """
    Bug: delete_page_chunks() fires delete_by_query without wait_for_completion,
         so the delete may still be running when index_documents() starts,
         leaving duplicate chunks in the index.
    Fix: pass params={"wait_for_completion": "true"} (or the kwarg equivalent)
         so the call blocks until OpenSearch confirms the deletion.
    """

    def _make_store(self):
        from providers.store.opensearch_store import OpenSearchStore
        store = OpenSearchStore.__new__(OpenSearchStore)
        store.index = "test_index"
        store.client = MagicMock()
        return store

    # ── baseline ───────────────────────────────────────────────────────────────

    def test_delete_page_chunks_calls_delete_by_query(self):
        """Baseline: delete_by_query is called with the right document_id filter."""
        store = self._make_store()
        store.delete_page_chunks("doc-abc")

        store.client.delete_by_query.assert_called_once()
        body = store.client.delete_by_query.call_args.kwargs["body"]
        assert body["query"]["term"]["document_id"] == "doc-abc"

    # ── failing test (red before fix) ─────────────────────────────────────────

    def test_delete_page_chunks_passes_wait_for_completion(self):
        """
        FAILS before fix.
        delete_by_query must be called with wait_for_completion so the method
        blocks until the deletion is done before control returns to update_page().

        Accepted forms (opensearch-py accepts both):
          params={"wait_for_completion": "true"}   ← REST query param style
          wait_for_completion=True                 ← direct kwarg style
        """
        store = self._make_store()
        store.delete_page_chunks("doc-abc")

        call_kwargs = store.client.delete_by_query.call_args.kwargs
        params      = call_kwargs.get("params", {})

        wait_passed = (
            params.get("wait_for_completion") in ("true", True)
            or call_kwargs.get("wait_for_completion") in ("true", True)
        )

        assert wait_passed, (
            "delete_by_query must include wait_for_completion=True.\n"
            f"Actual call kwargs: {call_kwargs}"
        )
