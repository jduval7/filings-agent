"""Throttled, disk-cached JSON client for data.sec.gov.

Every response is cached by URL, so re-running the pipeline never re-downloads.
That's the first layer of idempotency: fetching is a pure function of the URL.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, cast

import httpx

from filings_agent.fsutil import atomic_write_text


class SecHttp:
    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        max_rps: float = 5.0,
        transport: httpx.BaseTransport | None = None,  # tests inject a mock transport
    ) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
        )
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._min_interval = 1.0 / max_rps
        self._last = 0.0
        self._lock = threading.Lock()

    def _cache_path(self, url: str) -> Path:
        return self._cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.json"

    def get_json(self, url: str) -> dict[str, Any]:
        path = self._cache_path(url)
        if path.exists():
            return cast(dict[str, Any], json.loads(path.read_text()))
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            resp = self._client.get(url)
            self._last = time.monotonic()
        resp.raise_for_status()
        data = cast(dict[str, Any], resp.json())
        atomic_write_text(path, json.dumps(data))  # a crash never leaves a half-written entry
        return data
