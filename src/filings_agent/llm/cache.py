"""Disk cache for LLM step outputs.

Keyed by everything that can change the answer: model id, prompt version, input, and the
repetition index (so n=3 repeats are 3 real calls). A hit returns the ORIGINAL call's token usage
and latency, so cached re-runs report true cost/speed instead of $0 / 0 ms.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from filings_agent.fsutil import atomic_write_text


class StepCache:
    def __init__(self, root: Path) -> None:
        self._root = root

    @staticmethod
    def key(**parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self._root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return cast(dict[str, Any], json.loads(path.read_text()))

    def put(self, key: str, value: dict[str, Any]) -> None:
        atomic_write_text(self._path(key), json.dumps(value, default=str))
