"""Filesystem helpers."""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a uniquely named temp file + rename: readers never see a half-written file, and
    concurrent writers of the same path (same cache key from two threads) can't collide on the
    temp file. Last writer wins, which is fine for content-addressed caches."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
