"""Atomic file writes."""

from __future__ import annotations

from pathlib import Path


def test_atomic_write_survives_concurrent_writers_of_the_same_path(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from filings_agent.fsutil import atomic_write_text

    target = tmp_path / "same.json"

    def write(i: int) -> None:
        atomic_write_text(target, f'{{"v": {i}}}')

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(write, range(200)))
    assert target.read_text().startswith('{"v": ')
    assert list(tmp_path.glob("*.tmp")) == []  # no temp files left behind
