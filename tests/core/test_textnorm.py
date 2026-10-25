"""Text normalization for verbatim checks."""

from __future__ import annotations

from filings_agent.textnorm import normalize


def test_normalize_is_strict_about_words() -> None:
    assert normalize("Inc\\.  “Quoted”  ten‑for‑one") == normalize('inc. "quoted" ten-for-one')
    assert normalize("ten-for-one") != normalize("five-for-one")
