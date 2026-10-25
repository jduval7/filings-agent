"""SEC HTTP client and filing source, against a mock transport (no network)."""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from filings_agent.sec import filings as filings_mod
from filings_agent.sec.filings import SecSource
from filings_agent.sec.http import SecHttp

TICKERS = {"0": {"ticker": "WMT", "cik_str": 104169}, "1": {"ticker": "GE", "cik_str": 40545}}


def _recent(rows: list[tuple[str, str, str, str]]) -> dict[str, list[str]]:
    return {
        "accessionNumber": [r[0] for r in rows],
        "form": [r[1] for r in rows],
        "filingDate": [r[2] for r in rows],
        "reportDate": [r[3] for r in rows],
    }


class FakeSec:
    """Routes data.sec.gov / www.sec.gov URLs to canned JSON and records every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: dict[str, Any] = {
            "/files/company_tickers.json": TICKERS,
            "/submissions/CIK0000104169.json": {
                "filings": {
                    "recent": _recent(
                        [
                            ("A-26", "10-K", "2026-03-13", "2026-01-31"),
                            ("Q-25", "10-Q", "2025-12-01", "2025-10-31"),
                            ("A-25", "10-K", "2025-03-14", "2025-01-31"),
                        ]
                    ),
                    "files": [
                        {"name": "CIK0000104169-submissions-001.json", "filingTo": "2024-12-31"},
                        {"name": "CIK0000104169-submissions-002.json", "filingTo": "2019-12-31"},
                    ],
                }
            },
            "/submissions/CIK0000104169-submissions-001.json": _recent(
                [
                    ("A-24", "10-K", "2024-03-15", "2024-01-31"),
                    ("A-23", "10-K", "2023-03-17", "2023-01-31"),
                    ("A-22", "10-K", "2022-03-18", "2022-01-31"),  # before filed_since
                    ("AMD-23", "10-K/A", "2023-05-01", "2023-01-31"),  # amendment: other form
                ]
            ),
            "/api/xbrl/companyfacts/CIK0000104169.json": {"facts": {"us-gaap": {}}},
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = self.routes.get(request.url.path)
        if body is None:
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json=body)


@pytest.fixture
def sec(tmp_path: Path) -> tuple[SecHttp, FakeSec]:
    fake = FakeSec()
    http = SecHttp("Test Agent test@example.com", tmp_path / "cache", max_rps=1000,
                   transport=httpx.MockTransport(fake.handler))  # fmt: skip
    return http, fake


def test_sends_user_agent_and_caches_by_url(sec: tuple[SecHttp, FakeSec]) -> None:
    http, fake = sec
    url = "https://www.sec.gov/files/company_tickers.json"
    first = http.get_json(url)
    second = http.get_json(url)
    assert first == second == TICKERS
    assert len(fake.requests) == 1  # second read came from disk
    assert fake.requests[0].headers["User-Agent"] == "Test Agent test@example.com"


def test_cache_entries_are_complete_json_files(
    sec: tuple[SecHttp, FakeSec], tmp_path: Path
) -> None:
    http, _ = sec
    http.get_json("https://www.sec.gov/files/company_tickers.json")
    files = list((tmp_path / "cache").glob("*"))
    assert len(files) == 1 and files[0].suffix == ".json"  # no .tmp left behind
    assert json.loads(files[0].read_text()) == TICKERS


def test_http_errors_raise_and_are_not_cached(sec: tuple[SecHttp, FakeSec], tmp_path: Path) -> None:
    http, fake = sec
    url = "https://data.sec.gov/missing.json"
    for _ in range(2):
        with pytest.raises(httpx.HTTPStatusError):
            http.get_json(url)
    assert len(fake.requests) == 2
    assert list((tmp_path / "cache").glob("*")) == []


def test_throttle_spaces_requests(tmp_path: Path) -> None:
    fake = FakeSec()
    http = SecHttp("UA x@y.z", tmp_path, max_rps=20, transport=httpx.MockTransport(fake.handler))
    t0 = time.monotonic()
    http.get_json("https://www.sec.gov/files/company_tickers.json")
    http.get_json("https://data.sec.gov/api/xbrl/companyfacts/CIK0000104169.json")
    http.get_json("https://data.sec.gov/submissions/CIK0000104169.json")
    assert time.monotonic() - t0 >= 2 / 20 * 0.9  # 3 requests -> at least 2 gaps of 50ms


def test_cik_lookup(sec: tuple[SecHttp, FakeSec], tmp_path: Path) -> None:
    http, _ = sec
    src = SecSource(http, tmp_path / "filings", "UA")
    assert src.cik("WMT") == 104169 and src.cik("GE") == 40545
    with pytest.raises(KeyError):
        src.cik("NOPE")


def test_list_filings_follows_older_pages_and_filters(
    sec: tuple[SecHttp, FakeSec], tmp_path: Path
) -> None:
    http, fake = sec
    src = SecSource(http, tmp_path / "filings", "UA")
    refs = src.list_filings("WMT", "10-K", date(2023, 1, 1))
    assert [r.accession_no for r in refs] == ["A-23", "A-24", "A-25", "A-26"]  # oldest first
    assert all(r.form == "10-K" and r.cik == 104169 for r in refs)
    assert refs[0].period_of_report == date(2023, 1, 31)
    paths = [r.url.path for r in fake.requests]
    assert "/submissions/CIK0000104169-submissions-001.json" in paths
    # the page that ends before filed_since is never fetched
    assert "/submissions/CIK0000104169-submissions-002.json" not in paths


def test_company_facts(sec: tuple[SecHttp, FakeSec], tmp_path: Path) -> None:
    http, _ = sec
    assert SecSource(http, tmp_path, "UA").company_facts("WMT") == {"facts": {"us-gaap": {}}}


def test_markdown_is_fetched_once_and_cached(
    sec: tuple[SecHttp, FakeSec], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    http, _ = sec
    calls: list[str] = []

    def fake_markdown(accession_no: str, user_agent: str) -> str:
        calls.append(accession_no)
        return "# 10-K text"

    monkeypatch.setattr(filings_mod, "_edgar_markdown", fake_markdown)
    src = SecSource(http, tmp_path / "filings", "UA")
    (ref,) = [r for r in src.list_filings("WMT", "10-K", date(2026, 1, 1))]
    assert src.markdown(ref) == "# 10-K text"
    assert src.markdown(ref) == "# 10-K text"
    assert calls == ["A-26"]
    assert (tmp_path / "filings" / "A-26.md").read_text() == "# 10-K text"
