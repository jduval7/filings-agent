"""Live sources behind an allowlist: SEC contact header, redirects, EDGAR search, passage
ranking, inline-XBRL stripping."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from filings_agent.research.sources import (
    Hit,
    LiveFetcher,
    NotAllowed,
    allowed,
    passages,
)
from tests.support.research_doubles import SEC_DOC

VIEWPOINT = "https://viewpoint.pwc.com/dt/us/en/pwc/accounting_guides/x.html"


def test_allowlist() -> None:
    assert allowed(SEC_DOC) and allowed(VIEWPOINT)
    assert allowed("https://efts.sec.gov/LATEST/search-index?q=x")
    for bad in (
        "https://www.fasb.org/page",  # refuses automated fetching anyway
        "http://www.sec.gov/insecure",
        "https://sec.gov.evil.example/phish",
        "https://example.com/?u=https://www.sec.gov/",
        "https://user:pw@www.sec.gov/x",  # credentials
        "https://www.sec.gov@evil.example/x",  # the host is evil.example
        "https://www.sec.gov:8443/x",  # another port
        "https://www.sec.gov:99999/x",  # malformed port: refused, not raised
        "https://[::1/x",  # malformed host
    ):
        assert not allowed(bad), bad
    assert allowed("https://www.sec.gov:443/x")


def test_requests_to_one_host_are_spaced_but_not_serialized(tmp_path: Path) -> None:
    """Review finding: one lock held across the whole request made five parallel investigations
    read one page at a time. The lock now only reserves a time slot."""
    import threading
    import time

    inside = 0
    peak = 0
    guard = threading.Lock()

    def slow(request: httpx.Request) -> httpx.Response:
        nonlocal inside, peak
        with guard:
            inside += 1
            peak = max(peak, inside)
        time.sleep(0.2)
        with guard:
            inside -= 1
        return httpx.Response(200, text="<html><title>t</title>ok</html>")

    f = LiveFetcher("Jane Doe jane@example.com", tmp_path, 1000.0, httpx.MockTransport(slow))
    urls = [f"https://viewpoint.pwc.com/p{i}.html" for i in range(4)]
    threads = [threading.Thread(target=f.document, args=(u,)) for u in urls]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak > 1


class Recorder:
    def __init__(self, routes: dict[str, httpx.Response]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.routes[str(request.url)]


def fetcher(tmp: Path, routes: dict[str, httpx.Response]) -> tuple[LiveFetcher, Recorder]:
    rec = Recorder(routes)
    return LiveFetcher("Jane Doe jane@example.com", tmp, 1000.0, httpx.MockTransport(rec)), rec


def test_sec_contact_goes_to_sec_only_and_pages_are_cached(tmp_path: Path) -> None:
    html = "<html><title>T</title><body><p>We adopted ASU 2018-12.</p>"
    html += f'<a href="{VIEWPOINT}">guide</a><a href="https://evil.example/">x</a></body></html>'
    f, rec = fetcher(
        tmp_path,
        {
            SEC_DOC: httpx.Response(200, text=html),
            VIEWPOINT: httpx.Response(200, text="<html><body>Guide text</body></html>"),
        },
    )
    doc = f.document(SEC_DOC)
    assert "adopted ASU 2018-12" in doc.text and doc.links == (VIEWPOINT,)  # evil link dropped
    f.document(VIEWPOINT)
    f.document(SEC_DOC)  # cached: no second request
    agents = [(str(r.url), r.headers["user-agent"]) for r in rec.requests]
    assert agents == [
        (SEC_DOC, "Jane Doe jane@example.com"),
        (VIEWPOINT, "filings-agent research"),
    ]


def test_refuses_disallowed_urls_and_redirects_off_the_allowlist(tmp_path: Path) -> None:
    f, rec = fetcher(
        tmp_path,
        {SEC_DOC: httpx.Response(302, headers={"location": "https://evil.example/x"})},
    )
    with pytest.raises(NotAllowed):
        f.document("https://evil.example/x")
    assert rec.requests == []  # refused before any request
    with pytest.raises(NotAllowed):
        f.document(SEC_DOC)


def test_edgar_search_builds_document_urls(tmp_path: Path) -> None:
    body = {
        "hits": {
            "hits": [
                {
                    "_id": "0000040545-24-000027:ge-20231231.htm",
                    "_source": {
                        "form": "10-K",
                        "file_date": "2024-02-02",
                        "ciks": ["0000040545"],
                        "display_names": ["GE  (CIK 0000040545)"],
                    },
                }
            ]
        }
    }
    url = "https://efts.sec.gov/LATEST/search-index?q=%22Long%20Duration%22&ciks=0000040545&forms=10-K"
    f, _ = fetcher(tmp_path, {url: httpx.Response(200, text=json.dumps(body))})
    (hit,) = f.edgar_search("Long Duration", 40545, ("10-K",))
    assert hit == Hit("10-K", "2024-02-02", "GE  (CIK 0000040545)", SEC_DOC)


def test_passages_rank_by_query_words() -> None:
    text = "Unrelated text. " * 200 + "We adopted ASU 2018-12 retrospectively. " + "More. " * 50
    (_, chunk), *_ = passages(text, "adopted ASU retrospectively")
    assert "adopted ASU 2018-12" in chunk
    assert passages(text, "zebra") == []


def test_inline_xbrl_hidden_facts_are_not_read_as_text() -> None:
    from filings_agent.research.sources import html_to_text

    html = (
        "<html><body><ix:header><ix:hidden>ge:GEHealthcareMember</ix:hidden></ix:header>"
        '<div style="display:none">us-gaap:SegmentDiscontinuedOperationsMember</div>'
        "<p>On January 1, 2023, we adopted ASU No. 2018-12.</p></body></html>"
    )
    _, text, _ = html_to_text(html, SEC_DOC)
    assert text == "On January 1, 2023, we adopted ASU No. 2018-12."


def test_sec_contact_is_read_only_when_sec_is_contacted(tmp_path: Path) -> None:
    """Building the app mustn't need SEC_USER_AGENT; research reaching sec.gov does."""
    asked: list[str] = []

    def contact() -> str:
        asked.append("x")
        return "Jane Doe jane@example.com"

    rec = Recorder(
        {
            VIEWPOINT: httpx.Response(200, text="<p>guide</p>"),
            SEC_DOC: httpx.Response(200, text="<p>10-K</p>"),
        }
    )
    f = LiveFetcher(contact, tmp_path, 1000.0, httpx.MockTransport(rec))
    f.document(VIEWPOINT)
    assert asked == []  # not needed for other hosts
    f.document(SEC_DOC)
    f.document(SEC_DOC)  # cached; and the contact is read once
    assert asked == ["x"] and rec.requests[-1].headers["user-agent"] == "Jane Doe jane@example.com"
