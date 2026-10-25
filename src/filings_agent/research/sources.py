"""Live, authoritative sources for the research agent, behind an allowlist.

Only two kinds of source, both fetched live:
- **SEC (primary source):** EDGAR full-text search over every filing since 2001 (the company's
  8-Ks, amended 10-Ks, 10-Qs, and other companies' disclosures of the same standard), and the
  filing documents themselves.
- **PwC Viewpoint (interpretive guidance):** accounting guides that reproduce and interpret the
  FASB codification. Checked 2026-10-21: fasb.org refuses automated fetching (403) and Deloitte
  DART renders its text with JavaScript (an empty page without a browser), so neither is usable.

Anything else is refused before a request is made. Fetched text is untrusted DATA: the agent's
prompt says so, and every quote it reports is verified against what was fetched.

The SEC requires a contact in the User-Agent, so the configured SEC contact goes to sec.gov
only; other sites get a generic agent string.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable
from urllib.parse import quote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from filings_agent.fsutil import atomic_write_text

SEC_HOSTS = frozenset({"www.sec.gov", "sec.gov", "efts.sec.gov"})
GUIDANCE_HOSTS = frozenset({"viewpoint.pwc.com"})
ALLOWED_HOSTS = SEC_HOSTS | GUIDANCE_HOSTS

_VP = "https://viewpoint.pwc.com/dt/us/en/pwc/accounting_guides/financial_statement_/financial_statement___18_US/chapter_30_accountin_US/"
# Starting points (verified to load 2026-10-21). Pages link onward; the agent may follow links
# that stay on an allowed host.
GUIDANCE_START: dict[str, str] = {
    "Correction of an error (restatements, revisions)": _VP + "307_correction_of_an_US.html",
    "Reclassifications": _VP + "308_reclassification_US.html",
    "Change in reporting entity": _VP + "306_change_in_a_repo_US.html",
}

EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index?q={q}"
# Part of every cache key: bump when extraction changes, so stale text is never reused.
# "ixbrl-1": inline-XBRL hidden facts stripped (the first live run read tag soup like
# "ge:GEHealthcareMember us-gaap:SegmentDiscontinuedOperationsMember" instead of the notes).
SOURCES_VERSION = "ixbrl-1"
GENERIC_AGENT = "filings-agent research"
_CHUNK, _OVERLAP = 1500, 200


class NotAllowed(Exception):
    """A URL outside the allowlist. Raised before any request is made."""


def allowed(url: str) -> bool:
    """https on an allowed host, default port, no credentials. Malformed URLs are refused."""
    try:
        p = urlparse(url)
        port = p.port  # raises ValueError on a malformed port
    except ValueError:
        return False
    return (
        p.scheme == "https"
        and (p.hostname or "") in ALLOWED_HOSTS
        and port in (None, 443)
        and p.username is None
        and p.password is None
    )


@dataclass(frozen=True)
class Document:
    url: str
    title: str
    text: str
    links: tuple[str, ...]  # absolute, allowlisted
    fetched_at: str  # ISO date: live sources change, so findings say when they were read


@dataclass(frozen=True)
class Hit:
    form: str
    filed: str
    company: str
    url: str


@runtime_checkable  # tool runtimes are validated by pydantic, which needs an isinstance check
class Fetcher(Protocol):
    def document(self, url: str) -> Document: ...
    def edgar_search(self, query: str, cik: int | None, forms: tuple[str, ...]) -> list[Hit]: ...


def html_to_text(html: str, base_url: str) -> tuple[str, str, tuple[str, ...]]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else base_url
    links: list[str] = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not isinstance(href, str) or href.startswith(("#", "javascript:", "mailto:")):
            continue
        absolute = urljoin(base_url, href).split("#")[0]
        if allowed(absolute) and absolute not in links:
            links.append(absolute)
    # 10-Ks are inline XBRL: a hidden header and thousands of display:none facts that read as
    # tag soup. Keep only what a person sees.
    for t in soup.find_all("ix:header"):
        t.decompose()
    for t in soup.find_all(style=re.compile(r"display:\s*none")):
        t.decompose()
    for t in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        t.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    return title, text, tuple(links)


def passages(text: str, query: str, k: int = 4) -> list[tuple[int, str]]:
    """The k chunks sharing the most query words: plain keyword ranking, deterministic."""
    words = {w for w in re.findall(r"[a-z0-9]{3,}", query.lower())}
    chunks = [
        (i, text[start : start + _CHUNK])
        for i, start in enumerate(range(0, max(len(text), 1), _CHUNK - _OVERLAP))
    ]
    scored = [(sum(c.lower().count(w) for w in words), i, c) for i, c in chunks if words]
    best = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))[:k]
    return [(i, c) for _, i, c in sorted(best, key=lambda s: s[1])]


class LiveFetcher:
    """httpx with a per-host politeness delay and an on-disk cache keyed by URL."""

    def __init__(
        self,
        sec_user_agent: str | Callable[[], str],  # a callable is read on the first SEC request
        cache_dir: Path,
        max_rps: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._sec_agent = sec_user_agent
        self._client = httpx.Client(
            timeout=30.0,
            follow_redirects=False,
            transport=transport,  # redirects re-checked
        )
        self._cache = cache_dir
        self._cache.mkdir(parents=True, exist_ok=True)
        self._interval = 1.0 / max_rps
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def _get(self, url: str, redirects: int = 3) -> httpx.Response:
        if not allowed(url):
            raise NotAllowed(f"{url} is not an allowed source ({', '.join(sorted(ALLOWED_HOSTS))})")
        host = urlparse(url).hostname or ""
        agent = GENERIC_AGENT
        if host in SEC_HOSTS:
            if callable(self._sec_agent):
                self._sec_agent = self._sec_agent()
            agent = self._sec_agent
        with self._lock:  # only to reserve a time slot per host; requests run in parallel
            slot = max(time.monotonic(), self._last.get(host, 0.0) + self._interval)
            self._last[host] = slot
        if (wait := slot - time.monotonic()) > 0:
            time.sleep(wait)
        resp = self._client.get(url, headers={"User-Agent": agent})
        if resp.is_redirect and redirects > 0:  # every hop must stay on the allowlist
            return self._get(urljoin(url, resp.headers["location"]), redirects - 1)
        resp.raise_for_status()
        return resp

    def _path(self, url: str) -> Path:
        key = f"{SOURCES_VERSION}|{url}"
        return self._cache / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"

    def _cached(self, url: str) -> dict[str, Any] | None:
        path = self._path(url)
        return cast(dict[str, Any], json.loads(path.read_text())) if path.exists() else None

    def _store(self, url: str, record: dict[str, Any]) -> None:
        path = self._path(url)
        atomic_write_text(path, json.dumps(record))

    def document(self, url: str) -> Document:
        if not allowed(url):
            raise NotAllowed(f"{url} is not an allowed source")
        cached = self._cached(url)
        if cached is not None:
            rec = cached
        else:
            resp = self._get(url)
            title, text, links = html_to_text(resp.text, url)
            rec: dict[str, Any] = {
                "title": title,
                "text": text,
                "links": list(links),
                "fetched_at": time.strftime("%Y-%m-%d"),
            }
            self._store(url, rec)
        return Document(url, rec["title"], rec["text"], tuple(rec["links"]), rec["fetched_at"])

    def edgar_search(self, query: str, cik: int | None, forms: tuple[str, ...]) -> list[Hit]:
        url = EDGAR_SEARCH.format(q=quote(f'"{query}"'))
        if cik is not None:
            url += f"&ciks={cik:010d}"
        if forms:
            url += "&forms=" + quote(",".join(forms), safe=",")
        rec = self._cached(url)
        if rec is None:
            rec = cast(dict[str, Any], self._get(url).json())
            self._store(url, rec)
        hits: list[Hit] = []
        for h in cast(list[dict[str, Any]], rec.get("hits", {}).get("hits", [])):
            src = cast(dict[str, Any], h.get("_source", {}))
            adsh, _, filename = str(h.get("_id", "")).partition(":")
            ciks = cast(list[str], src.get("ciks") or ["0"])
            hits.append(
                Hit(
                    form=str(src.get("form", "")),
                    filed=str(src.get("file_date", "")),
                    company=", ".join(cast(list[str], src.get("display_names") or [])),
                    url=f"https://www.sec.gov/Archives/edgar/data/{int(ciks[0])}/"
                    f"{adsh.replace('-', '')}/{filename}",
                )
            )
        return hits
