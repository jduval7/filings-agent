"""The research tools, called directly: changed filings, EDGAR search, allowlisted reading."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from filings_agent.research.sources import Hit
from filings_agent.research.tools import (
    ResearchContext,
    list_guidance,
    read_changed_filing,
    read_source,
    search_sec_filings,
)
from tests.support.cases import GE_NET_INCOME
from tests.support.research_doubles import SEC_DOC, FakeFetcher

INDEX = "https://www.sec.gov/Archives/edgar/data/40545/000004054524000027/x-index.htm"
FILING = "Intro. " * 300 + "On January 1, 2023, we adopted ASU No. 2018-12 retrospectively."


def load(accession_no: str) -> str:
    return FILING


def runtime(fetcher: FakeFetcher, **kw: Any) -> Any:
    ctx = ResearchContext(discrepancy=GE_NET_INCOME, cik=40545, fetcher=fetcher, **kw)
    return SimpleNamespace(context=ctx)


def call(tool: Any, *args: Any, **kwargs: Any) -> Any:
    return tool.func(*args, **kwargs)  # the plain function under @tool


def test_read_changed_filing_returns_passages_with_a_citable_url() -> None:
    rt = runtime(FakeFetcher({}), changed={"A-24": INDEX}, load_filing=load)
    content, artifact = call(read_changed_filing, "A-24", "adopted ASU 2018-12", runtime=rt)
    assert "adopted ASU No. 2018-12" in content and INDEX in content
    assert artifact and artifact[0]["url"] == INDEX


def test_read_changed_filing_refuses_other_filings_and_reports_misses() -> None:
    rt = runtime(FakeFetcher({}), changed={"A-24": INDEX}, load_filing=load)
    content, artifact = call(read_changed_filing, "B-99", "x", runtime=rt)
    assert "Not a changed filing" in content and "A-24" in content and artifact == []
    content, artifact = call(read_changed_filing, "A-24", "zebra", runtime=rt)
    assert content.startswith("Nothing about") and artifact == []


def test_search_scopes_to_the_company_unless_asked_not_to() -> None:
    hit = Hit("8-K", "2024-04-02", "GE", SEC_DOC)
    fetcher = FakeFetcher({}, hits=[hit])
    text = call(search_sec_filings, "spin off", runtime=runtime(fetcher), forms="8-K, 10-K")
    assert SEC_DOC in text and "8-K filed 2024-04-02" in text
    call(search_sec_filings, "ASU 2018-12", runtime=runtime(fetcher), this_company=False)
    assert fetcher.searches[0] == ("spin off", 40545, ("8-K", "10-K"))
    assert fetcher.searches[1][1] is None  # all companies
    assert "No SEC filings" in call(search_sec_filings, "x", runtime=runtime(FakeFetcher({})))


def test_without_the_company_id_a_company_search_says_so_and_searches_nothing() -> None:
    """Found 2026-10-23: with no CIK, "this company's filings" silently searched everyone's."""
    fetcher = FakeFetcher({}, hits=[Hit("8-K", "2024-04-02", "Other Co", SEC_DOC)])
    ctx = ResearchContext(discrepancy=GE_NET_INCOME, cik=None, fetcher=fetcher)
    text = call(search_sec_filings, "spin off", runtime=SimpleNamespace(context=ctx))
    assert "unknown" in text and fetcher.searches == []


def test_read_source_returns_passages_links_and_refuses_off_allowlist() -> None:
    other = "https://viewpoint.pwc.com/dt/us/en/pwc/accounting_guides/x.html"
    fetcher = FakeFetcher({SEC_DOC: FILING, other: "guide"})
    content, artifact = call(read_source, SEC_DOC, "adopted ASU", runtime=runtime(fetcher))
    assert "read 2026-10-21" in content and other in content  # links it may follow
    assert artifact[0]["fetched_at"] == "2026-10-21"
    content, artifact = call(read_source, "https://evil.example/x", "q", runtime=runtime(fetcher))
    assert content.startswith("Refused") and artifact == []
    content, _ = call(read_source, other, "zebra", runtime=runtime(fetcher))
    assert content.startswith("Nothing about")


def test_list_guidance_points_at_allowlisted_pages() -> None:
    text = call(list_guidance)
    assert "Correction of an error" in text and "viewpoint.pwc.com" in text
