"""Research doubles: findings and results builders, and an in-memory fetcher behind the real
allowlist."""

from __future__ import annotations

from filings_agent.research.findings import ResearchResult, VerifiedFinding
from filings_agent.research.sources import Document, Hit, NotAllowed, allowed
from filings_agent.steps.reconcile import ReconcileTask
from tests.support.cases import GE_NET_INCOME

SEC_DOC = "https://www.sec.gov/Archives/edgar/data/40545/000004054524000027/ge-20231231.htm"


def result(*findings: VerifiedFinding, round_: int = 1, hint: str = "") -> ResearchResult:
    return ResearchResult(GE_NET_INCOME.id, round_, hint, findings, 1, "GE filings", "", None)


def vf(cause: str, stance: str = "supports", other: str = "") -> VerifiedFinding:
    return VerifiedFinding(cause, other, stance, "q", SEC_DOC, "note", "2026-10-21")


class FakeFetcher:
    """Serves `docs` by URL, applies the real allowlist, returns canned EDGAR hits."""

    def __init__(self, docs: dict[str, str], hits: list[Hit] | None = None) -> None:
        self.docs = docs
        self.hits = hits or []
        self.searches: list[tuple[str, int | None, tuple[str, ...]]] = []

    def document(self, url: str) -> Document:
        if not allowed(url):
            raise NotAllowed(url)
        links = tuple(u for u in self.docs if u != url)
        return Document(url, "t", self.docs[url], links, "2026-10-21")

    def edgar_search(self, query: str, cik: int | None, forms: tuple[str, ...]) -> list[Hit]:
        self.searches.append((query, cik, forms))
        return self.hits


class FakeResearcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def research(self, task: ReconcileTask, hint: str, round_: int) -> ResearchResult:
        self.calls.append((task.discrepancy_id, hint, round_))
        return ResearchResult(
            task.discrepancy_id, round_, hint, (vf("other", other="error correction"),), 0,
            "8-Ks", "", None,
        )  # fmt: skip
