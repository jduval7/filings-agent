"""Find 10-K filings (SEC submissions API) and fetch their readable text (edgartools).

edgartools is untyped, so it's confined to `_edgar_markdown`; everything else is typed.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, cast

from filings_agent.domain import FilingRef
from filings_agent.fsutil import atomic_write_text
from filings_agent.sec.http import SecHttp

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
PAGE_URL = "https://data.sec.gov/submissions/{}"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


class SecSource:
    def __init__(self, http: SecHttp, filings_dir: Path, user_agent: str) -> None:
        self._http = http
        self._filings_dir = filings_dir
        self._filings_dir.mkdir(parents=True, exist_ok=True)
        self._user_agent = user_agent
        self._ciks: dict[str, int] | None = None

    def cik(self, ticker: str) -> int:
        if self._ciks is None:
            rows = cast(dict[str, dict[str, Any]], self._http.get_json(TICKERS_URL))
            self._ciks = {str(r["ticker"]): int(r["cik_str"]) for r in rows.values()}
        return self._ciks[ticker]

    def list_filings(self, ticker: str, form: str, filed_since: date) -> list[FilingRef]:
        cik = self.cik(ticker)
        sub = self._http.get_json(SUBMISSIONS_URL.format(cik=cik))["filings"]
        # "recent" holds only the latest ~1000 filings; frequent filers (Walmart) need older pages.
        pages = [cast(dict[str, list[str]], sub["recent"])]
        for extra in cast(list[dict[str, Any]], sub.get("files", [])):
            if date.fromisoformat(str(extra["filingTo"])) < filed_since:
                continue
            pages.append(
                cast(dict[str, list[str]], self._http.get_json(PAGE_URL.format(extra["name"])))
            )
        rows = [
            row
            for page in pages
            for row in zip(
                page["accessionNumber"],
                page["form"],
                page["filingDate"],
                page["reportDate"],
                strict=True,
            )
        ]
        out: list[FilingRef] = []
        for accn, f, filed, report in rows:
            if f != form or date.fromisoformat(filed) < filed_since:
                continue
            out.append(
                FilingRef(
                    ticker=ticker,
                    cik=cik,
                    accession_no=accn,
                    form=f,
                    filed=date.fromisoformat(filed),
                    period_of_report=date.fromisoformat(report),
                )
            )
        return sorted(out, key=lambda r: r.filed)

    def company_facts(self, ticker: str) -> dict[str, Any]:
        return self._http.get_json(COMPANY_FACTS_URL.format(cik=self.cik(ticker)))

    def markdown(self, filing: FilingRef) -> str:
        """Full filing as markdown (tables preserved), cached by accession number."""
        path = self._filings_dir / f"{filing.accession_no}.md"
        if path.exists():
            return path.read_text()
        text = _edgar_markdown(filing.accession_no, self._user_agent)
        atomic_write_text(path, text)
        return text


def _edgar_markdown(accession_no: str, user_agent: str) -> str:
    import edgar  # pyright: ignore[reportMissingTypeStubs]

    edgar.set_identity(user_agent)  # pyright: ignore[reportUnknownMemberType]
    filing: Any = edgar.get_by_accession_number(accession_no)  # pyright: ignore[reportUnknownMemberType]
    if filing is None:
        raise LookupError(f"filing not found: {accession_no}")
    return str(filing.markdown())
