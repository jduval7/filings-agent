"""One investigation per cause, fanned out by code: coverage, verification, no favorite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import ToolMessage

from filings_agent.research.agent import CAUSE_BRIEFS, CauseReport, Finding, verify_findings
from filings_agent.steps.reconcile import ReconcileTask
from tests.support.cases import GE_EVIDENCE, GE_NET_INCOME
from tests.support.research_doubles import SEC_DOC, FakeFetcher


def test_only_quotes_found_verbatim_in_an_allowed_source_survive() -> None:
    src = "On January 1, 2023, we adopted ASU No. 2018-12, Targeted Improvements."
    fake = FakeFetcher({SEC_DOC: src})

    def finding(quote: str, url: str = SEC_DOC) -> Finding:
        return Finding(stance="supports", quote=quote, source_url=url, note="n")

    report = CauseReport(
        findings=[
            finding("we adopted ASU No. 2018-12, Targeted Improvements"),  # verbatim: kept
            finding("we adopted ASU 2018-12 which restated net income"),  # invented: dropped
            finding("we adopted ASU No. 2018-12", "https://evil.example/x"),  # not allowed
        ],
        looked_at="GE 10-K",
    )
    kept, dropped = verify_findings("accounting_change", report, fake, cik=40545)
    assert [k.quote for k in kept] == ["we adopted ASU No. 2018-12, Targeted Improvements"]
    assert dropped == 2 and kept[0].fetched_at == "2026-10-21"


class ToolFake(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # tools are bound, never called
        return self


class FakeInvestigation:
    def __init__(self, cause: str, calls: list[str], fail: bool = False) -> None:
        self.cause, self.calls, self.fail = cause, calls, fail

    def invoke(self, inp: Any, config: Any = None, context: Any = None) -> dict[str, Any]:
        self.calls.append(self.cause)
        if self.fail:
            return {"messages": []}  # e.g. hit its call limit without a report
        quote = "we adopted ASU No. 2018-12"
        finding = Finding(stance="supports", quote=quote, source_url=SEC_DOC, note=self.cause)
        read = ToolMessage("passages", tool_call_id="t1", artifact=[{"url": SEC_DOC, "text": "x"}])
        return {
            "messages": [read],
            "structured_response": CauseReport(findings=[finding], looked_at="10-K"),
        }


def cik_one(ticker: str) -> int:
    return 40545  # GE, the filer in SEC_DOC


def test_every_cause_is_investigated_and_one_failure_does_not_sink_the_rest(
    tmp_path: Path,
) -> None:
    from filings_agent.config import ModelSpec
    from filings_agent.llm.cache import StepCache
    from filings_agent.research.agent import Researcher

    spec = ModelSpec(provider="bedrock", id="m", price=(3.0, 15.0))
    fake = FakeFetcher({SEC_DOC: "On January 1, 2023, we adopted ASU No. 2018-12."})
    r = Researcher(
        "m", spec, ToolFake(messages=iter([])), StepCache(tmp_path), fake, cik_one, lambda a: ""
    )
    calls: list[str] = []
    r._investigators = {  # pyright: ignore[reportPrivateUsage]
        c: FakeInvestigation(c, calls, fail=(c == "reclassification")) for c in CAUSE_BRIEFS
    }
    task = ReconcileTask(GE_NET_INCOME.id, GE_NET_INCOME, GE_EVIDENCE)
    res = r.research(task, "", 1)
    assert sorted(calls) == sorted(CAUSE_BRIEFS)  # code guarantees the coverage
    assert {f.cause for f in res.findings} == set(CAUSE_BRIEFS) - {"reclassification"}
    assert res.error and "reclassification" in res.error

    r.research(task, "", 1)  # a failed investigation isn't cached: the round can be retried
    assert len(calls) == 2 * len(CAUSE_BRIEFS)


def test_investigations_run_in_the_callers_context_so_traces_nest(tmp_path: Path) -> None:
    # LangChain and LangSmith find the parent run through contextvars. A plain ThreadPoolExecutor
    # doesn't copy them, so every investigation was traced as its own root trace (seen in
    # LangSmith: parent none) instead of under the research step.
    from contextvars import ContextVar

    from filings_agent.config import ModelSpec
    from filings_agent.llm.cache import StepCache
    from filings_agent.research.agent import Researcher

    parent: ContextVar[str] = ContextVar("parent", default="none")
    seen: list[str] = []

    class Recording(FakeInvestigation):
        def invoke(self, inp: Any, config: Any = None, context: Any = None) -> dict[str, Any]:
            seen.append(parent.get())
            return super().invoke(inp, config, context)

    spec = ModelSpec(provider="bedrock", id="m", price=(3.0, 15.0))
    fake = FakeFetcher({SEC_DOC: "On January 1, 2023, we adopted ASU No. 2018-12."})
    r = Researcher(
        "m", spec, ToolFake(messages=iter([])), StepCache(tmp_path), fake, cik_one, lambda a: ""
    )
    calls: list[str] = []
    r._investigators = {c: Recording(c, calls) for c in CAUSE_BRIEFS}  # pyright: ignore[reportPrivateUsage]
    parent.set("research-step")
    r.research(ReconcileTask(GE_NET_INCOME.id, GE_NET_INCOME, GE_EVIDENCE), "", 1)
    assert seen == ["research-step"] * len(CAUSE_BRIEFS)


def test_reports_have_no_place_for_a_favorite() -> None:
    assert set(CauseReport.model_fields) == {"findings", "looked_at", "open_questions"}
    assert "most_likely" not in Finding.model_fields


def test_investigators_are_told_where_the_material_move_happened() -> None:
    from datetime import date
    from decimal import Decimal

    from filings_agent.domain import Discrepancy, Metric
    from filings_agent.research.agent import moves

    d = Discrepancy(
        ticker="GE",
        metric=Metric.NET_INCOME,
        fiscal_year=2022,
        period_end=date(2022, 12, 31),
        values=(("a-23", Decimal(225)), ("b-24", Decimal(339)), ("c-25", Decimal(336))),
    )
    assert moves(d) == [
        "- in b-24: 225 -> 339 (+50.7%) <- LARGEST",
        "- in c-25: 339 -> 336 (-0.9%)",
    ]


GE_CIK = 40545
OTHER_FILER = "https://www.sec.gov/Archives/edgar/data/1099219/000109921924000010/met-10k.htm"
GUIDE = "https://viewpoint.pwc.com/dt/us/en/x.html"


def _report(*urls: str) -> CauseReport:
    quote = "we adopted ASU No. 2018-12"
    return CauseReport(
        findings=[Finding(stance="supports", quote=quote, source_url=u, note="n") for u in urls],
        looked_at="x",
    )


def test_a_finding_must_cite_a_source_the_investigation_read() -> None:
    """Review finding: a model could cite an allowed page it never read, and the verifier fetched
    it for the first time. Now only what the investigation read counts."""
    fake = FakeFetcher(
        {SEC_DOC: "we adopted ASU No. 2018-12.", GUIDE: "we adopted ASU No. 2018-12."}
    )
    kept, dropped = verify_findings("x", _report(SEC_DOC, GUIDE), fake, cik=GE_CIK, read={GUIDE})
    assert [k.source_url for k in kept] == [GUIDE] and dropped == 1
    # reports cached before reads were recorded (read=None) are still checked by everything else
    kept, _ = verify_findings("x", _report(SEC_DOC, GUIDE), fake, cik=GE_CIK, read=None)
    assert len(kept) == 2


def test_another_companys_filing_is_not_evidence_about_this_one() -> None:
    fake = FakeFetcher(
        {SEC_DOC: "we adopted ASU No. 2018-12.", OTHER_FILER: "we adopted ASU No. 2018-12."}
    )
    kept, dropped = verify_findings("x", _report(SEC_DOC, OTHER_FILER), fake, cik=GE_CIK)
    assert [k.source_url for k in kept] == [SEC_DOC] and dropped == 1
    _, dropped = verify_findings("x", _report(SEC_DOC), fake, cik=None)  # unknown company
    assert dropped == 1


@pytest.mark.parametrize(
    ("error", "is_transient"),
    [
        (httpx.ConnectError("down"), True),
        (httpx.ReadTimeout("slow"), True),
        (
            httpx.HTTPStatusError(
                "x", request=httpx.Request("GET", GUIDE), response=httpx.Response(503)
            ),
            True,
        ),  # fmt: skip
        (
            httpx.HTTPStatusError(
                "x", request=httpx.Request("GET", GUIDE), response=httpx.Response(429)
            ),
            True,
        ),  # fmt: skip
        (
            httpx.HTTPStatusError(
                "x", request=httpx.Request("GET", GUIDE), response=httpx.Response(404)
            ),
            False,
        ),  # fmt: skip
        (ValueError("bad argument"), False),
    ],
)
def test_outages_fail_the_investigation_other_errors_go_to_the_model(
    error: Exception, is_transient: bool
) -> None:
    from types import SimpleNamespace

    from filings_agent.research.agent import tool_errors_unless_transient

    def handler(_: Any) -> Any:
        raise error

    request: Any = SimpleNamespace(tool_call={"id": "c1", "name": "read_source", "args": {}})
    if is_transient:
        with pytest.raises(type(error)):
            tool_errors_unless_transient.wrap_tool_call(request, handler)
    else:
        msg = tool_errors_unless_transient.wrap_tool_call(request, handler)
        assert isinstance(msg, ToolMessage) and "Tool error" in str(msg.content)


def test_an_outage_is_reported_and_not_cached(tmp_path: Path) -> None:
    from filings_agent.config import ModelSpec
    from filings_agent.llm.cache import StepCache
    from filings_agent.research.agent import Researcher

    class Down:
        def invoke(self, inp: Any, config: Any = None, context: Any = None) -> Any:
            raise httpx.ConnectError("sec.gov unreachable")

    spec = ModelSpec(provider="bedrock", id="m", price=(3.0, 15.0))
    r = Researcher("m", spec, ToolFake(messages=iter([])), StepCache(tmp_path),
                   FakeFetcher({}), cik_one, lambda a: "")  # fmt: skip
    r._investigators = dict.fromkeys(CAUSE_BRIEFS, Down())  # pyright: ignore[reportPrivateUsage]
    task = ReconcileTask(GE_NET_INCOME.id, GE_NET_INCOME, GE_EVIDENCE)
    res = r.research(task, "", 1)
    assert res.findings == () and res.error and "source_error:ConnectError" in res.error
    assert list(tmp_path.rglob("*.json")) == []  # nothing cached: the round can be retried


def test_a_changed_filing_is_recognized_by_accession_whatever_the_filer_folder() -> None:
    """Reports cached before the CIK fix cite /data/0/...; they still point at our copy."""
    from filings_agent.research.agent import filing_url

    text = "On January 1, 2023, we adopted ASU No. 2018-12."
    local = {filing_url(320335, "0000320335-24-000006"): text}
    legacy = filing_url(None, "0000320335-24-000006")
    kept, dropped = verify_findings("x", _report(legacy), FakeFetcher({}), local, cik=320335)
    assert len(kept) == 1 and dropped == 0 and kept[0].fetched_at == "downloaded copy"
    other = filing_url(None, "0000320335-23-000001")  # not a changed filing
    _, dropped = verify_findings("x", _report(other), FakeFetcher({}), local, cik=320335)
    assert dropped == 1


def test_research_finds_the_cik_of_reconcile_only_companies(cfg: Any) -> None:
    import json

    from filings_agent.graph.runner import make_researcher

    gold = cfg.data_dir / "gold"
    gold.mkdir(parents=True, exist_ok=True)
    row = {"ticker": "GL", "cik": 320335, "accession_no": "0000320335-24-000006", "form": "10-K",
           "filed": "2024-02-27", "period_of_report": "2023-12-31"}  # fmt: skip
    (gold / "filings_all.jsonl").write_text(json.dumps(row) + "\n")
    (gold / "filings.jsonl").write_text("")  # the extraction list doesn't have GL
    r = make_researcher(cfg)
    assert r._cik_of("GL") == 320335  # pyright: ignore[reportPrivateUsage]
    assert r._cik_of("NOPE") is None  # pyright: ignore[reportPrivateUsage]


def _tool_agent(fetch: Any) -> Any:
    """A minimal agent with the research middleware: the model calls `fetch` once, then answers."""
    from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    from filings_agent.research.agent import investigator_middleware

    @tool
    def read_page(url: str) -> str:
        """Read a page."""
        return fetch(url)

    call = AIMessage("", tool_calls=[{"name": "read_page", "args": {"url": SEC_DOC}, "id": "t1"}])
    model = ToolFake(messages=iter([call, AIMessage("done")]))
    return create_agent(model=model, tools=[read_page], middleware=investigator_middleware(0.0))


def _http_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", SEC_DOC)
    return httpx.HTTPStatusError("x", request=req, response=httpx.Response(status, request=req))


def test_a_briefly_unavailable_source_is_retried_before_the_investigation_gives_up() -> None:
    attempts: list[int] = []

    def fetch(url: str) -> str:
        attempts.append(1)
        if len(attempts) <= 2:
            raise _http_error(503)
        return "the page"

    out = _tool_agent(fetch).invoke({"messages": [("user", "go")]})
    tool_reply = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert tool_reply.content == "the page" and len(attempts) == 3


def test_a_source_that_stays_down_fails_the_investigation_visibly() -> None:
    def fetch(url: str) -> str:
        raise _http_error(503)

    with pytest.raises(httpx.HTTPStatusError):  # raised, so it isn't cached as "no evidence"
        _tool_agent(fetch).invoke({"messages": [("user", "go")]})


def test_a_bad_request_still_goes_back_to_the_model_as_a_message() -> None:
    attempts: list[int] = []

    def fetch(url: str) -> str:
        attempts.append(1)
        raise _http_error(404)

    out = _tool_agent(fetch).invoke({"messages": [("user", "go")]})
    tool_reply = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert "Tool error" in str(tool_reply.content) and len(attempts) == 1  # not retried
