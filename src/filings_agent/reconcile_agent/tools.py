"""The reconcile agent's tools.

Per the LangChain docs: `@tool` with type hints and a docstring written for the model; per-run data
comes from `ToolRuntime.context` (hidden from the model), so tools are plain functions of
(arguments, context) and easy to unit-test.

  fact_history   which filings reported which value: tells the agent WHERE the number changed
  search_filing  RAG over one filing: finds the passage that explains WHY

Tools never raise into the model loop for bad arguments; they return a short error the model can
act on (e.g. an unknown filing id), and the agent's middleware turns unexpected exceptions into
the same kind of message.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from langchain.tools import ToolRuntime, tool

from filings_agent.domain import Discrepancy, FilingRef
from filings_agent.reconcile_agent.retrieval import FilingIndex


@dataclass(frozen=True)
class ReconcileContext:
    """Everything the tools may READ for one discrepancy. Passed as `context=` at invoke time.

    Immutable on purpose: the LangChain docs define runtime context as read-only configuration.
    (v1 of this agent wrote retrieved passages into it; those writes were lost, so correct quotes
    looked unverified. Retrieved passages now travel as tool-message artifacts instead.)
    """

    discrepancy: Discrepancy
    filings: dict[str, FilingRef]  # this company's filings, by accession number
    index: FilingIndex
    k: int = 5


def _describe(f: FilingRef | None, accession_no: str) -> str:
    if f is None:
        return accession_no
    return f"{accession_no} (10-K for fiscal year ending {f.period_of_report}, filed {f.filed})"


# ToolRuntime is generic over (context, state); this agent's tools only read context.
Runtime = ToolRuntime[ReconcileContext, dict[str, Any]]


@tool
def fact_history(runtime: Runtime) -> str:
    """Show the value of the disputed figure as reported in each annual report, oldest first.

    Use this first: the filings where the value CHANGED are the ones that explain why.
    """
    ctx = runtime.context
    d = ctx.discrepancy
    lines = [f"{d.metric.value} for fiscal year {d.fiscal_year} ({d.ticker}), as reported in:"]
    previous = None
    for accession_no, value in d.values:
        changed = previous is not None and value != previous
        marker = "  <- changed here" if changed else ""
        lines.append(
            f"- {_describe(ctx.filings.get(accession_no), accession_no)}: {value:,}{marker}"
        )
        previous = value
    return "\n".join(lines)


@tool(response_format="content_and_artifact")
def search_filing(
    query: str, accession_no: str, runtime: Runtime
) -> tuple[str, list[dict[str, Any]]]:
    """Search one annual report for passages explaining a change in a reported figure.

    Args:
        query: What to look for, e.g. "stock split share counts retroactively adjusted" or
            "discontinued operations spin-off prior periods recast".
        accession_no: The filing to search, as listed by fact_history (e.g. 0001045810-25-000023).
    """
    ctx = runtime.context
    if accession_no not in ctx.filings:
        known = ", ".join(sorted(ctx.filings))
        return f"Unknown filing {accession_no!r}. Filings for {ctx.discrepancy.ticker}: {known}", []
    passages = ctx.index.search(query, accession_no, k=ctx.k)
    if not passages:
        return f"No passages found in {accession_no} for {query!r}.", []
    # Content for the model; the structured passages ride along as the ToolMessage artifact
    # (the documented pattern for retrieval tools), so code can verify quotes against them.
    content = "\n\n".join(f"[{p.passage_id}] {p.text}" for p in passages)
    return content, [asdict(p) for p in passages]


TOOLS = [fact_history, search_filing]


def filings_after(filings: dict[str, FilingRef], since: date) -> list[str]:
    """Helper for evals: accession numbers filed on/after a date."""
    return sorted(a for a, f in filings.items() if f.filed >= since)
