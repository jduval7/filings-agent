"""Memo step: a short comparison memo for an analyst, where every number is checkable.

Code builds the table (latest value per company/metric/year, after human decisions) and every
derived number (year-over-year change), each with a display string like "$215.9B" or "+65.5%".
The LLM writes prose using ONLY those strings. A deterministic checker extracts every number
from the memo and rejects any that isn't in the table: an invented or mis-rounded number can't
reach the analyst. One retry with the failures fed back; if it still fails, the memo ships with
a NOT VERIFIED banner instead of silently.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from filings_agent.config import ModelSpec
from filings_agent.domain import METRIC_SPECS, ExtractedFact, Metric, Unit
from filings_agent.llm.cache import StepCache
from filings_agent.steps.extract import Usage
from filings_agent.steps.reconcile import Resolution

# ---- deterministic table ----------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    ticker: str
    metric: Metric
    fiscal_year: int
    value: Decimal
    display: str
    source: str  # accession number the value came from
    changed_by: str | None  # cause, if a later filing restated it


def display(value: Decimal, metric: Metric) -> str:
    unit = METRIC_SPECS[metric].unit
    v = abs(value)
    sign = "-" if value < 0 else ""
    if unit is Unit.SHARES:
        return (
            f"{sign}{v / Decimal(1e9):.2f}B shares"
            if v >= Decimal(1e9)
            else f"{sign}{v / Decimal(1e6):.0f}M shares"
        )
    if v >= Decimal(1e9):
        return f"{sign}${v / Decimal(1e9):.1f}B"
    return f"{sign}${v / Decimal(1e6):.0f}M"


def pct(new: Decimal, old: Decimal) -> str | None:
    if old == 0 or (old < 0) != (new < 0):
        return None  # growth from a loss, or across zero, isn't meaningful as a percent
    return f"{(new - old) / abs(old) * 100:+.1f}%"


def build_table(
    facts: list[ExtractedFact],
    resolutions: dict[str, Resolution],
    filed: dict[str, date] | None = None,
) -> tuple[list[Row], list[str]]:
    """One consistent basis per company: the figures from its LATEST filing.

    Comparing across filings mixes bases: NVIDIA FY2023 shares from the post-split filing vs
    FY2022 from a pre-split one gave "+889%"; GE debt before vs after its spin-offs gave a
    meaningless decline. So the table holds each company's latest filing only (every year in it
    is on the same basis), YoY changes are computed only between rows from the same filing, and
    restatements are listed as caveats. A human decision to use an original value is honoured,
    and that row is marked so it isn't compared against recast figures.
    """

    def recency(accn: str) -> tuple[date, str]:
        return ((filed or {}).get(accn, date.min), accn)

    latest: dict[str, str] = {}
    for f in facts:
        if f.ticker not in latest or recency(f.accession_no) > recency(latest[f.ticker]):
            latest[f.ticker] = f.accession_no
    rows: list[Row] = []
    for f in sorted(facts, key=lambda f: (f.ticker, f.metric.value, f.fiscal_year)):
        if f.accession_no != latest[f.ticker]:
            continue
        disc_id = f"{f.ticker}:{f.metric.value}:FY{f.fiscal_year}"
        res = resolutions.get(disc_id)
        value, source = f.value, f.accession_no
        if res is not None and res.use_accession != f.accession_no:  # human chose the original
            value, source = res.use_value, res.use_accession
        rows.append(
            Row(
                f.ticker,
                f.metric,
                f.fiscal_year,
                value,
                display(value, f.metric),
                source,
                res.cause.value if res else None,
            )  # fmt: skip
        )
    changes: list[str] = []
    index = {(r.ticker, r.metric, r.fiscal_year): r for r in rows}
    for r in rows:
        prev = index.get((r.ticker, r.metric, r.fiscal_year - 1))
        if prev is None or prev.source != r.source:  # different basis: no comparison
            continue
        if p := pct(r.value, prev.value):
            label = f"{r.ticker} {r.metric.value} FY{r.fiscal_year} vs FY{prev.fiscal_year}"
            changes.append(f"{label}: {p}")
    return rows, changes


# ---- the deterministic checker ----------------------------------------------------------------

_NUMBER = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?\s*(?:%|[BMK]\b|x\b)?")


def _canon(token: str) -> str:
    return re.sub(r"[\s,]", "", token).lower()


def allowed_numbers(rows: list[Row], changes: list[str]) -> set[str]:
    allowed: set[str] = set()
    for r in rows:
        for tok in _NUMBER.findall(r.display):
            allowed.add(_canon(tok))
        allowed.add(str(r.fiscal_year))
        allowed.add(f"fy{r.fiscal_year}")
    for c in changes:
        for tok in _NUMBER.findall(c.split(":")[-1]):
            allowed.add(_canon(tok))
            allowed.add(_canon(tok.lstrip("+")))
    return allowed


def unverified_numbers(text: str, allowed: set[str]) -> list[str]:
    """Numbers in the memo that aren't in the table. Small counting words (1-10) are allowed."""
    bad: list[str] = []
    for tok in _NUMBER.findall(text):
        c = _canon(tok)
        # signs must match: "-65.5%" is not "+65.5%" (found by review); an unsigned number may
        # match a "+" one
        if c in allowed or (
            not c.startswith("-") and (c.lstrip("+") in allowed or "+" + c.lstrip("+") in allowed)
        ):
            continue
        if re.fullmatch(r"\d", c) or c in {"10"}:
            continue  # "two companies", "3 years": not financial figures
        bad.append(tok.strip())
    return bad


# ---- the LLM part -----------------------------------------------------------------------------


class MemoSection(BaseModel):
    heading: str
    body: str


class MemoOutput(BaseModel):
    title: str
    summary: str = Field(description="Two or three sentences: the headline comparison.")
    sections: list[MemoSection] = Field(description="One short section per company or theme.")
    caveats: str = Field(description="Restatements the reader must know about, from the table.")


PROMPT = (
    "You write a short comparison memo for an investment analyst from a table of figures taken "
    "from companies' annual reports.\n"
    "Hard rule: every number you write must be copied exactly from the table or the list of "
    "changes (e.g. '$215.9B', '+65.5%', 'FY2024'). Never compute, round, or restate a number "
    "differently. If a comparison isn't in the table, describe it in words without a number.\n"
    "All figures for a company come from its latest annual report, so they're comparable with "
    "each other. Mention restated figures (the 'changed_by' column) in caveats: say what changed "
    "and why. Don't add background that isn't in the table (no history, no outlook)."
)


def _table_text(rows: list[Row], changes: list[str]) -> str:
    lines = ["ticker | metric | fiscal year | value | changed_by"]
    lines += [
        f"{r.ticker} | {r.metric.value} | FY{r.fiscal_year} | {r.display} | {r.changed_by or ''}"
        for r in rows
    ]
    return "\n".join(lines) + "\n\nYear-over-year changes:\n" + "\n".join(changes)


@dataclass(frozen=True)
class Memo:
    markdown: str
    verified: bool
    unverified: list[str]
    attempts: int
    usage: Usage


def render(out: MemoOutput, verified: bool, bad: list[str]) -> str:
    banner = "" if verified else (
        f"> **NOT VERIFIED**: these numbers aren't in the source table: {', '.join(bad)}. "
        "Trust the table, not the prose.\n\n"
    )  # fmt: skip
    body = "\n\n".join(f"## {s.heading}\n\n{s.body}" for s in out.sections)
    return f"# {out.title}\n\n{banner}{out.summary}\n\n{body}\n\n## Caveats\n\n{out.caveats}\n"


def table_text(rows: list[Row], changes: list[str]) -> str:
    """The exact material the memo writer (and the judge) sees."""
    return _table_text(rows, changes)


class MemoWriter:
    def __init__(
        self,
        model_name: str,
        spec: ModelSpec,
        chat: BaseChatModel,
        cache: StepCache | None = None,
    ) -> None:
        self.name = f"memo/{model_name}"
        self._model_name = model_name
        self._spec = spec
        self._structured = chat.with_structured_output(MemoOutput, include_raw=True)
        self._cache = cache

    def write(
        self, rows: list[Row], changes: list[str], max_attempts: int = 2, rep: int = 0
    ) -> Memo:
        key = StepCache.key(
            step="memo",
            model=self._spec.id,
            prompt=PROMPT,
            table=_table_text(rows, changes),
            attempts=max_attempts,
            rep=rep,
        )
        if self._cache is not None and (hit := self._cache.get(key)) is not None:
            u = hit["usage"]
            return Memo(hit["markdown"], hit["verified"], hit["unverified"], hit["attempts"],
                        Usage(self._model_name, u["input_tokens"], u["output_tokens"],
                              u["cost_usd"], u["latency_s"], cached=True))  # fmt: skip
        memo = self._write(rows, changes, max_attempts)
        if self._cache is not None:
            u = memo.usage
            self._cache.put(key, {
                "markdown": memo.markdown, "verified": memo.verified,
                "unverified": memo.unverified, "attempts": memo.attempts,
                "usage": {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                          "cost_usd": u.cost_usd, "latency_s": u.latency_s},
            })  # fmt: skip
        return memo

    def _write(self, rows: list[Row], changes: list[str], max_attempts: int) -> Memo:
        allowed = allowed_numbers(rows, changes)
        messages: list[BaseMessage] = [
            SystemMessage(PROMPT),
            HumanMessage(_table_text(rows, changes)),
        ]
        in_tok = out_tok = 0
        t0 = time.monotonic()
        out: MemoOutput | None = None
        bad: list[str] = []
        attempts = 0
        for _ in range(max_attempts):
            attempts += 1
            raw = cast(dict[str, Any], self._structured.invoke(messages))
            usage = cast(AIMessage, raw["raw"]).usage_metadata
            if usage:
                in_tok += usage["input_tokens"]
                out_tok += usage["output_tokens"]
            out = cast(MemoOutput | None, raw.get("parsed"))
            if out is None:
                bad = ["(no structured output)"]
                continue
            text = " ".join([out.title, out.summary, out.caveats, *(s.body for s in out.sections)])
            bad = unverified_numbers(text, allowed)
            if not bad:
                break
            messages.append(
                HumanMessage(
                    f"These numbers are not in the table: {', '.join(bad)}. Rewrite the memo using "
                    "only numbers copied exactly from the table."
                )
            )
        usage = Usage(
            self._model_name, in_tok, out_tok, self._spec.cost(in_tok, out_tok),
            time.monotonic() - t0, cached=False,
        )  # fmt: skip
        if out is None:
            return Memo(
                "# Memo unavailable\n\nThe memo writer returned no output.\n",
                False,
                bad,
                attempts,
                usage,
            )
        return Memo(render(out, not bad, bad), not bad, bad, attempts, usage)
