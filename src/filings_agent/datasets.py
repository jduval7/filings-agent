"""Per-step eval datasets, built from the answer key.

Each step gets its own dataset so it can be tested (and its model swapped) in isolation:

  extract    one example per (filing x statement): section text in, expected facts out.
  reconcile  one example per answer-key discrepancy (built on day 3).

A fact only counts as expected if its value actually appears in the section text the extractor
sees ("answerable"). GE's diluted shares live in a note, not on the statement, so for GE the
right answer is "not on this statement". That keeps the score honest: we never grade a step on
information it was never given.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from filings_agent.config import AppConfig
from filings_agent.domain import Discrepancy, FilingRef, GoldFact, Metric
from filings_agent.sec.sections import StatementKind, locate_statements

STATEMENT_METRICS: dict[StatementKind, tuple[Metric, ...]] = {
    StatementKind.INCOME: (
        Metric.REVENUE,
        Metric.OPERATING_INCOME,
        Metric.NET_INCOME,
        Metric.DILUTED_SHARES,
    ),
    StatementKind.BALANCE: (Metric.LONG_TERM_DEBT, Metric.CASH),
    StatementKind.CASH_FLOW: (Metric.OPERATING_CASH_FLOW,),
}

# A table can mix scales: Apple reports dollars in millions but shares in thousands.
_FACTORS = (Decimal(1), Decimal(1_000), Decimal(1_000_000), Decimal(1_000_000_000))


def _renderings(value: Decimal) -> set[str]:
    """Ways a value can be printed in a statement: '16,864,919' (thousands), '1,609.4' (Nike)."""
    out: set[str] = set()
    v = abs(value)
    for factor in _FACTORS:
        scaled = v / factor
        if scaled == scaled.to_integral_value():
            out.add(f"{int(scaled):,}")
        for places in (1, 2):
            q = Decimal(10) ** -places
            if scaled == scaled.quantize(q) and scaled != scaled.to_integral_value():
                out.add(f"{scaled:,.{places}f}")
    return {r for r in out if len(r.replace(",", "").replace(".", "")) >= 2}


def is_answerable(value: Decimal, text: str) -> bool:
    return any(r in text for r in _renderings(value))


class ExpectedFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric: Metric
    period_end: date
    value: Decimal


class ExtractExample(BaseModel):
    example_id: str
    split: str
    ticker: str
    accession_no: str
    fiscal_year_end: date
    statement: StatementKind
    section_text: str
    metrics: list[Metric]
    expected: list[ExpectedFact]
    # Metrics with nothing answerable on this statement: the right output is "not reported".
    expect_absent: list[Metric]


def build_extract_dataset(
    cfg: AppConfig, filings: list[FilingRef], gold: list[GoldFact], markdown_of: dict[str, str]
) -> list[ExtractExample]:
    by_filing: dict[str, list[GoldFact]] = {}
    for g in gold:
        by_filing.setdefault(g.accession_no, []).append(g)

    examples: list[ExtractExample] = []
    for f in filings:
        sections = locate_statements(markdown_of[f.accession_no])
        for kind, metrics in STATEMENT_METRICS.items():
            wanted = [m for m in metrics if m in cfg.metrics]
            if kind not in sections or not wanted:
                continue
            text = sections[kind].text
            expected = [
                ExpectedFact(metric=g.metric, period_end=g.period.end, value=g.value)
                for g in by_filing.get(f.accession_no, [])
                if g.metric in wanted and is_answerable(g.value, text)
            ]
            present = {e.metric for e in expected}
            examples.append(
                ExtractExample(
                    example_id=f"{f.accession_no}:{kind.value}",
                    split=cfg.split_of(f.ticker),
                    ticker=f.ticker,
                    accession_no=f.accession_no,
                    fiscal_year_end=f.period_of_report,
                    statement=kind,
                    section_text=text,
                    metrics=wanted,
                    expected=sorted(expected, key=lambda e: (e.metric, e.period_end)),
                    expect_absent=[m for m in wanted if m not in present],
                )
            )
    return examples


# Relative change below this is treated as an immaterial revision (auto-accepted, no human).
MATERIALITY = Decimal("0.005")


class ReconcileExample(BaseModel):
    """One answer-key discrepancy. `cause` is hand-labelled after reading the filing (day 3);
    None means not yet labelled, and such examples are excluded from cause-accuracy scoring."""

    example_id: str
    split: str
    ticker: str
    metric: Metric
    period_end: date
    # (accession_no, filed, value), oldest filing first
    values: list[tuple[str, date, Decimal]]
    ratio: Decimal
    material: bool
    cause: str | None = None


def build_reconcile_dataset(
    cfg: AppConfig, discrepancies: list[Discrepancy], filed: dict[str, date]
) -> list[ReconcileExample]:
    out: list[ReconcileExample] = []
    for d in discrepancies:
        vals = [v for _, v in d.values]
        spread = (max(vals) - min(vals)) / max(abs(v) for v in vals)
        out.append(
            ReconcileExample(
                example_id=f"{d.ticker}:{d.metric.value}:{d.period_end.isoformat()}",
                split=cfg.split_of(d.ticker),
                ticker=d.ticker,
                metric=d.metric,
                period_end=d.period_end,
                values=[(a, filed[a], v) for a, v in d.values],
                ratio=d.ratio,
                material=spread >= MATERIALITY,
            )
        )
    return out


def write_jsonl(path: Path, rows: list[ExtractExample] | list[ReconcileExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows))


def summarize_reconcile(examples: list[ReconcileExample]) -> str:
    c = Counter((e.split, "material" if e.material else "immaterial") for e in examples)
    return f"reconcile examples: {dict(sorted(c.items()))}"


def summarize(examples: list[ExtractExample]) -> str:
    by_split = Counter(e.split for e in examples)
    facts = Counter(e.split for e in examples for _ in e.expected)
    absent = Counter((e.ticker, m.value) for e in examples for m in e.expect_absent)
    lines = [
        f"extract examples: {dict(by_split)}; expected facts: {dict(facts)}",
        f"expect 'not reported': {dict(absent)}",
    ]
    return "\n".join(lines)
