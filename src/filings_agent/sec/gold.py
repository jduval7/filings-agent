"""Build the answer key from SEC XBRL company facts.

The agent NEVER sees this data. It reads the human-readable filing; we grade it against these
values. Each XBRL fact carries the accession number of the filing that reported it, so the key
is exact per filing, including the prior years a filing re-reports (where restatements show up).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Protocol, cast

from filings_agent.domain import (
    METRIC_SPECS,
    Discrepancy,
    FilingRef,
    GoldFact,
    Metric,
    MetricSpec,
    Period,
    PeriodKind,
)

ANNUAL_DAYS = range(350, 381)  # 52/53-week fiscal years land in here


@dataclass
class GoldResult:
    facts: list[GoldFact] = field(default_factory=lambda: list[GoldFact]())
    # (ticker, accession, metric) with no matching concept -> the metric isn't reported
    missing: list[tuple[str, str, Metric]] = field(
        default_factory=lambda: list[tuple[str, str, Metric]]()
    )
    warnings: list[str] = field(default_factory=lambda: list[str]())


def _entries(
    us_gaap: dict[str, Any], concept: str, spec: MetricSpec, accession_no: str
) -> list[tuple[Period, Decimal]]:
    node = cast(dict[str, Any] | None, us_gaap.get(concept))
    if node is None:
        return []
    units = cast(dict[str, list[dict[str, Any]]], node["units"])
    out: list[tuple[Period, Decimal]] = []
    for x in units.get(spec.unit.value, []):
        if x.get("accn") != accession_no or not str(x.get("form", "")).startswith("10-K"):
            continue
        end = date.fromisoformat(cast(str, x["end"]))
        start_raw = cast(str | None, x.get("start"))
        start = date.fromisoformat(start_raw) if start_raw else None
        if spec.kind is PeriodKind.DURATION:
            if start is None or (end - start).days not in ANNUAL_DAYS:
                continue
        elif start is not None:
            continue
        out.append((Period(start=start, end=end), Decimal(str(x["val"]))))
    return out


def build_gold(
    ticker: str, company_facts: dict[str, Any], filings: list[FilingRef], metrics: list[Metric]
) -> GoldResult:
    us_gaap = cast(dict[str, Any], company_facts.get("facts", {}).get("us-gaap", {}))
    result = GoldResult()
    for filing in filings:
        per_metric: dict[Metric, tuple[str, list[tuple[Period, Decimal]]]] = {}
        for metric in metrics:
            spec = METRIC_SPECS[metric]
            for concept in spec.concepts:
                found = _entries(us_gaap, concept, spec, filing.accession_no)
                if found:
                    per_metric[metric] = (concept, found)
                    break
            else:
                result.missing.append((ticker, filing.accession_no, metric))

        # Fiscal year-ends this filing covers, from its income-statement periods.
        fy_ends = sorted(
            {
                p.end
                for m, (_, found) in per_metric.items()
                if METRIC_SPECS[m].kind is PeriodKind.DURATION
                for p, _ in found
                if p.end <= filing.period_of_report
            },
            reverse=True,
        ) or [filing.period_of_report]
        # A balance sheet shows the current and prior year-end only.
        balance_dates = set(fy_ends[:2])

        for metric, (concept, found) in per_metric.items():
            by_end: dict[date, set[Decimal]] = defaultdict(set)
            periods: dict[date, Period] = {}
            for period, value in found:
                if period.end > filing.period_of_report:
                    continue
                if period.kind is PeriodKind.INSTANT and period.end not in balance_dates:
                    continue
                by_end[period.end].add(value)
                periods[period.end] = period
            for end, values in sorted(by_end.items()):
                if len(values) > 1:
                    result.warnings.append(
                        f"{ticker} {filing.accession_no} {metric} {end}: "
                        f"conflicting values {sorted(values)}; skipped"
                    )
                    continue
                result.facts.append(
                    GoldFact(
                        ticker=ticker,
                        metric=metric,
                        period=periods[end],
                        value=next(iter(values)),
                        accession_no=filing.accession_no,
                        concept=concept,
                    )
                )
    return result


class FactLike(Protocol):
    """Anything with a fact's identity and value: answer-key facts and extracted facts alike."""

    @property
    def ticker(self) -> str: ...
    @property
    def metric(self) -> Metric: ...
    @property
    def fiscal_year(self) -> int: ...
    @property
    def period_end(self) -> date: ...
    @property
    def accession_no(self) -> str: ...
    @property
    def value(self) -> Decimal: ...


def find_discrepancies(facts: Sequence[FactLike], filed: dict[str, date]) -> list[Discrepancy]:
    """Same (ticker, metric, fiscal year) with different values across filings.

    Pure code, no LLM: comparing numbers doesn't need a model. The graph reuses this on
    extracted facts to decide which wave-2 reconciliation branches to spawn.
    """
    groups: dict[tuple[str, Metric, int], dict[str, Decimal]] = defaultdict(dict)
    ends: dict[tuple[str, Metric, int], date] = {}
    for f in facts:
        key = (f.ticker, f.metric, f.fiscal_year)
        groups[key][f.accession_no] = f.value
        ends.setdefault(key, f.period_end)
    out: list[Discrepancy] = []
    for key, by_accn in sorted(groups.items()):
        if len(set(by_accn.values())) < 2:
            continue
        ticker, metric, fy = key
        ordered = sorted(by_accn.items(), key=lambda kv: filed[kv[0]])
        out.append(
            Discrepancy(
                ticker=ticker,
                metric=metric,
                fiscal_year=fy,
                period_end=ends[key],
                values=tuple(ordered),
            )
        )
    return out
