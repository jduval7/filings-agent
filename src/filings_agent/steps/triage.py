"""Triage: which reconcile answers are safe to skip human review.

The quote check (`verify_quote`) proves the evidence EXISTS. It does not prove the cause
explains THIS number. GE FY2022 net income passed with a real quote about the HealthCare
spin-off, but a spin-off can't change total net income (it moves amounts between continuing and
discontinued lines). Triage checks the links between cause, metric and number with accounting
rules code can verify, and gives each check one of three outcomes:

- confirmed:    a rule ties the answer to the number (a stock split leaves an exact
                whole-number ratio, and the quote names the same ratio);
- contradicted: a rule says the answer can't be right (a spin-off vs net income);
- not_covered:  no rule can prove or disprove it.

Easy lane ("auto"): every check confirmed, including at least one numeric fingerprint. Only
then is the answer trusted without a human. Everything else goes to review, failing safe:
"not covered" means "a human looks", never "probably fine". A planned hard lane (a research
step that lays out the competing causes for the reviewer) would replace plain review for those;
see DECISIONS.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from filings_agent.domain import Discrepancy, Metric
from filings_agent.steps.reconcile import MATERIALITY, Cause, Resolution, change_filings
from filings_agent.textnorm import normalize

# Weighted-average diluted shares aren't an exact multiple after a split (AVGO x10.0047,
# AMZN x19.9922): share issuance and rounding move the average slightly.
SPLIT_TOLERANCE = Decimal("0.005")
# ...and never more than a quarter share either way, so a ratio is near at most one whole number.
# (Found by a property test: 129.5 was within 0.5% of both 129 and 130 and matched 130.)
SPLIT_MAX_DRIFT = Decimal("0.25")


class CheckStatus(StrEnum):
    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"
    NOT_COVERED = "not_covered"


@dataclass(frozen=True)
class Check:
    name: str
    status: CheckStatus
    detail: str


Lane = Literal["auto", "review"]


@dataclass(frozen=True)
class Triage:
    lane: Lane
    checks: tuple[Check, ...]


# Metrics a cause CANNOT move, with the accounting reason and its source (shown on the review card).
# General rules, not per company. Checked 2026-10-23 against Deloitte's DART roadmaps:
#   ASC 260-10-55-12: a stock split adjusts share counts and per-share amounts retroactively; it
#     changes the number of shares, not any dollar amount.
#     https://dart.deloitte.com/USDART/home/codification/presentation/asc260-10/roadmap-earnings-per-share/chapter-8-other-considerations/8-2-shareholder-distributions
#   ASC 205-20-45-3A: results of discontinued operations are reported "as a separate component of
#     income", i.e. inside net income. Reclassifying a business into discontinued operations moves
#     amounts between lines; net income (and the parent's share of it) is unchanged.
#     https://dart.deloitte.com/USDART/home/codification/presentation/asc205-20/roadmap-disposals-long-lived-assets-discontinued-operations/chapter-7-presentation-disclosure-requirements-for/7-3-income-statement-presentation-discontinued
#   Spin-offs and share counts: no codification paragraph needed. A pro-rata spin-off distributes
#     the subsidiary's shares; the parent's own share count doesn't change. Labelled as reasoning.
SPLIT_RULE = (
    "a stock split changes share counts, not dollar amounts (ASC 260-10-55-12: splits restate "
    "share counts and per-share amounts only)"
)
CANNOT_MOVE: dict[Cause, dict[Metric, str]] = {
    Cause.STOCK_SPLIT: {m: SPLIT_RULE for m in Metric if m is not Metric.DILUTED_SHARES},
    Cause.DISCONTINUED_OPERATIONS: {
        Metric.NET_INCOME: "net income attributable to the company includes discontinued "
        "operations, so a spin-off or sale moves amounts between lines but not the total "
        "(ASC 205-20-45-3A: discontinued operations are a separate component of income)",
        Metric.DILUTED_SHARES: "a spin-off or sale doesn't change the company's share count "
        "(reasoning: a pro-rata spin-off distributes the subsidiary's shares, not the parent's)",
    },
}


def _ratios(d: Discrepancy) -> list[Decimal]:
    """Newer/older value for each filing where the number changed."""
    vals = dict(d.values)
    accns = [a for a, _ in d.values]
    out: list[Decimal] = []
    for a in change_filings(d):
        prev = vals[accns[accns.index(a) - 1]]
        if prev != 0:
            out.append(vals[a] / prev)
    return out


def split_ratio(r: Decimal) -> int | None:
    """N if r is a whole-number N-for-1 split (N >= 2), or a 1-for-N reverse split (as -N)."""
    for x, sign in ((r, 1), (1 / r if r else Decimal(0), -1)):
        n = round(x)
        if n >= 2 and abs(x - n) <= min(SPLIT_TOLERANCE * n, SPLIT_MAX_DRIFT):
            return sign * n
    return None


_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "twenty-five": 25, "thirty": 30,
    "forty": 40, "fifty": 50, "hundred": 100, "one-hundred": 100, "one": 1,
}  # fmt: skip
_NUM = r"(\d+|" + "|".join(sorted(map(re.escape, _WORDS), key=len, reverse=True)) + r")"
_STATED_RATIO = re.compile(rf"\b{_NUM}-for-{_NUM}\b|\b(\d+):(\d+)\b")


def _num(s: str) -> int:
    return int(s) if s.isdigit() else _WORDS[s]


def stated_ratios(quote: str) -> set[int]:
    """Split ratios a quote names: "ten-for-one" -> 10, "1-for-20" -> -20 (reverse)."""
    out: set[int] = set()
    for m in _STATED_RATIO.finditer(normalize(quote)):
        a, b = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        x, y = _num(a), _num(b)
        if y == 1 and x >= 2:
            out.add(x)
        elif x == 1 and y >= 2:
            out.add(-y)
    return out


def _split_checks(d: Discrepancy, quote: str) -> list[Check]:
    found = {r: split_ratio(r) for r in _ratios(d)}
    fits = {n for n in found.values() if n is not None}
    shown = ", ".join(f"x{r:.4f}" for r in found) or "no change"
    if not fits:
        return [
            Check("split_ratio", CheckStatus.CONTRADICTED, f"{shown}: not a whole-number split")
        ]
    # EVERY move must be a split or immaterial: 100 -> 1000 (split) -> 1150 (a 15% restatement)
    # must not ride through on the split (found by review).
    unexplained = [r for r, n in found.items() if n is None and abs(r - 1) >= MATERIALITY]
    if unexplained:
        other = ", ".join(f"x{r:.4f}" for r in unexplained)
        return [
            Check("split_ratio", CheckStatus.CONTRADICTED, f"{other} isn't a split: another change")
        ]
    checks = [Check("split_ratio", CheckStatus.CONFIRMED, f"{shown} matches a split")]
    named = stated_ratios(quote)
    if not named:
        checks.append(Check("quote_names_ratio", CheckStatus.NOT_COVERED, "quote names no ratio"))
    elif named & fits:
        checks.append(
            Check("quote_names_ratio", CheckStatus.CONFIRMED, f"quote names {sorted(named)}")
        )
    else:
        checks.append(
            Check(
                "quote_names_ratio",
                CheckStatus.CONTRADICTED,
                f"quote names {sorted(named)}, the number moved by {sorted(fits)}",
            )
        )
    return checks


def cause_checks(d: Discrepancy, cause: Cause, quote: str) -> list[Check]:
    """Can `cause` explain this change, given the numbers and the quoted text? Used for the
    agent's answer (triage) and for every candidate cause on the review card."""
    reason = CANNOT_MOVE.get(cause, {}).get(d.metric)
    if reason:
        return [Check("cause_fits_metric", CheckStatus.CONTRADICTED, reason)]
    checks = [Check("cause_fits_metric", CheckStatus.CONFIRMED, f"{cause} can move {d.metric}")]
    if cause is Cause.STOCK_SPLIT:
        checks.extend(_split_checks(d, quote))
    else:
        checks.append(Check("fingerprint", CheckStatus.NOT_COVERED, f"no numeric rule for {cause}"))
    return checks


def triage(d: Discrepancy, r: Resolution) -> Triage:
    if not r.material:
        return Triage("auto", (Check("immaterial", CheckStatus.CONFIRMED, "under threshold"),))
    if r.cause is Cause.UNKNOWN or not r.verified:
        return Triage("review", (Check("quote", CheckStatus.CONTRADICTED, "no verified answer"),))
    checks = [Check("quote", CheckStatus.CONFIRMED, "quote found in the evidence")]
    checks.extend(cause_checks(d, r.cause, r.evidence_quote))
    lane: Lane = "auto" if all(c.status is CheckStatus.CONFIRMED for c in checks) else "review"
    return Triage(lane, tuple(checks))


def apply_triage(d: Discrepancy, r: Resolution, auto_approve: bool) -> Resolution:
    """Attach the triage. With `auto_approve` off, every material change is still reviewed
    (the triage is shown to the reviewer but doesn't skip anyone)."""
    t = triage(d, r)
    skip = auto_approve and t.lane == "auto"
    return replace(r, triage=t, needs_review=r.needs_review and not skip)
