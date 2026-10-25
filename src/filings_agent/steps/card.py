"""The review card: what a reviewer sees for one material change.

Design rules (DECISIONS.md, 2026-10-21 triage entry; research in reports/):
- **Numbers first.** Old and new value, the change, and which filing changed it.
- **Every candidate cause, never a favorite.** Causes are listed in a fixed order, each with the
  sentences from the filing that mention it and whether it fits the numbers. The agent's own
  answer and its hand-picked quote are NOT shown: putting them under one cause would quietly
  make it the favorite, and reviewers anchor on AI proposals (automation bias). The agent's
  answer still drives triage and the evals.
- **Teach while checking.** A ruled-out cause says why in one line ("a spin-off can't change
  total net income: ...").
- **The reviewer decides, or escalates.** Picking a cause is the decision; "escalate" is always
  allowed for cases where no candidate fits.

Built by code only, from evidence the pipeline already gathered. No LLM call.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from filings_agent.domain import Discrepancy
from filings_agent.research.findings import ResearchResult, VerifiedFinding
from filings_agent.steps.reconcile import EVIDENCE_FAMILIES, Cause, change_filings
from filings_agent.steps.triage import CheckStatus, cause_checks

CANDIDATES = (
    Cause.STOCK_SPLIT,
    Cause.DISCONTINUED_OPERATIONS,
    Cause.ACCOUNTING_CHANGE,
    Cause.RECLASSIFICATION,
)
Fit = Literal["fits", "possible", "ruled_out", "no_evidence"]
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
_MAX_QUOTE = 320
# Research findings shown per cause. More findings for one cause would read as a favorite.
MAX_FINDINGS_PER_CAUSE = 2


def _clean(passage: str) -> str:
    """Display text: drop markdown escapes ("ten\\-for\\-one"), heading marks and HTML tags.
    Heading marks are removed, not split on: GE prints the ASU title as a heading mid-sentence."""
    text = re.sub(r"\\(.)", r"\1", passage)
    text = re.sub(r"<[^>]+>|#{1,6}\s+", " ", text)
    return re.sub(r"\s+", " ", text)


def sentences_for(cause: Cause, evidence: tuple[str, ...], limit: int = 2) -> list[str]:
    """Whole sentences from the evidence that mention this cause."""
    patterns = [re.compile(p, re.I) for p in EVIDENCE_FAMILIES.get(cause.value, [])]
    out: list[str] = []
    for passage in evidence:
        for s in _SENTENCE.split(_clean(passage)):
            s = s.strip()
            if s and s not in out and any(p.search(s) for p in patterns):
                out.append(s if len(s) <= _MAX_QUOTE else s[: _MAX_QUOTE - 1] + "…")
                if len(out) >= limit:
                    return out
    return out


def _finding(f: VerifiedFinding) -> dict[str, str]:
    # Quote and source are verified; the note is the investigator's reading, NOT verified (a real
    # quote can come with a wrong conclusion), so it's labelled that way and shown last.
    return {
        "stance": f.stance,
        "quote": f.quote,
        "source": f.source_url,
        "read": f.fetched_at,
        "unverified_reading": f.note,
    }


def _capped(found: list[VerifiedFinding]) -> tuple[list[dict[str, str]], int]:
    """At most MAX_FINDINGS_PER_CAUSE, taking for and against in turn so neither side is hidden."""
    sides = [[f for f in found if f.stance == s] for s in ("supports", "contradicts")]
    picked: list[VerifiedFinding] = []
    for i in range(MAX_FINDINGS_PER_CAUSE):
        for side in sides:
            if i < len(side) and len(picked) < MAX_FINDINGS_PER_CAUSE:
                picked.append(side[i])
    return [_finding(f) for f in picked], len(found) - len(picked)


def candidate(
    d: Discrepancy,
    cause: Cause,
    evidence: tuple[str, ...],
    found: tuple[VerifiedFinding, ...] = (),
) -> dict[str, Any]:
    quotes = sentences_for(cause, evidence)
    researched, hidden = _capped([f for f in found if f.cause == cause.value])
    checks = cause_checks(d, cause, " ".join(quotes))
    against = [c for c in checks if c.status is CheckStatus.CONTRADICTED]
    fit: Fit
    if against:  # a rule rules it out even if the filing mentions it: that's the lesson
        fit, why = "ruled_out", against[0].detail
    elif not quotes and not any(f["stance"] == "supports" for f in researched):
        fit, why = "no_evidence", "not mentioned in the filing(s) where the number changed"
    elif all(c.status is CheckStatus.CONFIRMED for c in checks):
        fit, why = "fits", "; ".join(c.detail for c in checks[1:])
    else:
        fit, why = "possible", f"{cause.value} can move {d.metric.value}; no numeric check"
    return {
        "cause": cause.value,
        "fit": fit,
        "why": why,
        "evidence": quotes,
        # Verified findings, for and against, capped. They inform the reviewer; an investigator's
        # "contradicts" never rules a cause out (only code rules do).
        "research": researched,
        "research_not_shown": hidden,
    }


def build_card(
    d: Discrepancy,
    evidence: tuple[str, ...],
    research: Sequence[ResearchResult] = (),
    research_rounds_left: int = 0,
) -> dict[str, Any]:
    """Fits come from code rules only. Research findings are shown as evidence (and can move a
    cause from "no evidence" to "possible"), but never override a rule."""
    first, last = d.values[0][1], d.values[-1][1]
    pct = f" ({(last - first) / abs(first):+.1%})" if first else ""
    found = tuple(f for r in research for f in r.findings)
    return {
        "id": d.id,
        "change": f"{d.metric.value} FY{d.fiscal_year}: {first:,} -> {last:,}{pct}",
        "values": [[a, str(v)] for a, v in d.values],
        "changed_in": change_filings(d),
        "candidates": [candidate(d, c, evidence, found) for c in CANDIDATES],
        # Causes outside the four that research found (error correction, segment change, ...)
        "other_causes": [
            {"name": f.other_cause or "other", **_finding(f)}
            for f in [f for f in found if f.cause == "other"][: 2 * MAX_FINDINGS_PER_CAUSE]
        ],
        "research_rounds": [
            {
                "round": r.round,
                "hint": r.hint,
                "looked_at": r.looked_at,
                "open_questions": r.open_questions,
                "findings": len(r.findings),
                "discarded_unverified": r.dropped,
                "error": r.error,
            }
            for r in research
        ],
        "research_rounds_left": research_rounds_left,
        "decide": {
            "pick": "set correct_cause to the cause you judge right, and action to 'accept' "
            "(use the recast value) or 'use_original'",
            "escalate": "action 'escalate' if no candidate explains it; the number is published "
            "flagged as unresolved",
            "research": "action 'research' (note = optional hint, e.g. 'check for an 8-K') to "
            "have the research agent dig further and show this card again"
            + ("" if research_rounds_left > 0 else " (no rounds left)"),
        },
    }


def _research_lines(f: dict[str, str]) -> list[str]:
    return [
        f"> {f['quote']}",
        f"[source]({f['source']}), read {f['read']}",
        f"<details><summary>investigator's reading (unverified)</summary>"
        f"{f['unverified_reading']}</details>",
    ]


_MARK = {"fits": "✓ fits", "possible": "? possible", "ruled_out": "✗ ruled out", "no_evidence": "–"}


def render_card(card: dict[str, Any]) -> str:
    """Markdown for the terminal or a notebook."""
    lines = [f"### {card['id']}", f"**{card['change']}**  ", f"changed in: {card['changed_in']}"]
    lines += ["", "| cause | fits the numbers? | why |", "|---|---|---|"]
    for c in card["candidates"]:
        lines.append(f"| {c['cause']} | {_MARK[c['fit']]} | {c['why']} |")
    for c in card["candidates"]:
        if c["evidence"]:
            lines += ["", f"*{c['cause']}* in the filing:"]
            lines += [f"> {q}" for q in c["evidence"]]
        for f in c.get("research", []):
            lines += ["", f"*{c['cause']}*, research {f['stance']}:", *_research_lines(f)]
        if c.get("research_not_shown"):
            lines.append(f"({c['research_not_shown']} more findings for {c['cause']} not shown)")
    for o in card.get("other_causes", []):
        lines += ["", f"*other: {o['name']}*, research {o['stance']}:", *_research_lines(o)]
    for r in card.get("research_rounds", []):
        hint = f" (hint: {r['hint']})" if r["hint"] else ""
        lines += [
            "",
            f"Research round {r['round']}{hint}: {r['findings']} verified findings, "
            f"{r['discarded_unverified']} discarded as unverifiable. {r['looked_at']}",
        ]
        if r["open_questions"]:
            lines.append(f"Still open: {r['open_questions']}")
        if r["error"]:
            lines.append(f"Research failed: {r['error']}")
    left = card.get("research_rounds_left", 0)
    more = f", or research further ({left} rounds left)" if left else ""
    lines += ["", f"Decide: pick the cause (accept / use_original), escalate{more}."]
    return "\n".join(lines)
