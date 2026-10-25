"""Reconcile step: a number changed between two filings. Why, and which value do we use?

Same division of labour as extraction:
- code decides WHETHER to ask (materiality is arithmetic, not judgment): immaterial revisions
  are auto-accepted with no LLM call;
- code gathers EVIDENCE deterministically (keyword windows from the newer filing: stock split,
  discontinued operations, recast, reclassified, retrospective...);
- the LLM classifies the CAUSE from that evidence and must quote it;
- code VERIFIES the quote is really in the evidence, otherwise the answer is downgraded to
  "unknown" and goes to a human.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, cast

from botocore.exceptions import ClientError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from filings_agent.config import ModelSpec
from filings_agent.domain import METRIC_SPECS, Discrepancy
from filings_agent.llm.cache import StepCache
from filings_agent.steps.extract import MODEL_FAILURE_CODES, Usage
from filings_agent.textnorm import normalize

if TYPE_CHECKING:
    from filings_agent.steps.triage import Triage

MATERIALITY = Decimal("0.005")  # relative spread below this: immaterial, auto-accepted


class Cause(StrEnum):
    STOCK_SPLIT = "stock_split"
    DISCONTINUED_OPERATIONS = "discontinued_operations"  # spin-offs, divestitures
    ACCOUNTING_CHANGE = "accounting_change"  # new standard adopted retrospectively
    RECLASSIFICATION = "reclassification"  # items moved between lines/segments
    IMMATERIAL_REVISION = "immaterial_revision"
    UNKNOWN = "unknown"


class ReconcileOutput(BaseModel):
    cause: Cause
    explanation: str = Field(description="One or two sentences, plain English.")
    evidence_quote: str = Field(
        description="A short verbatim quote (under 300 characters) from the evidence that "
        "supports the cause. Copy it exactly. Empty only if cause is 'unknown'."
    )


def spread(d: Discrepancy) -> Decimal:
    vals = [v for _, v in d.values]
    biggest = max(abs(v) for v in vals)
    return (max(vals) - min(vals)) / biggest if biggest else Decimal(0)


def is_material(d: Discrepancy) -> bool:
    return spread(d) >= MATERIALITY


# ---- deterministic evidence gathering ---------------------------------------------------------

# One pattern family per cause, so every cause gets a chance to be represented.
EVIDENCE_FAMILIES: dict[str, list[str]] = {
    # Filings arrive as markdown with escaped hyphens and stray spaces: "20 \-for\-1 stock
    # split" (Amazon, Chipotle). A plain [- ] never matched those.
    "stock_split": [
        r"\w+\s*\\?[-\u2010\u2011 ]\s*for\s*\\?[-\u2010\u2011 ]\s*\w+"
        r"\s+(?:forward\s+)?(?:stock\s+)?split",
        r"stock\s+split",
    ],
    "discontinued_operations": [
        r"spin[- ]?off|separation\s+of\s+(?:the|our)",
        r"discontinued\s+operations\s+for\s+all\s+periods",
        r"discontinued\s+operations",
    ],
    "accounting_change": [
        r"adopted\s+(?:ASU|Accounting\s+Standards\s+Update|the\s+new\s+(?:accounting|standard))",
        r"retrospective(?:ly)?\s+(?:basis|adopt|applied|method|transition)",
    ],
    "reclassification": [
        r"(?:recast|reclassif)\w*[^.]{0,160}(?:conform|current\s+(?:year|period)\s+presentation)",
        r"reclassif",
    ],
}
_WINDOW = 450


def _is_boilerplate(snippet: str) -> bool:
    """Tables of contents and number tables: lots of pipes/digits, few words."""
    letters = sum(c.isalpha() for c in snippet)
    return snippet.count("|") > 12 or letters < 0.5 * len(snippet)


def gather_evidence(newer_filing_md: str, max_snippets: int = 8, per_family: int = 2) -> list[str]:
    """Up to `per_family` text windows per cause family, taken round-robin.

    v1 filled every slot from the first family that matched; GE mentions "discontinued
    operations" everywhere, so its accounting-standard change (ASU 2018-12) never reached the
    model and net income was misattributed. Round-robin gives each cause a fair shot.
    """
    taken: list[tuple[int, int]] = []
    by_family: dict[str, list[str]] = {f: [] for f in EVIDENCE_FAMILIES}
    for family, patterns in EVIDENCE_FAMILIES.items():
        for pat in patterns:
            for m in re.finditer(pat, newer_filing_md, re.I):
                if len(by_family[family]) >= per_family:
                    break
                a = max(0, m.start() - _WINDOW)
                b = min(len(newer_filing_md), m.end() + _WINDOW)
                if any(a < tb and ta < b for ta, tb in taken):
                    continue
                snippet = re.sub(r"\s+", " ", newer_filing_md[a:b]).strip()
                if _is_boilerplate(snippet):
                    continue
                taken.append((a, b))
                by_family[family].append(snippet)
    out: list[str] = []
    for i in range(per_family):  # interleave so truncation never drops a whole family
        out.extend(snips[i] for snips in by_family.values() if len(snips) > i)
    return out[:max_snippets]


def change_filings(d: Discrepancy) -> list[str]:
    """Filings where the value actually changed vs the previous filing.

    The newest filing alone isn't enough: GE's FY2022 net income went 225 -> 339 in the 2024
    10-K (accounting change) and only 339 -> 336 in the 2025 one. Evidence from the 2025 filing
    can't explain the material change.
    """
    return [d.values[i][0] for i in range(1, len(d.values)) if d.values[i][1] != d.values[i - 1][1]]


def evidence_for(
    d: Discrepancy, markdown_of: Callable[[str], str], max_snippets: int = 8
) -> tuple[str, ...]:
    """Evidence from every filing where the number changed, interleaved, deduplicated."""
    per_filing = [gather_evidence(markdown_of(a), max_snippets) for a in change_filings(d)]
    out: list[str] = []
    for i in range(max_snippets):
        for snippets in per_filing:
            if i < len(snippets) and snippets[i] not in out:
                out.append(snippets[i])
    return tuple(out[:max_snippets])


def _norm(s: str) -> str:
    return normalize(s)


QUOTE_SPAN = 800  # a stitched quote's fragments must all sit within one passage of this many chars
_FRAGMENT_SPLIT = re.compile(r"(?<=[.;:])\s+|\s*(?:\.\.\.|…)\s*")


def verify_quote(quote: str, evidence: list[str]) -> bool:
    """True if the quote is verbatim evidence.

    Models often stitch two real sentences and drop one in between (seen with GE: "...(GE
    HealthCare). The historical results..." skipping "See Notes 2 and 3..."). So a quote may be
    several fragments, but EVERY fragment must appear word for word, in order, and all of them
    within one passage (QUOTE_SPAN characters). Stitching nearby real text passes; inventing
    text, even a short inserted sentence ("It was not."), fails, and so does gluing together
    sentences from far apart in a 10-K (found by review: short fragments used to be skipped and
    fragments could come from anywhere in the document).
    """
    q = _norm(quote)
    if len(q) < 12:
        return False
    fragments = [f for f in (_norm(x) for x in _FRAGMENT_SPLIT.split(q)) if f]
    if not fragments:
        return False
    for e in (_norm(x) for x in evidence):
        start = e.find(fragments[0])
        while start >= 0:
            pos, ok = start + len(fragments[0]), True
            for frag in fragments[1:]:
                found = e.find(frag, pos)
                if found < 0 or found + len(frag) - start > QUOTE_SPAN:
                    ok = False
                    break
                pos = found + len(frag)
            if ok and pos - start <= QUOTE_SPAN:
                return True
            start = e.find(fragments[0], start + 1)
    return False


# ---- step I/O ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconcileTask:
    discrepancy_id: str
    discrepancy: Discrepancy
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class Resolution:
    discrepancy_id: str
    cause: Cause
    explanation: str
    evidence_quote: str
    material: bool
    verified: bool  # quote found in evidence (always True for auto-accepted immaterial)
    use_value: Decimal  # value from the newest filing: comparisons use as-recast figures
    use_accession: str
    needs_review: bool
    usage: Usage | None = None
    error: str | None = None
    triage: Triage | None = None  # set after reconcile (steps/triage.py); None = not triaged


def auto_resolution(discrepancy_id: str, d: Discrepancy) -> Resolution:
    accn, value = d.values[-1]
    return Resolution(
        discrepancy_id=discrepancy_id,
        cause=Cause.IMMATERIAL_REVISION,
        explanation=f"Spread {spread(d):.3%} is under the {MATERIALITY:.1%} materiality threshold.",
        evidence_quote="",
        material=False,
        verified=True,
        use_value=value,
        use_accession=accn,
        needs_review=False,
    )


class Reconciler(Protocol):
    name: str

    def reconcile(self, task: ReconcileTask, rep: int = 0) -> Resolution: ...


PROMPTS: dict[str, str] = {
    "v1": (
        "A company reported the same number differently in two of its annual reports. Decide why,"
        " using ONLY the evidence excerpts from the newer report.\n"
        "Causes:\n"
        "- stock_split: share counts changed by a split ratio (e.g. x10, x3);"
        " dollar amounts don't.\n"
        "- discontinued_operations: a business was spun off or sold and prior years were recast"
        " to exclude it.\n"
        "- accounting_change: a new accounting standard was adopted retrospectively.\n"
        "- reclassification: amounts were moved between lines or segments to conform to the"
        " current presentation.\n"
        "- unknown: the evidence doesn't explain the change. Prefer unknown over guessing.\n"
        "Quote the evidence exactly."
    ),
    # v2: v1 quotes stitched sentences together (dropping text in between), which the verifier
    # rejects. Ask for one contiguous passage.
    "v2": (
        "A company reported the same number differently in two of its annual reports. Decide why,"
        " using ONLY the evidence excerpts from the newer report.\n"
        "Causes:\n"
        "- stock_split: share counts changed by a split ratio (e.g. x10, x3);"
        " dollar amounts don't.\n"
        "- discontinued_operations: a business was spun off or sold and prior years were recast"
        " to exclude it.\n"
        "- accounting_change: a new accounting standard was adopted retrospectively.\n"
        "- reclassification: amounts were moved between lines or segments to conform to the"
        " current presentation.\n"
        "- unknown: the evidence doesn't explain the change. Prefer unknown over guessing.\n"
        "evidence_quote must be ONE contiguous passage copied character for character from a"
        " single excerpt (one or two sentences, under 300 characters). Do not join sentences"
        " from different places or skip words."
    ),
}
# v3: from a production reviewer correction (GE net income: agent said spin-off, reviewer said
# accounting change). The added rule is general accounting, not company-specific.
PROMPTS["v3"] = PROMPTS["v2"] + (
    "\nAccounting rule: net income attributable to the company INCLUDES discontinued"
    " operations, so a spin-off or sale by itself does not change prior years' net income"
    " (it moves amounts between continuing and discontinued lines). If net income itself was"
    " restated, look for an accounting standard adopted retrospectively (e.g. an ASU) and"
    " prefer accounting_change when the evidence shows one. Revenue, operating income, cash and"
    " debt DO change with a spin-off."
)


def build_messages(task: ReconcileTask, prompt: str) -> list[SystemMessage | HumanMessage]:
    d = task.discrepancy
    vals = "\n".join(f"- filing {a}: {v:,}" for a, v in d.values)
    ev = "\n\n".join(f"[{i + 1}] {e}" for i, e in enumerate(task.evidence)) or "(none found)"
    human = (
        f"Company: {d.ticker}\nMetric: {d.metric.value} ({METRIC_SPECS[d.metric].definition})\n"
        f"Fiscal period ending: {d.period_end.isoformat()}\n"
        f"Reported values, oldest filing first:\n{vals}\n"
        f"Ratio newest/oldest: {d.ratio:.4f}\n\n"
        f"Evidence from the newest filing:\n{ev}"
    )
    return [SystemMessage(PROMPTS[prompt]), HumanMessage(human)]


class LLMReconciler:
    def __init__(
        self,
        model_name: str,
        spec: ModelSpec,
        chat: BaseChatModel,
        cache: StepCache,
        prompt_version: str,
    ) -> None:
        self.name = f"reconcile/{model_name}/{prompt_version}"
        self._model_name = model_name
        self._spec = spec
        self._structured = chat.with_structured_output(ReconcileOutput, include_raw=True)
        self._cache = cache
        self._prompt = prompt_version

    def _call(self, task: ReconcileTask, rep: int) -> dict[str, Any]:
        key = StepCache.key(
            step="reconcile",
            model=self._spec.id,
            prompt=self._prompt,
            schema=ReconcileOutput.model_json_schema(),
            discrepancy=task.discrepancy.model_dump(mode="json"),
            evidence=task.evidence,
            rep=rep,
        )
        if (hit := self._cache.get(key)) is not None:
            return {**hit, "cached": True}
        t0 = time.monotonic()
        try:
            d = task.discrepancy
            raw = cast(
                dict[str, Any],
                self._structured.invoke(
                    build_messages(task, self._prompt),
                    config={
                        "run_name": f"reconcile {d.id}",
                        "tags": [
                            "step:reconcile",
                            f"model:{self._model_name}",
                            f"ticker:{d.ticker}",
                        ],
                        "metadata": {
                            "step": "reconcile",
                            "discrepancy_id": d.id,
                            "ticker": d.ticker,
                            "metric": d.metric.value,
                            "model": self._model_name,
                            "prompt_version": self._prompt,
                            "rep": rep,
                        },
                    },
                ),
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in MODEL_FAILURE_CODES:
                raise
            return {
                "parsed": None,
                "error": f"model_error:{code}",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_s": time.monotonic() - t0,
                "cached": False,
            }
        msg = cast(AIMessage, raw["raw"])
        usage = msg.usage_metadata or {"input_tokens": 0, "output_tokens": 0}
        parsed = cast(ReconcileOutput | None, raw.get("parsed"))
        record: dict[str, Any] = {
            "parsed": parsed.model_dump(mode="json") if parsed else None,
            "error": str(raw.get("parsing_error")) if parsed is None else None,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "latency_s": time.monotonic() - t0,
        }
        self._cache.put(key, record)
        return {**record, "cached": False}

    def reconcile(self, task: ReconcileTask, rep: int = 0) -> Resolution:
        d = task.discrepancy
        rec = self._call(task, rep)
        usage = Usage(
            model=self._model_name,
            input_tokens=int(rec["input_tokens"]),
            output_tokens=int(rec["output_tokens"]),
            cost_usd=self._spec.cost(int(rec["input_tokens"]), int(rec["output_tokens"])),
            latency_s=float(rec["latency_s"]),
            cached=bool(rec["cached"]),
        )
        accn, value = d.values[-1]
        if rec["parsed"] is None:
            return Resolution(
                task.discrepancy_id, Cause.UNKNOWN, "Model returned no usable answer.", "",
                True, False, value, accn, True, usage, rec["error"],
            )  # fmt: skip
        out = ReconcileOutput.model_validate(rec["parsed"])
        verified = out.cause is Cause.UNKNOWN or verify_quote(
            out.evidence_quote, list(task.evidence)
        )
        cause = out.cause if verified else Cause.UNKNOWN
        return Resolution(
            discrepancy_id=task.discrepancy_id,
            cause=cause,
            explanation=out.explanation if verified else f"Unverified claim: {out.explanation}",
            evidence_quote=out.evidence_quote,
            material=True,
            verified=verified,
            use_value=value,
            use_accession=accn,
            needs_review=True,  # every material change gets a human decision
            usage=usage,
        )
