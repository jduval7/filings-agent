"""Extraction step: one financial statement in, verified facts out.

Division of labour: the LLM *reads* (which row, which column, what's printed, what scale applies);
code does the *math* (parse "(1,234)", apply the scale) and *checks* (the printed number and row
label must literally appear in the statement). A cheap model can misread, but it can't
miscalculate, and every misread that invents text gets caught.

`Extractor` is a Protocol so strategies are swappable: per-statement LLM extraction (default),
whole-filing single call (the baseline), and whatever the sweep wants to try next.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol, cast

from botocore.exceptions import ClientError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from filings_agent.config import ModelSpec
from filings_agent.domain import METRIC_SPECS, ExtractedFact, Metric, Unit, fiscal_year_of
from filings_agent.llm.cache import StepCache
from filings_agent.sec.sections import SCALE_FACTOR, Scale
from filings_agent.textnorm import normalize

# Bedrock error codes that mean "the model produced bad output", as opposed to throttling/outages.
MODEL_FAILURE_CODES = frozenset({"ModelErrorException", "ValidationException"})

# ---- what the LLM returns ---------------------------------------------------------------------


class ReportedValue(BaseModel):
    metric: Metric
    row_label: str = Field(description="The row label exactly as printed, e.g. 'Total revenues'.")
    printed_value: str = Field(
        description="The number exactly as printed in the cell, e.g. '215,938' or '(1,234)' or "
        "'1,609.4'. Copy it; do not convert or rescale it."
    )
    period_end: date = Field(
        description="End date of the fiscal year (or balance-sheet date) for this column, "
        "formatted YYYY-MM-DD."
    )
    scale: Scale = Field(
        description="Scale that applies to THIS row: usually the statement's '(In millions)', but "
        "some rows override it, e.g. share counts 'in thousands'. 'units' if not scaled."
    )


class ExtractionOutput(BaseModel):
    values: list[ReportedValue] = Field(
        description="One entry per requested metric per fiscal year shown in the statement."
    )
    not_reported: list[Metric] = Field(
        default_factory=lambda: list[Metric](),
        description="Requested metrics that do not appear on this statement.",
    )


# ---- step input/output ------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractTask:
    task_id: str
    ticker: str
    accession_no: str
    fiscal_year_end: date
    statement: str
    section_text: str
    metrics: tuple[Metric, ...]


@dataclass(frozen=True)
class Rejection:
    metric: str
    printed_value: str
    row_label: str
    reason: str


@dataclass(frozen=True)
class Usage:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    cached: bool


@dataclass
class ExtractResult:
    task_id: str
    facts: list[ExtractedFact] = field(default_factory=lambda: list[ExtractedFact]())
    rejected: list[Rejection] = field(default_factory=lambda: list[Rejection]())
    not_reported: list[Metric] = field(default_factory=lambda: list[Metric]())
    usage: Usage | None = None
    error: str | None = None  # schema/parse failure: the model didn't return usable output


class Extractor(Protocol):
    name: str

    def extract(self, task: ExtractTask, rep: int = 0) -> ExtractResult: ...


# ---- deterministic parts: parse, scale, verify ------------------------------------------------

_NUM = re.compile(r"\(?-?\$?\s*\d[\d,]*(?:\.\d+)?\)?")


def parse_printed(printed: str) -> Decimal | None:
    """'(1,234)' -> -1234, '$ 215,938' -> 215938, '1,609.4' -> 1609.4, '—' -> None."""
    s = printed.strip().replace("$", "").replace(" ", "")
    # "(41" happens too: some filings put the closing paren in the next cell (Microsoft)
    negative = s.startswith(("(", "-"))
    digits = s.strip("()-").replace(",", "")
    if not digits:
        return None
    try:
        value = Decimal(digits)
    except InvalidOperation:
        return None
    return -value if negative else value


def _norm(text: str) -> str:
    return normalize(text)


def verify(rv: ReportedValue, section_text: str) -> str | None:
    """Reason the value is rejected, or None if it checks out against the source text.

    Checked: the number as printed (as a whole number: "234" doesn't match inside "1,234"), its
    sign (parentheses mean negative), the row label, and for dollar figures the statement's scale.
    Not checked: that number and label sit on the same row and in the right year column.
    """
    printed = rv.printed_value.strip().replace("$", "").strip()
    negative = printed.startswith("(") or printed.startswith("-")
    core = printed.strip("()").lstrip("-").strip()
    if not core:
        return f"printed value {rv.printed_value!r} not found in statement"
    number = rf"(?<![\d.,]){re.escape(core)}(?![\d]|[.,]\d)"
    bare = [m for m in re.finditer(number, section_text) if not _in_parens(section_text, m)]
    in_parens = [m for m in re.finditer(number, section_text) if _in_parens(section_text, m)]
    if not bare and not in_parens:
        return f"printed value {rv.printed_value!r} not found in statement"
    if negative and not in_parens and not re.search(rf"-\s*\$?\s*{re.escape(core)}", section_text):
        return (
            f"printed value {rv.printed_value!r} is negative but the statement prints it positive"
        )
    if not negative and not bare:
        return f"printed value {rv.printed_value!r} is positive but the statement prints ({core})"
    if _norm(rv.row_label) not in _norm(section_text):
        return f"row label {rv.row_label!r} not found in statement"
    stated = _SCALE_MARK.search(section_text[:_SCALE_WINDOW])
    if (
        stated
        and METRIC_SPECS[rv.metric].unit is Unit.USD  # share rows often carry their own scale
        and rv.scale != Scale(stated.group(1).lower())
    ):
        return f"scale {rv.scale} but the statement is in {stated.group(1).lower()}"
    return None


_SCALE_MARK = re.compile(r"(?:\bin\s+|\(\s*)(thousands|millions|billions)\b", re.I)
_SCALE_WINDOW = 1_200


def _in_parens(text: str, m: re.Match[str]) -> bool:
    """Is this occurrence printed as a negative, "(1,234)" or "( 1,234 )" or "$ (1,234)"?"""
    before = text[max(0, m.start() - 4) : m.start()].rstrip(" $")
    after = text[m.end() : m.end() + 3].lstrip(" ")
    return before.endswith("(") and after.startswith(")")


# ---- the LLM strategy -------------------------------------------------------------------------

PROMPTS: dict[str, str] = {
    "v1": (
        "You extract numbers from one financial statement of a company's annual report (10-K).\n"
        "Rules:\n"
        "- Report each requested metric for EVERY fiscal year column shown.\n"
        "- Copy printed_value exactly as printed (keep parentheses for negatives); never compute.\n"
        "- period_end is the column's fiscal year-end date as YYYY-MM-DD. If a column header shows"
        " only a year, use the company's fiscal year-end month and day.\n"
        "- scale is what applies to that row. Watch for rows that override the heading (e.g."
        " shares 'in thousands' inside a table 'in millions').\n"
        "- If a metric is not on this statement, list it in not_reported. Never guess or use a"
        " different line as a substitute."
    ),
    # v2: written from the v1 dev-split failure analysis (DECISIONS.md). General rules only, no
    # company-specific hints, so it can't overfit dev companies.
    "v2": (
        "You extract numbers from one financial statement of a company's annual report (10-K).\n"
        "Rules:\n"
        "1. Columns: return one entry per requested metric for EVERY fiscal-year column in the"
        " statement (income and cash-flow statements usually show 3 years, balance sheets 2)."
        " Missing a prior-year column is an error.\n"
        "2. Copy printed_value exactly as printed (keep parentheses for negatives);"
        " never compute.\n"
        "3. period_end is that column's fiscal year-end date as YYYY-MM-DD. If a header shows"
        " only a year, use the fiscal year-end month and day of this report.\n"
        "4. Scale: the heading's scale (e.g. 'In millions') applies to EVERY row, share counts"
        " included, unless the heading or row explicitly gives shares their own scale (e.g."
        " 'shares in thousands'). 'Except per share data/amounts' means per-share DOLLAR amounts"
        " (EPS) are not scaled; share COUNTS still use the heading's scale.\n"
        "5. net_income means net income attributable to the company itself (often 'Net income"
        " attributable to <Company>'). Not the consolidated total that includes noncontrolling"
        " interests, and not the amount available to common shareholders after preferred"
        " dividends. If only one net income line exists, use it.\n"
        "6. If a metric is not on this statement, list it in not_reported. Never substitute a"
        " different line."
    ),
}


def _metric_lines(metrics: tuple[Metric, ...]) -> str:
    return "\n".join(f"- {m.value}: {METRIC_SPECS[m].definition}" for m in metrics)


def build_messages(task: ExtractTask, prompt_version: str) -> list[SystemMessage | HumanMessage]:
    human = (
        f"Company: {task.ticker}\n"
        f"Fiscal year of this report ended: {task.fiscal_year_end.isoformat()}\n"
        f"Statement: {task.statement}\n\n"
        f"Metrics to extract:\n{_metric_lines(task.metrics)}\n\n"
        f"Statement text:\n{task.section_text}"
    )
    return [SystemMessage(PROMPTS[prompt_version]), HumanMessage(human)]


class LLMStatementExtractor:
    """Default strategy: one structured-output call per statement."""

    def __init__(
        self,
        model_name: str,
        spec: ModelSpec,
        chat: BaseChatModel,
        cache: StepCache,
        prompt_version: str,
    ) -> None:
        self.name = f"statement/{model_name}/{prompt_version}"
        self._model_name = model_name
        self._spec = spec
        self._structured = chat.with_structured_output(ExtractionOutput, include_raw=True)
        self._cache = cache
        self._prompt_version = prompt_version

    def _call(self, task: ExtractTask, rep: int) -> dict[str, Any]:
        key = StepCache.key(
            step="extract",
            model=self._spec.id,
            prompt=self._prompt_version,
            schema=ExtractionOutput.model_json_schema(),
            text=task.section_text,
            metrics=[m.value for m in task.metrics],
            fy_end=task.fiscal_year_end,
            rep=rep,
        )
        if (hit := self._cache.get(key)) is not None:
            return {**hit, "cached": True}
        t0 = time.monotonic()
        try:
            raw = cast(
                dict[str, Any],
                self._structured.invoke(
                    build_messages(task, self._prompt_version),
                    config={
                        # Filterable in LangSmith: "every GE balance sheet on gpt-oss-120b".
                        "run_name": f"extract {task.ticker} {task.statement}",
                        "tags": [
                            "step:extract",
                            f"model:{self._model_name}",
                            f"ticker:{task.ticker}",
                        ],
                        "metadata": {
                            "step": "extract",
                            "ticker": task.ticker,
                            "accession_no": task.accession_no,
                            "statement": task.statement,
                            "model": self._model_name,
                            "prompt_version": self._prompt_version,
                            "rep": rep,
                        },
                    },
                ),
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in MODEL_FAILURE_CODES:
                raise  # throttling/outages: the caller's retry policy handles these
            # The model itself failed (e.g. malformed tool call). That's a quality outcome to
            # measure, not a crash. Not cached, so a later run gets a fresh attempt.
            return {
                "parsed": None,
                "error": f"model_error:{code}",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_s": time.monotonic() - t0,
                "cached": False,
            }
        latency = time.monotonic() - t0
        msg = cast(AIMessage, raw["raw"])
        usage = msg.usage_metadata or {"input_tokens": 0, "output_tokens": 0}
        parsed = cast(ExtractionOutput | None, raw.get("parsed"))
        record: dict[str, Any] = {
            "parsed": parsed.model_dump(mode="json") if parsed else None,
            "error": str(raw.get("parsing_error")) if parsed is None else None,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "latency_s": latency,
        }
        self._cache.put(key, record)
        return {**record, "cached": False}

    def extract(self, task: ExtractTask, rep: int = 0) -> ExtractResult:
        rec = self._call(task, rep)
        result = ExtractResult(
            task_id=task.task_id,
            usage=Usage(
                model=self._model_name,
                input_tokens=int(rec["input_tokens"]),
                output_tokens=int(rec["output_tokens"]),
                cost_usd=self._spec.cost(int(rec["input_tokens"]), int(rec["output_tokens"])),
                latency_s=float(rec["latency_s"]),
                cached=bool(rec["cached"]),
            ),
        )
        if rec["parsed"] is None:
            result.error = rec["error"] or "no structured output"
            return result
        out = ExtractionOutput.model_validate(rec["parsed"])
        result.not_reported = [m for m in out.not_reported if m in task.metrics]
        seen: set[str] = set()
        for rv in out.values:
            if rv.metric not in task.metrics:
                continue
            reason = verify(rv, task.section_text)
            number = parse_printed(rv.printed_value)
            if reason is None and number is None:
                reason = f"unparseable value {rv.printed_value!r}"
            if reason is not None or number is None:
                result.rejected.append(
                    Rejection(rv.metric.value, rv.printed_value, rv.row_label, reason or "")
                )
                continue
            fact = ExtractedFact(
                ticker=task.ticker,
                metric=rv.metric,
                fiscal_year=fiscal_year_of(rv.period_end),
                period_end=rv.period_end,
                value=number * SCALE_FACTOR[rv.scale],
                accession_no=task.accession_no,
                statement=task.statement,
                row_label=rv.row_label,
                printed_value=rv.printed_value,
                scale=rv.scale.value,
                model=self._model_name,
            )
            if fact.id in seen:  # model repeated itself: first answer wins
                continue
            seen.add(fact.id)
            result.facts.append(fact)
        return result
