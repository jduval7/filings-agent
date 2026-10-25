"""LLM-as-judge, only where code can't grade.

Numbers, quotes and causes have right answers: code grades those (exact, free, deterministic).
What's left is judgment: does an explanation actually justify its cause; is a memo's prose faithful
to its table, are its caveats right, could an analyst act on it. That's what the judge grades.

Design choices, each for reliability:
- **Binary criteria with a reason each**, not a 1-10 score. Yes/no questions are easier to answer
  consistently and to calibrate against a human.
- **The judge is never the model that wrote the output** (enforced in config validation).
- **Pairwise comparisons run in both orders**; a winner only counts if it wins both ways, which
  cancels position bias.
- **The judge must be calibrated** against human labels before its scores are trusted (see
  evals/calibration.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from filings_agent.config import ModelSpec
from filings_agent.llm.cache import StepCache
from filings_agent.steps.extract import Usage

# ---- rubrics ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Criterion:
    name: str
    question: str


@dataclass(frozen=True)
class Rubric:
    name: str
    version: str
    criteria: tuple[Criterion, ...]

    def names(self) -> list[str]:
        return [c.name for c in self.criteria]


RECONCILE_RUBRIC = Rubric(
    name="reconcile_explanation",
    version="v1",
    criteria=(
        Criterion(
            "supported",
            "Is every factual claim in the explanation supported by the evidence excerpts? "
            "(Background knowledge not in the evidence = fail.)",
        ),
        Criterion(
            "specific",
            "Does the explanation name the specific event (e.g. the split ratio, the business "
            "spun off, the accounting standard) rather than a generic reason?",
        ),
        Criterion(
            "consistent",
            "Is the chosen cause label consistent with the explanation and with the size and "
            "direction of the change (e.g. a ~10x share change for a 10-for-1 split)?",
        ),
    ),
)

MEMO_RUBRIC = Rubric(
    name="memo",
    version="v1",
    criteria=(
        Criterion(
            "faithful",
            "Are all qualitative claims (trends, rankings, 'strongest', 'declined') supported by "
            "the table? Claims about causes, history or outlook not in the table = fail.",
        ),
        Criterion(
            "caveats_correct",
            "Are restated figures (the changed_by column) mentioned in the caveats, with the "
            "right reason, and nothing restated that wasn't?",
        ),
        Criterion(
            "comparable",
            "Are comparisons fair: same metric, same basis, no comparing a company's figures "
            "across different reports or mixing units?",
        ),
        Criterion(
            "actionable",
            "Could an analyst act on it: the headline comparison is clear in the first two "
            "sentences, it's concise, and it's organized?",
        ),
    ),
)

RUBRICS = {r.name: r for r in (RECONCILE_RUBRIC, MEMO_RUBRIC)}


# ---- outputs ----------------------------------------------------------------------------------


class CriterionVerdict(BaseModel):
    criterion: str
    reason: str = Field(description="One sentence citing the specific text that decided it.")
    passed: bool


class JudgeOutput(BaseModel):
    verdicts: list[CriterionVerdict]


class PairwiseOutput(BaseModel):
    reason: str = Field(description="Two sentences: the decisive difference.")
    winner: Literal["A", "B", "tie"]


@dataclass(frozen=True)
class JudgeResult:
    rubric: str
    passed: dict[str, bool]  # criterion -> pass
    reasons: dict[str, str]
    usage: Usage
    error: str | None = None

    @property
    def pass_rate(self) -> float:
        return sum(self.passed.values()) / len(self.passed) if self.passed else 0.0


@dataclass(frozen=True)
class PairwiseResult:
    winner: Literal["A", "B", "tie"]
    consistent: bool  # same winner in both orders
    reasons: tuple[str, str]
    usage: Usage


# ---- the judge --------------------------------------------------------------------------------

SYSTEM = (
    "You are a strict reviewer of an AI system's output for financial analysts. Grade ONLY "
    "against the criteria given, using ONLY the material provided. For each criterion give a "
    "one-sentence reason citing the specific text that decided it, then pass true/false. When "
    "in doubt, fail: a false pass is worse than a false fail."
)

PAIRWISE_SYSTEM = (
    "You compare two memos written from the same table of figures for an investment analyst. "
    "Prefer the memo that is more faithful to the table, gets restatement caveats right, and is "
    "clearer and more concise. Length is not quality. Answer A, B, or tie."
)


class LLMJudge:
    def __init__(self, model_name: str, spec: ModelSpec, chat: BaseChatModel, cache: StepCache):
        self.model_name = model_name
        self._spec = spec
        self._grade = chat.with_structured_output(JudgeOutput, include_raw=True)
        self._compare = chat.with_structured_output(PairwiseOutput, include_raw=True)
        self._cache = cache

    def _invoke(
        self, runnable: Any, messages: list[Any], key: str, run_name: str
    ) -> tuple[dict[str, Any] | None, Usage]:
        hit = self._cache.get(key)
        if hit is not None:
            u = hit["usage"]
            usage = Usage(
                self.model_name,
                u["input_tokens"],
                u["output_tokens"],
                u["cost_usd"],
                u["latency_s"],
                cached=True,
            )
            return hit["parsed"], usage
        t0 = time.monotonic()
        raw = cast(
            dict[str, Any],
            runnable.invoke(
                messages,
                config={"run_name": run_name, "tags": ["judge", f"model:{self.model_name}"]},
            ),
        )
        latency = time.monotonic() - t0
        meta = cast(AIMessage, raw["raw"]).usage_metadata or {"input_tokens": 0, "output_tokens": 0}
        in_tok, out_tok = meta["input_tokens"], meta["output_tokens"]
        usage = Usage(
            self.model_name,
            in_tok,
            out_tok,
            self._spec.cost(in_tok, out_tok),
            latency,
            cached=False,
        )
        parsed_model = cast(BaseModel | None, raw.get("parsed"))
        parsed = parsed_model.model_dump(mode="json") if parsed_model else None
        if parsed is not None:
            self._cache.put(
                key,
                {
                    "parsed": parsed,
                    "usage": {
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "cost_usd": usage.cost_usd,
                        "latency_s": latency,
                    },
                },
            )
        return parsed, usage

    def grade(self, rubric: Rubric, material: str, output: str) -> JudgeResult:
        criteria = "\n".join(f"- {c.name}: {c.question}" for c in rubric.criteria)
        messages = [
            SystemMessage(SYSTEM),
            HumanMessage(
                f"Criteria (answer every one, using these exact names):\n{criteria}\n\n"
                f"Material the output was based on:\n{material}\n\nOutput to grade:\n{output}"
            ),
        ]
        key = StepCache.key(
            step="judge",
            model=self._spec.id,
            rubric=rubric.name,
            version=rubric.version,
            material=material,
            output=output,
        )
        parsed, usage = self._invoke(self._grade, messages, key, f"judge {rubric.name}")
        if parsed is None:  # one retry: gpt-oss occasionally skips the tool call (~1%)
            parsed, usage = self._invoke(self._grade, messages, key, f"judge {rubric.name} retry")
        if parsed is None:
            return JudgeResult(rubric.name, {}, {}, usage, error="no structured output")
        out = JudgeOutput.model_validate(parsed)
        by_name = {v.criterion: v for v in out.verdicts}
        missing = [n for n in rubric.names() if n not in by_name]
        if missing:  # an incomplete verdict is a failed grading, not a partial pass
            return JudgeResult(rubric.name, {}, {}, usage, error=f"missing criteria: {missing}")
        return JudgeResult(
            rubric.name,
            {n: by_name[n].passed for n in rubric.names()},
            {n: by_name[n].reason for n in rubric.names()},
            usage,
        )

    def compare(self, material: str, a: str, b: str) -> PairwiseResult:
        """Run both orders; a winner counts only if it wins both ways (cancels position bias)."""

        def ask(first: str, second: str, tag: str) -> tuple[PairwiseOutput | None, Usage]:
            messages = [
                SystemMessage(PAIRWISE_SYSTEM),
                HumanMessage(f"Table:\n{material}\n\nMemo A:\n{first}\n\nMemo B:\n{second}"),
            ]
            key = StepCache.key(
                step="pairwise", model=self._spec.id, material=material, a=first, b=second
            )
            parsed, usage = self._invoke(self._compare, messages, key, f"pairwise memo {tag}")
            return (PairwiseOutput.model_validate(parsed) if parsed else None), usage

        ab, u1 = ask(a, b, "A-B")
        ba, u2 = ask(b, a, "B-A")
        usage = Usage(
            self.model_name,
            u1.input_tokens + u2.input_tokens,
            u1.output_tokens + u2.output_tokens,
            u1.cost_usd + u2.cost_usd,
            u1.latency_s + u2.latency_s,
            u1.cached and u2.cached,
        )
        first = ab.winner if ab else "tie"
        swapped = {"A": "B", "B": "A", "tie": "tie"}[ba.winner] if ba else "tie"
        consistent = first == swapped
        winner: Literal["A", "B", "tie"] = first if consistent else "tie"
        return PairwiseResult(
            winner, consistent, (ab.reason if ab else "", ba.reason if ba else ""), usage
        )
