"""Deterministic scoring for the extraction step. No LLM in the grading path.

Facts are matched on (metric, fiscal year). Outcomes per requested metric/year:
  correct       value equals the answer key exactly
  wrong_value   right metric/year, wrong number (misread, wrong scale, wrong line)
  missing       expected but not produced
  hallucinated  produced a value for a metric that isn't on the statement (should abstain)
Extra years not in the key (e.g. a 4th column) are ignored rather than penalised.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any

from filings_agent.domain import Metric, fiscal_year_of


@dataclass(frozen=True)
class ExtractScore:
    expected: int
    correct: int
    wrong_value: int
    missing: int
    hallucinated: int
    abstain_expected: int
    abstain_correct: int
    rejected: int
    schema_error: bool

    @property
    def precision(self) -> float:
        denom = self.correct + self.wrong_value + self.hallucinated
        return self.correct / denom if denom else (1.0 if not self.schema_error else 0.0)

    @property
    def recall(self) -> float:
        return self.correct / self.expected if self.expected else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "precision": self.precision, "recall": self.recall, "f1": self.f1}


def score_extraction(
    expected: list[dict[str, Any]],
    expect_absent: list[str],
    predicted: list[dict[str, Any]],
    rejected: int = 0,
    schema_error: bool = False,
) -> ExtractScore:
    """`expected`: [{metric, period_end, value}]; `predicted`: [{metric, fiscal_year, value}]."""
    key: dict[tuple[str, int], Decimal] = {
        (str(e["metric"]), fiscal_year_of(_as_date(e["period_end"]))): Decimal(str(e["value"]))
        for e in expected
    }
    pred: dict[tuple[str, int], Decimal] = {}
    for p in predicted:
        pred.setdefault((str(p["metric"]), int(p["fiscal_year"])), Decimal(str(p["value"])))

    correct = sum(1 for k, v in key.items() if pred.get(k) == v)
    wrong = sum(1 for k, v in key.items() if k in pred and pred[k] != v)
    absent = set(expect_absent)
    predicted_metrics = {m for m, _ in pred}
    hallucinated = sum(1 for (m, _) in pred if m in absent)
    return ExtractScore(
        expected=len(key),
        correct=correct,
        wrong_value=wrong,
        missing=len(key) - correct - wrong,
        hallucinated=hallucinated,
        abstain_expected=len(absent),
        # A failed call predicts nothing; that is not a correct "not reported".
        abstain_correct=0 if schema_error else len(absent - predicted_metrics),
        rejected=rejected,
        schema_error=schema_error,
    )


def _as_date(v: Any) -> Any:
    from datetime import date

    return v if isinstance(v, date) else date.fromisoformat(str(v))


def micro(scores: list[ExtractScore]) -> dict[str, float]:
    """Dataset-level metrics: pooled counts, not an average of per-example ratios."""
    c = sum(s.correct for s in scores)
    w = sum(s.wrong_value for s in scores)
    h = sum(s.hallucinated for s in scores)
    e = sum(s.expected for s in scores)
    a_exp = sum(s.abstain_expected for s in scores)
    a_ok = sum(s.abstain_correct for s in scores)
    precision = c / (c + w + h) if c + w + h else 0.0
    recall = c / e if e else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "abstain_accuracy": a_ok / a_exp if a_exp else 1.0,
        "schema_error_rate": sum(s.schema_error for s in scores) / len(scores) if scores else 0.0,
    }


__all__ = ["ExtractScore", "Metric", "micro", "score_extraction"]
