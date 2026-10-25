# LangSmith's client is only partly typed; same boundary as the other experiment modules.
# pyright: reportUnknownMemberType=false
"""Research eval: does reviewer-triggered research help on the hard cases? A LangSmith experiment.

Hard cases = labelled restatements outside the easy lane (everything but stock splits), one per
company and metric so near-duplicates (GE revenue FY2021/22/23) don't eat the budget. GE net
income FY2022, the case that motivated the review card, is always included.

What "helps" means here, given the card never names a favorite:
  true_cause_supported     at least one verified finding SUPPORTS the labelled cause (the reviewer
                           gets evidence for the right answer)
  true_cause_contradicted  a verified finding CONTRADICTS the labelled cause (the dangerous kind:
                           pushes the reviewer away from the right answer)
  wrong_causes_supported   how many OTHER causes got supporting findings (noise the reviewer has
                           to wade through)
  true_cause_open_on_card  the labelled cause is still fits/possible on the card after research
                           (code rules decide fits; this must stay 1)
  discarded_rate           share of findings dropped because the quote wasn't in the cited source
"""

from __future__ import annotations

import statistics
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, cast

from langsmith import Client
from langsmith.evaluation import EvaluationResult, EvaluationResults
from langsmith.schemas import Example, Run

from filings_agent.config import AppConfig
from filings_agent.domain import Discrepancy
from filings_agent.evals.extract_experiments import dataset_name
from filings_agent.evals.reconcile_local import LabeledTask, load_tasks
from filings_agent.research.findings import ResearchResult, VerifiedFinding
from filings_agent.steps.card import build_card
from filings_agent.steps.reconcile import Cause, ReconcileTask

_NS = uuid.UUID("6b1d2f0e-3c4a-4e8b-9f7d-2a5c8e1b0d46")
MUST_INCLUDE = "GE:net_income:FY2022"


SETS: dict[str, tuple[str, ...]] = {
    "hard": ("dev", "test"),  # the first research eval (GE was used while developing it)
    "fresh": ("fresh",),  # companies added after every choice was frozen
}


def hard_cases(cfg: AppConfig, which: str = "hard") -> list[LabeledTask]:
    """One labelled non-split case per (company, metric) from the splits in SETS[which]."""
    picked: dict[tuple[str, str], LabeledTask] = {}
    for split in SETS[which]:
        for t in load_tasks(cfg, split):
            d = t.task.discrepancy
            if t.task.discrepancy_id.endswith("#corr") or t.label is Cause.STOCK_SPLIT:
                continue
            key = (d.ticker, d.metric.value)
            if key not in picked or t.task.discrepancy_id == MUST_INCLUDE:
                picked[key] = t
    return sorted(picked.values(), key=lambda t: t.task.discrepancy_id)


def upload_research_dataset(cfg: AppConfig, client: Client, which: str = "hard") -> tuple[str, int]:
    name = dataset_name("research", which)
    ds = (
        client.read_dataset(dataset_name=name)
        if client.has_dataset(dataset_name=name)
        else client.create_dataset(
            name, description="Hard restatement cases (non-splits) for reviewer-triggered research"
        )
    )
    existing = {str(e.id) for e in client.list_examples(dataset_id=ds.id)}
    new: list[dict[str, Any]] = []
    cases = hard_cases(cfg, which)
    for t in cases:
        ex_id = str(uuid.uuid5(_NS, t.task.discrepancy_id))
        if ex_id in existing:
            continue
        new.append(
            {
                "id": ex_id,
                "inputs": {
                    "discrepancy": t.task.discrepancy.model_dump(mode="json"),
                    "evidence": list(t.task.evidence),
                },
                "outputs": {"cause": t.label.value},
                "metadata": {"ticker": t.task.discrepancy.ticker, "split": "mixed"},
            }
        )
    if new:
        client.create_examples(dataset_id=ds.id, examples=new)
    return name, len(cases)


def make_target(researcher: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        d = Discrepancy.model_validate(inputs["discrepancy"])
        evidence = tuple(cast(list[str], inputs["evidence"]))
        res = cast(ResearchResult, researcher.research(ReconcileTask(d.id, d, evidence), "", 1))
        card = build_card(d, evidence, [res], 0)
        return {
            "findings": [asdict(f) for f in res.findings],
            "dropped": res.dropped,
            "card_fits": {c["cause"]: c["fit"] for c in card["candidates"]},
            "error": res.error,
            "cost_usd": res.usage.cost_usd if res.usage else 0.0,
            "latency_s": res.usage.latency_s if res.usage else 0.0,
        }

    return target


def score(out: dict[str, Any], label: str) -> dict[str, float]:
    """Pure scoring of one research output against the labelled cause (unit-tested)."""
    findings = [VerifiedFinding(**f) for f in cast(list[dict[str, Any]], out.get("findings", []))]
    supported = {f.cause for f in findings if f.stance == "supports"}
    contradicted = {f.cause for f in findings if f.stance == "contradicts"}
    total = len(findings) + int(out.get("dropped", 0))
    fit = cast(dict[str, str], out.get("card_fits", {})).get(label, "")
    return {
        "true_cause_supported": float(label in supported),
        "true_cause_contradicted": float(label in contradicted),
        "wrong_causes_supported": float(len(supported - {label, "other"})),
        "true_cause_open_on_card": float(fit in ("fits", "possible")),
        "discarded_rate": int(out.get("dropped", 0)) / total if total else 0.0,
        "cost_usd": float(out.get("cost_usd", 0.0)),
        "latency_s": float(out.get("latency_s", 0.0)),
    }


def research_evaluator(run: Run, example: Example) -> EvaluationResults:
    out = cast(dict[str, Any], run.outputs or {})
    label = str((example.outputs or {}).get("cause", ""))
    return {"results": [EvaluationResult(key=k, score=v) for k, v in score(out, label).items()]}


def research_summary(runs: list[Run], examples: list[Example]) -> EvaluationResults:
    rows = [
        score(cast(dict[str, Any], r.outputs or {}), str((e.outputs or {}).get("cause", "")))
        for r, e in zip(runs, examples, strict=True)
    ]
    keys = ("true_cause_supported", "true_cause_contradicted", "true_cause_open_on_card")
    results = [
        EvaluationResult(key=f"{k}_rate", score=statistics.fmean(r[k] for r in rows))
        for k in keys
        if rows
    ]
    results.append(EvaluationResult(key="total_cost_usd", score=sum(r["cost_usd"] for r in rows)))
    return {"results": results}


def run_research_experiment(
    cfg: AppConfig, client: Client, split: str = "dev", concurrency: int = 2
) -> dict[str, Any]:
    """`split`: "fresh" runs the fresh cases; anything else the original hard set (dev + test)."""
    from filings_agent.graph.runner import make_researcher

    which = "fresh" if split == "fresh" else "hard"
    name, n = upload_research_dataset(cfg, client, which)
    researcher = make_researcher(cfg)
    results = client.evaluate(
        make_target(researcher),
        data=name,
        evaluators=[research_evaluator],
        summary_evaluators=[research_summary],
        experiment_prefix=f"research-{cfg.research.model}-{researcher.name.split('/')[-1]}",
        description=f"Reviewer-triggered research on {which} cases (no hint, round 1)",
        metadata={
            "step": "research",
            "model": cfg.research.model,
            "harness": researcher.name,
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        max_concurrency=concurrency,
    )
    per_case: dict[str, dict[str, float]] = {}
    for r in results:
        out = cast(dict[str, Any], r["run"].outputs or {})
        ex = r["example"]
        case = Discrepancy.model_validate((ex.inputs or {})["discrepancy"]).id
        per_case[case] = score(out, str((ex.outputs or {}).get("cause", "")))
    return {"experiment": results.experiment_name, "n": n, "per_case": per_case}
