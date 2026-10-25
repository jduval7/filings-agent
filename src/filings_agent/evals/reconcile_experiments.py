"""LangSmith wiring for the reconcile step: labelled datasets and experiments.

Inputs are the exact reconcile inputs (discrepancy + evidence), outputs the verified cause, so
an experiment replays precisely what the step sees in production.
"""

# LangSmith client methods are partially untyped; this module is a boundary that accepts it.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import statistics
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from langsmith import Client
from langsmith.evaluation import EvaluationResult, EvaluationResults
from langsmith.schemas import Example, Run

from filings_agent.config import AppConfig
from filings_agent.domain import Discrepancy
from filings_agent.evals.agent_eval import load_anchors, trajectory_ok
from filings_agent.evals.extract_experiments import dataset_name
from filings_agent.evals.reconcile_local import load_tasks
from filings_agent.graph.build import Reconciler
from filings_agent.reconcile_agent.agent import AgentReconciler
from filings_agent.steps.reconcile import Cause, ReconcileTask, change_filings
from filings_agent.textnorm import normalize

_NS = uuid.UUID("2c7e0d4a-9b1f-4f3e-8a6d-5e2b7c9f1d33")


def upload_reconcile_datasets(cfg: AppConfig, client: Client) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("dev", "test"):
        name = dataset_name("reconcile", split)
        ds = (
            client.read_dataset(dataset_name=name)
            if client.has_dataset(dataset_name=name)
            else client.create_dataset(
                name, description=f"Restatement cause, labelled from filing text ({split})"
            )
        )
        existing = {str(e.id) for e in client.list_examples(dataset_id=ds.id)}
        new: list[dict[str, Any]] = []
        update: list[dict[str, Any]] = []
        tasks = load_tasks(cfg, split)
        anchors = load_anchors()
        for t in tasks:
            ex_id = str(uuid.uuid5(_NS, t.task.discrepancy_id))
            payload: dict[str, Any] = {
                "id": ex_id,
                "inputs": {
                    "discrepancy": t.task.discrepancy.model_dump(mode="json"),
                    "evidence": list(t.task.evidence),
                },
                "outputs": {
                    "cause": t.label.value,
                    "anchors": anchors.get(t.task.discrepancy_id.removesuffix("#corr"), []),
                    "change_filings": change_filings(t.task.discrepancy),
                },
                "metadata": {
                    "ticker": t.task.discrepancy.ticker,
                    "metric": t.task.discrepancy.metric.value,
                    "source": "reviewer" if t.task.discrepancy_id.endswith("#corr") else "label",
                },
            }
            (update if ex_id in existing else new).append(payload)
        if new:
            client.create_examples(dataset_id=ds.id, examples=new)
        if update:
            client.update_examples(dataset_id=ds.id, updates=update)
        counts[name] = len(tasks)
    return counts


def _outputs(obj: Run | Example) -> dict[str, Any]:
    return cast(dict[str, Any], obj.outputs or {})


def make_target(rec: Reconciler, rep: int) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Works for the pipeline and the agent; the agent also reports its tool calls and what
    retrieval returned, for the tool-use and retrieval evaluators."""

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        d = Discrepancy.model_validate(inputs["discrepancy"])
        task = ReconcileTask(d.id, d, tuple(inputs["evidence"]))
        extra: dict[str, Any] = {}
        if isinstance(rec, AgentReconciler):
            outcome = rec.investigate(task, rep)
            res = outcome.resolution
            extra = {
                "tool_calls": [
                    {"name": c.name, "accession_no": c.accession_no, "query": c.query}
                    for c in outcome.tool_calls
                ],
                "retrieved": [p.text for p in outcome.passages],
            }
        else:
            res = rec.reconcile(task, rep=rep)
        return {
            "cause": res.cause.value,
            "verified": res.verified,
            "explanation": res.explanation,
            "evidence_quote": res.evidence_quote,
            "cost_usd": res.usage.cost_usd if res.usage else 0.0,
            "latency_s": res.usage.latency_s if res.usage else 0.0,
            **extra,
        }

    return target


def agent_evaluator(run: Run, example: Example) -> EvaluationResults:
    """Retrieval + tool use; skipped (no results) for pipeline runs."""
    out, ref = _outputs(run), _outputs(example)
    if "tool_calls" not in out:
        return {"results": []}
    calls = cast(list[dict[str, str]], out["tool_calls"])
    texts = [normalize(t) for t in cast(list[str], out.get("retrieved", []))]
    anchors = cast(list[str], ref.get("anchors", []))
    searched = {c["accession_no"] for c in calls if c["name"] == "search_filing"}
    msgs = [{"role": "user", "content": "q"}] + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": c["name"], "arguments": "{}"}}],
        }
        for c in calls
    ]
    results = [
        EvaluationResult(
            key="searched_change_filing",
            score=int(bool(searched & set(cast(list[str], ref.get("change_filings", []))))),
        ),
        EvaluationResult(key="trajectory_superset", score=int(trajectory_ok(msgs))),
        EvaluationResult(
            key="tool_calls", score=sum(1 for c in calls if c["name"] != "AgentAnswer")
        ),
    ]
    if anchors:
        hit = any(normalize(a) in t for a in anchors for t in texts)
        results.append(EvaluationResult(key="retrieval_hit", score=int(hit)))
    return {"results": results}


def reconcile_evaluator(run: Run, example: Example) -> EvaluationResults:
    out, ref = _outputs(run), _outputs(example)
    return {
        "results": [
            EvaluationResult(key="cause_correct", score=int(out.get("cause") == ref.get("cause"))),
            EvaluationResult(key="quote_verified", score=int(bool(out.get("verified")))),
            EvaluationResult(key="unknown", score=int(out.get("cause") == Cause.UNKNOWN.value)),
            EvaluationResult(key="cost_usd", score=float(out.get("cost_usd", 0.0))),
            EvaluationResult(key="latency_s", score=float(out.get("latency_s", 0.0))),
        ]
    }


def reconcile_summary(runs: list[Run], examples: list[Example]) -> EvaluationResults:
    correct = [
        _outputs(r).get("cause") == _outputs(e).get("cause")
        for r, e in zip(runs, examples, strict=True)
    ]
    cost = sum(float(_outputs(r).get("cost_usd", 0.0)) for r in runs)
    return {
        "results": [
            EvaluationResult(
                key="cause_accuracy", score=statistics.fmean(correct) if correct else 0
            ),
            EvaluationResult(key="total_cost_usd", score=cost),
        ]
    }


def run_reconcile_sweep(
    cfg: AppConfig, client: Client, models: list[str], split: str, prompt: str, reps: int,
    concurrency: int, strategy: str = "pipeline", retrieval: str | None = None,
) -> list[dict[str, Any]]:  # fmt: skip
    """`prompt` is the prompt version (pipeline) or the agent version (agent)."""
    from filings_agent.reconcile_agent.factory import make_index, make_reconciler

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict[str, Any]] = []
    for model in models:
        index = make_index(cfg, retrieval) if strategy == "agent" else None
        rec = make_reconciler(cfg, model, strategy, prompt, index)
        for rep in range(reps):
            results = client.evaluate(
                make_target(rec, rep),
                data=dataset_name("reconcile", split),
                evaluators=[reconcile_evaluator, agent_evaluator],
                summary_evaluators=[reconcile_summary],
                experiment_prefix=(
                    f"reconcile-{strategy}-{model}-{prompt}"
                    + (f"-{index.mode}" if index else "")
                    + f"-r{rep}"
                ),
                description=f"reconcile step | {model} | prompt {prompt} | rep {rep} | {split}",
                metadata={
                    "step": "reconcile",
                    "model": model,
                    "model_id": cfg.model(model).id,
                    "prompt_version": prompt,
                    "rep": rep,
                    "split": split,
                    "sweep": stamp,
                },  # fmt: skip
                max_concurrency=concurrency,
            )
            outs = [(_outputs(r["run"]), _outputs(r["example"])) for r in results]
            acc = statistics.fmean(o.get("cause") == e.get("cause") for o, e in outs) if outs else 0
            row = {
                "model": model, "prompt": prompt, "rep": rep, "split": split,
                "experiment": results.experiment_name, "n": len(outs), "cause_accuracy": acc,
                "cost_usd": sum(float(o.get("cost_usd", 0.0)) for o, _ in outs),
            }  # fmt: skip
            rows.append(row)
    return rows
