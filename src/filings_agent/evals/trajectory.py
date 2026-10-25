"""Trajectory evaluation: did the agent take the right PATH, not just produce the right answer?

For a graph agent, the path is part of correctness: a company with restated figures must go
through reconcile and stop for human review; a clean company must skip both. Final-answer evals
can't see that a run silently skipped review.

  run_with_trajectory  streams node updates and records the path (resuming through review)
  expected_path        the path a scenario should take
  end-to-end dataset   scenarios (ticker sets) with expected path + expected material
                       restatements from the answer key; evaluated as a LangSmith experiment
"""

# LangGraph/LangSmith are partially untyped; this module is a boundary that accepts it.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import date
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from filings_agent.config import AppConfig
from filings_agent.domain import FilingRef, GoldFact
from filings_agent.evals.extract_experiments import dataset_name
from filings_agent.graph.build import Deps, build_graph
from filings_agent.sec.gold import find_discrepancies
from filings_agent.steps.reconcile import is_material

_NS = uuid.UUID("9a3e1f7c-2d4b-4e8a-b6c5-0f1d2e3a4b5c")

SCENARIOS: tuple[tuple[str, ...], ...] = (("GE", "NVDA"), ("AAPL", "MSFT"), ("KO",), ("NVDA",))


def dedupe(path: list[str]) -> list[str]:
    """Collapse fan-out repeats: extract x24 -> extract."""
    out: list[str] = []
    for node in path:
        if not out or out[-1] != node:
            out.append(node)
    return out


def expected_path(review_expected: bool) -> list[str]:
    head = ["plan", "extract", "find_discrepancies"]
    if review_expected:
        return [*head, "reconcile", "interrupt", "review", "memo", "publish"]
    return [*head, "review", "memo", "publish"]


def _stream_path(graph: Any, inp: Any, config: RunnableConfig, deps: Deps) -> list[str]:
    path: list[str] = []
    for chunk in graph.stream(inp, config, context=deps, stream_mode="updates"):
        for node in cast(dict[str, Any], chunk):
            path.append("interrupt" if node == "__interrupt__" else node)
    return path


def run_with_trajectory(
    deps: Deps, tickers: list[str], thread_id: str = "trajectory"
) -> tuple[dict[str, Any], list[str], bool]:
    """Run to completion, auto-approving review. Returns (final state, path, review_triggered)."""
    from filings_agent.graph.runner import serializer

    graph = build_graph(InMemorySaver(serde=serializer()))  # same type allowlist as production
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    path = _stream_path(graph, {"tickers": tickers}, config, deps)
    review_triggered = "interrupt" in path
    if review_triggered:
        pending = graph.get_state(config).tasks[0].interrupts[0].value["pending"]
        decisions = {p["id"]: {"action": "accept", "note": "trajectory eval"} for p in pending}
        path += _stream_path(graph, Command(resume=decisions), config, deps)
    state = cast(dict[str, Any], graph.get_state(config).values)
    return state, path, review_triggered


def expected_material(cfg: AppConfig, tickers: tuple[str, ...]) -> list[str]:
    gold = [
        GoldFact.model_validate_json(x)
        for x in (cfg.data_dir / "gold" / "gold.jsonl").read_text().splitlines()
    ]
    filed: dict[str, date] = {
        f.accession_no: f.filed
        for f in (
            FilingRef.model_validate_json(x)
            for x in (cfg.data_dir / "gold" / "filings.jsonl").read_text().splitlines()
        )
    }
    facts = [g for g in gold if g.ticker in tickers]
    return sorted(d.id for d in find_discrepancies(facts, filed) if is_material(d))


def upload_e2e_dataset(cfg: AppConfig, client: Any) -> int:
    name = dataset_name("e2e", "dev")
    if client.has_dataset(dataset_name=name):
        ds = client.read_dataset(dataset_name=name)
    else:
        ds = client.create_dataset(name, description="Scenarios: expected path + restatements")
    existing = {str(e.id) for e in client.list_examples(dataset_id=ds.id)}
    new: list[dict[str, Any]] = []
    update: list[dict[str, Any]] = []
    for tickers in SCENARIOS:
        material = expected_material(cfg, tickers)
        ex_id = str(uuid.uuid5(_NS, "+".join(tickers)))
        payload = {
            "id": ex_id,
            "inputs": {"tickers": list(tickers)},
            "outputs": {
                "path": expected_path(bool(material)),
                "material": material,
                "review": bool(material),
            },
            "metadata": {"scenario": "+".join(tickers)},
        }
        (update if ex_id in existing else new).append(payload)
    if new:
        client.create_examples(dataset_id=ds.id, examples=new)
    if update:
        client.update_examples(dataset_id=ds.id, updates=update)
    return len(SCENARIOS)


def make_e2e_target(make_deps: Callable[[], Deps]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        tickers = list(inputs["tickers"])
        state, path, review = run_with_trajectory(make_deps(), tickers, "+".join(tickers))
        material = sorted(k for k, r in state.get("resolutions", {}).items() if r.material)
        return {
            "path": dedupe(path),
            "raw_path_len": len(path),
            "review": review,
            "material": material,
            "facts": len(state.get("facts", {})),
            "errors": len(state.get("errors", {})),
        }

    return target


def _out(obj: Any) -> dict[str, Any]:
    return cast(dict[str, Any], obj.outputs or {})


def e2e_evaluators() -> list[Callable[..., Any]]:
    from langsmith.evaluation import EvaluationResult

    def trajectory(run: Any, example: Any) -> Any:
        out, ref = _out(run), _out(example)
        return EvaluationResult(
            key="trajectory_match",
            score=int(out.get("path") == ref.get("path")),
            comment=f"got {out.get('path')}",
        )

    def review_gate(run: Any, example: Any) -> Any:
        out, ref = _out(run), _out(example)
        return EvaluationResult(
            key="review_correct", score=int(out.get("review") == ref.get("review"))
        )

    def restatements(run: Any, example: Any) -> Any:
        got = set(cast(list[str], _out(run).get("material", [])))
        want = set(cast(list[str], _out(example).get("material", [])))
        recall = len(got & want) / len(want) if want else 1.0
        precision = len(got & want) / len(got) if got else 1.0
        return {
            "results": [
                EvaluationResult(
                    key="restatement_recall", score=recall, comment=f"missed: {sorted(want - got)}"
                ),
                EvaluationResult(
                    key="restatement_precision",
                    score=precision,
                    comment=f"extra (likely extraction errors): {sorted(got - want)}",
                ),
            ]
        }

    return [trajectory, review_gate, restatements]


def run_e2e_experiment(cfg: AppConfig, client: Any, make_deps: Callable[[], Deps]) -> str:
    results = client.evaluate(
        make_e2e_target(make_deps),
        data=dataset_name("e2e", "dev"),
        evaluators=e2e_evaluators(),
        experiment_prefix="e2e",
        description="Whole-graph scenarios: path, review gate, restatements caught",
        metadata={
            "step": "e2e",
            "extractor": cfg.roles.extractor,
            "reconciler": cfg.roles.reconciler,
            "prompts": cfg.prompts.model_dump(),
        },
        max_concurrency=1,
    )
    return str(results.experiment_name)
