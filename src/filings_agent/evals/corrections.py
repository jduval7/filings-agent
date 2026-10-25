"""Reviewer corrections -> regression tests (the "failure traces become tests" loop).

When a reviewer overrides the agent's cause, publish stores the correction with the exact
reconcile input the model saw (discrepancy + evidence) and the LangSmith run id. Export then:
  1. adds it to the LangSmith dataset `filings-agent/reconcile-corrections` (deterministic ids,
     so exporting twice never duplicates; ids already there are skipped, because LangSmith
     answers a duplicate id with 409 and a half-finished export would otherwise never finish),
  2. attaches reviewer feedback to that run's trace, so the failure is visible where it happened,
  3. appends it to configs/reconcile_corrections.jsonl (committed, so CI sees it), which the
     reconcile eval and the CI gate include from then on.
So next week's eval contains this week's mistakes.
"""

# LangSmith client methods are partially untyped; this module is a boundary that accepts it.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from filings_agent.storage.ledger import Ledger

DATASET = "filings-agent/reconcile-corrections"
_NS = uuid.UUID("8f1d2b6e-3c1a-4e7a-9a0b-2f6c9d1e5a77")


class LangSmithLike(Protocol):
    def has_dataset(self, *, dataset_name: str) -> bool: ...
    def create_dataset(self, dataset_name: str, *, description: str) -> Any: ...
    def read_dataset(self, *, dataset_name: str) -> Any: ...
    def list_examples(self, *, dataset_id: Any, example_ids: list[str]) -> Iterable[Any]: ...
    def create_examples(self, *, dataset_id: Any, examples: list[dict[str, Any]]) -> Any: ...
    def create_feedback(self, run_id: Any, key: str, **kwargs: Any) -> Any: ...
    def read_project(self, *, project_name: str) -> Any: ...


def example_id(discrepancy_id: str, thread_id: str) -> str:
    return str(uuid.uuid5(_NS, f"{discrepancy_id}|{thread_id}"))


def export_corrections(
    ledger: Ledger,
    client: LangSmithLike | None,
    local_path: Path,
    project_name: str = "filings-agent",
) -> int:
    pending = ledger.unexported_corrections()
    if not pending:
        return 0
    examples: list[dict[str, Any]] = []
    for disc_id, thread_id, c in pending:
        examples.append(
            {
                "id": example_id(disc_id, thread_id),
                "inputs": json.loads(c.task_json),
                "outputs": {"cause": c.correct_cause},
                "metadata": {
                    "discrepancy_id": disc_id,
                    "thread_id": thread_id,
                    "model_cause": c.model_cause,
                    "note": c.note,
                    "source": "reviewer",
                },
            }
        )
    if client is not None:
        if client.has_dataset(dataset_name=DATASET):
            ds = client.read_dataset(dataset_name=DATASET)
        else:
            ds = client.create_dataset(
                DATASET, description="Reconcile cases a human reviewer corrected in production"
            )
        # A retry after a partial failure (examples created, then feedback failed) must not send
        # the same ids again.
        ids = [str(ex["id"]) for ex in examples]
        there = {str(e.id) for e in client.list_examples(dataset_id=ds.id, example_ids=ids)}
        new = [ex for ex in examples if str(ex["id"]) not in there]
        if new:
            client.create_examples(dataset_id=ds.id, examples=new)
        # Feedback without the trace's project (session) id is deprecated in LangSmith and will
        # stop working; resolve it once from the tracing project.
        session_id = client.read_project(project_name=project_name).id
        for disc_id, thread_id, c in pending:
            if c.run_id:
                client.create_feedback(
                    c.run_id,
                    key="reviewer_cause_correction",
                    score=0,
                    value=c.correct_cause,
                    correction={"cause": c.correct_cause, "discrepancy_id": disc_id},
                    comment=f"agent said {c.model_cause}; reviewer: {c.correct_cause}. {c.note}",
                    trace_id=c.run_id,  # root run id == trace id
                    session_id=session_id,
                    # deterministic: re-exporting can't create duplicate feedback
                    feedback_id=uuid.uuid5(_NS, f"feedback|{disc_id}|{thread_id}"),
                )
    existing: set[str] = set()
    if local_path.exists():
        existing = {json.loads(line)["id"] for line in local_path.read_text().splitlines() if line}
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("a") as f:
        for ex in examples:
            if ex["id"] not in existing:
                f.write(json.dumps(ex) + "\n")
    ledger.mark_exported([(d, t) for d, t, _ in pending])
    return len(examples)
