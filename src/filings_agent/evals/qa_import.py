"""QA catches -> regression tests: the second door into the CI gate.

Reviewers on the review screen correct causes as they decide (evals/corrections.py). QA reviewers
work the LangSmith annotation queue instead, and find mistakes the analysts didn't. For one of
those to guard future changes it has to reach the gate, which reads the committed corrections
file. This is that path:

  1. In the trace, open the `reconcile` step for the wrong change and click Add to Dataset,
     choosing `filings-agent/reconcile-qa`. The example's input is that step's input: the
     discrepancy and the exact evidence the model saw.
  2. Set the example's reference output to the right cause: {"cause": "accounting_change"}
     (or edit the cause inside the copied `resolutions` output). A person decides the answer.
  3. `make import-qa` checks each example and appends the new ones to
     configs/reconcile_corrections.jsonl, the file every reconcile eval and the gate include.
     Commit it: the diff is the review of what became a test.

Nothing is imported silently: an example without a reconcile task or with a cause that isn't one
of the candidate causes is reported and skipped.
"""

# LangSmith client methods are partially untyped; this module is a boundary that accepts it.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Protocol, cast

from filings_agent.domain import Discrepancy
from filings_agent.steps.card import CANDIDATES

QA_DATASET = "filings-agent/reconcile-qa"
QA_DESCRIPTION = (
    "Reconcile mistakes found by QA in the annotation queue. Add a trace's `reconcile` step and "
    'set the reference output to {"cause": "<the right cause>"}; `make import-qa` turns them into '
    "regression tests for the CI gate."
)
PICKABLE = frozenset(c.value for c in CANDIDATES)
_NS = uuid.UUID("5a0e7c2b-91d4-4f3e-8b6a-2c1d9e0f4b73")


class QaClient(Protocol):
    def has_dataset(self, *, dataset_name: str) -> bool: ...
    def create_dataset(self, dataset_name: str, *, description: str) -> Any: ...
    def list_examples(self, *, dataset_name: str) -> Any: ...


def ensure_dataset(client: QaClient) -> None:
    """The dataset QA picks in "Add to Dataset" must exist before anyone can pick it."""
    if not client.has_dataset(dataset_name=QA_DATASET):
        client.create_dataset(QA_DATASET, description=QA_DESCRIPTION)


def to_row(example: Any) -> tuple[dict[str, Any] | None, str | None]:
    """(a corrections-file row, None) or (None, why it can't be a test)."""
    inputs = cast(dict[str, Any], example.inputs or {})
    outputs = cast(dict[str, Any], example.outputs or {})
    task = cast(dict[str, Any], inputs.get("task") or inputs)
    if "discrepancy" not in task or "evidence" not in task:
        return None, "not a reconcile step (add the trace's `reconcile` span, not the whole run)"
    try:
        d = Discrepancy.model_validate(task["discrepancy"])
    except Exception as e:
        return None, f"the discrepancy doesn't parse: {e}"
    cause = outputs.get("cause")
    if cause is None:  # the copied output, with the cause edited in place
        res = cast(dict[str, Any], outputs.get("resolutions") or {}).get(d.id) or {}
        cause = cast(dict[str, Any], res).get("cause")
    if cause not in PICKABLE:
        return None, f"cause {cause!r} isn't one of {sorted(PICKABLE)}: set the reference output"
    row = {
        "id": str(uuid.uuid5(_NS, str(example.id))),
        "inputs": {"discrepancy": d.model_dump(mode="json"), "evidence": list(task["evidence"])},
        "outputs": {"cause": cause},
        "metadata": {
            "discrepancy_id": d.id,
            "source": "qa",
            "dataset": QA_DATASET,
            "example_id": str(example.id),
        },  # fmt: skip
    }
    return row, None


def import_qa(client: QaClient, path: Path) -> dict[str, Any]:
    """Append QA's new examples to the corrections file. Idempotent: an example already
    imported is skipped. Returns what happened, including every example that was refused."""
    ensure_dataset(client)
    existing: set[str] = set()
    if path.exists():
        existing = {json.loads(line)["id"] for line in path.read_text().splitlines() if line}
    added: list[str] = []
    refused: dict[str, str] = {}
    new_rows: list[dict[str, Any]] = []
    for ex in client.list_examples(dataset_name=QA_DATASET):
        row, problem = to_row(ex)
        if row is None:
            refused[str(ex.id)] = str(problem)
        elif row["id"] not in existing:
            new_rows.append(row)
            existing.add(row["id"])
            added.append(f"{row['metadata']['discrepancy_id']} -> {row['outputs']['cause']}")
    if new_rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for row in new_rows:
                f.write(json.dumps(row) + "\n")
    return {"added": added, "refused": refused, "file": str(path)}
