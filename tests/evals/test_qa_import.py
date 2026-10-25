"""QA catches in LangSmith become regression tests the gate runs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from filings_agent.evals.qa_import import QA_DATASET, import_qa
from filings_agent.evals.reconcile_local import load_corrections
from filings_agent.steps.reconcile import Cause

TASK = {
    "discrepancy_id": "GE:net_income:FY2022",
    "discrepancy": {
        "ticker": "GE",
        "metric": "net_income",
        "fiscal_year": 2022,
        "period_end": "2022-12-31",
        "values": [["A1", "225000000"], ["A2", "339000000"]],
    },  # fmt: skip
    "evidence": ["On January 1, 2023, we adopted ASU No. 2018-12"],
}


class FakeLangSmith:
    def __init__(self, examples: list[Any]) -> None:
        self.examples = examples
        self.created: list[str] = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        return bool(self.created)

    def create_dataset(self, dataset_name: str, *, description: str) -> Any:
        self.created.append(dataset_name)

    def list_examples(self, *, dataset_name: str) -> list[Any]:
        assert dataset_name == QA_DATASET
        return self.examples


def ex(id_: str, inputs: dict[str, Any], outputs: dict[str, Any]) -> Any:
    return SimpleNamespace(id=id_, inputs=inputs, outputs=outputs)


def test_a_qa_catch_becomes_a_test_the_gate_runs(tmp_path: Path) -> None:
    """Review finding: the queue's Add to Dataset never reached the CI gate."""
    edited_in_place = {"resolutions": {"GE:net_income:FY2022": {"cause": "accounting_change"}}}
    client = FakeLangSmith([
        ex("e1", {"task": TASK}, {"cause": "accounting_change"}),
        ex("e2", {"task": TASK}, edited_in_place),
        ex("e3", {"tickers": ["GE"]}, {"status": "published"}),  # the whole run, not the step
        ex("e4", {"task": TASK}, {"cause": "vibes"}),
    ])  # fmt: skip
    path = tmp_path / "corrections.jsonl"
    result = import_qa(client, path)
    assert client.created == [QA_DATASET]  # created so QA can pick it in Add to Dataset
    assert result["added"] == ["GE:net_income:FY2022 -> accounting_change"] * 2
    assert set(result["refused"]) == {"e3", "e4"} and "reconcile" in result["refused"]["e3"]

    tasks = load_corrections(path)  # what every reconcile eval and the gate read
    assert len(tasks) == 2 and all(t.label is Cause.ACCOUNTING_CHANGE for t in tasks)
    assert tasks[0].task.evidence == tuple(TASK["evidence"])  # the exact input the model saw
    assert json.loads(path.read_text().splitlines()[0])["metadata"]["source"] == "qa"

    again = import_qa(client, path)  # importing twice adds nothing
    assert again["added"] == [] and len(path.read_text().splitlines()) == 2
