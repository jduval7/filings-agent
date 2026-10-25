"""Reviewer corrections become regression tests (LangSmith dataset + local file)."""

from __future__ import annotations

import contextlib
import uuid
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from filings_agent.storage.ledger import Ledger
from tests.support.graph_doubles import invoke, review_env
from tests.support.langsmith_doubles import check_feedback


class FakeLangSmith:
    def __init__(self) -> None:
        self.examples: dict[str, dict[str, Any]] = {}
        self.feedback: list[tuple[Any, str, dict[str, Any]]] = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        return True

    def create_dataset(self, dataset_name: str, *, description: str) -> Any:  # pragma: no cover
        raise AssertionError

    def read_dataset(self, *, dataset_name: str) -> Any:
        return type("DS", (), {"id": "ds1"})()

    def list_examples(self, *, dataset_id: Any, example_ids: list[str]) -> list[Any]:
        return [type("Ex", (), {"id": i})() for i in example_ids if i in self.examples]

    def create_examples(self, *, dataset_id: Any, examples: list[dict[str, Any]]) -> Any:
        for ex in examples:
            if ex["id"] in self.examples:  # what LangSmith does with a duplicate id
                raise RuntimeError("409 Conflict: example already exists")
            self.examples[ex["id"]] = ex

    def create_feedback(self, run_id: Any, key: str, **kwargs: Any) -> Any:
        check_feedback(kwargs)
        self.feedback.append((run_id, key, kwargs))

    def read_project(self, *, project_name: str) -> Any:
        return type("Project", (), {"id": f"project-id-of-{project_name}"})()


def test_reviewer_correction_becomes_a_regression_test(tmp_path: Path) -> None:
    from filings_agent.evals.corrections import export_corrections
    from filings_agent.evals.reconcile_local import load_corrections

    with review_env(tmp_path) as (graph, deps, _rec, ledger):
        invoke(graph, {"tickers": ["AAA"]}, deps)
        decision = {
            "AAA:revenue:FY2023": {
                "action": "accept",
                "note": "it was an accounting change",
                "correct_cause": "accounting_change",
            }
        }
        root = uuid.UUID("11111111-1111-1111-1111-111111111111")
        cfg: RunnableConfig = {
            "configurable": {"thread_id": "review-1", "root_run_id": str(root)},
            "run_id": root,
        }
        graph.invoke(Command(resume=decision), cfg, context=deps)
        assert ledger.counts()["corrections"] == 1

        client = FakeLangSmith()
        local = tmp_path / "corrections.jsonl"
        assert export_corrections(ledger, client, local) == 1
        assert export_corrections(ledger, client, local) == 0  # exporting twice: no duplicates

    (ex,) = client.examples.values()
    assert ex["outputs"] == {"cause": "accounting_change"}
    assert ex["metadata"]["model_cause"] == "discontinued_operations"
    (fb,) = client.feedback
    assert fb[0] == "11111111-1111-1111-1111-111111111111"  # feedback lands on the right trace
    assert fb[1] == "reviewer_cause_correction" and fb[2]["score"] == 0
    assert fb[2]["session_id"] == "project-id-of-filings-agent"  # not the deprecated form
    (task,) = load_corrections(local)
    assert task.label.value == "accounting_change"
    assert task.task.discrepancy.id == "AAA:revenue:FY2023"  # replays the production input


def test_export_without_langsmith_still_writes_local_tests(tmp_path: Path) -> None:
    from filings_agent.evals.corrections import example_id, export_corrections
    from filings_agent.storage.ledger import Correction

    ledger = Ledger(tmp_path / "l.sqlite")
    task_json = '{"discrepancy": {}, "evidence": []}'
    c = Correction("X:revenue:FY2023", "unknown", "stock_split", "", task_json, None)
    ledger.publish("t", [], [], corrections=[c])
    local = tmp_path / "corr.jsonl"
    assert export_corrections(ledger, None, local) == 1
    row = local.read_text().splitlines()[0]
    assert example_id("X:revenue:FY2023", "t") in row
    assert export_corrections(ledger, None, local) == 0  # nothing pending any more


def test_an_export_that_failed_halfway_finishes_on_retry(tmp_path: Path) -> None:
    """Review finding: examples created, then feedback failed, so nothing was marked exported;
    the retry re-sent the same example ids, which LangSmith refuses (409): stuck for good."""
    from filings_agent.evals.corrections import export_corrections
    from filings_agent.storage.ledger import Correction

    ledger = Ledger(tmp_path / "l.sqlite")
    task_json = '{"discrepancy": {}, "evidence": []}'
    c = Correction("X:revenue:FY2023", "unknown", "stock_split", "", task_json, "run-1")
    ledger.publish("t", [], [], corrections=[c])

    class FlakyFeedback(FakeLangSmith):
        fail = True

        def create_feedback(self, run_id: Any, key: str, **kwargs: Any) -> Any:
            if self.fail:
                raise ConnectionError("LangSmith unreachable")
            return super().create_feedback(run_id, key, **kwargs)

    client = FlakyFeedback()
    local = tmp_path / "corr.jsonl"
    with contextlib.suppress(ConnectionError):
        export_corrections(ledger, client, local)
    assert len(client.examples) == 1 and ledger.unexported_corrections()  # half done
    client.fail = False
    assert export_corrections(ledger, client, local) == 1  # finishes, no 409
    assert len(client.examples) == 1 and len(client.feedback) == 1
    assert ledger.unexported_corrections() == []
