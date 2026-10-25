"""Judge-calibration labels: filled in a LangSmith queue, pulled back into the labels file."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from filings_agent.config import AppConfig
from filings_agent.evals import calibration_queue as cq
from filings_agent.evals.calibration import CalibrationItem
from filings_agent.steps.judge import MEMO_RUBRIC, RECONCILE_RUBRIC

ITEMS = [
    CalibrationItem(item_id="rec-1", rubric="reconcile_explanation", source_model="nova-lite",
                    material="evidence", output="spin-off", context={"label": "accounting_change"}),
    CalibrationItem(item_id="memo-1", rubric="memo", source_model="claude-haiku-4.5",
                    material="table", output="memo text", context={}),
]  # fmt: skip


class FakeLangSmith:
    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self.queues: dict[str, list[str]] = {}
        self.feedback: dict[str, list[Any]] = {}
        self.configs: dict[str, Any] = {"label.supported": {"type": "continuous"}}

    def create_run(self, **kw: Any) -> None:
        self.runs[kw["id"]] = kw

    def list_annotation_queues(self, *, name: str) -> list[Any]:
        return [SimpleNamespace(id=name)] if name in self.queues else []

    def create_annotation_queue(self, *, name: str, **kw: Any) -> Any:
        self.queues[name] = []
        return SimpleNamespace(id=name, **kw)

    def add_runs_to_annotation_queue(self, queue_id: str, *, run_ids: list[str]) -> None:
        self.queues[queue_id] += run_ids

    def list_feedback_configs(self) -> list[Any]:
        return [SimpleNamespace(feedback_key=k, feedback_config=v) for k, v in self.configs.items()]

    def create_feedback_config(self, key: str, *, feedback_config: Any) -> None:
        self.configs[key] = feedback_config

    def update_feedback_config(self, key: str, *, feedback_config: Any) -> None:
        self.configs[key] = feedback_config

    def list_feedback(self, *, run_ids: list[str]) -> list[Any]:
        return self.feedback.get(run_ids[0], [])


@pytest.fixture
def items(monkeypatch: pytest.MonkeyPatch) -> None:
    def load_items(cfg: Any) -> list[CalibrationItem]:
        return ITEMS

    monkeypatch.setattr(cq, "load_items", load_items)


@pytest.mark.usefixtures("items")
def test_items_go_to_one_queue_per_rubric_blind_and_once(cfg: AppConfig) -> None:
    client = FakeLangSmith()
    assert cq.queue_items(cfg, client) == {"reconcile_explanation": 1, "memo": 1}
    assert cq.queue_items(cfg, client) == {"reconcile_explanation": 0, "memo": 0}  # idempotent
    assert client.queues[cq.queue_name(RECONCILE_RUBRIC)] == [cq.run_id(ITEMS[0])]
    run = client.runs[cq.run_id(ITEMS[0])]
    shown = str(run["inputs"]) + str(run["outputs"])
    assert "nova-lite" not in shown  # blind: the labeller isn't told which model wrote it
    assert "accounting_change" not in shown  # nor the answer key: the judge doesn't see it either
    assert run["inputs"] == {"material": "evidence"}
    # every question has a pass/fail definition, or LangSmith leaves it out of the form
    assert all(client.configs[f"label.{n}"] == cq.PASS_FAIL for n in RECONCILE_RUBRIC.names())
    keys = [r["feedback_key"] for r in cq.rubric_items(MEMO_RUBRIC)]
    assert keys == [
        "label.faithful",
        "label.caveats_correct",
        "label.comparable",
        "label.actionable",
    ]


@pytest.mark.usefixtures("items")
def test_pulled_scores_become_labels_and_unscored_ones_stay(cfg: AppConfig, tmp_path: Path) -> None:
    client = FakeLangSmith()
    client.feedback[cq.run_id(ITEMS[0])] = [
        SimpleNamespace(key="label.supported", score=0),
        SimpleNamespace(key="label.specific", score=1),
        SimpleNamespace(key="online.cost_usd", score=0.1),  # not a label: ignored
    ]
    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump({"rec-1": {"consistent": True}}))
    assert cq.pull_labels(cfg, client, path) == {"reconcile_explanation": 3, "memo": 0}
    labels = yaml.safe_load(path.read_text())
    assert labels["rec-1"] == {"consistent": True, "supported": False, "specific": True}
    assert labels["memo-1"] == dict.fromkeys(MEMO_RUBRIC.names())
