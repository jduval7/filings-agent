"""Local reconcile scoring: labels, accuracy, reviewer-correction replays."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from filings_agent.config import AppConfig
from filings_agent.evals import reconcile_local as rl
from filings_agent.steps.reconcile import Cause, ReconcileOutput, ReconcileTask
from tests.support.eval_data import EVIDENCE, D, patch_chat, returns


def test_load_labels_and_tasks_skip_unlabelled_and_missing_filings(cfg: AppConfig) -> None:
    labels = cfg.data_dir / "labels.yaml"
    labels.write_text(yaml.safe_dump({D.id: {"cause": "stock_split"}, "X:y:FY1": {"cause": None}}))
    assert rl.load_labels(labels) == {D.id: Cause.STOCK_SPLIT, "X:y:FY1": None}
    row = {
        "example_id": "NVDA:diluted_shares:2024-01-28", "split": "dev", "ticker": "NVDA",
        "metric": "diluted_shares", "period_end": "2024-01-28",
        "values": [["N1", "2023-02-01", "2494000000"], ["N2", "2025-02-01", "24940000000"]],
        "ratio": "10", "material": True, "cause": None,
    }  # fmt: skip
    (cfg.data_dir / "datasets").mkdir(parents=True)
    (cfg.data_dir / "datasets" / "reconcile.jsonl").write_text(json.dumps(row) + "\n")
    assert rl.load_tasks(cfg, "dev", labels) == []  # filings not downloaded yet
    (cfg.data_dir / "filings").mkdir()
    for a in ("N1", "N2"):
        (cfg.data_dir / "filings" / f"{a}.md").write_text(
            "Words about the company and its business. " * 20 + EVIDENCE[0]
        )
    (task,) = rl.load_tasks(cfg, "dev", labels)
    assert task.label is Cause.STOCK_SPLIT and task.task.discrepancy_id == D.id
    assert any("stock split" in e for e in task.task.evidence)
    assert rl.load_tasks(cfg, "test", labels) == []


def test_score_reconciler_accuracy_and_mistakes(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = [rl.LabeledTask(ReconcileTask(D.id, D, EVIDENCE), Cause.STOCK_SPLIT)]
    monkeypatch.setattr(rl, "load_tasks", returns(tasks))
    good = ReconcileOutput(cause=Cause.STOCK_SPLIT, explanation="x",
                           evidence_quote="we executed a ten-for-one stock split")  # fmt: skip
    bad = ReconcileOutput(cause=Cause.RECLASSIFICATION, explanation="x",
                          evidence_quote="we executed a ten-for-one stock split")  # fmt: skip
    from filings_agent.reconcile_agent import factory

    patch_chat(monkeypatch, factory, [good, bad])
    row = rl.score_reconciler(cfg, "nova-micro", "v2", "dev", reps=2)
    assert row["cause_accuracy"] == 0.5 and row["cause_accuracy_sd"] > 0
    assert row["mistakes"] == {D.id: "reclassification (label stock_split)"}
    assert "| nova-micro | v2 | 1 | 2 |" in rl.reconcile_table([row])


def test_load_corrections_replays_inputs(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    assert rl.load_corrections(path) == []
    row = {"inputs": {"discrepancy": D.model_dump(mode="json"), "evidence": list(EVIDENCE)},
           "outputs": {"cause": "stock_split"}}  # fmt: skip
    path.write_text(json.dumps(row) + "\n\n")
    (t,) = rl.load_corrections(path)
    assert t.task.discrepancy_id == f"{D.id}#corr" and t.task.evidence == EVIDENCE


def test_every_fresh_label_records_how_it_was_verified() -> None:
    """The fresh labels were drafted by the coding assistant and checked by the author against each
    company's adoption table. Each one says how (tie_out or narrative), so results can be reported
    on all labels and on the ones that tie out."""
    from filings_agent.config import load_config

    fresh_companies = set(load_config().splits.fresh)
    raw = yaml.safe_load(rl.LABELS.read_text())
    fresh = {
        k for k, v in raw.items() if v and v.get("cause") and k.split(":")[0] in fresh_companies
    }
    verified = rl.load_verification()
    assert fresh and fresh <= set(verified)
    assert set(verified.values()) <= {"tie_out", "narrative"}
    assert all(raw[k].get("verified_note") for k in fresh)
