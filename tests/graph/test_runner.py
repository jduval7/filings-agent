"""Runner: thread identity, status reporting, and real graph runs with fake dependencies."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from filings_agent.config import AppConfig
from filings_agent.domain import FilingRef
from filings_agent.graph import runner
from filings_agent.graph.build import Deps
from filings_agent.storage.ledger import Ledger
from tests.support.graph_doubles import (
    METRICS,
    FakeExtractor,
    FakeReconciler,
    FakeStore,
    RestatingExtractor,
    TwoFilingStore,
)


def test_run_identity_covers_the_filings_and_every_setting_that_shapes_the_result(
    cfg: AppConfig,
) -> None:
    """Review finding: identity was tickers + extractor + prompt, so a new 10-K or a new
    reconcile model replayed the old result."""
    key = runner.request_key(runner.run_identity(cfg, ["GE", "NVDA"], "m", "v2"))
    assert key == runner.request_key(runner.run_identity(cfg, ["NVDA", "GE"], "m", "v2"))
    assert len(key) == 16
    assert key != runner.request_key(runner.run_identity(cfg, ["GE", "NVDA"], "m2", "v2"))
    assert key != runner.request_key(runner.run_identity(cfg, ["GE", "NVDA"], "m", "v1"))
    other_reconciler = cfg.model_copy(
        update={"roles": cfg.roles.model_copy(update={"reconciler": "nova-pro"})}
    )
    assert key != runner.request_key(
        runner.run_identity(other_reconciler, ["GE", "NVDA"], "m", "v2")
    )
    auto = cfg.model_copy(
        update={
            "reconcile": cfg.reconcile.model_copy(
                update={"auto_approve": not cfg.reconcile.auto_approve}
            )
        }
    )
    assert key != runner.request_key(runner.run_identity(auto, ["GE", "NVDA"], "m", "v2"))

    ref = FilingRef(ticker="GE", cik=1, accession_no="A9", form="10-K", filed=date(2026, 2, 1),
                    period_of_report=date(2025, 12, 31))  # fmt: skip
    (cfg.data_dir / "gold").mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "gold" / "filings.jsonl").write_text(ref.model_dump_json() + "\n")
    assert (
        runner.filings_on_disk(cfg, ["GE"]) == ["A9"] and runner.filings_on_disk(cfg, ["KO"]) == []
    )
    assert key != runner.request_key(
        runner.run_identity(cfg, ["GE", "NVDA"], "m", "v2")
    )  # new 10-K


def test_local_filing_store(tmp_path: Path) -> None:
    ref = FilingRef(ticker="GE", cik=1, accession_no="A1", form="10-K", filed=date(2025, 2, 1),
                    period_of_report=date(2024, 12, 31))  # fmt: skip
    (tmp_path / "gold").mkdir()
    (tmp_path / "gold" / "filings.jsonl").write_text(ref.model_dump_json() + "\n")
    (tmp_path / "filings").mkdir()
    (tmp_path / "filings" / "A1.md").write_text("# 10-K")
    store = runner.LocalFilingStore(tmp_path)
    assert store.filings("GE") == [ref] and store.filings("NVDA") == []
    assert store.markdown(ref) == "# 10-K"


def _snapshot(
    values: dict[str, Any], interrupts: tuple[Any, ...] = (), nxt: tuple[str, ...] = ()
) -> Any:
    task = SimpleNamespace(interrupts=[SimpleNamespace(value=v) for v in interrupts])
    return SimpleNamespace(values=values, tasks=[task], next=nxt)


@pytest.mark.parametrize(
    ("snapshot", "status"),
    [
        (_snapshot({}), "not_found"),
        (_snapshot({"tickers": ["GE"]}, nxt=("extract",)), "interrupted"),
        (
            _snapshot({"tickers": ["GE"]}, interrupts=({"pending": [{"id": "x"}]},)),
            "awaiting_review",
        ),
        (_snapshot({"tickers": ["GE"], "publication": {"facts": 1}}), "published"),
        (_snapshot({"tickers": ["GE"]}), "done"),
    ],
)
def test_summarize_status(snapshot: Any, status: str) -> None:
    assert runner.summarize("t", snapshot)["status"] == status


def test_get_status_of_unknown_thread(cfg: AppConfig) -> None:
    assert runner.get_status(cfg, "nope")["status"] == "not_found"
    assert runner.pending_review(cfg, "nope") == []


def _fake_deps(cfg: AppConfig, deps: Deps) -> Any:
    def make(*_: Any, **__: Any) -> Deps:
        return deps

    return make


def test_run_analysis_end_to_end_with_events(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    deps = Deps(FakeStore(), FakeExtractor(), METRICS, ledger=Ledger(cfg.data_dir / "l.sqlite"))
    monkeypatch.setattr(runner, "make_deps", _fake_deps(cfg, deps))
    events: list[dict[str, Any]] = []
    summary = runner.run_analysis(cfg, ["AAA"], "nova-micro", "v2", thread_id="r1",
                                  on_event=events.append)  # fmt: skip
    assert summary["status"] == "published" and summary["facts"] == 6 and summary["branches"] == 6
    assert summary["run_id"] and summary["wall_s"] >= 0
    kinds = [e["event"] for e in events]
    assert kinds[0] == "planned" and kinds.count("branch_done") == 6 and kinds[-1] == "published"
    assert runner.get_status(cfg, "r1")["status"] == "published"  # persisted in the checkpointer


def test_run_analysis_pauses_then_resumes_with_decisions(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    deps = Deps(TwoFilingStore(), RestatingExtractor(), METRICS, reconciler=FakeReconciler(),
                ledger=Ledger(cfg.data_dir / "l.sqlite"))  # fmt: skip
    monkeypatch.setattr(runner, "make_deps", _fake_deps(cfg, deps))
    paused = runner.run_analysis(cfg, ["AAA"], "nova-micro", "v2", thread_id="r2")
    assert paused["status"] == "awaiting_review"
    assert runner.pending_review(cfg, "r2") == ["AAA:revenue:FY2023"]
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert printed[0]["event"] == "planned"  # default event sink prints JSON lines
    done = runner.run_analysis(cfg, ["AAA"], "nova-micro", "v2", thread_id="r2",
                               decisions={"AAA:revenue:FY2023": {"action": "accept"}})  # fmt: skip
    assert done["status"] == "published"
    assert done["resolutions"]["AAA:cash:FY2023"]["cause"] == "immaterial_revision"


def test_rerunning_a_finished_request_replays_it_instead_of_discarding_new_work(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding: new input on a published thread re-paid the work, kept the old decisions
    and then published nothing. Now the stored result comes back and nothing runs."""
    extractor = FakeExtractor()
    deps = Deps(FakeStore(), extractor, METRICS, ledger=Ledger(cfg.data_dir / "l.sqlite"))
    monkeypatch.setattr(runner, "make_deps", _fake_deps(cfg, deps))
    first = runner.run_analysis(cfg, ["AAA"], "nova-micro", "v2", thread_id="r3",
                                on_event=lambda _: None)  # fmt: skip
    calls = extractor.calls
    events: list[dict[str, Any]] = []
    again = runner.run_analysis(cfg, ["AAA"], "nova-micro", "v2", thread_id="r3",
                                on_event=events.append)  # fmt: skip
    assert again["replayed"] is True and again["status"] == first["status"] == "published"
    assert extractor.calls == calls and events == []  # nothing ran


def test_new_input_on_a_thread_waiting_for_review_is_refused(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    deps = Deps(TwoFilingStore(), RestatingExtractor(), METRICS, reconciler=FakeReconciler(),
                ledger=Ledger(cfg.data_dir / "l.sqlite"))  # fmt: skip
    monkeypatch.setattr(runner, "make_deps", _fake_deps(cfg, deps))

    def quiet(_: dict[str, Any]) -> None:
        return None

    runner.run_analysis(cfg, ["AAA"], "nova-micro", "v2", thread_id="r4", on_event=quiet)
    with pytest.raises(ValueError, match="unfinished"):
        runner.run_analysis(cfg, ["AAA"], "nova-micro", "v2", thread_id="r4", on_event=quiet)
    assert runner.pending_review(cfg, "r4") == ["AAA:revenue:FY2023"]  # still waiting, intact


def test_a_fresh_run_gets_its_own_thread(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    deps = Deps(FakeStore(), FakeExtractor(), METRICS, ledger=Ledger(cfg.data_dir / "l.sqlite"))
    monkeypatch.setattr(runner, "make_deps", _fake_deps(cfg, deps))

    def quiet(_: dict[str, Any]) -> None:
        return None

    a = runner.run_analysis(cfg, ["AAA"], "nova-micro", "v2", on_event=quiet)
    b = runner.run_analysis(cfg, ["AAA"], "nova-micro", "v2", on_event=quiet, rep=7)
    assert b["thread_id"] == a["thread_id"] + "-r7" and "replayed" not in b


def test_a_thread_leased_by_another_process_is_not_run(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding: the CLI and the server (or two workers) could run one thread at once."""
    from filings_agent.storage.leases import Leases, ThreadBusy

    extractor = FakeExtractor()
    deps = Deps(FakeStore(), extractor, METRICS, ledger=Ledger(cfg.data_dir / "l.sqlite"))
    monkeypatch.setattr(runner, "make_deps", _fake_deps(cfg, deps))
    leases = Leases(cfg.data_dir / "leases.sqlite")
    assert leases.acquire("busy", "the-server")
    with pytest.raises(ThreadBusy):
        runner.run_analysis(cfg, ["AAA"], "nova-micro", "v2", thread_id="busy",
                            on_event=lambda _: None)  # fmt: skip
    assert extractor.calls == 0
    # The owner itself re-enters its lease (the API takes it, then the run uses it).
    done = runner.run_analysis(cfg, ["AAA"], "nova-micro", "v2", thread_id="busy",
                               on_event=lambda _: None, lease_owner="the-server")  # fmt: skip
    assert done["status"] == "published" and leases.holder("busy") is None  # released


def test_runs_share_one_ledger_connection_per_file(tmp_path: Path) -> None:
    """Review finding: every run opened a ledger connection and never closed it."""
    assert runner.shared_ledger(tmp_path / "l.sqlite") is runner.shared_ledger(
        tmp_path / "l.sqlite"
    )


def test_checkpoint_serializer_allows_our_types_only() -> None:
    allowed = {name for _, name in runner.CHECKPOINT_TYPES}
    assert {"ExtractTask", "ExtractedFact", "Resolution", "Discrepancy", "Usage"} <= allowed


def test_make_deps_wires_real_components(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    (cfg.data_dir / "gold").mkdir(parents=True)
    (cfg.data_dir / "gold" / "filings.jsonl").write_text("")
    deps = runner.make_deps(cfg, "nova-micro", "v2", rep=7)
    assert deps.extractor.name == "statement/nova-micro/v2"
    assert deps.reconciler is not None
    assert deps.reconciler.name == f"reconcile/{cfg.roles.reconciler}/{cfg.prompts.reconcile}"
    assert deps.memo_writer is not None and deps.ledger is not None and deps.rep == 7
    assert deps.memo_writer._cache is not None  # pyright: ignore[reportPrivateUsage]  # crash-safe
