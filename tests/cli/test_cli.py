"""CLI: every command parses and dispatches; build-data runs end to end against a fake SEC."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from filings_agent import cli
from filings_agent.domain import FilingRef
from filings_agent.evals import extract_experiments as ee
from filings_agent.evals import reconcile_experiments as re_
from filings_agent.evals.gate import Check
from filings_agent.evals.local import RepScore


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    raw = yaml.safe_load(Path("configs/default.yaml").read_text())
    raw["data_dir"] = str(tmp_path / "data")
    raw["spend"]["ledger"] = str(tmp_path / "spend.json")
    raw["spend"]["shared_ledger"] = str(tmp_path / "shared.json")
    raw["companies"], raw["reconcile_companies"] = ["GE", "WMT"], ["AMZN"]
    raw["splits"] = {"dev": ["GE", "AMZN"], "test": ["WMT"]}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _main(monkeypatch: pytest.MonkeyPatch, config_file: Path, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["filings-agent", "--config", str(config_file), *args])
    monkeypatch.setattr(cli, "_langsmith", lambda: None)  # no LangSmith client in tests
    cli.main()


def _ret(value: Any) -> Any:
    def fn(*_: Any, **__: Any) -> Any:
        return value

    return fn


def test_spend(
    monkeypatch: pytest.MonkeyPatch, config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _main(monkeypatch, config_file, "spend")
    out = json.loads(capsys.readouterr().out)
    assert out["project_usd"] == 0 and out["project_cap"] == 30.0


def test_score_extract_prints_table(
    monkeypatch: pytest.MonkeyPatch, config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from filings_agent.evals import local

    rep = RepScore(0, {"f1": 0.9, "precision": 0.9, "recall": 0.9, "abstain_accuracy": 1.0,
                       "schema_error_rate": 0.0}, 0.01, 1.0, 2.0, 3, {})  # fmt: skip
    monkeypatch.setattr(local, "score_model", _ret([rep]))
    _main(monkeypatch, config_file, "score", "extract", "--models", "nova-micro,nova-lite")
    out = capsys.readouterr().out
    assert "| nova-micro | v1 | 1 | 0.900" in out and "| nova-lite |" in out


def test_score_reconcile_prints_table(
    monkeypatch: pytest.MonkeyPatch, config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from filings_agent.evals import reconcile_local

    row: dict[str, Any] = {"model": "m", "prompt": "v2", "n": 3, "reps": 1, "cause_accuracy": 1.0,
           "cause_accuracy_sd": 0.0, "verified_rate": 1.0, "unknown_rate": 0.0,
           "cost_per_run_usd": 0.1, "p95_s": 1.0, "mistakes": {}, "auto_rate": 0.5,
           "auto_wrong": []}  # fmt: skip
    monkeypatch.setattr(reconcile_local, "score_reconciler", _ret(row))
    _main(
        monkeypatch, config_file, "score", "reconcile", "--models", "nova-micro", "--prompt", "v2"
    )
    assert "| m | v2 | 3 | 1 | 1.00" in capsys.readouterr().out


@pytest.mark.parametrize(("ok", "code"), [(True, 0), (False, 1)])
def test_gate_exit_code(
    monkeypatch: pytest.MonkeyPatch, config_file: Path, ok: bool, code: int
) -> None:
    from filings_agent.evals import gate

    checks = [Check("extract.f1", 0.99 if ok else 0.5, 0.97, None, 0.02)]
    monkeypatch.setattr(gate, "run_gate", _ret((ok, checks, {})))
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, config_file, "gate")
    assert exc.value.code == code


def test_sweep_rejects_unknown_model_before_spending(
    monkeypatch: pytest.MonkeyPatch, config_file: Path
) -> None:
    from filings_agent.evals import extract_experiments

    called: list[bool] = []

    def fake_sweep(*_: Any, **__: Any) -> list[Any]:
        called.append(True)
        return []

    monkeypatch.setattr(extract_experiments, "run_extract_sweep", fake_sweep)
    with pytest.raises(KeyError, match="gpt-imaginary"):
        _main(monkeypatch, config_file, "sweep", "extract", "--models", "nova-micro,gpt-imaginary")
    assert called == []


def test_run_and_review_dispatch(
    monkeypatch: pytest.MonkeyPatch, config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from filings_agent.graph import runner

    calls: list[dict[str, Any]] = []

    def fake_run(
        cfg: Any,
        tickers: list[str],
        model: str,
        prompt: str,
        thread: Any = None,
        resume: bool = False,
        concurrency: int = 8,
        **kw: Any,
    ) -> dict[str, Any]:
        calls.append(
            {
                "tickers": tickers,
                "model": model,
                "prompt": prompt,
                "thread": thread,
                "resume": resume,
                **kw,
            }  # fmt: skip
        )
        return {"status": "ok"}

    monkeypatch.setattr(runner, "run_analysis", fake_run)
    monkeypatch.setattr(runner, "pending_review", _ret(["A:revenue:FY2023", "B:cash:FY2024"]))
    _main(monkeypatch, config_file, "run", "--tickers", "ge, nvda")
    assert calls[-1] == {"tickers": ["GE", "NVDA"], "model": "gpt-oss-120b", "prompt": "v2",
                         "thread": None, "resume": False, "rep": 0}  # fmt: skip
    _main(
        monkeypatch, config_file, "run", "--tickers", "GE", "--fresh", "--resume", "--thread", "t"
    )
    assert calls[-1]["rep"] > 0 and calls[-1]["resume"] and calls[-1]["thread"] == "t"
    _main(monkeypatch, config_file, "review", "--thread", "t", "--tickers", "GE", "--accept-all",
          "--reviewer", "alice")  # fmt: skip
    decisions = calls[-1]["decisions"]
    assert {k: (d["action"], d["note"], d["actor"]) for k, d in decisions.items()} == {
        "A:revenue:FY2023": ("accept", "accept-all", "alice"),
        "B:cash:FY2024": ("accept", "accept-all", "alice"),
    }
    assert all(d["decided_at"].startswith("20") for d in decisions.values())
    assert '"status": "ok"' in capsys.readouterr().out


def test_export_corrections_upload_and_serve_dispatch(
    monkeypatch: pytest.MonkeyPatch, config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import uvicorn

    from filings_agent.evals import corrections, extract_experiments, reconcile_experiments

    monkeypatch.setattr(corrections, "export_corrections", _ret(2))
    _main(monkeypatch, config_file, "export-corrections")
    assert "exported 2 correction(s)" in capsys.readouterr().out
    from filings_agent.evals import memo_eval, trajectory

    monkeypatch.setattr(extract_experiments, "upload_extract_datasets", _ret({"a": 1}))
    monkeypatch.setattr(reconcile_experiments, "upload_reconcile_datasets", _ret({"b": 2}))
    monkeypatch.setattr(memo_eval, "upload_memo_datasets", _ret({"c": 3}))
    monkeypatch.setattr(trajectory, "upload_e2e_dataset", _ret(4))
    _main(monkeypatch, config_file, "upload-datasets")
    assert json.loads(capsys.readouterr().out) == {
        "a": 1,
        "b": 2,
        "c": 3,
        "filings-agent/e2e-dev": 4,
    }
    served: list[tuple[str, int]] = []

    def fake_uvicorn_run(app: Any, host: str, port: int) -> None:
        served.append((host, port))

    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
    _main(monkeypatch, config_file, "serve", "--port", "9999")
    assert served == [("127.0.0.1", 9999)]
    # Review finding: no authentication. Without tokens, only loopback is served.
    monkeypatch.delenv("FILINGS_API_TOKENS", raising=False)
    with pytest.raises(SystemExit, match="without authentication"):
        _main(monkeypatch, config_file, "serve", "--host", "0.0.0.0")  # nosec B104 - refused
    monkeypatch.setenv("FILINGS_API_TOKENS", "alice:alice-token-0123456789")
    _main(monkeypatch, config_file, "serve", "--host", "0.0.0.0")  # nosec B104 - authenticated
    assert served[-1] == ("0.0.0.0", 8000)  # nosec B104


ROWS = "\n".join(f"| Line {i} | {1000 + i:,} |  | {900 + i:,} |  |" for i in range(8))


MD = (
    "**Consolidated Statements of Income**\n(In millions)\n"
    f"|  | 2024 |  | 2023 |  |\n{ROWS}\n| Total revenues | $ | 5,000 | $ | 4,000 |\n"
    "| Net income | $ | 700 | $ | 600 |\n"
)


class FakeSource:
    def __init__(self) -> None:
        self.markdown_calls: list[str] = []

    def list_filings(self, ticker: str, form: str, filed_since: date) -> list[FilingRef]:
        return [FilingRef(ticker=ticker, cik=1, accession_no=f"{ticker}-1", form="10-K",
                          filed=date(2025, 2, 1), period_of_report=date(2024, 12, 31))]  # fmt: skip

    def company_facts(self, ticker: str) -> dict[str, Any]:
        rev = [
            {"val": 5_000_000_000, "accn": f"{ticker}-1", "start": "2024-01-01",
             "end": "2024-12-31", "form": "10-K"},
            {"val": 4_000_000_000, "accn": f"{ticker}-1", "start": "2023-01-01",
             "end": "2023-12-31", "form": "10-K"},
        ]  # fmt: skip
        return {"facts": {"us-gaap": {"Revenues": {"units": {"USD": rev}}}}}

    def markdown(self, filing: FilingRef) -> str:
        self.markdown_calls.append(filing.accession_no)
        return MD


def test_build_data_writes_answer_key_datasets_and_report(
    monkeypatch: pytest.MonkeyPatch, config_file: Path, tmp_path: Path
) -> None:
    src = FakeSource()
    monkeypatch.setattr(cli, "_source", _ret(src))
    _main(monkeypatch, config_file, "build-data")
    gold = tmp_path / "data" / "gold"
    assert len((gold / "filings.jsonl").read_text().splitlines()) == 2  # GE + WMT
    assert len((gold / "gold.jsonl").read_text().splitlines()) == 4  # 2 years x 2 companies
    report = (gold / "REPORT.md").read_text()
    assert "## GE (dev)" in report and "income_statement=ok[millions]" in report
    assert "Reconcile-only companies: 1" in report
    extract = (tmp_path / "data" / "datasets" / "extract.jsonl").read_text().splitlines()
    assert {json.loads(r)["split"] for r in extract} == {"dev", "test"}
    assert "AMZN-1" in src.markdown_calls  # reconcile-only filings are downloaded too


def test_cli_sweep_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    import yaml

    from filings_agent import cli

    raw = yaml.safe_load(Path("configs/default.yaml").read_text())
    raw["data_dir"] = str(tmp_path / "d")
    raw["spend"]["ledger"] = str(tmp_path / "s.json")
    raw["spend"]["shared_ledger"] = str(tmp_path / "sh.json")
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw))

    def fake_extract_sweep(*_: Any, **__: Any) -> list[dict[str, Any]]:
        return [{"sweep": "S", "model": "nova-micro", "prompt_version": "v2", "rep": 0, "f1": 1.0,
                 "precision": 1.0, "recall": 1.0, "abstain_accuracy": 1.0,
                 "schema_error_rate": 0.0, "cost_usd": 0.01, "latency_p50_s": 1.0,
                 "latency_p95_s": 2.0}]  # fmt: skip

    def fake_reconcile_sweep(*_: Any, **__: Any) -> list[dict[str, Any]]:
        return [{"model": "nova-micro", "cause_accuracy": 1.0}]

    monkeypatch.setattr(ee, "run_extract_sweep", fake_extract_sweep)
    monkeypatch.setattr(re_, "run_reconcile_sweep", fake_reconcile_sweep)
    monkeypatch.setattr(cli, "_langsmith", lambda: None)
    for args in (["sweep", "extract", "--models", "nova-micro", "--split", "test"],
                 ["sweep", "reconcile", "--models", "nova-micro", "--split", "test"]):  # fmt: skip
        monkeypatch.setattr(sys, "argv", ["filings-agent", "--config", str(path), *args])
        cli.main()
    out = capsys.readouterr().out
    assert out.count("NOTE: test split") == 2  # loud warning: final confirmation only
    assert "| nova-micro | v2 | 0 | 1.000" in out and '"cause_accuracy": 1.0' in out
    assert (tmp_path / "d" / "sweeps" / "extract-S.md").exists()


CARD: dict[str, Any] = {
    "id": "GE:net_income:FY2022",
    "change": "net_income FY2022: 225 -> 339 (+50.7%)",
    "changed_in": ["A"],
    "candidates": [{"cause": "accounting_change", "fit": "possible", "why": "w", "evidence": []}],
}


# (argv, {module.attr: fake return value}, expected in stdout)
DISPATCH: list[tuple[list[str], dict[str, Any], str]] = [
        (["cards", "--eval"], {"evals.reconcile_local.card_eval": {"split": "dev", "n": 1}},
         '"n": 1'),
        (["cards", "--thread", "t1"], {"graph.runner.pending_cards": [CARD]},
         "net_income FY2022: 225 -> 339"),
        (["cards", "--thread", "t2"], {"graph.runner.pending_cards": []},
         "nothing waiting for review"),
        (["score", "memo", "--models", "claude-haiku-4.5"],
         {"evals.memo_eval.score_memo": {"model": "m", "verdicts": []},
          "evals.memo_eval.save_report": None}, '"model": "m"'),
        (["compare", "memo", "--models", "a,b"],
         {"evals.memo_eval.compare_memos": {"a_wins": 1, "details": []}}, '"a_wins": 1'),
        (["compare", "memo", "--experiments", "e1,e2"],
         {"evals.memo_eval.run_pairwise_experiment": "pairwise-exp"}, "pairwise-exp"),
        (["calibrate", "prepare"], {"evals.calibration.prepare": [1, 2]}, "2 items"),
        (["calibrate", "score", "--upload"],
         {"evals.calibration.score": {}, "evals.calibration.report_markdown": "REPORT",
          "evals.calibration.save": "r.json", "evals.calibration.upload_calibration": None,
          "evals.calibration.run_calibration_experiment": "cal-exp"}, "cal-exp"),
        (["queue", "setup"],
         {"evals.online.ensure_queue": SimpleNamespace(name="q", id="1")}, "queue ready: q"),
        (["sweep", "memo", "--models", "claude-haiku-4.5"],
         {"evals.memo_eval.run_memo_sweep": [{"exp": "m1"}]}, '"exp": "m1"'),
        (["sweep", "e2e"], {"evals.trajectory.run_e2e_experiment": "e2e-exp"}, "e2e-exp"),
        (["sweep", "research"],
         {"evals.research_experiments.run_research_experiment": {"experiment": "r-exp"}},
         "r-exp"),
        (["score", "reconcile", "--reconciler", "agent", "--models", "claude-sonnet-4.6"],
         {"evals.agent_eval.score_agent": ({"retrieval_mode": "vector", "n": 1}, []),
          "evals.agent_eval.table": "AGENT TABLE"}, "AGENT TABLE"),
]  # fmt: skip


@pytest.mark.parametrize(("argv", "fakes", "expected"), DISPATCH)
def test_command_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    config_file: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    fakes: dict[str, Any],
    expected: str,
) -> None:
    import importlib

    for target, value in fakes.items():
        module, attr = target.rsplit(".", 1)
        monkeypatch.setattr(importlib.import_module(f"filings_agent.{module}"), attr, _ret(value))
    _main(monkeypatch, config_file, *argv)
    assert expected in capsys.readouterr().out


def test_card_preview_rejects_unknown_cases(
    monkeypatch: pytest.MonkeyPatch, config_file: Path
) -> None:
    from filings_agent.evals import reconcile_local

    monkeypatch.setattr(reconcile_local, "load_tasks", _ret([]))
    with pytest.raises(SystemExit, match="not in the labelled datasets"):
        _main(monkeypatch, config_file, "cards", "--case", "NOPE:revenue:FY2020")


def test_online_evaluator_create_posts_the_rule_with_a_backfill(
    monkeypatch: pytest.MonkeyPatch, config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from filings_agent.evals import online_judge

    seen: dict[str, Any] = {}

    def create_rule(api: Any, project_id: str, sampling: float, since: str | None) -> str:
        seen.update(project=project_id, sampling=sampling, since=since)
        return "rule-1"

    client = SimpleNamespace(api_url="https://ls.test", read_project=_ret(SimpleNamespace(id="p1")))
    monkeypatch.setattr(online_judge, "create_rule", create_rule)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    argv = ["online-evaluator", "--create", "demo", "--sampling", "0.5", "--backfill-hours", "6"]
    monkeypatch.setattr(sys, "argv", ["filings-agent", "--config", str(config_file), *argv])
    monkeypatch.setattr(cli, "_langsmith", lambda: client)
    cli.main()
    out = capsys.readouterr().out
    assert "rule-1 on demo (sampling 50%)" in out
    assert seen["project"] == "p1" and seen["sampling"] == 0.5 and seen["since"]
