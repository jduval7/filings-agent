"""Shared fixtures: a real AppConfig pointed at temp dirs, so no test touches real ledgers/data."""

from __future__ import annotations

import os
from pathlib import Path

# Tests never send traces. Importing the Studio entry point or running the CLI loads .env, which
# turned tracing on mid-session and used up the LangSmith monthly trace quota (2026-10-22).
# load_dotenv() never overrides variables that are already set, so this wins.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import pytest

from filings_agent.config import AppConfig, load_config


def make_cfg(tmp: Path, project_cap: float = 30.0, monthly_cap: float = 100.0) -> AppConfig:
    base = load_config()
    spend = base.spend.model_copy(
        update={
            "ledger": tmp / "spend.json",
            "shared_ledger": tmp / "shared.json",
            "project_cap_usd": project_cap,
            "monthly_cap_usd": monthly_cap,
            "max_rpm_per_model": 0,  # no rate-limit sleeps in tests unless a test asks
        }
    )
    return base.model_copy(update={"data_dir": tmp, "spend": spend})


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    return make_cfg(tmp_path)


@pytest.fixture(autouse=True)
def _no_committed_corrections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never read the real reviewer corrections (configs/), only what they write."""
    monkeypatch.setattr(
        "filings_agent.evals.reconcile_local.CORRECTIONS", tmp_path / "corrections.jsonl"
    )
