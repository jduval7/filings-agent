"""The reconciler factory: one place builds what runs and what's evaluated; no silent fallbacks."""

from __future__ import annotations

from pathlib import Path

import pytest

from filings_agent.config import AppConfig
from filings_agent.domain import FilingRef
from filings_agent.reconcile_agent import factory


def test_missing_filing_list_is_an_error_not_a_fallback(cfg: AppConfig) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        factory.filings_by_ticker(cfg)


def test_filings_are_grouped_by_ticker_and_unknown_tickers_fail(
    cfg: AppConfig, tmp_path: Path
) -> None:
    ref = FilingRef(ticker="GE", cik=40545, accession_no="A1", form="10-K",
                    filed="2024-02-02", period_of_report="2023-12-31")  # type: ignore[arg-type]  # fmt: skip
    (cfg.data_dir / "gold").mkdir(parents=True)
    (cfg.data_dir / "gold" / "filings_all.jsonl").write_text(ref.model_dump_json() + "\n")
    assert set(factory.filings_for(cfg, "GE")) == {"A1"}
    with pytest.raises(LookupError, match="no filings for KO"):
        factory.filings_for(cfg, "KO")


def test_index_uses_configured_retrieval(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    assert factory.make_index(cfg, "hybrid").mode == "hybrid"
    assert factory.make_index(cfg).mode == cfg.retrieval.mode
