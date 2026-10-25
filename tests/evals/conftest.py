"""Fixtures for the eval tests."""

from __future__ import annotations

import pytest

from filings_agent.config import AppConfig
from filings_agent.datasets import write_jsonl
from tests.support.eval_data import SECTION, extract_example


@pytest.fixture
def data(cfg: AppConfig) -> AppConfig:
    write_jsonl(
        cfg.data_dir / "datasets" / "extract.jsonl",
        [extract_example(), extract_example("test", "B1")],
    )
    (cfg.data_dir / "filings").mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "filings" / "A1.md").write_text("intro\n" + SECTION)
    return cfg
