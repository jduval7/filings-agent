"""Config validation: the guard rails that stop a bad config before it costs anything."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from filings_agent.config import AppConfig, ModelSpec, SecConfig, load_config


def _raw() -> dict[str, Any]:
    return yaml.safe_load(Path("configs/default.yaml").read_text())


def test_default_config_loads_and_is_consistent() -> None:
    cfg = load_config()
    assert set(cfg.splits.dev) | set(cfg.splits.test) | set(cfg.splits.fresh) == set(
        cfg.companies
    ) | set(cfg.reconcile_companies)
    assert cfg.roles.extractor in cfg.models and cfg.roles.reconciler in cfg.models
    assert cfg.split_of("GE") == "dev" and cfg.split_of("WMT") == "test"


def test_company_in_both_splits_is_rejected() -> None:
    raw = _raw()
    raw["splits"]["test"].append("GE")
    with pytest.raises(ValidationError, match="more than one split"):
        AppConfig.model_validate(raw)


def test_splits_must_cover_every_company() -> None:
    raw = _raw()
    raw["splits"]["dev"].remove("GE")
    with pytest.raises(ValidationError, match="must equal"):
        AppConfig.model_validate(raw)


def test_role_must_name_a_configured_model() -> None:
    raw = _raw()
    raw["roles"]["extractor"] = "gpt-9-imaginary"
    with pytest.raises(ValidationError, match="not in models"):
        AppConfig.model_validate(raw)


@pytest.mark.parametrize("role", ["reconciler", "memo_writer"])
def test_judge_cannot_grade_its_own_work(role: str) -> None:
    raw = _raw()
    raw["roles"]["judge"] = raw["roles"][role]
    with pytest.raises(ValidationError, match="judge"):
        AppConfig.model_validate(raw)


def test_unknown_model_lookup_names_the_options() -> None:
    with pytest.raises(KeyError, match="known"):
        load_config().model("nope")


def test_model_cost_uses_per_million_prices() -> None:
    spec = ModelSpec(provider="bedrock", id="x", price=(1.0, 5.0))
    assert spec.cost(1_000_000, 200_000) == pytest.approx(2.0)


def test_sec_user_agent_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="SEC requires a contact"):
        SecConfig().user_agent()
    monkeypatch.setenv("SEC_USER_AGENT", "  Jane Doe jane@example.com  ")
    assert SecConfig().user_agent() == "Jane Doe jane@example.com"


def test_prompt_versions_exist_for_configured_steps() -> None:
    from filings_agent.steps import extract, reconcile

    cfg = load_config()
    assert cfg.prompts.extract in extract.PROMPTS
    assert cfg.prompts.reconcile in reconcile.PROMPTS


def test_fresh_split_is_its_own_split_and_unknown_tickers_fail() -> None:
    cfg = load_config()
    assert cfg.split_of("MET") == "fresh" and cfg.split_of("GE") == "dev"
    assert cfg.split_of("WMT") == "test"
    with pytest.raises(KeyError, match="in no split"):
        cfg.split_of("ZZZZ")
