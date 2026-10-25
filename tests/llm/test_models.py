"""Model factory: config name -> chat model, always wired through the spend guard."""

from __future__ import annotations

import pytest
from langchain_aws import ChatBedrockConverse

from filings_agent.config import AppConfig
from filings_agent.llm import models


def test_every_configured_model_builds_with_the_guard_installed(cfg: AppConfig) -> None:
    for name, spec in cfg.models.items():
        chat = models.chat_model(cfg, name, max_tokens=123)
        assert isinstance(chat, ChatBedrockConverse)
        assert chat.model_id == spec.id
        assert chat.temperature == 0  # determinism: every measured call is temperature 0
        assert chat.max_tokens == 123
        assert getattr(chat.client, "_spend_guard_installed", False), name


def test_unknown_model_name_fails_fast(cfg: AppConfig) -> None:
    with pytest.raises(KeyError):
        models.chat_model(cfg, "not-a-model")


def test_clients_are_shared_per_region(cfg: AppConfig) -> None:
    a = models.chat_model(cfg, "nova-micro")
    b = models.chat_model(cfg, "nova-lite")
    assert isinstance(a, ChatBedrockConverse) and isinstance(b, ChatBedrockConverse)
    assert a.client is b.client  # one guarded client, one connection pool
