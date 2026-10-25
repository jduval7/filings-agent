"""Config name -> LangChain chat model. The only place that knows about providers.

Every Bedrock client is created with the spend guard attached, so no call can bypass it.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config as BotoConfig
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel

from filings_agent.config import AppConfig
from filings_agent.llm.spend_guard import get_guard, install

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient


@functools.cache
def _bedrock_client(region: str) -> BedrockRuntimeClient:
    # Retries are LangGraph's job (RetryPolicy per node), so botocore makes one attempt only.
    # Its own retries are invisible to the spend guard (botocore emits the events it hooks once per
    # call, not per attempt), and they multiplied LangGraph's: up to 2 x 4 sends per step.
    # "adaptive" still paces requests client-side after throttling.
    return boto3.client(  # pyright: ignore[reportUnknownMemberType]
        "bedrock-runtime",
        region_name=region,
        config=BotoConfig(retries={"max_attempts": 1, "mode": "adaptive"}, read_timeout=120),
    )


def bedrock_client(cfg: AppConfig) -> BedrockRuntimeClient:
    """The shared Bedrock client with the spend guard installed (chat models and embeddings)."""
    return install(_bedrock_client(cfg.bedrock.region), get_guard(cfg))


def chat_model(cfg: AppConfig, name: str, *, max_tokens: int = 4096) -> BaseChatModel:
    spec = cfg.model(name)
    match spec.provider:
        case "bedrock":
            client = install(_bedrock_client(cfg.bedrock.region), get_guard(cfg))
            return ChatBedrockConverse(
                client=client,
                model=spec.id,
                region_name=cfg.bedrock.region,
                temperature=0,
                max_tokens=max_tokens,
            )
