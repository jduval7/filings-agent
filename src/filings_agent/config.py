"""Typed config loaded from YAML. One file decides companies, splits, metrics (and later models)."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

from filings_agent.domain import Metric


class SecConfig(BaseModel):
    user_agent_env: str = "SEC_USER_AGENT"
    max_requests_per_second: float = 5.0

    def user_agent(self) -> str:
        ua = os.environ.get(self.user_agent_env, "").strip()
        if not ua:
            raise RuntimeError(
                f"Set {self.user_agent_env}='Your Name you@example.com' (SEC requires a contact)."
            )
        return ua


class FilingsConfig(BaseModel):
    form: str = "10-K"
    filed_since: date
    # Reconcile-only companies go back further: their splits/spin-offs happened in 2022-2024.
    reconcile_filed_since: date


class Splits(BaseModel):
    dev: list[str]
    test: list[str]
    # Added after every choice was frozen, to check generalization: never tuned on, never in the
    # CI gate (which scores dev). Reported separately.
    fresh: list[str] = []


class ModelSpec(BaseModel):
    provider: Literal["bedrock"]
    id: str
    price: tuple[float, float]  # USD per 1M tokens (input, output)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.price[0] + output_tokens * self.price[1]) / 1e6


class Roles(BaseModel):
    extractor: str
    reconciler: str
    memo_writer: str
    judge: str


class Prompts(BaseModel):
    """Prompt version per step. Used by the runner, the API and the CI gate alike, so what's
    evaluated is exactly what runs (a class default once silently ran reconcile v1 in prod)."""

    extract: str = "v2"
    reconcile: str = "v2"


class ReconcileConfig(BaseModel):
    strategy: Literal["pipeline", "agent"] = "pipeline"
    agent_version: str = "v1"
    # Skip human review for answers triage fully confirms (steps/triage.py). Off by default in
    # code so nothing is auto-approved unless the config says so.
    auto_approve: bool = False


class ResearchConfig(BaseModel):
    """The reviewer-triggered research agent (research/agent.py). Off unless the config says so."""

    enabled: bool = False
    model: str = "claude-sonnet-4.6"
    max_rounds: int = 2  # per case, then the reviewer must decide or escalate


class EmbeddingsConfig(BaseModel):
    id: str
    price: tuple[float, float]  # USD per 1M tokens (input, output)
    # Embeddings cost ~$0.000002/call; an LLM-sized rate limit would only slow indexing down.
    max_rpm: float = 1200


class RetrievalConfig(BaseModel):
    chunk_size: int = 1500
    chunk_overlap: int = 200
    k: int = 5
    mode: Literal["vector", "hybrid"] = "vector"


class BedrockConfig(BaseModel):
    region: str = "us-east-1"
    profile_env: str = "AWS_PROFILE"


class SpendConfig(BaseModel):
    project_cap_usd: float
    ledger: Path
    shared_ledger: Path | None = None
    monthly_cap_usd: float = 100.0
    max_rpm_per_model: float = 300.0


class ApiConfig(BaseModel):
    """Who may call the HTTP API. Tokens are secrets, so they come from the environment:
    `FILINGS_API_TOKENS="alice:<token>,bob:<token>"`. The name is what the ledger records as the
    reviewer. Unset: local mode (the server refuses to listen on anything but loopback)."""

    tokens_env: str = "FILINGS_API_TOKENS"

    def tokens(self) -> dict[str, str]:
        """token -> user name."""
        out: dict[str, str] = {}
        for entry in os.environ.get(self.tokens_env, "").split(","):
            name, sep, token = entry.strip().partition(":")
            if not entry.strip():
                continue
            if not sep or not name or len(token) < 16:
                raise RuntimeError(
                    f"{self.tokens_env}: each entry is name:token, token at least 16 characters"
                )
            out[token] = name
        return out


class AppConfig(BaseModel):
    sec: SecConfig
    data_dir: Path
    # Where runs live (checkpoints, the run list, the ledger, leases). Defaults to data_dir. The
    # smoke, browser and crash tests point it at a throwaway folder (FILINGS_STATE_DIR) so their
    # runs never show up in the review screen; they still read the shared filings and LLM cache.
    state_dir: Path | None = None
    filings: FilingsConfig
    companies: list[str]
    # Only used for the reconcile dataset (answer key + notes); never run through extraction.
    reconcile_companies: list[str] = []
    splits: Splits
    metrics: list[Metric]
    models: dict[str, ModelSpec]
    roles: Roles
    prompts: Prompts = Prompts()
    reconcile: ReconcileConfig = ReconcileConfig()
    research: ResearchConfig = ResearchConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig(
        id="amazon.titan-embed-text-v2:0", price=(0.02, 0.0)
    )
    retrieval: RetrievalConfig = RetrievalConfig()
    bedrock: BedrockConfig = BedrockConfig()
    api: ApiConfig = ApiConfig()
    spend: SpendConfig

    @model_validator(mode="after")
    def _splits_cover_companies(self) -> AppConfig:
        dev, test, fresh = set(self.splits.dev), set(self.splits.test), set(self.splits.fresh)
        overlap = (dev & test) | (dev & fresh) | (test & fresh)
        if overlap:
            raise ValueError(f"companies in more than one split: {sorted(overlap)}")
        everyone = set(self.companies) | set(self.reconcile_companies)
        if dev | test | fresh != everyone:
            raise ValueError("splits.dev + test + fresh must equal companies + reconcile_companies")
        for role, name in self.roles.model_dump().items():
            if name not in self.models:
                raise ValueError(f"roles.{role} = {name!r} is not in models")
        # The judge grades reconcile explanations and memos (extraction is graded by code), so it
        # must not be either of the models that wrote them: no self-grading.
        if self.roles.judge in {self.roles.reconciler, self.roles.memo_writer}:
            raise ValueError("the judge must not be a model it judges")
        return self

    def state(self, name: str) -> Path:
        """A run-state file: checkpoints.sqlite, runs.sqlite, ledger.sqlite, leases.sqlite."""
        return (self.state_dir or self.data_dir) / name

    def model(self, name: str) -> ModelSpec:
        if name not in self.models:
            raise KeyError(f"unknown model {name!r}; known: {sorted(self.models)}")
        return self.models[name]

    def split_of(self, ticker: str) -> str:
        for name in ("dev", "test", "fresh"):
            if ticker in getattr(self.splits, name):
                return name
        raise KeyError(f"{ticker} is in no split")


def load_config(path: Path | str = "configs/default.yaml") -> AppConfig:
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    if os.environ.get("FILINGS_STATE_DIR"):
        raw["state_dir"] = os.environ["FILINGS_STATE_DIR"]
    return AppConfig.model_validate(raw)
