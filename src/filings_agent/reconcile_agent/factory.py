"""Build the configured reconciler. One place decides pipeline vs agent, used by the graph runner,
the API, the CLI and the evals, so what's evaluated is exactly what runs."""

from __future__ import annotations

import functools
from collections import defaultdict
from pathlib import Path

from langchain_aws import BedrockEmbeddings

from filings_agent.config import AppConfig
from filings_agent.domain import FilingRef
from filings_agent.graph.build import Reconciler
from filings_agent.llm.cache import StepCache
from filings_agent.llm.models import bedrock_client, chat_model
from filings_agent.reconcile_agent.agent import AgentReconciler
from filings_agent.reconcile_agent.retrieval import CachedEmbeddings, FilingIndex
from filings_agent.steps.reconcile import LLMReconciler


@functools.cache
def _filings(path: Path) -> dict[str, dict[str, FilingRef]]:
    by_ticker: dict[str, dict[str, FilingRef]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        f = FilingRef.model_validate_json(line)
        by_ticker[f.ticker][f.accession_no] = f
    return dict(by_ticker)


def filings_by_ticker(cfg: AppConfig) -> dict[str, dict[str, FilingRef]]:
    """All filings incl. reconcile-only companies (written by `build-data`).

    No silent fallback: a missing list once meant the agent got "unknown filing" for every
    reconcile-only company and quietly answered "unknown" (see DECISIONS.md)."""
    path = cfg.data_dir / "gold" / "filings_all.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `make data` (build-data writes it)")
    return _filings(path)


def filings_for(cfg: AppConfig, ticker: str) -> dict[str, FilingRef]:
    filings = filings_by_ticker(cfg).get(ticker)
    if not filings:
        raise LookupError(f"no filings for {ticker}: is it in companies/reconcile_companies?")
    return filings


def make_index(cfg: AppConfig, mode: str | None = None) -> FilingIndex:
    embeddings = CachedEmbeddings(
        BedrockEmbeddings(client=bedrock_client(cfg), model_id=cfg.embeddings.id),
        cfg.data_dir / "cache" / "embeddings",
        cfg.embeddings.id,
    )
    return FilingIndex(
        lambda accn: (cfg.data_dir / "filings" / f"{accn}.md").read_text(),
        embeddings,
        cfg.retrieval.chunk_size,
        cfg.retrieval.chunk_overlap,
        mode or cfg.retrieval.mode,
    )


def make_reconciler(
    cfg: AppConfig,
    model: str | None = None,
    strategy: str | None = None,
    version: str | None = None,
    index: FilingIndex | None = None,
) -> Reconciler:
    """`version` is the prompt version (pipeline) or agent version (agent)."""
    name = model or cfg.roles.reconciler
    cache = StepCache(cfg.data_dir / "cache" / "llm")
    if (strategy or cfg.reconcile.strategy) == "agent":
        return AgentReconciler(
            name,
            cfg.model(name),
            chat_model(cfg, name),
            cache,
            index or make_index(cfg),
            lambda ticker: filings_for(cfg, ticker),
            version or cfg.reconcile.agent_version,
            cfg.retrieval.k,
        )
    return LLMReconciler(
        name, cfg.model(name), chat_model(cfg, name), cache, version or cfg.prompts.reconcile
    )
