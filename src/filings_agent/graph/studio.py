"""Entry point for LangGraph Studio / `langgraph dev` (see langgraph.json).

The server provides persistence and passes context as JSON, so the graph gets its dependencies
(filing store, models from config, ledger) from a default built on first use.

    AWS_PROFILE=... uv run langgraph dev
    -> https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
"""

from __future__ import annotations

import functools

from dotenv import load_dotenv

from filings_agent.config import load_config
from filings_agent.graph.build import Deps, build_graph

load_dotenv()


@functools.cache
def _deps() -> Deps:
    from filings_agent.graph.runner import make_deps

    cfg = load_config()
    return make_deps(cfg, cfg.roles.extractor, cfg.prompts.extract)


graph = build_graph(default_deps=_deps, typed_context=False)
