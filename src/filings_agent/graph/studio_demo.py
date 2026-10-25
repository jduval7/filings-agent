"""LangGraph Studio on the demo's recordings: the real graph, no Bedrock, no network.

    make demo-studio      (its own langgraph.demo.json: the guard below is process-wide)
    -> https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

Input {"tickers": ["GE"]}: Studio draws the graph, runs plan and the 12 extract branches, the
second fan-out, and stops at review with the cards. Resume with decisions in Studio's interrupt
panel, e.g. {"GE:net_income:FY2022": {"action": "accept", "correct_cause": "accounting_change"},
...} for every card. Model answers come from demo/data; anything not recorded is refused. Runs
keep their own state (demo/state/studio), so they never touch the review screen's scenes.
"""

from __future__ import annotations

import functools
import os

from filings_agent import demo
from filings_agent.graph.build import Deps, build_graph


@functools.cache
def _deps() -> Deps:
    """Built on the first run. It switches the offline guard on for the whole server process
    (Studio runs the graph here, not inside a request we could wrap), and never off. Traces go to
    the demo's project when a LangSmith key is set."""
    from filings_agent.graph.runner import make_deps

    demo.offline(tracing=bool(os.environ.get("LANGSMITH_API_KEY"))).__enter__()
    cfg = demo.demo_config(state=demo.STATE / "studio")
    return make_deps(cfg, cfg.roles.extractor, cfg.prompts.extract)


graph = build_graph(default_deps=_deps, typed_context=False)
