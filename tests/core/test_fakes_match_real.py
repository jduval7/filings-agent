"""Every fake must look like the real thing it stands in for.

Two production bugs came from fakes that were looser than reality: a feedback source value the real
LangSmith client rejects, and a run status the real runner never returns. This test compares each
fake's methods and parameter names with the real class's, so a fake can't quietly accept a call the
real one would refuse (or need an argument the real one never gets).

The one place tests import from other test modules: some fakes are local to the test that uses them.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pytest
from langsmith import Client

from filings_agent.graph.runner import get_status, run_analysis
from filings_agent.reconcile_agent.agent import AgentReconciler
from filings_agent.research.agent import Researcher
from filings_agent.research.sources import LiveFetcher
from filings_agent.sec.filings import SecSource
from filings_agent.steps.extract import LLMStatementExtractor
from filings_agent.steps.judge import LLMJudge
from filings_agent.steps.memo import MemoWriter
from filings_agent.steps.reconcile import LLMReconciler


def _params(fn: Callable[..., Any]) -> tuple[dict[str, inspect.Parameter], bool]:
    sig = inspect.signature(fn)
    named = {
        n: p
        for n, p in sig.parameters.items()
        if n != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    open_ended = any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in sig.parameters.values())
    return named, open_ended


def compatible(fake: Callable[..., Any], real: Callable[..., Any], where: str) -> list[str]:
    """Problems that would let the fake diverge from the real callable."""
    problems: list[str] = []
    fake_params, fake_open = _params(fake)
    real_params, real_open = _params(real)
    for name in fake_params:
        if name not in real_params and not real_open:
            problems.append(f"{where}: the real one has no parameter {name!r}")
    for name, p in real_params.items():
        if p.default is p.empty and name not in fake_params and not fake_open:
            problems.append(f"{where}: the fake can't take required parameter {name!r}")
    return problems


def methods_compatible(fake_cls: type, real_cls: type) -> list[str]:
    problems: list[str] = []
    for name, fn in vars(fake_cls).items():
        if name.startswith("_") or not callable(fn):
            continue
        real = getattr(real_cls, name, None)
        where = f"{fake_cls.__name__}.{name} vs {real_cls.__name__}"
        if real is None:
            problems.append(f"{where}: the real class has no such method")
            continue
        problems += compatible(fn, real, where)
    return problems


def _fake(module: str, name: str) -> type:
    import importlib

    return getattr(importlib.import_module(module), name)


PAIRS: list[tuple[str, str, type]] = [
    # LangSmith
    ("tests.support.langsmith_doubles", "FakeClient", Client),
    ("tests.support.langsmith_doubles", "EvaluatingClient", Client),
    ("tests.evals.test_online", "FakeLS", Client),
    ("tests.evals.test_corrections", "FakeLangSmith", Client),
    # our own seams
    ("tests.support.research_doubles", "FakeFetcher", LiveFetcher),
    ("tests.support.graph_doubles", "FakeExtractor", LLMStatementExtractor),
    ("tests.support.graph_doubles", "RestatingExtractor", LLMStatementExtractor),
    ("tests.support.graph_doubles", "FakeReconciler", LLMReconciler),
    ("tests.support.graph_doubles", "FakeReconciler", AgentReconciler),
    ("tests.graph.test_branches", "BrokenReconciler", LLMReconciler),
    ("tests.steps.test_triage", "SplitReconciler", LLMReconciler),
    ("tests.support.research_doubles", "FakeResearcher", Researcher),
    ("tests.evals.test_memo_runners", "FakeWriter", MemoWriter),
    ("tests.evals.test_memo_runners", "FakeJudge", LLMJudge),
    ("tests.cli.test_cli", "FakeSource", SecSource),
]


@pytest.mark.parametrize(
    ("module", "name", "real"), PAIRS, ids=[f"{n}~{r.__name__}" for _, n, r in PAIRS]
)
def test_fake_matches_real(module: str, name: str, real: type) -> None:
    assert methods_compatible(_fake(module, name), real) == []


def test_fake_store_matches_the_real_store() -> None:
    from filings_agent.graph.runner import LocalFilingStore

    assert (
        methods_compatible(_fake("tests.support.graph_doubles", "FakeStore"), LocalFilingStore)
        == []
    )


def test_api_fake_backend_matches_the_real_runner() -> None:
    backend: Any = _fake("tests.api.test_api", "FakeBackend")
    assert compatible(backend.run, run_analysis, "FakeBackend.run vs run_analysis") == []
    assert compatible(backend.get_status, get_status, "FakeBackend.get_status vs get_status") == []


def test_the_checker_catches_a_loose_fake() -> None:
    """The checker itself: a fake with a made-up parameter must be flagged."""

    class Loose:
        def read_dataset(self, *, dataset_name: str, made_up: int = 0) -> None: ...

    assert any("made_up" in p for p in methods_compatible(Loose, Client))
