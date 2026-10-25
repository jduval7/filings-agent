"""Online checks, the annotation queue, and reviewer decisions as LangSmith feedback."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from filings_agent.evals import online
from tests.support.langsmith_doubles import check_feedback


def _returns(value: Any) -> Any:
    def fn(*_: Any, **__: Any) -> Any:
        return value

    return fn


def test_checks_and_routing() -> None:
    clean = {
        "status": "published",
        "errors": {},
        "resolutions": {},
        "memo": {"verified": True},
        "cost_usd": 0.1,
    }
    assert online.needs_human(online.checks(clean)) == []
    bad = {
        "status": "awaiting_review",
        "errors": {"t1": "x"},
        "resolutions": {"a": {"cause": "unknown", "verified": False}},
        "memo": {"verified": False},
        "cost_usd": 0.2,
    }
    reasons = online.needs_human(online.checks(bad))
    assert reasons == [
        "awaiting review",
        "1 branch error(s)",
        "1 unknown cause(s)",
        "memo numbers not verified",
    ]
    assert online.checks({"status": "done"})["online.memo_verified"] == -1.0  # no memo: n/a


class FakeLS:
    def __init__(self) -> None:
        self.feedback: list[tuple[str, dict[str, Any]]] = []
        self.queued: list[str] = []
        self.queues: list[Any] = []
        self.configs: dict[str, dict[str, Any]] = {}

    def list_feedback_configs(self) -> list[Any]:
        return [SimpleNamespace(feedback_key=k, feedback_config=c) for k, c in self.configs.items()]

    def create_feedback_config(self, feedback_key: str, *, feedback_config: dict[str, Any]) -> None:
        self.configs[feedback_key] = feedback_config

    def update_feedback_config(self, feedback_key: str, *, feedback_config: dict[str, Any]) -> None:
        self.configs[feedback_key] = feedback_config

    def read_project(self, *, project_name: str) -> Any:
        return SimpleNamespace(id="proj")

    def create_feedback(self, run_id: str, key: str, **kw: Any) -> None:
        check_feedback(kw)  # validate like the real client (it rejected "app")
        self.feedback.append((key, kw))

    def list_annotation_queues(self, *, name: str) -> list[Any]:
        return self.queues

    def create_annotation_queue(self, **kw: Any) -> Any:
        q = SimpleNamespace(id="q1", name=kw["name"])
        self.queues.append(q)
        return q

    def add_runs_to_annotation_queue(self, queue_id: str, *, run_ids: list[str]) -> None:
        self.queued += run_ids


def test_post_routes_flagged_runs_to_the_queue_with_idempotent_feedback() -> None:
    client = FakeLS()
    summary: dict[str, Any] = {
        "run_id": "r1",
        "status": "awaiting_review",
        "errors": {},
        "resolutions": {},
        "memo": None,
        "cost_usd": 0.0,
    }
    reasons = online.post(client, "r1", summary, "filings-agent")
    assert reasons == ["awaiting review"] and client.queued == ["r1"]
    keys = [k for k, _ in client.feedback]
    assert "online.awaiting_review" in keys and "online.routed_to_review" in keys
    assert all(kw["session_id"] == "proj" and kw["feedback_id"] for _, kw in client.feedback)
    online.post(client, "r1", summary, "filings-agent")
    assert len(client.queues) == 1  # queue is reused, not recreated


def test_monitor_never_raises_and_respects_tracing_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert online.monitor({"run_id": "r"}) == []

    class Broken:
        def read_project(self, **_: Any) -> Any:
            raise RuntimeError("LangSmith down")

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert online.monitor({"run_id": "r", "status": "done"}, client=Broken()) == []


def test_queue_routing_retries_until_the_trace_is_ingested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(online.time, "sleep", _returns(None))
    calls: list[int] = []

    class Lagging(FakeLS):
        def add_runs_to_annotation_queue(self, queue_id: str, *, run_ids: list[str]) -> None:
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("Resource not found for /annotation-queues/q1/runs")
            super().add_runs_to_annotation_queue(queue_id, run_ids=run_ids)

    client = Lagging()
    summary: dict[str, Any] = {"status": "awaiting_review", "errors": {}, "resolutions": {}}
    assert online.post(client, "r1", summary, "p") == ["awaiting review"]
    assert client.queued == ["r1"] and len(calls) == 3


def test_finished_run_posts_what_the_reviewers_did() -> None:
    client = FakeLS()
    summary: dict[str, Any] = {
        "run_id": "r2",
        "status": "published",  # what a real finished run reports
        "errors": {},
        "memo": None,
        "cost_usd": 0.0,
        "research_rounds": 1,
        "resolutions": {
            "GE:net_income:FY2022": {"cause": "discontinued_operations", "lane": "review"},
            "GE:revenue:FY2021": {"cause": "discontinued_operations", "lane": "review"},
            "NVDA:diluted_shares:FY2023": {"cause": "stock_split", "lane": "auto"},
            "X:cash:FY2023": {"cause": "unknown", "lane": "review"},
        },
        "decisions": {
            "GE:net_income:FY2022": {"action": "accept", "correct_cause": "accounting_change"},
            "GE:revenue:FY2021": {"action": "accept", "correct_cause": "discontinued_operations"},
            "X:cash:FY2023": {"action": "escalate"},
        },
    }
    scores, detail = online.review_scores(summary)
    assert scores == {
        "review.decisions": 3.0,
        "review.escalated": 1.0,
        "review.research_rounds": 1.0,
        "review.auto_approved": 1.0,
        "review.picked": 2.0,  # agreement's denominator: the escalation isn't in it
        "review.agent_agreement": 0.5,  # one of the two picks matched the agent's hidden answer
    }
    assert detail[0] == {
        "id": "GE:net_income:FY2022",
        "action": "accept",
        "picked": "accounting_change",
        "agent": "discontinued_operations",
    }
    online.post(client, "r2", summary, "filings-agent")
    posted = {k: kw for k, kw in client.feedback}
    assert posted["review.agent_agreement"]["feedback_source_type"] == "api"
    assert posted["review.agent_agreement"]["score"] == 0.5


def test_review_feedback_waits_for_the_finished_run() -> None:
    client = FakeLS()
    summary: dict[str, Any] = {
        "run_id": "r3", "status": "awaiting_review", "errors": {}, "memo": None,
        "cost_usd": 0.0, "resolutions": {}, "decisions": {"a": {"action": "research"}},
    }  # fmt: skip
    online.post(client, "r3", summary, "filings-agent")
    assert not any(k.startswith("review.") for k, _ in client.feedback)


def test_runs_without_a_ledger_also_post_review_feedback() -> None:
    client = FakeLS()
    summary: dict[str, Any] = {
        "run_id": "r4", "status": "done", "errors": {}, "memo": None, "cost_usd": 0.0,
        "resolutions": {"a": {"cause": "stock_split", "lane": "auto"}}, "decisions": {},
    }  # fmt: skip
    online.post(client, "r4", summary, "filings-agent")
    assert any(k == "review.auto_approved" for k, _ in client.feedback)


def test_a_failed_trace_flush_is_reported_not_silent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import langchain_core.tracers.langchain as lc

    def broken() -> None:
        raise RuntimeError("network down")

    monkeypatch.setattr(lc, "wait_for_all_tracers", broken)
    online._flush_traces()  # pyright: ignore[reportPrivateUsage]
    assert "trace flush failed" in capsys.readouterr().err


def test_every_queue_rubric_key_has_a_score_definition() -> None:
    # LangSmith hides a rubric key from the reviewer's form until it has a feedback config.
    ls = FakeLS()
    ls.configs["human.causes_correct"] = {"type": "continuous"}  # stale: replaced
    online.ensure_queue(ls)
    assert set(ls.configs) == {i["feedback_key"] for i in online.RUBRIC_ITEMS}
    assert ls.configs == online.FEEDBACK_CONFIGS
