"""API behaviour with a fake run function (no LLM, no network)."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from filings_agent.api.app import create_app
from filings_agent.config import AppConfig, load_config


class FakeBackend:
    """Stands in for run_analysis/get_status: first run pauses for review, review publishes."""

    def __init__(self) -> None:
        self.status: dict[str, dict[str, Any]] = {}
        self.runs = 0
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def run(
        self,
        cfg: AppConfig,
        tickers: list[str],
        model: str,
        prompt: str,
        thread_id: str,
        on_event: Any,
        decisions: dict[str, Any] | None = None,
        rep: int = 0,
        resume: bool = False,
        lease_owner: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            self.runs += 1
            self.calls.append({"thread_id": thread_id, "rep": rep, "decisions": decisions,
                               "lease_owner": lease_owner, "run_id": run_id})  # fmt: skip
        on_event({"event": "planned", "branches": 3})
        for i in range(3):
            on_event({"event": "branch_done", "task_id": f"t{i}"})
        if decisions is None:
            self.status[thread_id] = {
                "status": "awaiting_review",
                "awaiting_review": [{"id": "X:revenue:FY2023"}],
            }
        else:
            self.status[thread_id] = {
                "status": "published",
                "awaiting_review": [],
                "decisions": decisions,
            }
        return self.status[thread_id]

    def get_status(self, cfg: AppConfig, thread_id: str) -> dict[str, Any]:
        return self.status.get(thread_id, {"status": "not_found", "awaiting_review": []})


@pytest.fixture
def env(tmp_path: Path) -> Generator[tuple[TestClient, FakeBackend]]:
    cfg = load_config().model_copy(update={"data_dir": tmp_path})
    backend = FakeBackend()
    app = create_app(cfg, run_fn=backend.run, status_fn=backend.get_status)
    # One event loop for the whole test, like the real server. Without `with`, TestClient tears
    # its loop down after each request and cancels runs still in flight (found 2026-10-23).
    with TestClient(app) as client:
        yield client, backend


def _wait_for(
    client: TestClient, thread_id: str, status: str, timeout: float = 5
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/analyses/{thread_id}").json()
        if body["status"] == status:
            return body
        time.sleep(0.02)
    raise AssertionError(f"never reached {status}")


BODY = {"tickers": ["nvda", "GE"]}


def _review(client: TestClient, tid: str, decisions: dict[str, Any]) -> Any:
    """Submit decisions the way the UI does: with the token of the cards it loaded."""
    token = client.get(f"/analyses/{tid}").json().get("review_token") or "none"
    return client.post(
        f"/analyses/{tid}/review", json={"decisions": decisions, "review_token": token}
    )


def test_same_idempotency_key_starts_one_run(env: tuple[TestClient, FakeBackend]) -> None:
    client, backend = env
    first = client.post("/analyses", json=BODY, headers={"Idempotency-Key": "abc"}).json()
    _wait_for(client, first["thread_id"], "awaiting_review")
    second = client.post("/analyses", json=BODY, headers={"Idempotency-Key": "abc"}).json()
    assert first["created"] is True and second["created"] is False
    assert first["thread_id"] == second["thread_id"]
    assert backend.runs == 1


def test_key_reused_with_different_body_is_rejected(env: tuple[TestClient, FakeBackend]) -> None:
    client, _ = env
    client.post("/analyses", json=BODY, headers={"Idempotency-Key": "k"})
    resp = client.post("/analyses", json={"tickers": ["AAPL"]}, headers={"Idempotency-Key": "k"})
    assert resp.status_code == 422  # IETF idempotency-key draft: same key, different payload


def test_no_key_derives_identity_from_the_request(env: tuple[TestClient, FakeBackend]) -> None:
    client, backend = env
    a = client.post("/analyses", json={"tickers": ["GE", "NVDA"]}).json()
    _wait_for(client, a["thread_id"], "awaiting_review")
    b = client.post("/analyses", json={"tickers": ["nvda", "ge"]}).json()  # same request
    assert a["thread_id"] == b["thread_id"] and backend.runs == 1


def test_events_stream_replays_and_ends(env: tuple[TestClient, FakeBackend]) -> None:
    client, _ = env
    created = client.post("/analyses", json=BODY).json()
    with client.stream("GET", created["events_url"]) as resp:
        text = "".join(resp.iter_text())
    assert text.count("event: branch_done") == 3
    assert "event: end" in text


def test_review_flow(env: tuple[TestClient, FakeBackend]) -> None:
    client, backend = env
    tid = client.post("/analyses", json=BODY).json()["thread_id"]
    waiting = _wait_for(client, tid, "awaiting_review")
    assert [p["id"] for p in waiting["awaiting_review"]] == ["X:revenue:FY2023"]

    bad = _review(
        client, tid, {"nope": {"action": "accept", "correct_cause": "discontinued_operations"}}
    )
    assert bad.status_code == 422  # an unknown id and a missing one: never a silent approval

    ok = _review(
        client,
        tid,
        {
            "X:revenue:FY2023": {
                "action": "use_original",
                "note": "as filed",
                "correct_cause": "discontinued_operations",
            }
        },
    )
    assert ok.status_code == 202
    done = _wait_for(client, tid, "published")
    assert done["decisions"]["X:revenue:FY2023"]["action"] == "use_original"

    again = _review(
        client,
        tid,
        {"X:revenue:FY2023": {"action": "accept", "correct_cause": "discontinued_operations"}},
    )
    assert again.status_code == 409
    assert backend.runs == 2


def test_unknown_analysis_is_404(env: tuple[TestClient, FakeBackend]) -> None:
    client, _ = env
    assert client.get("/analyses/doesnotexist").status_code == 404


def test_client_retry_resumes_an_interrupted_run(env: tuple[TestClient, FakeBackend]) -> None:
    client, backend = env
    tid = client.post("/analyses", json=BODY, headers={"Idempotency-Key": "crash"}).json()[
        "thread_id"
    ]
    _wait_for(client, tid, "awaiting_review")
    backend.status[tid] = {"status": "interrupted", "awaiting_review": []}  # server died mid-run
    again = client.post("/analyses", json=BODY, headers={"Idempotency-Key": "crash"}).json()
    assert again["status"] == "resuming" and again["created"] is False
    _wait_for(client, tid, "awaiting_review")
    assert backend.runs == 2


def test_unknown_extractor_is_rejected_before_anything_runs(
    env: tuple[TestClient, FakeBackend],
) -> None:
    client, backend = env
    resp = client.post("/analyses", json={"tickers": ["GE"], "extractor": "gpt-imaginary"})
    assert resp.status_code == 422 and backend.runs == 0


def test_request_validation(env: tuple[TestClient, FakeBackend]) -> None:
    client, _ = env
    assert client.post("/analyses", json={"tickers": []}).status_code == 422
    assert client.post("/analyses", json={}).status_code == 422
    tid = client.post("/analyses", json=BODY).json()["thread_id"]
    _wait_for(client, tid, "awaiting_review")
    assert _review(client, tid, {"X:revenue:FY2023": {"action": "approve-ish"}}).status_code == 422
    bad_cause = {"X:revenue:FY2023": {"action": "accept", "correct_cause": "vibes"}}
    assert _review(client, tid, bad_cause).status_code == 422


def test_review_of_unknown_or_running_analysis(env: tuple[TestClient, FakeBackend]) -> None:
    client, backend = env
    body = {
        "decisions": {
            "X:revenue:FY2023": {"action": "accept", "correct_cause": "discontinued_operations"}
        },
        "review_token": "any",
    }
    assert client.post("/analyses/nope/review", json=body).status_code == 404

    gate = threading.Event()
    original = backend.run

    def slow_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        gate.wait(5)
        return original(*args, **kwargs)

    backend.run = slow_run  # type: ignore[method-assign]
    app = create_app(client.app.state.manager.cfg, run_fn=slow_run, status_fn=backend.get_status)  # type: ignore[attr-defined]
    with TestClient(app) as c2:
        tid = c2.post("/analyses", json=BODY).json()["thread_id"]
        assert c2.post(f"/analyses/{tid}/review", json=body).status_code == 409  # still running
        assert c2.get(f"/analyses/{tid}").json()["status"] == "running"
        gate.set()
        _wait_for(c2, tid, "awaiting_review")


def test_events_for_a_run_not_live_in_this_process(env: tuple[TestClient, FakeBackend]) -> None:
    client, _ = env
    tid = client.post("/analyses", json=BODY).json()["thread_id"]
    _wait_for(client, tid, "awaiting_review")
    client.app.state.manager.live.clear()  # type: ignore[attr-defined]  # e.g. after a restart
    with client.stream("GET", f"/analyses/{tid}/events") as resp:
        text = "".join(resp.iter_text())
    assert "event: status" in text and "awaiting_review" in text
    assert client.get("/analyses/nope/events").status_code == 404
    assert client.get("/health").json() == {"status": "ok"}


def test_events_after_a_review_follow_the_resumed_run(env: tuple[TestClient, FakeBackend]) -> None:
    """The UI subscribes again after submitting decisions: it must see the resumed run's events,
    not the paused run's "end" replayed."""
    client, _ = env
    tid = client.post("/analyses", json=BODY).json()["thread_id"]
    _wait_for(client, tid, "awaiting_review")
    _review(
        client,
        tid,
        {"X:revenue:FY2023": {"action": "accept", "correct_cause": "discontinued_operations"}},
    )
    _wait_for(client, tid, "published")
    with client.stream("GET", f"/analyses/{tid}/events") as resp:
        text = "".join(resp.iter_text())
    assert text.count("event: end") == 1 and '"status": "published"' in text


def test_recent_analyses_newest_first(env: tuple[TestClient, FakeBackend]) -> None:
    client, _ = env
    a = client.post("/analyses", json={"tickers": ["GE"]}).json()["thread_id"]
    _wait_for(client, a, "awaiting_review")
    time.sleep(0.01)
    b = client.post("/analyses", json={"tickers": ["NVDA"]}).json()["thread_id"]
    _wait_for(client, b, "awaiting_review")
    runs = client.get("/analyses").json()
    assert [r["thread_id"] for r in runs] == [b, a]
    assert runs[0]["tickers"] == ["NVDA"] and runs[0]["waiting"] == 1


def test_serves_the_review_ui_when_built(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>review ui</html>")
    cfg = load_config().model_copy(update={"data_dir": tmp_path})
    backend = FakeBackend()
    client = TestClient(create_app(cfg, backend.run, backend.get_status, web_dist=dist))
    assert "review ui" in client.get("/").text
    assert client.get("/health").json() == {"status": "ok"}  # the API still wins


def test_run_counts_as_finished_at_end_even_while_monitoring(tmp_path: Path) -> None:
    """Posting feedback to LangSmith takes seconds; status must not say "running" meanwhile."""
    import threading as th

    from filings_agent.api.app import RunManager

    release = th.Event()

    def slow_monitor(summary: dict[str, Any]) -> None:
        release.wait(5)

    cfg = load_config().model_copy(update={"data_dir": tmp_path})
    backend = FakeBackend()
    app = create_app(cfg, run_fn=backend.run, status_fn=backend.get_status)
    manager: RunManager = app.state.manager
    manager.monitor_fn = slow_monitor
    client = TestClient(app).__enter__()  # one loop for the test (see the env fixture)
    tid = client.post("/analyses", json=BODY).json()["thread_id"]
    _wait_for(client, tid, "awaiting_review")  # reached while the monitor is still blocked
    release.set()


def test_status_links_every_run_of_the_thread_and_meta_degrades_gracefully(
    env: tuple[TestClient, FakeBackend], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API picks each invocation's trace id before it runs, so a run that crashes is linked
    too, and a thread's start, review and resume are all listed."""
    client, backend = env
    tid = _waiting_card(client)
    _review(client, tid, ACCEPT)
    body = _wait_for(client, tid, "published")
    assert [r["kind"] for r in body["runs"]] == ["start", "review"]
    assert [r["run_id"] for r in body["runs"]] == [c["run_id"] for c in backend.calls]
    assert body["last_run_id"] == backend.calls[-1]["run_id"]
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert client.get("/meta").json() == {"langsmith_project_url": None, "demo": False}


def test_langsmith_project_url_is_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import langsmith

    from filings_agent.api.app import langsmith_project_url

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert langsmith_project_url() is None  # tracing off: no link

    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    class Down:
        def read_project(self, **_: Any) -> Any:
            raise ConnectionError("unreachable")

    monkeypatch.setattr(langsmith, "Client", Down)
    assert langsmith_project_url() is None  # unreachable: no link, no error

    class Up:
        def read_project(self, **_: Any) -> Any:
            return SimpleNamespace(tenant_id="org", id="proj")

    monkeypatch.setattr(langsmith, "Client", Up)
    assert langsmith_project_url() == "https://smith.langchain.com/o/org/projects/p/proj"


def _waiting_card(client: TestClient) -> str:
    tid = client.post("/analyses", json=BODY).json()["thread_id"]
    _wait_for(client, tid, "awaiting_review")
    return tid


def test_accept_needs_a_cause_and_research_needs_rounds(
    env: tuple[TestClient, FakeBackend],
) -> None:
    """Otherwise an API client publishes the agent's hidden answer, or a review is accepted
    (202) and then fails in the background."""
    client, _ = env
    tid = _waiting_card(client)
    no_cause = _review(client, tid, {"X:revenue:FY2023": {"action": "accept"}})
    assert no_cause.status_code == 422 and "correct_cause" in no_cause.text
    no_rounds = _review(client, tid, {"X:revenue:FY2023": {"action": "research"}})
    assert no_rounds.status_code == 422 and "no research rounds" in no_rounds.text


def test_a_stale_review_token_is_refused(env: tuple[TestClient, FakeBackend]) -> None:
    client, _ = env
    tid = _waiting_card(client)
    token = client.get(f"/analyses/{tid}").json()["review_token"]
    assert token
    decision = {"X:revenue:FY2023": {"action": "escalate"}}
    stale = client.post(
        f"/analyses/{tid}/review", json={"decisions": decision, "review_token": "old"}
    )
    assert stale.status_code == 409 and "changed" in stale.text
    ok = client.post(f"/analyses/{tid}/review", json={"decisions": decision, "review_token": token})
    assert ok.status_code == 202


def test_blank_tickers_are_rejected(env: tuple[TestClient, FakeBackend]) -> None:
    client, _ = env
    assert client.post("/analyses", json={"tickers": [" ", ""]}).status_code == 422


def test_the_agents_cause_never_leaves_the_server_while_pending() -> None:
    from filings_agent.api.app import without_pending_causes

    status: dict[str, Any] = {
        "awaiting_review": [{"id": "A"}],
        "resolutions": {"A": {"cause": "stock_split"}, "B": {"cause": "immaterial_revision"}},
    }
    out = without_pending_causes(status)
    assert out["resolutions"]["A"]["cause"] is None  # pending: hidden
    assert out["resolutions"]["B"]["cause"] == "immaterial_revision"  # decided by code: shown
    assert status["resolutions"]["A"]["cause"] == "stock_split"  # the input isn't mutated


ACCEPT = {"X:revenue:FY2023": {"action": "accept", "correct_cause": "discontinued_operations"}}


def test_a_review_without_the_card_token_is_refused(env: tuple[TestClient, FakeBackend]) -> None:
    """Review finding: the token was optional, so an API client could skip the stale-cards check."""
    client, _ = env
    tid = _waiting_card(client)
    resp = client.post(f"/analyses/{tid}/review", json={"decisions": ACCEPT})
    assert resp.status_code == 422 and "review_token" in resp.text


def test_the_server_stamps_who_decided_and_when(env: tuple[TestClient, FakeBackend]) -> None:
    client, backend = env
    tid = _waiting_card(client)
    forged = {"X:revenue:FY2023": {**ACCEPT["X:revenue:FY2023"], "actor": "mallory"}}
    assert _review(client, tid, forged).status_code == 202
    _wait_for(client, tid, "published")
    (decision,) = backend.calls[-1]["decisions"].values()
    assert decision["actor"] == "local"  # the payload's "actor" is ignored (not a model field)
    assert decision["decided_at"].startswith("20")


def test_fresh_is_a_new_analysis_and_a_keyed_fresh_retry_is_not(
    env: tuple[TestClient, FakeBackend],
) -> None:
    """Review finding: `fresh` never started anything once the thread existed."""
    client, backend = env
    first = client.post("/analyses", json=BODY).json()["thread_id"]
    _wait_for(client, first, "awaiting_review")
    again = client.post("/analyses", json={**BODY, "fresh": True}).json()
    assert again["created"] is True and again["thread_id"] != first
    _wait_for(client, again["thread_id"], "awaiting_review")
    assert backend.calls[-1]["rep"] != 0  # new LLM calls, not the cached ones
    keyed = {**BODY, "fresh": True}
    a = client.post("/analyses", json=keyed, headers={"Idempotency-Key": "re-1"}).json()
    _wait_for(client, a["thread_id"], "awaiting_review")
    b = client.post("/analyses", json=keyed, headers={"Idempotency-Key": "re-1"}).json()
    assert a["thread_id"] == b["thread_id"] and b["created"] is False  # a retry, not a third run


def test_new_inputs_start_a_new_analysis(tmp_path: Path) -> None:
    """Review finding: identity was the tickers + extractor, so after a new 10-K (or a new
    reconcile model) "Analyse GE" replayed the first GE run forever."""
    filings = ["A1"]

    def identity(cfg: AppConfig, tickers: list[str], extractor: str, prompt: str) -> dict[str, Any]:
        return {"tickers": tickers, "filings": list(filings)}

    cfg = load_config().model_copy(update={"data_dir": tmp_path})
    backend = FakeBackend()
    app = create_app(cfg, backend.run, backend.get_status, web_dist=None, identity_fn=identity)
    with TestClient(app) as client:
        a = client.post("/analyses", json=BODY).json()
        _wait_for(client, a["thread_id"], "awaiting_review")
        assert client.post("/analyses", json=BODY).json()["thread_id"] == a["thread_id"]
        filings.append("A2")  # GE filed a new 10-K
        b = client.post("/analyses", json=BODY).json()
        assert b["created"] is True and b["thread_id"] != a["thread_id"]


def test_a_thread_running_in_another_process_is_not_started_again(
    env: tuple[TestClient, FakeBackend],
) -> None:
    """Review finding: the per-thread locks lived in one process's memory, so a second worker
    (or the CLI) could run the same thread at the same time."""
    client, backend = env
    from filings_agent.api.app import RunManager
    from filings_agent.storage.leases import Leases

    manager = cast(RunManager, client.app.state.manager)  # type: ignore[attr-defined]

    other = Leases(manager.cfg.data_dir / "leases.sqlite")  # another process, same database
    thread_id = manager.identity_fn(manager.cfg, ["GE", "NVDA"], manager.cfg.roles.extractor,
                                    manager.cfg.prompts.extract)  # fmt: skip
    import hashlib
    import json as _json

    tid = hashlib.sha256(_json.dumps(thread_id, sort_keys=True).encode()).hexdigest()[:16]
    assert other.acquire(tid, "worker-2")
    resp = client.post("/analyses", json=BODY).json()
    assert resp["thread_id"] == tid and resp["created"] is False and resp["status"] == "running"
    assert backend.runs == 0
    other.release(tid, "worker-2")
    assert client.post("/analyses", json=BODY).json()["created"] is True


def test_a_review_is_refused_while_another_process_runs_the_thread(
    env: tuple[TestClient, FakeBackend],
) -> None:
    client, _ = env
    tid = _waiting_card(client)
    from filings_agent.storage.leases import Leases

    token = client.get(f"/analyses/{tid}").json()["review_token"]
    other = Leases(client.app.state.manager.cfg.data_dir / "leases.sqlite")  # type: ignore[attr-defined]
    assert other.acquire(tid, "cli-review")  # e.g. `filings-agent review` on the same thread
    assert client.get(f"/analyses/{tid}").json()["status"] == "running"
    resp = client.post(f"/analyses/{tid}/review", json={"decisions": ACCEPT, "review_token": token})
    assert resp.status_code == 409


def test_runs_release_their_lease(env: tuple[TestClient, FakeBackend]) -> None:
    client, backend = env
    tid = _waiting_card(client)
    from filings_agent.storage.leases import Leases

    leases = cast(Leases, client.app.state.manager.leases)  # type: ignore[attr-defined]
    assert backend.calls[0]["lease_owner"]  # the run was handed the lease the request took
    deadline = time.monotonic() + 5
    while leases.holder(tid) is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert leases.holder(tid) is None


TOKENS = "alice:alice-token-0123456789,bob:bob-token-0123456789"


@pytest.fixture
def authed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, FakeBackend]]:
    monkeypatch.setenv("FILINGS_API_TOKENS", TOKENS)
    cfg = load_config().model_copy(update={"data_dir": tmp_path})
    backend = FakeBackend()
    with TestClient(create_app(cfg, run_fn=backend.run, status_fn=backend.get_status)) as client:
        yield client, backend


def test_with_tokens_every_api_call_needs_one(authed: tuple[TestClient, FakeBackend]) -> None:
    """Review finding: anyone who could reach the port could start runs and submit reviews."""
    client, backend = authed
    assert client.post("/analyses", json=BODY).status_code == 401
    assert client.get("/analyses").status_code == 401
    assert client.get("/analyses/x").status_code == 401
    assert client.get("/analyses/x/events").status_code == 401
    assert (
        client.post(
            "/analyses/x/review", json={"decisions": ACCEPT, "review_token": "t"}
        ).status_code
        == 401
    )
    assert client.get("/meta").status_code == 401
    wrong = {"Authorization": "Bearer not-a-real-token-at-all"}
    assert client.get("/analyses", headers=wrong).status_code == 401
    assert client.get("/health").status_code == 200  # liveness stays open
    assert backend.runs == 0


def test_the_token_names_the_reviewer(authed: tuple[TestClient, FakeBackend]) -> None:
    client, backend = authed
    alice = {"Authorization": "Bearer alice-token-0123456789"}
    tid = client.post("/analyses", json=BODY, headers=alice).json()["thread_id"]
    _wait_for_with(client, tid, "awaiting_review", alice)
    token = client.get(f"/analyses/{tid}", headers=alice).json()["review_token"]
    bob = {"Authorization": "Bearer bob-token-0123456789"}
    resp = client.post(f"/analyses/{tid}/review", headers=bob,
                       json={"decisions": ACCEPT, "review_token": token})  # fmt: skip
    assert resp.status_code == 202
    _wait_for_with(client, tid, "published", bob)
    assert backend.calls[-1]["decisions"]["X:revenue:FY2023"]["actor"] == "bob"


def test_the_ui_signs_in_once_and_uses_a_cookie(authed: tuple[TestClient, FakeBackend]) -> None:
    """EventSource can't send an Authorization header, so the UI trades the token for a cookie."""
    client, _ = authed
    assert client.post("/auth/session", json={"token": "nope"}).status_code == 401
    resp = client.post("/auth/session", json={"token": "alice-token-0123456789"})
    assert resp.status_code == 200 and resp.json() == {"user": "alice"}
    assert "httponly" in resp.headers["set-cookie"].lower()
    assert client.get("/auth/me").json() == {"user": "alice", "auth": True}
    assert client.get("/analyses").status_code == 200
    client.delete("/auth/session")
    assert client.get("/analyses").status_code == 401


def test_bad_token_config_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    from filings_agent.config import ApiConfig

    monkeypatch.setenv("FILINGS_API_TOKENS", "alice:short")
    with pytest.raises(RuntimeError, match="at least 16"):
        ApiConfig().tokens()
    monkeypatch.setenv("FILINGS_API_TOKENS", "")
    assert ApiConfig().tokens() == {}


def _wait_for_with(
    client: TestClient, thread_id: str, status: str, headers: dict[str, str]
) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        body = client.get(f"/analyses/{thread_id}", headers=headers).json()
        if body["status"] == status:
            return body
        time.sleep(0.02)
    raise AssertionError(f"never reached {status}")


def test_losing_a_start_race_to_another_process_is_reported_not_failed(
    env: tuple[TestClient, FakeBackend],
) -> None:
    """Two workers both saw "not started"; the loser's run finds the lease taken."""
    client, backend = env
    from filings_agent.storage.leases import ThreadBusy

    def busy_run(*_: Any, **__: Any) -> dict[str, Any]:
        raise ThreadBusy("thread is running in another process")

    app = create_app(client.app.state.manager.cfg, run_fn=busy_run, status_fn=backend.get_status)  # type: ignore[attr-defined]
    with TestClient(app) as c2:
        tid = c2.post("/analyses", json={"tickers": ["KO"]}).json()["thread_id"]
        with c2.stream("GET", f"/analyses/{tid}/events") as resp:
            text = "".join(resp.iter_text())
    assert "running_elsewhere" in text and "failed" not in text


def test_an_interrupted_run_resumes_from_the_button(env: tuple[TestClient, FakeBackend]) -> None:
    """The review UI's Resume button: same as a client retry with the same key, but by id."""
    client, backend = env
    tid = _waiting_card(client)
    assert client.post(f"/analyses/{tid}/resume").status_code == 409  # not interrupted
    backend.status[tid] = {"status": "interrupted", "awaiting_review": []}  # an outage mid-run
    resp = client.post(f"/analyses/{tid}/resume")
    assert resp.status_code == 202 and resp.json()["status"] == "resuming"
    _wait_for(client, tid, "awaiting_review")
    assert backend.calls[-1]["thread_id"] == tid
    assert client.post("/analyses/nope/resume").status_code == 404


def test_a_case_shows_every_decision_and_the_one_that_stands(
    env: tuple[TestClient, FakeBackend],
) -> None:
    from decimal import Decimal

    from filings_agent.graph.runner import shared_ledger
    from filings_agent.steps.reconcile import Cause, Resolution
    from filings_agent.storage.ledger import Decided

    client, _ = env
    cfg = cast(AppConfig, client.app.state.manager.cfg)  # type: ignore[attr-defined]
    ledger = shared_ledger(cfg.state("ledger.sqlite"))
    pick = Resolution("GE:net_income:FY2022", Cause.ACCOUNTING_CHANGE, "e", "q", True, True,
                      Decimal(1), "A2", False)  # fmt: skip
    ledger.publish("A", [], [Decided(pick, "accept", "A1,A2", "alice")])
    ledger.publish("B", [], [Decided(pick, "escalate", "A1,A2", "bob")])
    body = client.get("/cases/GE:net_income:FY2022").json()
    assert [h["actor"] for h in body["history"]] == ["alice", "bob"]
    assert body["current"][0]["actor"] == "alice" and body["current"][0]["later_escalations"] == 1


def test_demo_routes_exist_only_in_demo_mode(tmp_path: Path) -> None:
    import json as _json

    cfg = load_config().model_copy(update={"data_dir": tmp_path})
    backend = FakeBackend()
    with TestClient(create_app(cfg, backend.run, backend.get_status, web_dist=None)) as client:
        assert client.get("/demo").status_code == 404
    state = tmp_path / "demo-state"
    (state / "replays").mkdir(parents=True)
    (state / "scenes.json").write_text(_json.dumps({"review": {"thread_id": "t-review", "run": 1}}))
    events = [
        {"event": "planned", "branches": 1, "tasks": ["A1:balance_sheet"]},
        {"event": "branch_done", "task_id": "A1:balance_sheet"},
        {"event": "end"},
    ]
    (state / "replays" / "fanout.json").write_text(_json.dumps(events))
    app = create_app(cfg, backend.run, backend.get_status, web_dist=None, demo_state=state)
    with TestClient(app) as client:
        assert client.get("/meta").json()["demo"] is True
        scenes = {s["id"]: s for s in client.get("/demo").json()["scenes"]}
        assert scenes["review"]["thread_id"] == "t-review" and scenes["fanout"]["kind"] == "replay"
        assert scenes["crash"]["thread_id"] is None  # not seeded
        # offline (no LangSmith key): workspace pages aren't linked, public ones are
        evals = scenes["evals"]["langsmith"]
        assert evals[0]["links"][0]["where"] == "public"
        assert all(not i["links"] for i in scenes["annotations"]["langsmith"])
        with client.stream("GET", "/demo/replay/fanout") as resp:
            text = "".join(resp.iter_text())
        assert text.count('"replay": true') == 3 and "event: end" in text
        assert client.get("/demo/replay/nope").status_code == 404
        assert client.post("/demo/evals/reset").status_code == 404  # links have nothing to reset
