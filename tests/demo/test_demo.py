"""Demo mode: every scene seeds offline from the committed recordings, with no credentials."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest
from langsmith.utils import LangSmithNotFoundError

from filings_agent import demo
from filings_agent.graph import build, runner


@pytest.fixture
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """The crash scene's outage retries with the production backoff (seconds); not here."""
    from langgraph.types import RetryPolicy

    fast = RetryPolicy(max_attempts=2, initial_interval=0.01, retry_on=build.is_transient)
    real = build.build_graph

    def with_fast_retries(cp: Any) -> Any:
        return real(cp, retry=fast)

    monkeypatch.setattr(runner, "build_graph", with_fast_retries)


@pytest.fixture
def no_credentials(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    for var in ("AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "LANGSMITH_API_KEY",
                "SEC_USER_AGENT", "FILINGS_API_TOKENS"):  # fmt: skip
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    yield


@pytest.mark.usefixtures("fast_retries", "no_credentials")
def test_every_scene_seeds_offline_into_its_state(tmp_path: Path) -> None:
    cfg = demo.demo_config(state=tmp_path / "state")
    rec = demo.Recorder()
    with demo.offline(), rec.watching():
        scenes = demo.seed(cfg)
        status = {
            s: runner.get_status(cfg, str(v["thread_id"]))["status"] for s, v in scenes.items()
        }
    assert status == {
        "fanout": "awaiting_review", "easy-lane": "published", "review": "awaiting_review",
        "research": "awaiting_review", "published": "published", "history": "published",
        "crash": "interrupted",
    }  # fmt: skip
    current = runner.shared_ledger(cfg.state("ledger.sqlite")).current("GE:net_income:FY2022")
    assert current[0]["actor"] == "alice" and current[0]["later_escalations"] == 1
    events = (tmp_path / "state" / "replays" / "fanout.json").read_text()
    assert '"planned"' in events and '"tasks"' in events  # the replay can lay out the fan-out

    # Recording what the seeding read reproduces exactly the committed recordings.
    out = tmp_path / "exported"
    counts = demo.export(rec, demo.DATA, out)
    assert counts["filings"] == 8 and counts["cache/llm"] == len(
        list((demo.DATA / "cache" / "llm").rglob("*.json"))
    )

    # A reset re-runs one scene under a new key: a new thread, the old one left alone.
    with demo.offline():
        again = demo.seed(cfg, ("review",))
    assert (
        again["review"]["run"] == 2
        and again["review"]["thread_id"] != scenes["review"]["thread_id"]
    )


def test_offline_refuses_unrecorded_calls_and_pages(tmp_path: Path) -> None:
    from filings_agent.llm import spend_guard
    from filings_agent.research.sources import LiveFetcher

    guard = spend_guard.SpendGuard(demo.demo_config(state=tmp_path))
    fetcher = LiveFetcher("demo test@example.com", tmp_path / "pages")
    with demo.offline():
        with pytest.raises(spend_guard.SpendLimitExceeded, match="wasn't recorded"):
            guard.reserve("us.anthropic.claude-sonnet-4-6", "hi", 10)
        with pytest.raises(demo.NotRecorded):
            fetcher.document("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany")


def test_the_outage_lets_the_first_statements_through_then_throttles() -> None:
    from botocore.exceptions import ClientError

    class Inner:
        name = "x"

        def extract(self, task: Any, rep: int = 0) -> Any:
            return "ok"

    outage = demo._Outage(Inner(), after=2)  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert [outage.extract(None), outage.extract(None)] == ["ok", "ok"]  # pyright: ignore[reportArgumentType]
    with pytest.raises(ClientError, match="ThrottlingException"):
        outage.extract(None)  # pyright: ignore[reportArgumentType]
    assert build.is_transient(ClientError({"Error": {"Code": "ThrottlingException"}}, "Converse"))


def test_scenes_have_everything_the_ui_shows() -> None:
    assert [s.id for s in demo.SCENES][: len(demo.SEED_ORDER) - 1] != []
    for s in demo.SCENES:
        assert s.title and s.shows and s.decision
        assert (s.kind == "case") == (s.case is not None)
        assert s.langsmith, f"{s.id}: every scene points somewhere in LangSmith"
        for item in s.langsmith:  # every link resolves to a known page (or the scene's own run)
            from filings_agent.evals import langsmith_links as ll

            known = {demo.THIS_RUN, demo.STUDIO, "demo_queue", "demo_project", "queue",
                     "corrections", "project", *ll.TRACES, *ll.COMPARISONS}  # fmt: skip
            assert item.workspace is None or item.workspace in known, item.workspace
            assert item.public is None or item.public in ll.PUBLIC, item.public
    assert set(demo.SEED_ORDER) == {
        s.id for s in demo.SCENES if s.kind in ("run", "replay", "case")
    }
    assert (
        demo.thread_for("demo-review-1") == demo.thread_for("demo-review-1") != demo.thread_for("x")
    )


def test_studio_runs_the_demo_graph_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """`make demo-studio`: the same graph on the recordings, guarded, with its own state."""
    from contextlib import contextmanager

    from filings_agent.graph import studio_demo

    guarded: list[bool] = []

    @contextmanager
    def fake_offline(tracing: bool = False) -> Generator[None]:
        guarded.append(tracing)
        yield

    built: list[Any] = []
    monkeypatch.setattr(demo, "offline", fake_offline)

    def fake_make_deps(cfg: Any, *_: Any) -> str:
        built.append(cfg)
        return "deps"

    monkeypatch.setattr(runner, "make_deps", fake_make_deps)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    studio_demo._deps.cache_clear()  # pyright: ignore[reportPrivateUsage]
    assert studio_demo._deps() == "deps"  # pyright: ignore[reportPrivateUsage]
    assert guarded == [False] and built[0].state_dir == demo.STATE / "studio"
    assert {"plan", "extract", "review"} <= set(studio_demo.graph.get_graph().nodes)
    studio_demo._deps.cache_clear()  # pyright: ignore[reportPrivateUsage]


class FakeLangSmith:
    """Just enough of langsmith.Client for the link lookups."""

    def __init__(self, projects: set[str]) -> None:
        self.projects = projects

    def read_project(self, *, project_name: str) -> Any:
        from types import SimpleNamespace

        if project_name not in self.projects:
            raise LookupError(project_name)
        return SimpleNamespace(tenant_id="org", id=f"p-{project_name}")

    def read_dataset(self, *, dataset_name: str) -> Any:
        from types import SimpleNamespace

        if dataset_name.endswith("memo-dev"):
            raise LangSmithNotFoundError(dataset_name)  # what the real client raises
        return SimpleNamespace(id=f"d-{dataset_name.split('/')[-1]}")

    def list_projects(self, *, reference_dataset_id: Any) -> list[Any]:
        from datetime import datetime
        from types import SimpleNamespace

        names = ["reconcile-claude-sonnet-4.6-v2-a", "reconcile-agent-v1-b", "research-x", "e2e-1",
                 "extract-gpt-oss-120b-v2-c"]  # fmt: skip
        return [
            SimpleNamespace(name=n, id=f"e-{n}", start_time=datetime(2026, 10, 1)) for n in names
        ]

    def list_annotation_queues(self, *, name: str) -> list[Any]:
        from types import SimpleNamespace

        return [SimpleNamespace(id=f"q-{name}")]


def test_workspace_links_finds_each_page_and_skips_what_is_missing() -> None:
    from filings_agent.evals import langsmith_links as ll

    assert ll.workspace_links(FakeLangSmith(set())) == {}  # no project yet: nothing to link
    links = ll.workspace_links(FakeLangSmith({"filings-agent"}))
    assert links["project"].endswith("/o/org/projects/p/p-filings-agent")
    assert links["trace_fanout"].endswith(ll.TRACES["trace_fanout"])
    assert (
        "selectedSessions=e-reconcile-claude-sonnet-4.6-v2-a,e-reconcile-agent-v1-b"
        in (links["compare_pipeline_agent"])
    )
    assert "compare_memo_judged" not in links  # its dataset doesn't exist here
    assert links["queue"].endswith("q-filings-agent: review") and "corrections" in links


def test_connected_scenes_link_their_own_traces_and_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import time

    import langsmith
    from fastapi.testclient import TestClient

    from filings_agent.api.app import create_app
    from filings_agent.config import load_config

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(
        langsmith, "Client", lambda: FakeLangSmith({"filings-agent", demo.DEMO_PROJECT})
    )
    state = tmp_path / "state"
    state.mkdir()
    (state / "scenes.json").write_text(json.dumps({"crash": {"thread_id": "t-crash", "run": 1}}))
    cfg = load_config().model_copy(update={"data_dir": tmp_path})

    def status(cfg: Any, tid: str) -> dict[str, Any]:
        return {"status": "interrupted", "awaiting_review": []}

    app = create_app(
        cfg, run_fn=lambda *a, **k: {}, status_fn=status, web_dist=None, demo_state=state
    )  # pyright: ignore[reportUnknownLambdaType]
    app.state.manager.registry.add_run("t-crash", "r1", "start")
    app.state.manager.registry.add_run("t-crash", "r2", "resume")
    body: dict[str, Any] = {}
    scenes: dict[str, dict[str, Any]] = {}
    with TestClient(app) as client:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:  # the links are looked up in the background
            body = cast(dict[str, Any], client.get("/demo").json())
            scenes = {str(s["id"]): s for s in body["scenes"]}
            if scenes["crash"]["langsmith"][0]["links"]:
                break
            time.sleep(0.05)
    assert body["langsmith"] == "connected"
    crash = cast(list[dict[str, str]], scenes["crash"]["langsmith"][0]["links"])
    assert [(link["text"], link["where"]) for link in crash] == [
        ("start", "this run"),
        ("resume", "this run"),
    ]
    assert crash[1]["url"].endswith(f"p-{demo.DEMO_PROJECT}/r/r2")
    queue = cast(dict[str, str], scenes["annotations"]["langsmith"][0]["links"][0])
    assert queue["where"] == "your workspace" and demo.DEMO_QUEUE in queue["url"]
    items = cast(list[dict[str, Any]], scenes["fanout"]["langsmith"])
    studio = next(i for i in items if "Studio" in i["label"])
    assert studio["links"][0]["url"] == demo.STUDIO_URL
