"""Demo mode: every scene of the walkthrough, with no credentials and no network.

The scenes are the real system: the real graph, API and review UI, run on filings and model
answers recorded into `demo/data/` (committed). Nothing is mocked; what's recorded is the input.
Any model call or web fetch that wasn't recorded is refused instead of attempted, so a demo can
never spend money or quietly need AWS, LangSmith or the SEC contact header.

    uv run python scripts/demo.py serve     # seed the scenes (if needed) and serve them
    uv run python scripts/demo.py record    # author only: re-record demo/data from data/ (AWS)

Each scene is one decision point, opened in a known state. A scene can be reset (a new run of
the same recipe, from the recordings) after it has been clicked through live.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from unittest import mock

from botocore.exceptions import ClientError

from filings_agent.config import AppConfig, load_config
from filings_agent.steps.extract import Extractor, ExtractResult, ExtractTask

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "demo"
DATA = DEMO_DIR / "data"  # committed: filings, answer-key lists, recorded model answers and pages
STATE = DEMO_DIR / "state"  # gitignored: the scenes' runs, rebuilt by `seed`
TICKERS = ("GE", "NVDA")
NOTICE = (
    "Demo mode: the real graph and API, on recorded filings and recorded model answers. "
    "Nothing calls a model or the web; anything not recorded is refused."
)

Kind = Literal["run", "replay", "case", "links"]


@dataclass(frozen=True)
class InLangSmith:
    """A LangSmith page for a scene. `workspace` is a key of langsmith_links.workspace_links (shown
    when the demo is connected to your LangSmith), `public` a key of langsmith_links.PUBLIC (works
    for anyone, no login). "this_run" is the scene's own run, traced when connected."""

    label: str
    point_at: str  # what to show on that page
    workspace: str | None = None
    public: str | None = None


THIS_RUN = "this_run"
STUDIO = "studio"  # LangGraph Studio on the recordings: `make demo-studio`, then this page
STUDIO_URL = "https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"
DEMO_PROJECT = "filings-agent-demo"  # the demo's traces, kept apart from real runs
DEMO_QUEUE = "filings-agent demo: review"


@dataclass(frozen=True)
class Scene:
    id: str
    title: str
    shows: str  # what you'll see
    decision: str  # the design decision it demonstrates
    kind: Kind
    case: str | None = None
    langsmith: tuple[InLangSmith, ...] = ()


SCENES: tuple[Scene, ...] = (
    Scene(
        "fanout", "The fan-out",
        "GE from the start: code finds each statement in each 10-K and runs one model call per "
        "statement, in parallel. Shown as a replay of a recorded run.",
        "One call per statement instead of one per report: F1 0.40 became 0.995.",
        "replay",
        langsmith=(
            InLangSmith(
                "A live GE run, traced",
                "Expand the root: plan, then 12 extract spans started together. Click one and "
                "open Metadata: step, ticker, statement, model, prompt version. Every call is "
                "tagged, so 'every GE balance sheet on gpt-oss' is a filter.",
                workspace="trace_fanout", public="trace_fanout",
            ),
            InLangSmith(
                "This scene's run", "The same graph shape for the run on screen: the 12 extract "
                "branches, then one reconcile branch per changed figure, stopping at review.",
                workspace=THIS_RUN,
            ),
            InLangSmith(
                "The graph, live in LangGraph Studio",
                "Run `make demo-studio` first, then open Studio and input "
                '{"tickers": ["GE"]}. Studio draws the graph, runs the fan-out node by node on the '
                "recordings and stops at review; resume it with decisions in the interrupt panel.",
                workspace=STUDIO,
            ),
        ),
    ),
    Scene(
        "easy-lane", "The easy lane",
        "NVDA: two stock splits, both confirmed by code, published with nothing to review.",
        "Only answers code can prove skip the reviewer.",
        "run",
        langsmith=(
            InLangSmith(
                "This run's trace", "Feedback tab: the online.* checks posted automatically on "
                "every run (branch errors, unknown causes, unverified quotes, cost). Nothing "
                "routed to review.", workspace=THIS_RUN,
            ),
        ),
    ),
    Scene(
        "review", "The review",
        "GE paused with 10 cards. Each lists every possible reason with its evidence; code rules "
        "some out. The agent's own answer is not on the page.",
        "The reviewer judges evidence, not the agent's proposal.",
        "run",
        langsmith=(
            InLangSmith(
                "Why the answer is hidden",
                "The tool-using agent searched, found a real sentence about the spin-off, and "
                "answered confidently. The quote check passed it: the quote exists, it just "
                "doesn't explain the number.",
                workspace="trace_agent_wrong", public="trace_agent_wrong",
            ),
            InLangSmith(
                "This run's trace", "It ends at review (interrupt). Feedback: "
                "online.routed_to_review, and the run is in the demo's annotation queue (scene 9).",
                workspace=THIS_RUN,
            ),
        ),
    ),
    Scene(
        "research", "Research",
        "GE net income after the reviewer asked for research: one investigation per possible "
        "reason, every quote checked against the page it came from.",
        "Code runs one bounded investigation per reason, so every reason gets looked at.",
        "run",
        langsmith=(
            InLangSmith(
                "The accounting-change investigation", "One create_agent per cause: this one "
                "reads GE's filing and PwC guidance and finds the ASU 2018-12 adoption. Tool calls "
                "and their results are spans.", workspace="trace_research", public="trace_research",
            ),
            InLangSmith(
                "Research, evaluated", "The research-hard experiment: one row per hard case, our "
                "evaluators as columns (true cause supported, contradicted, discarded, cost).",
                workspace="compare_research_hard", public="dataset_research_hard",
            ),
        ),
    ),
    Scene(
        "published", "Published once",
        "GE after every decision: the escalations, the reviewer's picks, and a memo whose numbers "
        "all come from the table.",
        "Code builds the numbers, the model writes the prose, and results publish exactly once.",
        "run",
        langsmith=(
            InLangSmith(
                "Reviewer decisions on the trace", "Feedback tab: review.decisions, "
                "review.escalated, review.agent_agreement, review.picked. Charted over time, "
                "these are the production signal.",
                workspace="trace_publish", public="trace_publish",
            ),
            InLangSmith(
                "This run's traces", "The start (pauses), the research round, then the resume that "
                "writes the memo and publishes: one trace per invocation of the thread.",
                workspace=THIS_RUN,
            ),
            InLangSmith(
                "The online evaluator",
                "The project's Evaluators tab: the memo-quality evaluator LangSmith runs on every "
                "published run (an OpenAI judge, our memo rubric, set up from `filings-agent "
                "online-evaluator`). Open the final review trace: judge_faithful, "
                "judge_caveats_correct, judge_comparable, judge_actionable sit next to online.*.",
                workspace="demo_project",
            ),
        ),
    ),
    Scene(
        "crash", "Crash and resume",
        "GE stopped halfway through the fan-out by a simulated Bedrock outage. Resume finishes it: "
        "statements already read are not read again.",
        "A checkpoint after every step, plus a cache of every model answer.",
        "run",
        langsmith=(
            InLangSmith(
                "The failed run and the resume", "The first trace: 6 extract spans succeed, 6 fail "
                "with ThrottlingException after the retry policy. After Resume, the second trace "
                "runs only the 6 that failed.", workspace=THIS_RUN,
            ),
        ),
    ),
    Scene(
        "history", "Two reviewers, two runs",
        "GE net income FY2022: Alice picked the accounting change in one run; Bob escalated it in "
        "a later one. Both are kept, and Alice's pick still stands.",
        "Every decision is kept, and a later run can't erase a person's pick.",
        "case",
        case="GE:net_income:FY2022",
        langsmith=(
            InLangSmith(
                "Corrections become tests",
                "When a reviewer picks a different cause than the agent, the case lands in this "
                "dataset with the exact input the model saw, and every eval and the CI gate "
                "include it from then on.",
                workspace="corrections",
            ),
        ),
    ),
    Scene(
        "annotations", "Annotations",
        "Runs that need a person land in a LangSmith annotation queue with a rubric: are the "
        "causes right, is the memo acceptable, and if not, which step failed.",
        "Online checks route runs to people; their scores become feedback on the trace.",
        "links",
        langsmith=(
            InLangSmith(
                "The demo's review queue",
                "Open a run, read the reconcile spans, score the rubric (human.causes_correct, "
                "human.memo_acceptable, human.root_cause). The scores land on the run's trace "
                "next to the automatic online.* checks. A wrong cause: Add to Dataset on that "
                "reconcile span (filings-agent/reconcile-qa); `make import-qa` makes it a test.",
                workspace="demo_queue",
            ),
            InLangSmith(
                "The production queue", "The same queue for real runs, with what reviewers "
                "scored so far.", workspace="queue",
            ),
        ),
    ),
    Scene(
        "evals", "The evals",
        "The LangSmith experiments behind the design.",
        "The evals decided the architecture.",
        "links",
        langsmith=(
            InLangSmith(
                "Pipeline vs agent on reconcile", "One row per restatement, red where they "
                "disagree: every agent miss is GE. Click a GE cell to open that run's trace.",
                workspace="compare_pipeline_agent", public="dataset_reconcile",
            ),
            InLangSmith(
                "The ranking flips", "gpt-oss is the best extractor and one of the worst "
                "reconcilers: why each step has its own model.", workspace="compare_ranking_flips",
            ),
            InLangSmith(
                "Held-out companies", "Haiku scored 0.999 on dev and 0.928 on test; gpt-oss held. "
                "Why the test split is run once.",
                workspace="compare_heldout", public="dataset_extract_test",
            ),
            InLangSmith(
                "Fresh insurers", "Research on 6 cases never seen while tuning, with the caveat: "
                "one accounting standard.",
                workspace="compare_research_fresh", public="dataset_research_fresh",
            ),
            InLangSmith(
                "Memo writers, judged", "The LLM judge tied 5 of 5 with 40% position bias: not "
                "calibrated yet, so indicative only.", workspace="compare_memo_judged",
            ),
        ),
    ),
)  # fmt: skip
SCENE_BY_ID = {s.id: s for s in SCENES}

# Demo reviewers. Seeding signs in as them so the ledger records who decided; the demo server
# itself runs in local mode (no tokens).
REVIEWERS = {"alice": "alice-demo-token-0001", "bob": "bob-demo-token-00001"}


def demo_config(data: Path = DATA, state: Path = STATE) -> AppConfig:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    spend = cfg.spend.model_copy(
        update={"ledger": state / "spend_ledger.json", "shared_ledger": None}
    )
    return cfg.model_copy(update={"data_dir": data, "state_dir": state, "spend": spend})


# ---- offline: refuse anything not recorded ------------------------------------------------------


class NotRecorded(RuntimeError):
    """The demo needed a model answer or a web page that isn't in demo/data."""


@contextmanager
def offline(tracing: bool = False) -> Generator[None]:
    """No model calls, no web fetches. A cache miss fails loudly with a reason. With `tracing`
    (a LangSmith key is set), runs are traced to the demo's own project and monitored: online
    checks, reviewer feedback and the demo's annotation queue."""
    from filings_agent.llm import spend_guard
    from filings_agent.research import sources

    def refuse_call(self: Any, model_id: str, payload: Any, max_tokens: int | None) -> Any:
        raise spend_guard.SpendLimitExceeded(
            f"demo mode: a call to {model_id} wasn't recorded, so it was refused (nothing sent). "
            "Re-record with `scripts/demo.py record`."
        )

    def refuse_fetch(self: Any, url: str, redirects: int = 3) -> Any:
        raise NotRecorded(f"demo mode: {url} wasn't recorded, so it wasn't fetched")

    if tracing:
        os.environ.update({"LANGSMITH_TRACING": "true", "LANGSMITH_PROJECT": DEMO_PROJECT,
                           "LANGSMITH_REVIEW_QUEUE": DEMO_QUEUE})  # fmt: skip
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
    with (
        mock.patch.object(spend_guard.SpendGuard, "reserve", refuse_call),
        mock.patch.object(sources.LiveFetcher, "_get", refuse_fetch),
    ):
        yield


# ---- record: capture exactly what the scenes read ------------------------------------------------


class Recorder:
    """Notes every model answer (StepCache) and web page (LiveFetcher) the scenes read or write."""

    def __init__(self) -> None:
        self.llm: set[Path] = set()
        self.pages: set[Path] = set()
        self._lock = threading.Lock()

    @contextmanager
    def watching(self) -> Generator[None]:
        from filings_agent.llm.cache import StepCache
        from filings_agent.research.sources import LiveFetcher

        real_get, real_put = StepCache.get, StepCache.put
        real_cached, real_store = LiveFetcher._cached, LiveFetcher._store  # pyright: ignore[reportPrivateUsage]
        rec = self

        def get(self: StepCache, key: str) -> dict[str, Any] | None:
            out = real_get(self, key)
            if out is not None:
                rec._note(rec.llm, self._path(key))  # pyright: ignore[reportPrivateUsage]
            return out

        def put(self: StepCache, key: str, value: dict[str, Any]) -> None:
            real_put(self, key, value)
            rec._note(rec.llm, self._path(key))  # pyright: ignore[reportPrivateUsage]

        def cached(self: LiveFetcher, url: str) -> dict[str, Any] | None:
            out = real_cached(self, url)
            if out is not None:
                rec._note(rec.pages, self._path(url))  # pyright: ignore[reportPrivateUsage]
            return out

        def store(self: LiveFetcher, url: str, record: dict[str, Any]) -> None:
            real_store(self, url, record)
            rec._note(rec.pages, self._path(url))  # pyright: ignore[reportPrivateUsage]

        with (
            mock.patch.object(StepCache, "get", get),
            mock.patch.object(StepCache, "put", put),
            mock.patch.object(LiveFetcher, "_cached", cached),
            mock.patch.object(LiveFetcher, "_store", store),
        ):
            yield

    def _note(self, into: set[Path], path: Path) -> None:
        with self._lock:
            into.add(path)


def export(rec: Recorder, source: Path, dest: Path = DATA) -> dict[str, int]:
    """Copy what the scenes used from `source` (the full data dir) into demo/data."""
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "gold").mkdir(parents=True)
    counts: dict[str, int] = {}
    for name in ("filings.jsonl", "filings_all.jsonl"):
        rows = [r for r in (source / "gold" / name).read_text().splitlines() if r]
        kept = [r for r in rows if json.loads(r)["ticker"] in TICKERS]
        (dest / "gold" / name).write_text("".join(r + "\n" for r in kept))
        counts[name] = len(kept)
    accessions = {
        json.loads(r)["accession_no"]
        for r in (dest / "gold" / "filings_all.jsonl").read_text().splitlines()
    }
    (dest / "filings").mkdir()
    for a in sorted(accessions):
        shutil.copy2(source / "filings" / f"{a}.md", dest / "filings" / f"{a}.md")
    counts["filings"] = len(accessions)
    for kind, paths in (("llm", rec.llm), ("research", rec.pages)):
        for p in paths:
            rel = p.relative_to(source / "cache" / kind)
            (dest / "cache" / kind / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest / "cache" / kind / rel)
        counts[f"cache/{kind}"] = len(paths)
    return counts


# ---- the simulated outage (crash scene) ---------------------------------------------------------


class _Outage:
    """Wraps the extractor: the first `after` statements are read, then Bedrock "throttles" every
    call, past the retry policy, so the run stops halfway with the finished branches saved."""

    def __init__(self, inner: Extractor, after: int) -> None:
        self.name = inner.name
        self._inner = inner
        self._left = after
        self._lock = threading.Lock()

    def extract(self, task: ExtractTask, rep: int = 0) -> ExtractResult:
        with self._lock:
            ok = self._left > 0
            self._left -= 1 if ok else 0
        if not ok:
            raise ClientError(
                {
                    "Error": {
                        "Code": "ThrottlingException",
                        "Message": "Rate exceeded (demo outage)",
                    }
                },
                "Converse",
            )
        return self._inner.extract(task, rep)


OUTAGE_THREADS: set[str] = set()  # threads whose next fresh run hits the simulated outage


def demo_run_fn() -> Callable[..., dict[str, Any]]:
    """run_analysis, except a thread marked for the outage gets the failing extractor once."""
    from filings_agent.graph import runner

    def run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        thread_id = str(args[4]) if len(args) > 4 else str(kwargs.get("thread_id"))
        fresh = not kwargs.get("resume") and kwargs.get("decisions") is None
        if not (fresh and thread_id in OUTAGE_THREADS):
            return runner.run_analysis(*args, **kwargs)
        OUTAGE_THREADS.discard(thread_id)
        real = runner.make_deps

        def make(*a: Any, **k: Any) -> Any:
            deps = real(*a, **k)
            return dataclasses.replace(deps, extractor=_Outage(deps.extractor, after=6))

        with mock.patch.object(runner, "make_deps", make):
            return runner.run_analysis(*args, **kwargs)

    return run


# ---- recipes: each scene's run, driven through the real API -------------------------------------


class Http(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...
    def post(self, url: str, **kwargs: Any) -> Any: ...
    def stream(self, method: str, url: str, **kwargs: Any) -> Any: ...


def thread_for(key: str) -> str:
    """The thread an Idempotency-Key maps to (api.app._thread_id)."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _auth(who: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {REVIEWERS[who]}"}


def _settle(http: Http, tid: str, want: Callable[[dict[str, Any]], bool], who: str = "alice",
            timeout: float = 120) -> dict[str, Any]:  # fmt: skip
    deadline = time.monotonic() + timeout
    body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        body = cast(dict[str, Any], http.get(f"/analyses/{tid}", headers=_auth(who)).json())
        if body.get("status") not in {"running", "resuming"} and want(body):
            return body
        time.sleep(0.1)
    raise RuntimeError(f"{tid} never reached the expected state (last: {body.get('status')})")


def _start(http: Http, key: str, tickers: list[str]) -> str:
    resp = http.post(
        "/analyses", json={"tickers": tickers}, headers={**_auth("alice"), "Idempotency-Key": key}
    )
    resp.raise_for_status()
    return str(resp.json()["thread_id"])


def _review(http: Http, tid: str, decisions: dict[str, Any], who: str) -> None:
    token = http.get(f"/analyses/{tid}", headers=_auth(who)).json()["review_token"]
    resp = http.post(
        f"/analyses/{tid}/review",
        json={"decisions": decisions, "review_token": token},
        headers=_auth(who),
    )
    resp.raise_for_status()


def _first_round(cards: list[str], net_income_2022: dict[str, Any]) -> dict[str, Any]:
    """The GE decisions used by the review scenes (the same ones the smoke test makes, so the
    research round is the recorded one)."""
    d: dict[str, Any] = {
        k: {"action": "accept", "correct_cause": "discontinued_operations"} for k in cards
    }
    d["GE:net_income:FY2021"] = {"action": "accept", "correct_cause": "accounting_change"}
    d["GE:revenue:FY2021"] = {"action": "use_original", "correct_cause": "discontinued_operations"}
    d["GE:long_term_debt:FY2023"] = {"action": "escalate", "note": "none of these fit"}
    d["GE:operating_cash_flow:FY2021"] = {"action": "escalate", "note": "none of these fit"}
    d["GE:net_income:FY2022"] = net_income_2022
    return d


def _cards(body: dict[str, Any]) -> list[str]:
    return [str(c["id"]) for c in body["awaiting_review"]]


def _waiting(body: dict[str, Any]) -> bool:
    return body["status"] == "awaiting_review"


def _published(body: dict[str, Any]) -> bool:
    return body["status"] == "published"


def recipe(scene: str, http: Http, run: int, replays: Path) -> str:
    """Create the scene's run (the `run`-th time: a reset uses a new key). Returns its thread."""
    key = f"demo-{scene}-{run}"
    if scene == "fanout":
        tid = _start(http, key, ["GE"])
        events: list[dict[str, Any]] = []
        with http.stream("GET", f"/analyses/{tid}/events", headers=_auth("alice")) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
                    if events[-1].get("event") == "end":
                        break
        replays.mkdir(parents=True, exist_ok=True)
        (replays / "fanout.json").write_text(json.dumps(events, indent=1))
        _settle(http, tid, _waiting)
        return tid
    if scene == "easy-lane":
        tid = _start(http, key, ["NVDA"])
        _settle(http, tid, _published)
        return tid
    if scene == "review":
        tid = _start(http, key, ["GE"])
        _settle(http, tid, _waiting)
        return tid
    if scene in ("research", "published"):
        tid = _start(http, key, ["GE"])
        cards = _cards(_settle(http, tid, _waiting))
        _review(http, tid, _first_round(cards, {"action": "research"}), "alice")
        _settle(http, tid, lambda b: _waiting(b) and b.get("research_rounds") == 1)
        if scene == "published":
            final = {
                "GE:net_income:FY2022": {
                    "action": "accept",
                    "correct_cause": "accounting_change",
                    "note": "ASU 2018-12, adopted retrospectively",
                }
            }
            _review(http, tid, final, "alice")
            _settle(http, tid, _published)
        return tid
    if scene == "history":
        # Bob reviews GE in a later run and escalates net income FY2022, which Alice picked in the
        # "published" scene. Both decisions stay in the ledger; Alice's pick stands.
        tid = _start(http, key, ["GE"])
        cards = _cards(_settle(http, tid, _waiting, who="bob"))
        bob = {"action": "escalate", "note": "the evidence points two ways; needs a second look"}
        _review(http, tid, _first_round(cards, bob), "bob")
        _settle(http, tid, _published, who="bob")
        return tid
    if scene == "crash":
        OUTAGE_THREADS.add(thread_for(key))
        tid = _start(http, key, ["GE"])
        _settle(http, tid, lambda b: b["status"] == "interrupted", timeout=180)
        return tid
    raise KeyError(f"no recipe for scene {scene!r}")


# Order matters: "history" must run after "published" (Bob's escalation comes after Alice's pick).
SEED_ORDER = ("fanout", "easy-lane", "review", "research", "published", "history", "crash")


def scenes_file(state: Path = STATE) -> Path:
    return state / "scenes.json"


def load_scenes(state: Path = STATE) -> dict[str, dict[str, Any]]:
    path = scenes_file(state)
    return cast(dict[str, dict[str, Any]], json.loads(path.read_text())) if path.exists() else {}


def save_scene(state: Path, scene: str, thread_id: str, run: int) -> None:
    scenes = load_scenes(state)
    scenes[scene] = {"thread_id": thread_id, "run": run}
    state.mkdir(parents=True, exist_ok=True)
    scenes_file(state).write_text(json.dumps(scenes, indent=2))


def tracing_on() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() in {"1", "true"}


def demo_monitor() -> Callable[[dict[str, Any]], object] | None:
    """Online checks and reviewer feedback on each finished run, when traced."""
    if not tracing_on():
        return None
    from filings_agent.evals.online import monitor

    return monitor


@contextmanager
def seeding_client(cfg: AppConfig) -> Generator[Http]:
    """The real API in-process, with the demo reviewers signed in."""
    from fastapi.testclient import TestClient

    from filings_agent.api.app import create_app

    tokens = ",".join(f"{who}:{tok}" for who, tok in REVIEWERS.items())
    with mock.patch.dict(os.environ, {cfg.api.tokens_env: tokens}):
        app = create_app(cfg, run_fn=demo_run_fn(), web_dist=None, monitor_fn=demo_monitor())
    with TestClient(app) as client:
        yield cast(Http, client)
        # monitoring posts in the background: let it finish before the event loop closes
        deadline = time.monotonic() + 120
        while app.state.manager._background and time.monotonic() < deadline:  # pyright: ignore[reportPrivateUsage]
            time.sleep(0.2)


def clean_langsmith() -> None:
    """A full rebuild empties the demo's annotation queue (it comes back with this build's runs).
    The demo's project is kept on purpose: its online evaluator (`online-evaluator --create`) is a
    rule on it and would be deleted with it; old traces expire with the project's retention
    (14 days), and each scene links straight to its own traces. Only the demo's queue is touched.
    Best effort."""
    if not tracing_on():
        return
    try:
        from langsmith import Client

        client = Client()
        for q in client.list_annotation_queues(name=DEMO_QUEUE):
            client.delete_annotation_queue(q.id)
    except Exception as e:
        print(f"[demo] couldn't clear the demo's annotation queue: {e}", flush=True)


def seed(cfg: AppConfig, only: tuple[str, ...] = SEED_ORDER) -> dict[str, dict[str, Any]]:
    """Build the scenes' runs into cfg.state_dir. With `only`, re-run just those (a reset).
    A full build also empties the demo's LangSmith queue, so it holds this build's runs only."""
    state = cfg.state_dir or cfg.data_dir
    if only == SEED_ORDER:
        clean_langsmith()
    if only == SEED_ORDER and state.exists():
        shutil.rmtree(state)
    state.mkdir(parents=True, exist_ok=True)
    with seeding_client(cfg) as http:
        for scene in only:
            run = int(load_scenes(state).get(scene, {}).get("run", 0)) + 1
            save_scene(state, scene, recipe(scene, http, run, state / "replays"), run)
    return load_scenes(state)
