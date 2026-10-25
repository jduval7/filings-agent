"""HTTP API.

  POST /analyses                 start a run. `Idempotency-Key` header = the run's identity:
                                 same key + same body -> the existing run (no second fan-out);
                                 same key + different body -> 422 (IETF idempotency draft).
                                 No key: the identity is the request's inputs (tickers, filings
                                 on disk, models, prompts, settings, graph version), so a new
                                 10-K or a config change starts a new analysis. `fresh: true`
                                 always starts a new one (re-analyze).
  GET  /analyses                 recent analyses, newest first (for the review UI)
  GET  /analyses/{id}            status, results, what's waiting for review, last_run_id (trace)
  GET  /meta                     LangSmith project URL, for "open trace" links (best-effort)
  GET  /analyses/{id}/events     live progress as Server-Sent Events (replays from the start)
  POST /analyses/{id}/review     submit human decisions; the run resumes and publishes
  POST /analyses/{id}/resume     continue a run that stopped mid-way (crash, outage); finished
                                 steps are checkpointed and not run again
  GET  /cases/{discrepancy_id}   one case's decision history from the ledger, and what stands
  GET  /demo, /demo/replay/{id}  demo mode only (scripts/demo.py): the scenes, and a paced replay
                                 of a recorded run's progress events
  POST /auth/session             exchange an API token for a session cookie (the UI; EventSource
                                 can't send headers). DELETE ends it. GET /auth/me: who am I.
  GET  /health
  GET  /                         the review UI (web/, built with `make web`), if built

Every /analyses, /meta and /auth/me call needs `Authorization: Bearer <token>` (or the session
cookie) when FILINGS_API_TOKENS is set; the token's name is recorded as the reviewer of every
decision. Unset, the server runs in local mode and only listens on loopback (cli `serve`).

Runs execute in a worker thread (the graph is sync); events cross back to the event loop with
call_soon_threadsafe. State lives in the LangGraph checkpointer, so a restarted server can still
report, review, and resume any run. Which process runs a thread is decided by a lease in the
database (storage/leases.py), not by this process's memory, so several workers (or the CLI next
to the server) never run the same thread twice.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import sqlite3
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from filings_agent.config import AppConfig
from filings_agent.storage.leases import Leases, ThreadBusy

RunFn = Callable[..., dict[str, Any]]
StatusFn = Callable[[AppConfig, str], dict[str, Any]]
IdentityFn = Callable[[AppConfig, list[str], str, str], dict[str, Any]]
SESSION_COOKIE = "filings_agent_session"
LOCAL_USER = "local"


# ---- request / response models ----------------------------------------------------------------


class CreateAnalysis(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=20)
    extractor: str | None = Field(default=None, description="config model name; default role")
    prompt: str | None = Field(default=None, description="defaults to prompts.extract")
    fresh: bool = Field(
        default=False, description="re-analyze: a new run with new LLM calls, even if one exists"
    )

    @field_validator("tickers")
    @classmethod
    def _upper(cls, v: list[str]) -> list[str]:
        out = sorted({t.strip().upper() for t in v if t.strip()})
        if not out:  # min_length runs before this: [" "] would otherwise start an empty run
            raise ValueError("at least one non-blank ticker")
        return out


class AnalysisCreated(BaseModel):
    thread_id: str
    created: bool  # False: this key already had a run; nothing new was started
    status: str
    status_url: str
    events_url: str


class DecisionIn(BaseModel):
    action: Literal["accept", "use_original", "escalate", "research"]
    note: str = ""
    correct_cause: (
        Literal["stock_split", "discontinued_operations", "accounting_change", "reclassification"]
        | None
    ) = None


class ReviewRequest(BaseModel):
    decisions: dict[str, DecisionIn] = Field(min_length=1)
    # The `review_token` from GET /analyses/{id}: which cards (and which research round of each)
    # the reviewer saw. A mismatch means the cards changed since (409). Required: a client that
    # skipped it could approve cards it never saw (found by review).
    review_token: str = Field(min_length=1)


class SessionIn(BaseModel):
    token: str = Field(min_length=1, max_length=500)


def review_token(cards: list[dict[str, Any]]) -> str | None:
    """Identifies the cards a reviewer is looking at, including research rounds already run."""
    if not cards:
        return None
    seen = sorted((str(c["id"]), len(c.get("research_rounds") or [])) for c in cards)
    return hashlib.sha256(json.dumps(seen).encode()).hexdigest()[:16]


def without_pending_causes(status: dict[str, Any]) -> dict[str, Any]:
    """The agent's cause for a change still waiting on a human stays on the server: the review
    must never show (or ship) the agent's answer before the decision. `summarize` already masks
    it; this keeps any other status source honest too."""
    cards = cast(list[dict[str, Any]], status.get("awaiting_review") or [])
    pending = {str(c["id"]) for c in cards}
    resolutions = cast(dict[str, dict[str, Any]], status.get("resolutions") or {})
    masked = {k: ({**r, "cause": None} if k in pending else r) for k, r in resolutions.items()}
    return {**status, "resolutions": masked}


# ---- run registry (persisted, so params survive a restart) ------------------------------------


class RunRegistry:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS runs (thread_id TEXT PRIMARY KEY, body_hash TEXT NOT NULL,"
            " params TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        # Every invocation's LangSmith root run, per thread: start, review, resume ("open trace"
        # links; a crashed run's trace is linked too, because the id is chosen before it runs).
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS run_history (thread_id TEXT NOT NULL, run_id TEXT NOT NULL,"
            " kind TEXT NOT NULL, started_at TEXT NOT NULL, PRIMARY KEY (thread_id, run_id))"
        )
        self._conn.commit()

    def add_run(self, thread_id: str, run_id: str, kind: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO run_history VALUES (?, ?, ?, ?)",
            (thread_id, run_id, kind, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    def runs_of(self, thread_id: str) -> list[dict[str, str]]:
        rows = self._conn.execute(
            "SELECT run_id, kind, started_at FROM run_history WHERE thread_id = ?"
            " ORDER BY started_at, rowid",
            (thread_id,),
        ).fetchall()
        return [{"run_id": r[0], "kind": r[1], "started_at": r[2]} for r in rows]

    def last_run(self, thread_id: str) -> str | None:
        runs = self.runs_of(thread_id)
        return runs[-1]["run_id"] if runs else None

    def get(self, thread_id: str) -> tuple[str, dict[str, Any]] | None:
        row = self._conn.execute(
            "SELECT body_hash, params FROM runs WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def recent(self, limit: int = 20) -> list[tuple[str, dict[str, Any], str]]:
        rows = self._conn.execute(
            "SELECT thread_id, params, created_at FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [(r[0], json.loads(r[1]), r[2]) for r in rows]

    def put(self, thread_id: str, body_hash: str, params: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO runs VALUES (?,?,?,?)",
            (thread_id, body_hash, json.dumps(params), datetime.now(UTC).isoformat()),
        )
        self._conn.commit()


# ---- live runs --------------------------------------------------------------------------------


@dataclass
class LiveRun:
    events: list[dict[str, Any]] = field(default_factory=lambda: list[dict[str, Any]]())
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()


class RunManager:
    def __init__(
        self,
        cfg: AppConfig,
        run_fn: RunFn,
        status_fn: StatusFn,
        monitor_fn: Callable[[dict[str, Any]], object] | None = None,
        identity_fn: IdentityFn | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_fn = run_fn
        self.status_fn = status_fn
        self.identity_fn: IdentityFn = identity_fn or _request_only_identity
        self.leases = Leases(cfg.state("leases.sqlite"))
        self.monitor_fn: Callable[[dict[str, Any]], object] = monitor_fn or _no_monitor
        self.registry = RunRegistry(cfg.state("runs.sqlite"))
        self.live: dict[str, LiveRun] = {}
        self._background: set[asyncio.Task[object]] = set()  # monitoring tasks, kept referenced
        self._project_url: str | None = None
        self._locks: dict[str, asyncio.Lock] = {}

    def project_url(self) -> str | None:
        """The LangSmith project page, if tracing is on and reachable. Cached; never raises."""
        if self._project_url is None:
            self._project_url = langsmith_project_url() or ""
        return self._project_url or None

    def lock(self, thread_id: str) -> asyncio.Lock:
        return self._locks.setdefault(thread_id, asyncio.Lock())

    def running(self, thread_id: str) -> bool:
        """Running here, or in another process (it holds the thread's lease)."""
        live = self.live.get(thread_id)
        return bool(live and live.running) or self.leases.holder(thread_id) is not None

    def _launch(self, thread_id: str, params: dict[str, Any], owner: str, **extra: Any) -> None:
        """Run with the lease `owner` already holds (taken by the caller, under the thread's lock,
        before it answered the request). Released when the run ends, however it ends."""
        loop = asyncio.get_running_loop()
        # A fresh event log per launch: after a review, a new subscriber must see the resumed run,
        # not replay the paused run's "end" and stop (found building the review UI).
        live = self.live[thread_id] = LiveRun()
        run_id = str(uuid.uuid4())
        kind = "resume" if extra.get("resume") else "review" if "decisions" in extra else "start"
        self.registry.add_run(thread_id, run_id, kind)

        def push(event: dict[str, Any]) -> None:
            def _append() -> None:
                live.events.append(event)
                live.changed.set()

            loop.call_soon_threadsafe(_append)

        async def body() -> None:
            try:
                summary = await asyncio.to_thread(
                    self.run_fn, self.cfg, params["tickers"], params["extractor"],
                    params["prompt"], thread_id, on_event=push, rep=params.get("rep", 0),
                    lease_owner=owner, run_id=run_id, **extra,
                )  # fmt: skip
                push({"event": "end", "status": summary.get("status")})
                # Monitoring (LangSmith feedback, seconds) runs after the run counts as finished:
                # otherwise a client that reloads on "end" still sees "running" (found in the UI).
                done = asyncio.create_task(asyncio.to_thread(self.monitor_fn, summary))
                self._background.add(done)
                done.add_done_callback(self._background.discard)
            except ThreadBusy:  # lost a race with another process: it is running the thread
                push({"event": "end", "status": "running_elsewhere"})
            except BaseException as e:  # surface failures to subscribers, then re-raise
                push({"event": "end", "status": "failed", "error": f"{type(e).__name__}: {e}"})
                raise
            finally:
                await asyncio.to_thread(self.leases.release, thread_id, owner)

        live.task = asyncio.create_task(body())
        live.task.add_done_callback(_log_failure)

    async def start(self, key: str | None, body: CreateAnalysis) -> AnalysisCreated:
        extractor = body.extractor or self.cfg.roles.extractor
        if extractor not in self.cfg.models:  # fail before anything starts
            raise HTTPException(422, f"unknown extractor {extractor!r}")
        prompt = body.prompt or self.cfg.prompts.extract
        # What the client asked for: a reused key must carry the same request.
        request = {"tickers": body.tickers, "extractor": extractor, "prompt": prompt,
                   "fresh": body.fresh}  # fmt: skip
        body_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        identity = await asyncio.to_thread(
            self.identity_fn, self.cfg, body.tickers, extractor, prompt
        )
        identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if key:
            # The key names this request, fresh or not: a client retry must reach the same run
            # (and the same LLM-cache repetition), even if the filings changed in between.
            thread_id = _thread_id(key)
            rep = (int(thread_id[:8], 16) or 1) if body.fresh else 0
        else:
            # No key: the inputs are the identity. `fresh` is a deliberate new analysis.
            rep = time.time_ns() // 1_000_000_000 if body.fresh else 0
            thread_id = identity_hash[:16] + (f"-r{rep}" if rep else "")
        params: dict[str, Any] = {
            "tickers": body.tickers,
            "extractor": extractor,
            "prompt": prompt,
            "rep": rep,
            "identity": identity,
        }
        async with self.lock(thread_id):
            existing = self.registry.get(thread_id)
            if existing and existing[0] != body_hash:
                raise HTTPException(422, "Idempotency-Key reused with a different request body")
            status = (await asyncio.to_thread(self.status_fn, self.cfg, thread_id))["status"]
            created = False
            if self.running(thread_id):
                status = "running"
            elif status in ("not_found", "interrupted"):
                owner = uuid.uuid4().hex
                if not await asyncio.to_thread(self.leases.acquire, thread_id, owner):
                    status = "running"  # another process took it between the check and here
                elif status == "not_found":
                    self.registry.put(thread_id, body_hash, params)
                    self._launch(thread_id, params, owner)
                    created, status = True, "running"
                else:
                    # The server died mid-run. A client retry resumes it: finished branches are
                    # checkpointed and not paid for again.
                    stored = existing[1] if existing else params
                    self._launch(thread_id, stored, owner, resume=True)
                    status = "resuming"
        return AnalysisCreated(
            thread_id=thread_id,
            created=created,
            status=status,
            status_url=f"/analyses/{thread_id}",
            events_url=f"/analyses/{thread_id}/events",
        )

    async def review(self, thread_id: str, req: ReviewRequest, actor: str) -> dict[str, Any]:
        async with self.lock(thread_id):
            reg = self.registry.get(thread_id)
            if reg is None:
                raise HTTPException(404, "unknown analysis")
            if self.running(thread_id):
                raise HTTPException(409, "run is still in progress")
            status = await asyncio.to_thread(self.status_fn, self.cfg, thread_id)
            if status["status"] != "awaiting_review":
                raise HTTPException(409, f"not awaiting review (status: {status['status']})")
            cards = {str(p["id"]): p for p in status["awaiting_review"]}
            if req.review_token != review_token(list(cards.values())):
                raise HTTPException(409, "the cards changed since you loaded them: reload")
            missing = set(cards) - set(req.decisions)
            unknown = set(req.decisions) - set(cards)
            if missing or unknown:
                raise HTTPException(
                    422, {"missing_decisions": sorted(missing), "unknown_ids": sorted(unknown)}
                )
            invalid: dict[str, str] = {}
            for k, d in req.decisions.items():
                if d.action in ("accept", "use_original") and d.correct_cause is None:
                    # otherwise the agent's hidden answer would be published unseen
                    invalid[k] = "pick the cause (correct_cause) to accept"
                if d.action == "research" and int(cards[k].get("research_rounds_left") or 0) <= 0:
                    invalid[k] = "no research rounds left: decide or escalate"
            if invalid:
                raise HTTPException(422, {"invalid_decisions": invalid})
            # Who and when come from the server, never from the payload.
            stamp = {"actor": actor, "decided_at": datetime.now(UTC).isoformat()}
            decisions = {
                k: {**v.model_dump(exclude_none=True), **stamp} for k, v in req.decisions.items()
            }
            owner = uuid.uuid4().hex
            if not await asyncio.to_thread(self.leases.acquire, thread_id, owner):
                raise HTTPException(409, "run is in progress in another process")
            self._launch(thread_id, reg[1], owner, decisions=decisions)
        return {"thread_id": thread_id, "status": "resuming"}

    async def resume(self, thread_id: str) -> dict[str, Any]:
        """A run that stopped mid-way (a crash, an outage past the retries) continues from its
        last checkpoint. The same thing a client retry with the same Idempotency-Key does."""
        async with self.lock(thread_id):
            reg = self.registry.get(thread_id)
            if reg is None:
                raise HTTPException(404, "unknown analysis")
            if self.running(thread_id):
                raise HTTPException(409, "run is still in progress")
            status = (await asyncio.to_thread(self.status_fn, self.cfg, thread_id))["status"]
            if status != "interrupted":
                raise HTTPException(409, f"nothing to resume (status: {status})")
            owner = uuid.uuid4().hex
            if not await asyncio.to_thread(self.leases.acquire, thread_id, owner):
                raise HTTPException(409, "run is in progress in another process")
            self._launch(thread_id, reg[1], owner, resume=True)
        return {"thread_id": thread_id, "status": "resuming"}

    async def stream(self, thread_id: str) -> AsyncIterator[str]:
        live = self.live.get(thread_id)
        if live is None:  # not running in this process: report final state once
            status = await asyncio.to_thread(self.status_fn, self.cfg, thread_id)
            yield _sse({"event": "status", **status})
            return
        sent = 0
        while True:
            while sent < len(live.events):
                event = live.events[sent]
                sent += 1
                yield _sse(event)
                if event.get("event") == "end":
                    return
            live.changed.clear()
            if sent < len(live.events):
                continue
            try:
                await asyncio.wait_for(live.changed.wait(), timeout=15)
            except TimeoutError:
                yield ": keep-alive\n\n"


def _no_monitor(_: dict[str, Any]) -> None:
    return None


def _log_failure(task: asyncio.Task[None]) -> None:
    """Nobody awaits a run's task; surface its failure instead of asyncio's "never retrieved"."""
    if not task.cancelled() and (e := task.exception()) is not None:
        import sys

        print(f"[api] run failed: {type(e).__name__}: {e}", file=sys.stderr)


def langsmith_project_url() -> str | None:
    """Best-effort: the UI links to traces only when tracing is on and LangSmith answers."""
    import os

    if os.environ.get("LANGSMITH_TRACING", "").lower() not in {"1", "true"}:
        return None
    try:
        from langsmith import Client

        project = Client().read_project(project_name=os.environ.get("LANGSMITH_PROJECT", "default"))
        return f"https://smith.langchain.com/o/{project.tenant_id}/projects/p/{project.id}"
    except Exception:  # an unreachable LangSmith must never break the review UI
        return None


def _request_only_identity(
    cfg: AppConfig, tickers: list[str], extractor: str, prompt: str
) -> dict[str, Any]:
    """For tests that inject a run function: no filings or graph to describe."""
    return {"tickers": sorted(tickers), "extractor": extractor, "prompt": prompt}


class Auth:
    """Bearer tokens (or the session cookie the UI gets for one) -> the user's name."""

    def __init__(self, tokens: dict[str, str]) -> None:
        self._tokens = tokens

    @property
    def enabled(self) -> bool:
        return bool(self._tokens)

    def user_for(self, token: str | None) -> str | None:
        if not token:
            return None
        found: str | None = None
        for known, name in self._tokens.items():  # constant time per token: no timing oracle
            if hmac.compare_digest(known.encode(), token.encode()):
                found = name
        return found

    def __call__(self, request: Request) -> str:
        """FastAPI dependency: the caller's name, or 401."""
        if not self.enabled:
            return LOCAL_USER
        header = request.headers.get("authorization", "")
        scheme, _, credentials = header.partition(" ")
        token = credentials.strip() if scheme.lower() == "bearer" else None
        user = self.user_for(token or request.cookies.get(SESSION_COOKIE))
        if user is None:
            raise HTTPException(401, "missing or unknown API token",
                                headers={"WWW-Authenticate": "Bearer"})  # fmt: skip
        return user


def _thread_id(key: str) -> str:
    # Keys are client-supplied: hash them so they're safe as ids and file-system keys.
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _sse(event: dict[str, Any]) -> str:
    return f"event: {event.get('event', 'message')}\ndata: {json.dumps(event, default=str)}\n\n"


# ---- demo mode --------------------------------------------------------------------------------


class _WorkspaceLinks:
    """The workspace pages (langsmith_links), looked up in the background when connected, so a
    request never waits on LangSmith. The demo's own project and queue appear only after its first
    traced run, so they're looked up again a minute later until found."""

    def __init__(self) -> None:
        self._links: dict[str, str] = {}
        self._at = 0.0
        self._busy = threading.Lock()

    def get(self) -> dict[str, str]:
        from filings_agent import demo

        if not demo.tracing_on():
            return {}
        age = time.time() - self._at
        if (age > 600 or ("demo_queue" not in self._links and age > 60)) and self._busy.acquire(
            blocking=False
        ):
            threading.Thread(target=self._refresh, name="langsmith-links", daemon=True).start()
        return self._links

    def _refresh(self) -> None:
        try:
            from langsmith import Client

            from filings_agent import demo
            from filings_agent.evals import langsmith_links as ll

            client = Client()
            found = ll.workspace_links(client)
            if proj := ll.project_url(client, demo.DEMO_PROJECT):
                found["demo_project"] = proj[1]
                if q := ll.queue_url(client, proj[0], demo.DEMO_QUEUE):
                    found["demo_queue"] = q
            self._links = found
        except Exception as e:  # LangSmith unreachable: scenes fall back to public links
            print(f"[demo] LangSmith links not loaded: {type(e).__name__}: {e}", file=sys.stderr)
        finally:
            self._at = time.time()
            self._busy.release()


REPLAY_PACE_S = {"planned": 0.8, "branch_done": 0.35, "branch_failed": 0.35, "wave1_done": 0.8}


def add_demo_routes(app: FastAPI, manager: RunManager, state: Path, user_dep: Any) -> None:
    """The scenes (filings_agent/demo.py), each scene's run and status, its LangSmith pages, and
    a paced replay."""
    from filings_agent import demo
    from filings_agent.evals.langsmith_links import PUBLIC as LANGSMITH_PUBLIC

    links = _WorkspaceLinks()
    links.get()  # start the first lookup now, while the server comes up

    def langsmith_items(scene: Any, thread_id: str | None) -> list[dict[str, Any]]:
        ws = links.get()
        out: list[dict[str, Any]] = []
        for item in scene.langsmith:
            found: list[dict[str, str]] = []
            if item.workspace == demo.THIS_RUN:
                demo_project = ws.get("demo_project")
                if demo_project and thread_id:
                    runs = manager.registry.runs_of(thread_id)
                    for r in runs:
                        same = [x for x in runs if x["kind"] == r["kind"]]
                        n = f" {same.index(r) + 1}" if len(same) > 1 else ""
                        url = f"{demo_project}/r/{r['run_id']}"
                        found.append({"text": f"{r['kind']}{n}", "url": url, "where": "this run"})
            elif item.workspace == demo.STUDIO:
                found.append({"text": "open Studio", "url": demo.STUDIO_URL, "where": "Studio"})
            elif item.workspace and item.workspace in ws:
                found.append({"text": "open", "url": ws[item.workspace], "where": "your workspace"})
            if item.public and not found:
                found.append(
                    {"text": "open", "url": LANGSMITH_PUBLIC[item.public], "where": "public"}
                )
            out.append({"label": item.label, "point_at": item.point_at, "links": found})
        return out

    @app.get("/demo")
    async def scenes(_user: str = user_dep) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        seeded = demo.load_scenes(state)
        out: list[dict[str, Any]] = []
        for s in demo.SCENES:
            tid = seeded.get(s.id, {}).get("thread_id")
            status = None
            if tid:
                status = (await asyncio.to_thread(manager.status_fn, manager.cfg, str(tid)))[
                    "status"
                ]
                if await asyncio.to_thread(manager.running, str(tid)):
                    status = "running"
            items = await asyncio.to_thread(langsmith_items, s, str(tid) if tid else None)
            out.append({
                "id": s.id, "title": s.title, "shows": s.shows, "decision": s.decision,
                "kind": s.kind, "case": s.case, "thread_id": tid, "status": status,
                "langsmith": items,
            })  # fmt: skip
        return {
            "notice": demo.NOTICE,
            "langsmith": "connected" if demo.tracing_on() else "offline",
            "scenes": out,
        }

    @app.post("/demo/{scene_id}/reset", status_code=202)
    async def reset(scene_id: str, _user: str = user_dep) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """A new run of the scene's recipe, from the recordings, after it was clicked through."""
        if scene_id not in demo.SCENE_BY_ID or demo.SCENE_BY_ID[scene_id].kind == "links":
            raise HTTPException(404, "unknown scene")
        only = ("published", "history") if scene_id == "history" else (scene_id,)
        result = await asyncio.to_thread(demo.seed, manager.cfg, only)
        return {"scene": scene_id, "thread_id": result[scene_id]["thread_id"]}

    @app.get("/demo/replay/{scene_id}")
    async def replay(scene_id: str, _user: str = user_dep) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        path = state / "replays" / f"{scene_id}.json"
        if not path.exists():
            raise HTTPException(404, "no recorded run for this scene")
        events = cast(list[dict[str, Any]], json.loads(path.read_text()))

        async def paced() -> AsyncIterator[str]:
            for e in events:
                yield _sse({**e, "replay": True})
                await asyncio.sleep(REPLAY_PACE_S.get(str(e.get("event")), 0.3))

        return StreamingResponse(paced(), media_type="text/event-stream")


# ---- app --------------------------------------------------------------------------------------


WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


def create_app(
    cfg: AppConfig,
    run_fn: RunFn | None = None,
    status_fn: StatusFn | None = None,
    web_dist: Path | None = WEB_DIST,
    identity_fn: IdentityFn | None = None,
    demo_state: Path | None = None,
    monitor_fn: Callable[[dict[str, Any]], object] | None = None,
) -> FastAPI:
    from filings_agent.evals.online import monitor
    from filings_agent.graph.runner import get_status, run_analysis, run_identity

    # Production wiring posts online checks; tests that inject fakes get no monitoring.
    production = run_fn is None
    manager = RunManager(
        cfg,
        run_fn or run_analysis,
        status_fn or get_status,
        monitor_fn or (monitor if production else None),
        identity_fn or (run_identity if production else None),
    )
    auth = Auth(cfg.api.tokens())
    app = FastAPI(title="filings-agent", version="0.1.0")
    app.state.manager = manager
    app.state.auth = auth
    user_dep = Depends(auth)  # the caller's name (401 without a valid token)

    @app.post("/auth/session")
    async def login(body: SessionIn, request: Request, response: Response) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        user = auth.user_for(body.token) if auth.enabled else LOCAL_USER
        if user is None:
            raise HTTPException(401, "unknown API token")
        if auth.enabled:
            response.set_cookie(
                SESSION_COOKIE, body.token, httponly=True, samesite="strict",
                secure=request.url.scheme == "https", max_age=12 * 3600,
            )  # fmt: skip
        return {"user": user}

    @app.delete("/auth/session")
    async def logout(response: Response) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        response.delete_cookie(SESSION_COOKIE)
        return {"status": "ok"}

    @app.get("/auth/me")
    async def me(user: str = user_dep) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return {"user": user, "auth": auth.enabled}

    @app.get("/health")
    async def health() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        return {"status": "ok"}

    @app.post("/analyses", status_code=202)
    async def create(  # pyright: ignore[reportUnusedFunction]
        body: CreateAnalysis,
        _user: str = user_dep,
        idempotency_key: Annotated[str | None, Header(max_length=200)] = None,
    ) -> AnalysisCreated:
        return await manager.start(idempotency_key, body)

    @app.get("/analyses")
    async def recent(_user: str = user_dep) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        out: list[dict[str, Any]] = []
        for thread_id, params, created_at in manager.registry.recent():
            s = await asyncio.to_thread(manager.status_fn, cfg, thread_id)
            running = await asyncio.to_thread(manager.running, thread_id)
            status = "running" if running else s["status"]
            out.append(
                {
                    "thread_id": thread_id,
                    "tickers": params["tickers"],
                    "created_at": created_at,
                    "status": status,
                    "waiting": len(s.get("awaiting_review") or []),
                }
            )
        return out

    @app.get("/analyses/{thread_id}")
    async def status(thread_id: str, _user: str = user_dep) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        if manager.registry.get(thread_id) is None:
            raise HTTPException(404, "unknown analysis")
        result = await asyncio.to_thread(manager.status_fn, cfg, thread_id)
        # A copy: never mutate what the status source returned (it may be shared/cached).
        out = {
            **without_pending_causes(result),
            "last_run_id": manager.registry.last_run(thread_id),
            "runs": manager.registry.runs_of(thread_id),
            "review_token": review_token(list(result.get("awaiting_review") or [])),
        }
        if await asyncio.to_thread(manager.running, thread_id):
            out["status"] = "running"
        return out

    @app.post("/analyses/{thread_id}/resume", status_code=202)
    async def resume(thread_id: str, _user: str = user_dep) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return await manager.resume(thread_id)

    @app.get("/cases/{discrepancy_id}")
    async def case(discrepancy_id: str, _user: str = user_dep) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        from filings_agent.graph.runner import shared_ledger

        ledger = shared_ledger(cfg.state("ledger.sqlite"))
        history = await asyncio.to_thread(ledger.history, discrepancy_id)
        current = await asyncio.to_thread(ledger.current, discrepancy_id)
        return {"discrepancy_id": discrepancy_id, "current": current, "history": history}

    if demo_state is not None:
        add_demo_routes(app, manager, demo_state, user_dep)

    @app.get("/meta")
    async def meta(_user: str = user_dep) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return {
            "langsmith_project_url": await asyncio.to_thread(manager.project_url),
            "demo": demo_state is not None,
        }

    @app.get("/analyses/{thread_id}/events")
    async def events(thread_id: str, _user: str = user_dep) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        if manager.registry.get(thread_id) is None:
            raise HTTPException(404, "unknown analysis")
        return StreamingResponse(manager.stream(thread_id), media_type="text/event-stream")

    @app.post("/analyses/{thread_id}/review", status_code=202)
    async def review(  # pyright: ignore[reportUnusedFunction]
        thread_id: str, req: ReviewRequest, user: str = user_dep
    ) -> dict[str, Any]:
        return await manager.review(thread_id, req, user)

    # The review UI, if built (`make web`). Mounted last so it never shadows the API.
    if web_dist is not None and (web_dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return app
