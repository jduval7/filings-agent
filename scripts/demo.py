"""The walkthrough's scenes, runnable by anyone: no AWS, no LangSmith, no SEC contact, no network.

    uv run python scripts/demo.py serve           # seed the scenes if needed, serve :8000
    uv run python scripts/demo.py serve --reset   # rebuild every scene first
    uv run python scripts/demo.py serve --offline # never connect to LangSmith, even with a key
    uv run python scripts/demo.py check           # seed a throwaway folder, verify every scene
    uv run python scripts/demo.py record          # author only: re-record demo/data from data/

`make demo` runs `serve` (and builds the review UI first if needed). See src/filings_agent/demo.py.

With a LangSmith key (LANGSMITH_API_KEY, in the environment or .env), the scenes' runs are traced
to the project "filings-agent-demo", monitored (online checks, reviewer feedback, the demo's
annotation queue), and every scene links to its own traces and your workspace pages. Without one,
everything still works offline and the scenes link to the public traces and datasets.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filings_agent import demo  # noqa: E402

EXPECTED = {  # scene -> the status its run must be in once seeded
    "fanout": "awaiting_review",
    "easy-lane": "published",
    "review": "awaiting_review",
    "research": "awaiting_review",
    "published": "published",
    "history": "published",
    "crash": "interrupted",
}


def check(state: Path) -> list[str]:
    """Seed every scene offline into `state` and say what's wrong with each (empty = all good)."""
    from filings_agent.graph.runner import get_status, shared_ledger

    cfg = demo.demo_config(state=state)
    with demo.offline():
        scenes = demo.seed(cfg)
        problems: list[str] = []
        for scene, want in EXPECTED.items():
            got = get_status(cfg, str(scenes[scene]["thread_id"]))
            if got["status"] != want:
                problems.append(f"{scene}: {got['status']}, expected {want}")
            if scene == "research" and len(got["awaiting_review"]) != 1:
                problems.append("research: expected the one researched card back")
        current = shared_ledger(cfg.state("ledger.sqlite")).current("GE:net_income:FY2022")
        if not current or current[0]["actor"] != "alice" or not current[0]["later_escalations"]:
            problems.append(f"history: expected Alice's pick, escalated later: {current}")
        if not (state / "replays" / "fanout.json").exists():
            problems.append("fanout: no recorded events for the replay")
    return problems


def langsmith_key() -> bool:
    """Only LangSmith settings are read from .env: the demo never uses AWS or the SEC contact."""
    import os

    from dotenv import dotenv_values

    for k, v in dotenv_values(ROOT / ".env").items():
        if k.startswith("LANGSMITH_") and k != "LANGSMITH_TRACING" and v and k not in os.environ:
            os.environ[k] = v
    return bool(os.environ.get("LANGSMITH_API_KEY"))


def serve(reset: bool, host: str, port: int, offline_only: bool) -> None:
    import uvicorn

    from filings_agent.api.app import WEB_DIST, create_app

    if not (WEB_DIST / "index.html").exists():
        sys.exit("The review UI isn't built: run `make web` first (or `make demo`, which does).")
    cfg = demo.demo_config()
    tracing = not offline_only and langsmith_key()
    mode = (
        f"LangSmith connected: traces in project {demo.DEMO_PROJECT}"
        if tracing
        else "LangSmith offline: scenes link to public traces"
    )
    with demo.offline(tracing=tracing):
        if reset or not demo.load_scenes():
            print(f"Seeding the demo scenes from demo/data ($0; {mode})...", flush=True)
            demo.seed(cfg)
        app = create_app(
            cfg, run_fn=demo.demo_run_fn(), demo_state=demo.STATE, monitor_fn=demo.demo_monitor()
        )
        print(f"Demo ready: http://{host}:{port}  ({mode})", flush=True)
        uvicorn.run(app, host=host, port=port, log_level="warning")


def record() -> None:
    """Run every scene on the full data/ with live Bedrock for anything not cached yet, then copy
    exactly what the scenes read into demo/data. Needs AWS_PROFILE; costs cents when cached."""
    from filings_agent.config import load_config

    rec = demo.Recorder()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(ROOT / "configs" / "default.yaml").model_copy(
            update={"state_dir": Path(tmp)}
        )
        with rec.watching():
            demo.seed(cfg)
        counts: dict[str, Any] = demo.export(rec, cfg.data_dir)
    print("recorded into demo/data:", counts)
    with tempfile.TemporaryDirectory() as tmp:
        problems = check(Path(tmp))
    print("offline check:", problems if problems else "OK")
    if problems:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="demo")
    sub = parser.add_subparsers(dest="cmd", required=True)
    srv = sub.add_parser("serve")
    srv.add_argument("--reset", action="store_true", help="rebuild every scene first")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--offline", action="store_true", help="never connect to LangSmith")
    sub.add_parser("check")
    sub.add_parser("record")
    args = parser.parse_args()
    if args.cmd == "serve":
        serve(args.reset, args.host, args.port, args.offline)
    elif args.cmd == "check":
        with tempfile.TemporaryDirectory() as tmp:
            problems = check(Path(tmp))
        print("demo check:", "OK, every scene is in its expected state" if not problems else "")
        for p in problems:
            print("  FAIL", p)
        sys.exit(1 if problems else 0)
    else:
        record()


if __name__ == "__main__":
    main()
