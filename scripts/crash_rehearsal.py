"""Crash rehearsal: kill the API mid-fan-out, restart it, retry the same request.

Expected: the retry (same Idempotency-Key) resumes the run; branches finished before the kill are
served from the LLM cache, so nothing is paid twice; the result is complete and published once.

    AWS_PROFILE=... uv run python scripts/crash_rehearsal.py [TICKER]

Uses `fresh: true`, so it makes real (cheap) LLM calls: about $0.02 per company on gpt-oss-120b.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"
ROOT = Path(__file__).resolve().parent.parent
STATE = tempfile.mkdtemp(prefix="filings-crash-")


def start_server() -> subprocess.Popen[bytes]:
    proc = subprocess.Popen(
        ["uv", "run", "filings-agent", "serve", "--port", str(PORT)],
        cwd=ROOT,
        # One throwaway folder for both server starts (the restart must find the run), never the
        # demo's run list. Filings and the LLM cache are shared, so the rehearsal stays $0.
        env={**os.environ, "FILINGS_STATE_DIR": STATE},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # so we can kill the whole process group
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{BASE}/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise SystemExit("server didn't start")


def watch(thread_id: str, stop_after: int | None = None) -> int:
    """Count finished branches from the SSE stream; stop early after `stop_after`."""
    done = 0
    with httpx.stream("GET", f"{BASE}/analyses/{thread_id}/events", timeout=300) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event.get("event") == "planned":
                print(f"  planned branches: {event['branches']}")
            elif event.get("event") == "branch_done":
                done += 1
            elif event.get("event") == "end":
                print(f"  run ended: {event.get('status')}")
                return done
            if stop_after is not None and done >= stop_after:
                return done
    return done


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "KO"
    body = {"tickers": [ticker], "fresh": True}
    headers = {"Idempotency-Key": f"crash-rehearsal-{ticker}-{int(time.time())}"}

    server = start_server()
    created = httpx.post(f"{BASE}/analyses", json=body, headers=headers).json()
    thread_id = created["thread_id"]
    print(f"run started: {thread_id}")
    finished = watch(thread_id, stop_after=4)
    os.killpg(server.pid, signal.SIGKILL)
    server.wait()
    print(f"  SIGKILL after {finished} branches finished")

    server = start_server()
    retry = httpx.post(f"{BASE}/analyses", json=body, headers=headers).json()
    print(f"client retry with the same key: status={retry['status']} created={retry['created']}")
    rerun = watch(thread_id)
    status = httpx.get(f"{BASE}/analyses/{thread_id}").json()
    cached = status["cached_branches"]
    print(
        f"  nodes re-run: {rerun}; served from LLM cache: {cached} (paid twice: 0); "
        f"new LLM calls: {status['branches'] - cached}; facts: {status['facts']}; "
        f"errors: {len(status['errors'])}; status: {status['status']}"
    )
    os.killpg(server.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
