"""Guard against LangSmith endpoints scheduled for removal (Jan 31, 2027).

`Client.read_run` (GET /runs/{id}) and `Client.list_runs` (POST /runs/query) are deprecated in
favour of `client.runs.retrieve()` / `client.runs.query()`. Feedback must carry `session_id`."""

from __future__ import annotations

import re
from pathlib import Path

DEPRECATED = re.compile(r"\.(read_run|list_runs)\(")


def test_no_deprecated_langsmith_run_endpoints_in_source() -> None:
    offenders = [
        f"{path}:{i}"
        for root in ("src", "scripts")
        for path in Path(root).rglob("*.py")
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if DEPRECATED.search(line)
    ]
    assert offenders == [], f"use client.runs.retrieve()/query() instead: {offenders}"


def test_feedback_always_carries_session_id() -> None:
    src = Path("src/filings_agent/evals/corrections.py").read_text()
    call = src[src.index("client.create_feedback(") :]
    assert "session_id=session_id" in call[: call.index(")\n")]


def test_tests_never_send_traces() -> None:
    """Loading .env (the Studio entry point and the CLI do) must not switch tracing on in tests."""
    import os

    from filings_agent.graph import studio  # its import runs load_dotenv()

    assert studio.graph is not None
    assert os.environ["LANGSMITH_TRACING"] == "false"
