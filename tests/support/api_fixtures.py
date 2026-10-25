"""Real API responses for the review UI's tests: the real graph behind the real FastAPI app.

    uv run python -m tests.support.api_fixtures --write    # after changing what the API returns

The graph runs offline with the usual doubles, behind `create_app`; the fixtures are the bodies of
`GET /analyses/{id}` at each stage, exactly as a browser receives them. `tests/api/test_contract.py`
fails when these files are stale, and the web tests render them, so a renamed or removed field
breaks a test on one side or the other instead of the demo.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from filings_agent.api.app import create_app
from filings_agent.config import AppConfig, load_config
from filings_agent.graph.build import Deps, build_graph
from filings_agent.graph.runner import serializer, summarize
from filings_agent.steps.extract import Usage
from filings_agent.steps.memo import Memo, Row
from filings_agent.storage.ledger import Ledger
from tests.support.graph_doubles import METRICS, FakeReconciler, RestatingExtractor, TwoFilingStore
from tests.support.research_doubles import FakeResearcher

OUT = Path(__file__).resolve().parents[2] / "web" / "src" / "fixtures"
CHANGE = "AAA:revenue:FY2023"


class FixedMemo:
    def write(self, rows: list[Row], changes: list[str], rep: int = 0) -> Memo:
        md = "# AAA summary\n\nRevenue was recast after a spin-off.\n\n## Caveats\n\nFY2023 recast."
        return Memo(md, True, [], 1, Usage("memo", 10, 5, 0.001, 1.0, False))


def build() -> dict[str, Any]:
    """{"awaiting": ..., "researched": ..., "published": ...}: GET /analyses/{id} at each stage."""
    with tempfile.TemporaryDirectory() as tmp:
        conn = sqlite3.connect(str(Path(tmp) / "cp.sqlite"), check_same_thread=False)
        ledger = Ledger(Path(tmp) / "ledger.sqlite")
        try:
            graph = build_graph(SqliteSaver(conn, serde=serializer()))
            deps = Deps(
                TwoFilingStore(), RestatingExtractor(), METRICS, reconciler=FakeReconciler(),
                ledger=ledger, memo_writer=cast(Any, FixedMemo()), researcher=FakeResearcher(),
                max_research_rounds=2,
            )  # fmt: skip
            runs = 0

            def config(thread_id: str) -> RunnableConfig:
                return {"configurable": {"thread_id": thread_id}}

            def run_fn(
                cfg: AppConfig,
                tickers: list[str],
                model: str,
                prompt: str,
                thread_id: str,
                on_event: Any = None,
                decisions: dict[str, Any] | None = None,
                rep: int = 0,
                resume: bool = False,
                lease_owner: str | None = None,
                run_id: str | None = None,
            ) -> dict[str, Any]:
                nonlocal runs
                runs += 1
                inp: Any = Command(resume=decisions) if decisions else {"tickers": tickers}
                graph.invoke(inp, config(thread_id), context=deps)  # pyright: ignore[reportUnknownMemberType]
                return {**status_fn(cfg, thread_id), "run_id": f"run-{runs}"}

            def status_fn(cfg: AppConfig, thread_id: str) -> dict[str, Any]:
                return summarize(thread_id, graph.get_state(config(thread_id)))

            cfg = load_config().model_copy(update={"data_dir": Path(tmp)})
            # one event loop for the whole flow, like the real server (see tests/api/test_api.py)
            client = TestClient(create_app(cfg, run_fn, status_fn, web_dist=None)).__enter__()
            tid = client.post(
                "/analyses", json={"tickers": ["AAA"]}, headers={"Idempotency-Key": "fixture"}
            ).json()["thread_id"]

            def settle(status: str) -> dict[str, Any]:
                for _ in range(500):
                    body: dict[str, Any] = client.get(f"/analyses/{tid}").json()
                    if body["status"] == status:
                        return body
                    time.sleep(0.01)
                raise AssertionError(f"never reached {status}")

            out = {"awaiting": settle("awaiting_review")}
            decide = {CHANGE: {"action": "research", "note": "check the 8-K"}}
            token = out["awaiting"]["review_token"]
            client.post(
                f"/analyses/{tid}/review", json={"decisions": decide, "review_token": token}
            )
            body: dict[str, Any] = {}
            for _ in range(500):  # wait for the research round to finish (one card again)
                body = client.get(f"/analyses/{tid}").json()
                if body["status"] == "awaiting_review" and body["research_rounds"] == 1:
                    break
                time.sleep(0.01)
            out["researched"] = body
            decide = {
                CHANGE: {
                    "action": "accept",
                    "correct_cause": "discontinued_operations",
                    "note": "spin-off recast",
                }
            }
            token = body["review_token"]
            client.post(
                f"/analyses/{tid}/review", json={"decisions": decide, "review_token": token}
            )
            out["published"] = settle("published")
            out["published"]["publication"]["published_at"] = "2026-10-23T12:00:00+00:00"
            for d in out["published"]["decisions"].values():  # stamped by the server: fix the time
                d["decided_at"] = "2026-10-23T12:00:00+00:00"
            # Trace ids and start times are chosen by the server per invocation: make them stable.
            for body in out.values():
                for i, r in enumerate(body["runs"]):
                    r["run_id"], r["started_at"] = f"run-{i + 1}", "2026-10-23T12:00:00+00:00"
                body["last_run_id"] = body["runs"][-1]["run_id"] if body["runs"] else None
            return out
        finally:
            conn.close()
            ledger.close()


def render(name: str, value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def main() -> None:
    fixtures = build()
    if "--write" in sys.argv:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, value in fixtures.items():
            (OUT / f"analysis-{name}.json").write_text(render(name, value))
        print(f"wrote {len(fixtures)} fixtures to {OUT}")
    else:
        print(json.dumps(fixtures, indent=2)[:2000])


if __name__ == "__main__":
    main()
