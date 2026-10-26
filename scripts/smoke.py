"""End-to-end smoke test: the real server, real graph and real (cached) models, driven over HTTP the
way the review UI drives it. Every bug found by hand at the seams (statuses, event streams, feedback
values, fresh environments) would fail here.

    make smoke                      # ~1 minute, ~$0 once the cache is warm (AWS_PROFILE needed)
    uv run python scripts/smoke.py --langsmith   # also check reviewer feedback reached LangSmith

Starts its own server on a spare port, uses fresh Idempotency-Keys (new threads every run), and
checks spend before and after. Exit code 1 on any failed check.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
MAX_COST_USD = 0.10  # cached runs cost ~$0; a fresh memo is a cent or two
FAILURES: list[str] = []


from filings_agent.preflight import friendly  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def spare_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def spend() -> float:
    out = subprocess.run(
        ["uv", "run", "filings-agent", "spend"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return float(json.loads(out[out.index("{") :])["project_usd"])


def events(client: httpx.Client, tid: str) -> list[dict[str, Any]]:
    """Follow the SSE stream until the run ends; return the events."""
    seen: list[dict[str, Any]] = []
    with client.stream("GET", f"/analyses/{tid}/events", timeout=600) as resp:
        for line in resp.iter_lines():
            if line.startswith("data: "):
                seen.append(json.loads(line[6:]))
                if seen[-1].get("event") in {"end", "status"}:
                    break
    return seen


def settle(
    client: httpx.Client, tid: str, until: Callable[[dict[str, Any]], bool]
) -> dict[str, Any]:
    for _ in range(600):
        body: dict[str, Any] = client.get(f"/analyses/{tid}").json()
        if until(body):
            return body
        time.sleep(0.5)
    raise SystemExit(f"timed out waiting on {tid}")


def start(client: httpx.Client, key: str, tickers: list[str]) -> dict[str, Any]:
    resp = client.post("/analyses", json={"tickers": tickers}, headers={"Idempotency-Key": key})
    return dict(resp.json())


@friendly("make smoke")
def main() -> int:
    port = spare_port()
    # Its runs go in a throwaway folder, not the demo's run list (filings and LLM cache are shared).
    state = tempfile.mkdtemp(prefix="filings-smoke-")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "FILINGS_STATE_DIR": state}
    server = subprocess.Popen(
        ["uv", "run", "filings-agent", "serve", "--port", str(port)],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    stamp = time.strftime("%Y%m%d%H%M%S")
    before = spend()
    try:
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60)
        for _ in range(60):
            try:
                if client.get("/health").status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.5)
        print(f"smoke against :{port}")

        print("server")
        check("health", client.get("/health").json() == {"status": "ok"})
        check("review UI served at /", "Restatement Review" in client.get("/").text,
              "build it with `make web`")  # fmt: skip

        print("NVDA: the easy lane")
        nvda = start(client, f"smoke-{stamp}-nvda", ["NVDA"])
        ev = events(client, nvda["thread_id"])
        names = [e.get("event") for e in ev]
        check("progress events stream", "planned" in names and "branch_done" in names)
        n = settle(client, nvda["thread_id"], lambda b: b["status"] not in {"running", "resuming"})
        auto = [k for k, r in n["resolutions"].items() if r.get("lane") == "auto"]
        check("published without review", n["status"] == "published" and not n["awaiting_review"],
              n["status"])  # fmt: skip
        check("both splits confirmed by code", len(auto) == 2, str(auto))
        check("trace id recorded", bool(n.get("last_run_id")))

        print("idempotency")
        again = start(client, f"smoke-{stamp}-nvda", ["NVDA"])
        check("same key + same body = same run", again["thread_id"] == nvda["thread_id"]
              and again["created"] is False)  # fmt: skip
        conflict = client.post("/analyses", json={"tickers": ["KO"]},
                               headers={"Idempotency-Key": f"smoke-{stamp}-nvda"})  # fmt: skip
        check("same key + different body = 422", conflict.status_code == 422)

        print("GE: every change needs a human")
        ge = start(client, f"smoke-{stamp}-ge", ["GE"])
        tid = ge["thread_id"]
        events(client, tid)
        g = settle(client, tid, lambda b: b["status"] == "awaiting_review")
        cards = {c["id"]: c for c in g["awaiting_review"]}
        check("10 cards", len(cards) == 10, str(len(cards)))
        answer_keys = ("cause", "explanation", "evidence_quote", "lane")
        leaked = [k for c in cards.values() for k in answer_keys if k in c]
        check("no card shows the agent's answer", not leaked, str(leaked))
        ni = cards.get("GE:net_income:FY2022", {})
        fits = {c["cause"]: c for c in ni.get("candidates", [])}
        disc = fits.get("discontinued_operations", {})
        cited = disc.get("fit") == "ruled_out" and "ASC 205-20-45-3A" in disc.get("why", "")
        check("spin-off ruled out for net income, with its citation", cited)

        print("review: refusals")
        token = g.get("review_token")
        no_token = client.post(f"/analyses/{tid}/review", json={"decisions": {
            k: {"action": "escalate"} for k in cards}})  # fmt: skip
        check("review without the cards' token rejected", no_token.status_code == 422)
        partial = client.post(f"/analyses/{tid}/review", json={"decisions": {
            "GE:net_income:FY2022": {"action": "escalate"}}, "review_token": token})  # fmt: skip
        check("partial review rejected", partial.status_code == 422)
        unknown = dict.fromkeys(cards, {"action": "escalate"}) | {
            "NOPE:x:FY1": {"action": "escalate"}
        }
        resp = client.post(f"/analyses/{tid}/review",
                           json={"decisions": unknown, "review_token": token})  # fmt: skip
        check("unknown id rejected", resp.status_code == 422)

        print("review: every kind of decision")
        decisions: dict[str, Any] = {
            k: {"action": "accept", "correct_cause": "discontinued_operations"} for k in cards
        }
        decisions["GE:net_income:FY2021"] = {
            "action": "accept",
            "correct_cause": "accounting_change",
        }
        decisions["GE:net_income:FY2022"] = {"action": "research"}
        decisions["GE:revenue:FY2021"] = {
            "action": "use_original",
            "correct_cause": "discontinued_operations",
        }
        decisions["GE:long_term_debt:FY2023"] = {"action": "escalate", "note": "smoke: none fit"}
        decisions["GE:operating_cash_flow:FY2021"] = {
            "action": "escalate",
            "note": "smoke: none fit",
        }
        stale = client.post(f"/analyses/{tid}/review",
                            json={"decisions": decisions, "review_token": "stale"})  # fmt: skip
        check("a review of cards that changed since is refused", stale.status_code == 409)
        no_cause = dict(decisions) | {"GE:revenue:FY2021": {"action": "accept"}}
        resp = client.post(f"/analyses/{tid}/review",
                           json={"decisions": no_cause, "review_token": token})  # fmt: skip
        check("accept without a cause rejected", resp.status_code == 422)
        resp = client.post(f"/analyses/{tid}/review",
                           json={"decisions": decisions, "review_token": token})  # fmt: skip
        check("review accepted", resp.status_code == 202, resp.text)
        busy = client.post(f"/analyses/{tid}/review",
                           json={"decisions": decisions, "review_token": token})  # fmt: skip
        check("a second review while running is refused", busy.status_code == 409)
        ev = events(client, tid)
        names = [e.get("event") for e in ev]
        check("resumed stream has exactly one end", names.count("end") == 1, str(names))
        check("research ran", "researched" in names, str(names))
        r = settle(client, tid, lambda b: b["status"] == "awaiting_review")
        back = r["awaiting_review"]
        check(
            "only the researched card comes back",
            [c["id"] for c in back] == ["GE:net_income:FY2022"],
        )
        rounds = back[0].get("research_rounds", []) if back else []
        found = bool(rounds) and rounds[0]["findings"] > 0
        check(
            "with verified findings and a round left",
            found and back[0]["research_rounds_left"] == 1,
        )

        print("publish")
        final = {"GE:net_income:FY2022": {"action": "accept", "correct_cause": "accounting_change",
                                         "note": "smoke: ASU 2018-12"}}  # fmt: skip
        body = {"decisions": final, "review_token": r["review_token"]}
        resp = client.post(f"/analyses/{tid}/review", json=body)
        check("final decision accepted", resp.status_code == 202, resp.text)
        events(client, tid)
        p = settle(client, tid, lambda b: b["status"] not in {"running", "resuming"})
        d = p.get("decisions", {})
        once = bool(p.get("publication")) and p["publication"]["newly_published"]
        check("published once", p["status"] == "published" and once)
        check("all 10 decisions recorded", len(d) == 10, str(len(d)))
        check("two escalations", sum(x["action"] == "escalate" for x in d.values()) == 2)
        check(
            "keep-original recorded", d.get("GE:revenue:FY2021", {}).get("action") == "use_original"
        )
        check("every decision names its reviewer", all(x.get("actor") for x in d.values()))
        check("memo verified", bool(p.get("memo") and p["memo"]["verified"]))
        replay = start(client, f"smoke-{stamp}-ge", ["GE"])
        check("replaying the request starts nothing", replay["created"] is False
              and replay["status"] == "published")  # fmt: skip
        recent = {r["thread_id"] for r in client.get("/analyses").json()}
        check("both runs listed", {nvda["thread_id"], tid} <= recent)

        if "--langsmith" in sys.argv:
            print("LangSmith")
            from dotenv import load_dotenv
            from langsmith import Client

            load_dotenv(ROOT / ".env")
            run_id = p.get("last_run_id")
            got: dict[str, float] = {}
            for _ in range(30):  # monitoring posts in the background after the run finishes
                got = {f.key: float(f.score or 0) for f in Client().list_feedback(run_ids=[run_id])
                       if f.key.startswith("review.")}  # fmt: skip
                if "review.decisions" in got:
                    break
                time.sleep(2)
            check("reviewer feedback on the trace", got.get("review.decisions") == 10.0
                  and got.get("review.escalated") == 2.0
                  and got.get("review.research_rounds") == 1.0, str(got))  # fmt: skip

        cost = spend() - before
        print("cost")
        check(f"spent ${cost:.4f} (cached runs)", cost <= MAX_COST_USD, f"limit ${MAX_COST_USD}")
    finally:
        server.terminate()
        server.wait(timeout=20)

    print(f"\n{'SMOKE PASSED' if not FAILURES else 'SMOKE FAILED: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
