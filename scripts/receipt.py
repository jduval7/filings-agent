"""The docs' numbers, recomputed. Every headline figure in README.md and EVALS.md is rebuilt from
the LLM cache and checked against the exact sentence that states it.

    make receipt          # $0: paid calls are blocked; a cache miss is reported, never paid for

Why: several wrong claims reached the docs by hand (a dataset size, the spend caps, "GE has no
splits"). Each claim below pairs a doc with a sentence generated from the recomputed value, so a
number that drifts, or a sentence edited away from the number, fails the receipt.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ["LANGSMITH_TRACING"] = "false"  # recomputing must not create traces

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
os.environ["LANGSMITH_TRACING"] = "false"

from filings_agent.llm import spend_guard  # noqa: E402


def _refuse(self: Any, model_id: str, payload: Any, max_tokens: int | None) -> Any:
    raise spend_guard.SpendLimitExceeded(f"receipt: cache miss for {model_id}; nothing was paid")


spend_guard.SpendGuard.reserve = _refuse  # type: ignore[method-assign]

from filings_agent.config import load_config  # noqa: E402

MODEL_EXTRACT, MODEL_RECONCILE = "gpt-oss-120b", "claude-sonnet-4.6"


def compute() -> dict[str, Any]:
    from filings_agent.evals.agent_eval import score_agent
    from filings_agent.evals.local import score_model
    from filings_agent.evals.reconcile_local import (
        card_eval,
        load_tasks,
        load_verification,
        score_reconciler,
    )
    from filings_agent.evals.research_experiments import hard_cases, make_target, score
    from filings_agent.graph.runner import make_researcher

    cfg = load_config()
    m: dict[str, Any] = {}
    # Fresh labels the company's own adoption table ties out (the rest are narrative-only).
    tied = {k for k, v in load_verification().items() if v == "tie_out"}

    test = score_model(cfg, MODEL_EXTRACT, "v2", "test", 1)[0]
    m["extract_test_f1"] = test.metrics["f1"]

    for split in ("dev", "test", "fresh"):
        r = score_reconciler(cfg, MODEL_RECONCILE, "v2", split, 1)
        m[f"pipe_{split}_n"] = r["n"]
        m[f"pipe_{split}_right"] = round(r["cause_accuracy"] * r["n"])
        m[f"pipe_{split}_auto_pct"] = round(100 * r["auto_rate"])
        m[f"pipe_{split}_auto_wrong"] = len(r["auto_wrong"])
        m[f"pipe_{split}_auto_n"] = r["auto_n"]
        if split == "fresh":
            ids = [t.task.discrepancy_id for t in load_tasks(cfg, split)]
            m["pipe_fresh_tied_n"] = sum(i in tied for i in ids)
            m["pipe_fresh_tied_right"] = sum(i in tied and i not in r["mistakes"] for i in ids)

    for split in ("dev", "fresh"):
        s, cases = score_agent(cfg, MODEL_RECONCILE, "v1", split, "vector", injections=False)
        m[f"agent_{split}_acc"] = s["cause_accuracy"]
        wrong = [c for c in cases if not c.correct]
        m[f"agent_{split}_confident_wrong"] = sum(c.cause.value != "unknown" for c in wrong)
        m[f"agent_{split}_anchor_hit"] = s["retrieval_hit_rate"]
        if split == "fresh":
            real = [c for c in cases if c.case_id in tied]
            m["agent_fresh_tied_n"] = len(real)
            m["agent_fresh_tied_right"] = sum(c.correct for c in real)

    hidden = 0
    total = 0
    fits: dict[str, int] = {}
    for split in ("dev", "test"):
        e = card_eval(cfg, split)
        hidden += len(e["true_cause_hidden"])
        total += e["n"]
        for k, v in e["candidate_fits"].items():
            fits[k] = fits.get(k, 0) + v
    m["cards_n"], m["cards_hidden"] = total, hidden
    shown = sum(fits.values())
    m["cards_ruled_out_pct"] = round(100 * fits.get("ruled_out", 0) / shown)
    m["cards_no_evidence_pct"] = round(100 * fits.get("no_evidence", 0) / shown)
    m["cards_open_pct"] = 100 - m["cards_ruled_out_pct"] - m["cards_no_evidence_pct"]
    fresh = card_eval(cfg, "fresh")
    m["cards_fresh_n"], m["cards_fresh_hidden"] = fresh["n"], len(fresh["true_cause_hidden"])

    target = make_target(make_researcher(cfg))
    for which in ("hard", "fresh"):
        picked = hard_cases(cfg, which)
        rows = [
            score(
                target({"discrepancy": t.task.discrepancy.model_dump(mode="json"),
                        "evidence": list(t.task.evidence)}),
                t.label.value,
            )
            for t in picked
        ]  # fmt: skip
        if which == "fresh":
            pairs = [
                (t, r) for t, r in zip(picked, rows, strict=True) if t.task.discrepancy_id in tied
            ]
            m["research_fresh_tied_n"] = len(pairs)
            m["research_fresh_tied_supported"] = int(
                sum(r["true_cause_supported"] for _, r in pairs)
            )
        m[f"research_{which}_n"] = len(rows)
        m[f"research_{which}_supported"] = int(sum(r["true_cause_supported"] for r in rows))
        m[f"research_{which}_contradicted"] = int(sum(r["true_cause_contradicted"] for r in rows))
        # the number that discriminates: research supports wrong causes too (it's asked to look)
        m[f"research_{which}_wrong_supported"] = round(
            sum(r["wrong_causes_supported"] for r in rows) / max(len(rows), 1), 1
        )
    m["research_total_n"] = m["research_hard_n"] + m["research_fresh_n"]
    m["pipe_auto_wrong"] = m["pipe_dev_auto_wrong"] + m["pipe_test_auto_wrong"]
    m["pipe_auto_n"] = m["pipe_dev_auto_n"] + m["pipe_test_auto_n"]
    m["agent_fresh_cw"] = m["agent_fresh_confident_wrong"]
    m["research_total_supported"] = m["research_hard_supported"] + m["research_fresh_supported"]

    m["labelled_total"] = sum(len(load_tasks(cfg, s)) for s in ("dev", "test", "fresh")) - sum(
        1 for t in load_tasks(cfg, "dev") if t.task.discrepancy_id.endswith("#corr")
    )

    out = subprocess.run(
        ["uv", "run", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True
    ).stdout
    found = re.search(r"(\d+) tests? collected", out)
    m["backend_tests"] = int(found.group(1)) if found else -1
    return m


# (doc, sentence built from the recomputed values). The sentence must appear in the doc verbatim.
CLAIMS: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
    ("README.md", lambda m: f"pyright strict, {m['backend_tests']} offline tests"),
    ("README.md", lambda m: f"pyright strict + {m['backend_tests']} tests (no network, no LLM)"),
    ("README.md", lambda m: f"F1 **{m['extract_test_f1']:.3f}** on held-out test companies"),
    (
        "README.md",
        lambda m: (
            f"**{m['pipe_test_right']}/{m['pipe_test_n']}** test, "
            f"{m['pipe_dev_right']}/{m['pipe_dev_n']} dev"
        ),
    ),
    (
        "README.md",
        lambda m: (
            f"{m['pipe_dev_auto_pct']}% dev / {m['pipe_test_auto_pct']}% test of "
            f"material changes skip review, **{m['pipe_auto_wrong']} wrong**"
        ),
    ),
    (
        "README.md",
        lambda m: f"true cause never ruled out ({m['cards_n'] - m['cards_hidden']}/{m['cards_n']})",
    ),
    (
        "README.md",
        lambda m: (
            f"true cause gets verified evidence on **{m['research_total_supported']}/"
            f"{m['research_total_n']}** hard cases"
        ),
    ),
    (
        "README.md",
        lambda m: (
            f"pipeline **{m['pipe_fresh_right']}/{m['pipe_fresh_n']}**, agent "
            f"{m['agent_fresh_confident_wrong']} confident-wrong"
        ),
    ),
    ("README.md", lambda m: f"samples are small ({m['labelled_total']} labelled"),
    (
        "EVALS.md",
        lambda m: (
            f"pipeline **{m['pipe_fresh_right']}/{m['pipe_fresh_n']}**; agent "
            f"{m['agent_fresh_acc']:.2f} with **{m['agent_fresh_cw']} confident-wrong**"
        ),
    ),
    (
        "EVALS.md",
        lambda m: (
            f"true cause never ruled out ({m['cards_n'] - m['cards_hidden']}/{m['cards_n']} cases)"
        ),
    ),
    (
        "EVALS.md",
        lambda m: (
            f"research supports the true cause **{m['research_fresh_supported']}/"
            f"{m['research_fresh_n']}**"
        ),
    ),
    (
        "EVALS.md",
        lambda m: (
            f"pipeline {m['pipe_dev_right']}/{m['pipe_dev_n']} dev, "
            f"{m['pipe_test_right']}/{m['pipe_test_n']} test, "
            f"{m['pipe_fresh_right']}/{m['pipe_fresh_n']} fresh"
        ),
    ),
    ("EVALS.md", lambda m: f"{m['labelled_total']} labelled restatements"),
    (
        "EVALS.md",
        lambda m: f"**{m['agent_dev_acc']:.2f}**, anchor hit {m['agent_dev_anchor_hit']:.2f}",
    ),
    (
        "README.md",
        lambda m: (
            f"{m['cards_ruled_out_pct']}% ruled out by code, {m['cards_no_evidence_pct']}% no "
            f"evidence, {m['cards_open_pct']}% left open"
        ),
    ),
    (
        "EVALS.md",
        lambda m: (
            f"{m['cards_ruled_out_pct']}% ruled out by code, {m['cards_no_evidence_pct']}% no "
            f"evidence, {m['cards_open_pct']}% left open"
        ),
    ),
    (
        "README.md",
        lambda m: (
            f"{m['research_hard_wrong_supported']} per hard case, "
            f"{m['research_fresh_wrong_supported']} per fresh case"
        ),
    ),
    ("README.md", lambda m: f"0 of {m['pipe_auto_n']} auto-approved"),
    (
        "README.md",
        lambda m: (
            f"On the {m['pipe_fresh_tied_n']} that add up: pipeline "
            f"{m['pipe_fresh_tied_right']}/{m['pipe_fresh_tied_n']}, agent "
            f"{m['agent_fresh_tied_right']}/{m['agent_fresh_tied_n']}, research "
            f"{m['research_fresh_tied_supported']}/{m['research_fresh_tied_n']}"
        ),
    ),
    (
        "EVALS.md",
        lambda m: (
            f"On the {m['pipe_fresh_tied_n']} labels that tie out: pipeline "
            f"{m['pipe_fresh_tied_right']}/{m['pipe_fresh_tied_n']}, agent "
            f"{m['agent_fresh_tied_right']}/{m['agent_fresh_tied_n']}, research "
            f"{m['research_fresh_tied_supported']}/{m['research_fresh_tied_n']}"
        ),
    ),
    (
        "EVALS.md",
        lambda m: f"contradicts it {m['research_fresh_contradicted']}/{m['research_fresh_n']}",
    ),
]


def main() -> int:
    if "--recompute" in sys.argv or not (ROOT / "data" / "receipt.json").exists():
        m = compute()
        (ROOT / "data" / "receipt.json").write_text(json.dumps(m, indent=2))
    else:
        m = json.loads((ROOT / "data" / "receipt.json").read_text())
    print("recomputed:", json.dumps(m, indent=2))
    bad = 0
    for doc, sentence in CLAIMS:
        text = re.sub(r"\s+", " ", (ROOT / doc).read_text())
        want = sentence(m)
        ok = want in text
        bad += not ok
        print(f"  {'OK  ' if ok else 'MISS'}  {doc}: {want}")
    print(
        "\nRECEIPT OK"
        if not bad
        else f"\nRECEIPT FAILED: {bad} claim(s) differ from the computed numbers"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
