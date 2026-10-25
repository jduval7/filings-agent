"""Local scoring loop for the reconcile step.

Inputs are answer-key discrepancies (not extracted ones), so this step is tested even if the
extractor is bad. Evidence comes from the newest filing, exactly as in the graph.
Only discrepancies with a verified label count toward cause accuracy.
"""

from __future__ import annotations

import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import yaml

from filings_agent.config import AppConfig
from filings_agent.datasets import ReconcileExample
from filings_agent.domain import Discrepancy, fiscal_year_of
from filings_agent.graph.build import Reconciler
from filings_agent.steps.reconcile import (
    Cause,
    ReconcileTask,
    Resolution,
    evidence_for,
)
from filings_agent.steps.triage import CheckStatus, triage


@dataclass(frozen=True)
class LabeledTask:
    task: ReconcileTask
    label: Cause


LABELS = Path("configs/reconcile_labels.yaml")
# Reviewer corrections, exported from production. Committed (data/ isn't), so CI's gate has them.
CORRECTIONS = Path("configs/reconcile_corrections.jsonl")


def load_labels(path: Path = LABELS) -> dict[str, Cause | None]:
    raw = cast(dict[str, dict[str, Any]], yaml.safe_load(path.read_text()) or {})
    return {k: (Cause(v["cause"]) if v.get("cause") else None) for k, v in raw.items()}


def load_verification(path: Path = LABELS) -> dict[str, str]:
    """How each label was checked, where recorded: "tie_out" (the company's own adoption table adds
    up for this figure) or "narrative" (the filing names the cause but ties out no numbers). Used
    to report results on all labels and on the strongest ones alone."""
    raw = cast(dict[str, dict[str, Any]], yaml.safe_load(path.read_text()) or {})
    return {k: str(v["verified_by"]) for k, v in raw.items() if v and v.get("verified_by")}


def load_tasks(cfg: AppConfig, split: str, labels_path: Path = LABELS) -> list[LabeledTask]:
    labels = load_labels(labels_path)
    rows = (cfg.data_dir / "datasets" / "reconcile.jsonl").read_text().splitlines()
    out: list[LabeledTask] = []
    md: dict[str, str] = {}
    for ex in (ReconcileExample.model_validate_json(r) for r in rows):
        if ex.split != split or not ex.material:
            continue
        d = Discrepancy(
            ticker=ex.ticker,
            metric=ex.metric,
            fiscal_year=fiscal_year_of(ex.period_end),
            period_end=ex.period_end,
            values=tuple((a, v) for a, _, v in ex.values),
        )
        label = labels.get(d.id)
        paths = [cfg.data_dir / "filings" / f"{a}.md" for a, _ in d.values]
        if label is None or not all(p.exists() for p in paths):
            continue  # unlabelled, or filings not downloaded yet

        def markdown_of(accession_no: str) -> str:
            if accession_no not in md:
                md[accession_no] = (cfg.data_dir / "filings" / f"{accession_no}.md").read_text()
            return md[accession_no]

        out.append(LabeledTask(ReconcileTask(d.id, d, evidence_for(d, markdown_of)), label))
    if split == "dev":  # production corrections are tuning data: they never touch test
        out.extend(load_corrections(CORRECTIONS))
    return out


def load_corrections(path: Path) -> list[LabeledTask]:
    """Reviewer-corrected cases, replayed with the exact input the model saw in production."""
    if not path.exists():
        return []
    out: list[LabeledTask] = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        d = Discrepancy.model_validate(row["inputs"]["discrepancy"])
        evidence = tuple(row["inputs"]["evidence"])
        out.append(
            LabeledTask(ReconcileTask(f"{d.id}#corr", d, evidence), Cause(row["outputs"]["cause"]))
        )
    return out


def lane_summary(cases: list[tuple[LabeledTask, Resolution]]) -> dict[str, Any]:
    """Per-lane scores. The easy lane's wrong answers are the "confident and wrong" ones:
    nobody reviews them, so `auto_wrong` is the number auto-approval rests on. `caught` = wrong
    answers a rule contradicted (sent to review for the right reason, not just by default)."""
    auto: list[bool] = []
    review: list[bool] = []
    auto_wrong: list[str] = []
    caught: list[str] = []
    for t, r in cases:
        tri = triage(t.task.discrepancy, r)
        ok = r.cause is t.label
        if tri.lane == "auto":
            auto.append(ok)
            if not ok:
                auto_wrong.append(t.task.discrepancy_id)
        else:
            review.append(ok)
            # the quote check predates triage; only the new link rules count as a catch
            if not ok and any(
                c.status is CheckStatus.CONTRADICTED and c.name != "quote" for c in tri.checks
            ):
                caught.append(t.task.discrepancy_id)
    n = len(cases)
    return {
        "auto_n": len(auto),
        "auto_rate": len(auto) / n if n else 0.0,
        "auto_accuracy": statistics.fmean(auto) if auto else 0.0,
        "review_accuracy": statistics.fmean(review) if review else 0.0,
        "auto_wrong": auto_wrong,
        "caught_by_rule": caught,
    }


def _run(rec: Reconciler, rep: int, t: LabeledTask) -> Resolution:
    return rec.reconcile(t.task, rep=rep)


def score_reconciler(
    cfg: AppConfig,
    model: str,
    prompt: str,
    split: str,
    reps: int,
    concurrency: int = 6,
    strategy: str = "pipeline",
) -> dict[str, Any]:
    """`prompt` is the prompt version (pipeline) or the agent version (agent)."""
    from filings_agent.reconcile_agent.factory import make_reconciler

    tasks = load_tasks(cfg, split)
    rec = make_reconciler(cfg, model, strategy, prompt)
    accs: list[float] = []
    verified: list[float] = []
    unknown: list[float] = []
    costs: list[float] = []
    lat: list[float] = []
    mistakes: dict[str, str] = {}
    lanes: list[dict[str, Any]] = []
    for rep in range(reps):
        with ThreadPoolExecutor(concurrency) as pool:
            results = list(pool.map(partial(_run, rec, rep), tasks))
        correct = [r.cause is t.label for t, r in zip(tasks, results, strict=True)]
        lanes.append(lane_summary(list(zip(tasks, results, strict=True))))
        accs.append(sum(correct) / len(tasks) if tasks else 0.0)
        verified.append(sum(r.verified for r in results) / len(tasks) if tasks else 0.0)
        unknown.append(
            sum(r.cause is Cause.UNKNOWN for r in results) / len(tasks) if tasks else 0.0
        )
        usages = [r.usage for r in results if r.usage]
        costs.append(sum(u.cost_usd for u in usages))
        lat.extend(u.latency_s for u in usages)
        for t, r, ok in zip(tasks, results, correct, strict=True):
            if not ok:
                mistakes[t.task.discrepancy_id] = f"{r.cause.value} (label {t.label.value})"
    return {
        "model": model,
        "strategy": strategy,
        "prompt": prompt,
        "split": split,
        "n": len(tasks),
        "reps": reps,
        "cause_accuracy": statistics.fmean(accs) if accs else 0.0,
        "cause_accuracy_sd": statistics.stdev(accs) if len(accs) > 1 else 0.0,
        "verified_rate": statistics.fmean(verified) if verified else 0.0,
        "unknown_rate": statistics.fmean(unknown) if unknown else 0.0,
        "cost_per_run_usd": statistics.fmean(costs) if costs else 0.0,
        "p95_s": sorted(lat)[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0,
        "mistakes": mistakes,
        "auto_n": max((int(x["auto_n"]) for x in lanes), default=0),
        "auto_rate": statistics.fmean(x["auto_rate"] for x in lanes) if lanes else 0.0,
        "auto_accuracy": statistics.fmean(x["auto_accuracy"] for x in lanes) if lanes else 0.0,
        "review_accuracy": statistics.fmean(x["review_accuracy"] for x in lanes) if lanes else 0.0,
        "auto_wrong": sorted({c for x in lanes for c in x["auto_wrong"]}),
        "caught_by_rule": sorted({c for x in lanes for c in x["caught_by_rule"]}),
    }


def reconcile_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| model | prompt | n | reps | cause acc (±sd) | quote verified | unknown | easy lane "
        "| easy-lane wrong | $/run | p95 s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (-r["cause_accuracy"], r["cost_per_run_usd"])):
        lines.append(
            f"| {r['model']} | {r['prompt']} | {r['n']} | {r['reps']} "
            f"| {r['cause_accuracy']:.2f} ±{r['cause_accuracy_sd']:.2f} | {r['verified_rate']:.2f} "
            f"| {r['unknown_rate']:.2f} | {r['auto_rate']:.2f} | {len(r['auto_wrong'])} "
            f"| {r['cost_per_run_usd']:.4f} | {r['p95_s']:.1f} |"
        )
    return "\n".join(lines)


def card_eval(cfg: AppConfig, split: str) -> dict[str, Any]:
    """Review cards against the labels. No LLM call.

    The card must never rule out the true cause (that would steer the reviewer away from the
    answer) and should narrow the candidates. Reviewer-correction replays are skipped: they
    duplicate a labelled case.
    """
    from filings_agent.steps.card import build_card

    tasks = [t for t in load_tasks(cfg, split) if not t.task.discrepancy_id.endswith("#corr")]
    true_fit: dict[str, int] = {}
    all_fits: dict[str, int] = {}  # every candidate on every card: how much the card narrows
    still_open: list[int] = []
    hidden: list[str] = []
    for t in tasks:
        card = build_card(t.task.discrepancy, t.task.evidence)
        fit = {c["cause"]: c["fit"] for c in card["candidates"]}
        true_fit[fit[t.label.value]] = true_fit.get(fit[t.label.value], 0) + 1
        for v in fit.values():
            all_fits[v] = all_fits.get(v, 0) + 1
        still_open.append(sum(v in ("fits", "possible") for v in fit.values()))
        if fit[t.label.value] in ("ruled_out", "no_evidence"):
            hidden.append(t.task.discrepancy_id)
    return {
        "split": split,
        "n": len(tasks),
        "true_cause_fit": true_fit,
        "true_cause_hidden": hidden,  # must be empty
        "candidate_fits": all_fits,
        "avg_open_candidates": statistics.fmean(still_open) if still_open else 0.0,
    }
