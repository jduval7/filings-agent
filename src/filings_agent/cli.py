"""Command line entry points.

filings-agent build-data   fetch filings + company facts, locate statements, build the answer key
                           and the per-step eval datasets
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from filings_agent.config import AppConfig, load_config
from filings_agent.datasets import (
    build_extract_dataset,
    build_reconcile_dataset,
    summarize,
    summarize_reconcile,
    write_jsonl,
)
from filings_agent.domain import FilingRef, GoldFact
from filings_agent.evals.online import monitor
from filings_agent.sec.filings import SecSource
from filings_agent.sec.gold import build_gold, find_discrepancies
from filings_agent.sec.http import SecHttp
from filings_agent.sec.sections import StatementKind, locate_statements

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


def _source(cfg: AppConfig) -> SecSource:
    ua = cfg.sec.user_agent()
    http = SecHttp(ua, cfg.data_dir / "cache" / "sec", cfg.sec.max_requests_per_second)
    return SecSource(http, cfg.data_dir / "filings", ua)


def build_data(cfg: AppConfig) -> None:
    src = _source(cfg)
    out_dir = cfg.data_dir / "gold"
    out_dir.mkdir(parents=True, exist_ok=True)

    filings: list[FilingRef] = []
    gold: list[GoldFact] = []
    lines: list[str] = ["# Data build report", ""]

    for ticker in cfg.companies:
        refs = src.list_filings(ticker, cfg.filings.form, cfg.filings.filed_since)
        filings.extend(refs)
        result = build_gold(ticker, src.company_facts(ticker), refs, cfg.metrics)
        gold.extend(result.facts)

        lines.append(f"## {ticker} ({cfg.split_of(ticker)}) — {len(refs)} filings")
        for ref in refs:
            secs = locate_statements(src.markdown(ref))
            found = ", ".join(
                f"{k.value}={'ok' if k in secs else 'MISSING'}"
                + (f"[{secs[k].scale}]" if k in secs and secs[k].scale else "")
                for k in StatementKind
            )
            n = sum(1 for g in result.facts if g.accession_no == ref.accession_no)
            lines.append(f"- {ref.accession_no} FY-end {ref.period_of_report}: {n} facts; {found}")
        missing = Counter(m.value for _, _, m in result.missing)
        if missing:
            lines.append(f"- not reported (no XBRL concept): {dict(missing)}")
        lines.extend(f"- WARN {w}" for w in result.warnings)
        lines.append("")

    # Reconcile-only companies: answer key + discrepancies, no extraction dataset.
    recon_filings: list[FilingRef] = []
    recon_gold: list[GoldFact] = []
    for ticker in cfg.reconcile_companies:
        refs = src.list_filings(ticker, cfg.filings.form, cfg.filings.reconcile_filed_since)
        recon_filings.extend(refs)
        for ref in refs:
            src.markdown(ref)  # reconcile reads the notes of the filings where numbers changed
        recon_gold.extend(build_gold(ticker, src.company_facts(ticker), refs, cfg.metrics).facts)
    lines.append(
        f"## Reconcile-only companies: {len(cfg.reconcile_companies)} "
        f"({len(recon_filings)} filings, {len(recon_gold)} facts)"
    )
    lines.append("")

    filed = {f.accession_no: f.filed for f in filings + recon_filings}
    discrepancies = find_discrepancies(gold + recon_gold, filed)
    lines.append(f"## Discrepancies in the answer key: {len(discrepancies)}")
    for d in discrepancies:
        vals = " -> ".join(f"{v:,}" for _, v in d.values)
        lines.append(f"- {d.ticker} {d.metric.value} {d.period_end}: {vals} (x{d.ratio:.3f})")

    (out_dir / "filings.jsonl").write_text("".join(f.model_dump_json() + "\n" for f in filings))
    (out_dir / "gold.jsonl").write_text("".join(g.model_dump_json() + "\n" for g in gold))
    (out_dir / "filings_all.jsonl").write_text(
        "".join(f.model_dump_json() + "\n" for f in filings + recon_filings)
    )
    (out_dir / "discrepancies.jsonl").write_text(
        "".join(d.model_dump_json() + "\n" for d in discrepancies)
    )
    extract = build_extract_dataset(
        cfg, filings, gold, {f.accession_no: src.markdown(f) for f in filings}
    )
    write_jsonl(cfg.data_dir / "datasets" / "extract.jsonl", extract)
    reconcile = build_reconcile_dataset(cfg, discrepancies, filed)
    write_jsonl(cfg.data_dir / "datasets" / "reconcile.jsonl", reconcile)
    from filings_agent.evals.memo_eval import build_memo_dataset, write_memo_dataset

    memo_examples = build_memo_dataset(cfg)
    write_memo_dataset(cfg.data_dir / "datasets" / "memo.jsonl", memo_examples)
    lines.append(f"memo examples: {len(memo_examples)}")
    lines.extend(["", "## Datasets", summarize(extract), summarize_reconcile(reconcile)])

    report = "\n".join(lines) + "\n"
    (out_dir / "REPORT.md").write_text(report)
    print(report)
    print(
        json.dumps(
            {
                "filings": len(filings),
                "gold_facts": len(gold),
                "discrepancies": len(discrepancies),
                "out": str(out_dir),
            }
        )
    )


def _langsmith() -> Any:
    from langsmith import Client

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "filings-agent")
    return Client()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="filings-agent")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-data", help="fetch filings, build answer key, report coverage")
    sub.add_parser("upload-datasets", help="push eval datasets to LangSmith")
    sub.add_parser("spend", help="show spend vs caps")
    sub.add_parser("export-corrections", help="push reviewer corrections to LangSmith")
    sub.add_parser("import-qa", help="QA's dataset examples -> regression tests for the gate")
    ev = sub.add_parser("online-evaluator", help="LangSmith's online evaluator for memos")
    ev.add_argument("--create", metavar="PROJECT", help="create it on this tracing project")
    ev.add_argument("--sampling", type=float, default=0.1, help="share of matching runs (0-1)")
    ev.add_argument("--backfill-hours", type=float, default=0, help="also score recent runs")
    gate = sub.add_parser("gate", help="eval regression gate for CI")
    gate.add_argument("--update-baseline", action="store_true")
    srv = sub.add_parser("serve", help="run the HTTP API")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    run = sub.add_parser("run", help="run the analysis graph on real filings")
    run.add_argument("--tickers", required=True, help="comma-separated, e.g. GE,NVDA")
    run.add_argument("--model", default=None, help="defaults to roles.extractor")
    run.add_argument("--prompt", default=None, help="defaults to prompts.extract in config")
    run.add_argument("--thread", default=None, help="defaults to a hash of the request")
    run.add_argument("--resume", action="store_true", help="continue an interrupted thread")
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--fresh", action="store_true", help="bypass the LLM cache (new calls)")
    rv = sub.add_parser("review", help="decide on material changes and resume a paused run")
    rv.add_argument("--thread", required=True)
    rv.add_argument("--tickers", required=True)
    rv.add_argument("--model", default=None)
    rv.add_argument("--prompt", default=None)
    rv.add_argument("--reviewer", default=None, help="recorded in the ledger (default: $USER)")
    group = rv.add_mutually_exclusive_group(required=True)
    group.add_argument("--accept-all", action="store_true", help="use recast values for all")
    group.add_argument("--decisions", type=Path, help='JSON: {"id": {"action": ..., "note": ...}}')
    cards = sub.add_parser("cards", help="show review cards (no LLM call)")
    src = cards.add_mutually_exclusive_group(required=True)
    src.add_argument("--thread", help="the cards a paused run is waiting on")
    src.add_argument("--case", help="preview dataset cases, e.g. GE:net_income:FY2022 (comma list)")
    src.add_argument("--eval", action="store_true", help="score the cards against the labels")
    score = sub.add_parser("score", help="local scoring loop over models x reps")
    score.add_argument("step", choices=["extract", "reconcile", "memo"])
    score.add_argument("--judge", default=None, help="memo: judge model (default roles.judge)")
    score.add_argument(
        "--reconciler",
        choices=["pipeline", "agent"],
        default="pipeline",
        help="reconcile: fixed pipeline (baseline) or tool-using agent",
    )
    score.add_argument("--version", default=None, help="reconcile agent version (default config)")
    score.add_argument(
        "--retrieval",
        choices=["vector", "hybrid"],
        default=None,
        help="reconcile agent: retrieval mode (default config)",
    )
    score.add_argument(
        "--no-injections",
        action="store_true",
        help="reconcile agent: skip the synthetic prompt-injection cases",
    )
    score.add_argument("--models", required=True)
    score.add_argument("--prompt", default="v1")
    score.add_argument("--split", default="dev", choices=["dev", "test", "fresh"])
    score.add_argument("--reps", type=int, default=1)
    score.add_argument("--concurrency", type=int, default=8)
    score.add_argument(
        "--strategy",
        default="statement",
        choices=["statement", "whole_filing"],
        help="statement = fan-out per located statement; whole_filing = baseline, one call/10-K",
    )
    cmp_ = sub.add_parser("compare", help="pairwise comparison (memo)")
    cmp_.add_argument("step", choices=["memo"])
    cmp_.add_argument("--models", help="two writer models, local pairwise: a,b")
    cmp_.add_argument("--experiments", help="two LangSmith memo experiment names: a,b")
    cmp_.add_argument("--split", default="dev", choices=["dev", "test"])
    cal = sub.add_parser("calibrate", help="judge calibration against your labels")
    cal.add_argument("action", choices=["prepare", "queue", "pull", "score"])
    cal.add_argument("--judge", default=None)
    cal.add_argument("--upload", action="store_true", help="also record in LangSmith")
    q = sub.add_parser("queue", help="LangSmith annotation queue")
    q.add_argument("action", choices=["setup"])
    sweep = sub.add_parser("sweep", help="run a step over several models as experiments")
    sweep.add_argument("step", choices=["extract", "reconcile", "memo", "e2e", "research"])
    sweep.add_argument("--models", default="", help="comma-separated names from config models")
    sweep.add_argument("--split", default="dev", choices=["dev", "test", "fresh"])
    sweep.add_argument("--prompt", default="v1")
    sweep.add_argument("--reps", type=int, default=1)
    sweep.add_argument("--concurrency", type=int, default=8)
    sweep.add_argument("--reconciler", choices=["pipeline", "agent"], default="pipeline")
    sweep.add_argument("--retrieval", choices=["vector", "hybrid"], default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "build-data":
        build_data(cfg)
    elif args.cmd == "upload-datasets":
        from filings_agent.evals.extract_experiments import upload_extract_datasets
        from filings_agent.evals.memo_eval import upload_memo_datasets
        from filings_agent.evals.reconcile_experiments import upload_reconcile_datasets
        from filings_agent.evals.trajectory import upload_e2e_dataset

        client = _langsmith()
        counts = {
            **upload_extract_datasets(cfg, client),
            **upload_reconcile_datasets(cfg, client),
            **upload_memo_datasets(cfg, client),
            "filings-agent/e2e-dev": upload_e2e_dataset(cfg, client),
        }
        print(json.dumps(counts, indent=2))
    elif args.cmd == "run":
        from filings_agent.graph.runner import run_analysis

        _langsmith()  # enable tracing
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        summary = run_analysis(
            cfg,
            tickers,
            args.model or cfg.roles.extractor,
            args.prompt or cfg.prompts.extract,
            args.thread,
            args.resume,
            args.concurrency,
            rep=int(time.time()) if args.fresh else 0,
        )
        summary["online_review_reasons"] = monitor(summary)
        print(json.dumps(summary, indent=2, default=str))
    elif args.cmd == "review":
        from filings_agent.graph.runner import pending_review, run_analysis

        _langsmith()
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.accept_all:
            ids = pending_review(cfg, args.thread)
            decisions = {i: {"action": "accept", "note": "accept-all"} for i in ids}
        else:
            decisions = json.loads(args.decisions.read_text())
        # who and when, as the API stamps them; the file's own values never win
        stamp = {
            "actor": args.reviewer or getpass.getuser(),
            "decided_at": datetime.now(UTC).isoformat(),
        }
        decisions = {k: {**v, **stamp} for k, v in decisions.items()}
        summary = run_analysis(
            cfg, tickers, args.model or cfg.roles.extractor, args.prompt or cfg.prompts.extract,
            args.thread,
            decisions=decisions,
        )  # fmt: skip
        summary["online_review_reasons"] = monitor(summary)
        print(json.dumps(summary, indent=2, default=str))
    elif args.cmd == "cards":
        from filings_agent.steps.card import build_card, render_card

        if args.eval:
            from filings_agent.evals.reconcile_local import card_eval

            for split in ("dev", "test", "fresh"):
                print(json.dumps(card_eval(cfg, split)))
            return
        if args.thread:
            from filings_agent.graph.runner import pending_cards

            shown = pending_cards(cfg, args.thread)
        else:
            from filings_agent.evals.reconcile_local import load_tasks

            wanted = {c.strip() for c in args.case.split(",") if c.strip()}
            tasks = {
                t.task.discrepancy_id: t.task
                for s in ("dev", "test", "fresh")
                for t in load_tasks(cfg, s)
            }
            missing = wanted - set(tasks)
            if missing:
                raise SystemExit(f"not in the labelled datasets: {sorted(missing)}")
            shown = [build_card(tasks[c].discrepancy, tasks[c].evidence) for c in sorted(wanted)]
        print("\n\n".join(render_card(c) for c in shown) or "nothing waiting for review")
    elif args.cmd == "score" and args.step == "memo":
        from filings_agent.evals.memo_eval import save_report, score_memo

        for model in [m.strip() for m in args.models.split(",") if m.strip()]:
            row = score_memo(cfg, model, args.split, args.judge)
            save_report(cfg.data_dir / "memo_eval" / f"{model}-{args.split}.json", row)
            print(json.dumps({k: v for k, v in row.items() if k != "verdicts"}))
    elif args.cmd == "compare":
        from filings_agent.evals.memo_eval import compare_memos, run_pairwise_experiment

        if args.experiments:
            a, b = [x.strip() for x in args.experiments.split(",")]
            print(run_pairwise_experiment(cfg, _langsmith(), a, b))
        else:
            a, b = [x.strip() for x in (args.models or "").split(",")]
            res = compare_memos(cfg, a, b, args.split)
            print(json.dumps({k: v for k, v in res.items() if k != "details"}, indent=2))
    elif args.cmd == "calibrate":
        from filings_agent.evals import calibration

        if args.action == "prepare":
            items = calibration.prepare(cfg)
            print(f"{len(items)} items -> {cfg.data_dir / 'calibration' / 'LABEL_ME.md'}")
            print("label them in LangSmith: filings-agent calibrate queue")
        elif args.action == "queue":
            from filings_agent.evals import calibration_queue as cq
            from filings_agent.evals.langsmith_links import project_url, queue_url
            from filings_agent.steps.judge import RUBRICS

            client = _langsmith()
            print("added to the queues:", cq.queue_items(cfg, client))
            org = (project_url(client, cq.PROJECT) or ("", ""))[0]
            for rubric in RUBRICS.values():
                print(f"  {cq.queue_name(rubric)}: {queue_url(client, org, cq.queue_name(rubric))}")
            print("label every item, then: filings-agent calibrate pull")
        elif args.action == "pull":
            from filings_agent.evals import calibration_queue as cq

            counts = cq.pull_labels(cfg, _langsmith())
            print(f"labelled questions per rubric: {counts} -> {calibration.LABELS_PATH}")
            print("then: filings-agent calibrate score")
        else:
            result = calibration.score(cfg, args.judge)
            print(calibration.report_markdown(result))
            print(f"report: {calibration.save(cfg, result)}")
            if args.upload:
                client = _langsmith()
                calibration.upload_calibration(cfg, client)
                print(
                    "experiment:", calibration.run_calibration_experiment(cfg, client, args.judge)
                )
    elif args.cmd == "queue":
        from filings_agent.evals.online import ensure_queue

        q = ensure_queue(_langsmith())
        print(f"annotation queue ready: {q.name} ({q.id})")
    elif args.cmd == "sweep" and args.step == "memo":
        from filings_agent.evals.memo_eval import run_memo_sweep

        models = [m.strip() for m in args.models.split(",") if m.strip()]
        for m in models:
            cfg.model(m)
        for row in run_memo_sweep(cfg, _langsmith(), models, args.split):
            print(json.dumps(row))
    elif args.cmd == "sweep" and args.step == "e2e":
        from filings_agent.evals.trajectory import run_e2e_experiment
        from filings_agent.graph.build import Deps
        from filings_agent.graph.runner import make_deps

        def deps() -> Deps:  # no ledger, no memo: the e2e eval mustn't publish
            d = make_deps(cfg, cfg.roles.extractor, cfg.prompts.extract)
            return Deps(
                d.store,
                d.extractor,
                d.metrics,
                reconciler=d.reconciler,
                auto_approve=d.auto_approve,
            )

        print(run_e2e_experiment(cfg, _langsmith(), deps))
    elif args.cmd == "score" and args.step == "reconcile" and args.reconciler == "agent":
        from dataclasses import asdict

        from filings_agent.evals.agent_eval import score_agent, table

        rows = []
        for model in [m.strip() for m in args.models.split(",") if m.strip()]:
            version = args.version or cfg.reconcile.agent_version
            summary, cases = score_agent(
                cfg, model, version, args.split, args.retrieval, injections=not args.no_injections
            )
            out = (
                cfg.data_dir
                / "agent_eval"
                / (f"{model}-{version}-{summary['retrieval_mode']}-{args.split}.json")
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {"summary": summary, "cases": [asdict(c) for c in cases]}, indent=2, default=str
                )
            )
            rows.append(summary)
            print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, (list, dict))}))
        print(table(rows))
    elif args.cmd == "score" and args.step == "reconcile":
        from filings_agent.evals.reconcile_local import reconcile_table, score_reconciler

        rows = []
        for model in [m.strip() for m in args.models.split(",") if m.strip()]:
            for prompt in [p.strip() for p in args.prompt.split(",") if p.strip()]:
                row = score_reconciler(cfg, model, prompt, args.split, args.reps, args.concurrency)
                rows.append(row)
                print(json.dumps(row), flush=True)
        print(reconcile_table(rows))
    elif args.cmd == "score":
        from filings_agent.evals.local import (
            score_model,
            score_whole_filing,
            summarize_reps,
            table,
        )

        scorer = score_whole_filing if args.strategy == "whole_filing" else score_model
        rows: list[dict[str, Any]] = []
        for model in [m.strip() for m in args.models.split(",") if m.strip()]:
            for prompt in [p.strip() for p in args.prompt.split(",") if p.strip()]:
                reps = scorer(cfg, model, prompt, args.split, args.reps, args.concurrency)
                label = prompt if args.strategy == "statement" else f"{prompt}/whole-filing"
                rows.append(summarize_reps(model, label, reps))
                print(json.dumps(rows[-1]), flush=True)
        print(table(rows))
    elif args.cmd == "export-corrections":
        from filings_agent.evals.corrections import export_corrections
        from filings_agent.evals.reconcile_local import CORRECTIONS
        from filings_agent.storage.ledger import Ledger

        n = export_corrections(
            Ledger(cfg.state("ledger.sqlite")),
            _langsmith(),
            CORRECTIONS,
            project_name=os.environ.get("LANGSMITH_PROJECT", "filings-agent"),
        )
        print(f"exported {n} correction(s)")
    elif args.cmd == "online-evaluator":
        from filings_agent.evals.online_judge import create_rule, setup_text

        if not args.create:
            print(setup_text())
        else:
            import httpx

            client = _langsmith()
            project = client.read_project(project_name=args.create)
            since = None
            if args.backfill_hours:
                from datetime import timedelta

                since = (datetime.now(UTC) - timedelta(hours=args.backfill_hours)).isoformat()
            key = {"x-api-key": os.environ["LANGSMITH_API_KEY"]}
            api = httpx.Client(base_url=client.api_url, headers=key, timeout=30)
            rule = create_rule(api, str(project.id), args.sampling, since)
            print(f"online evaluator {rule} on {args.create} (sampling {args.sampling:.0%})")
    elif args.cmd == "import-qa":
        from filings_agent.evals.qa_import import import_qa
        from filings_agent.evals.reconcile_local import CORRECTIONS

        result = import_qa(_langsmith(), CORRECTIONS)
        for line in result["added"]:
            print(f"added   {line}")
        for example_id, why in result["refused"].items():
            print(f"refused {example_id}: {why}")
        n = len(result["added"])
        print(f"{n} new test(s) in {result['file']}: commit it so the gate runs them")
    elif args.cmd == "gate":
        from filings_agent.evals.gate import run_gate

        ok, checks, _ = run_gate(
            cfg, Path("configs/eval_gate.yaml"), Path("evals/baseline.json"), args.update_baseline
        )
        print(f"extractor={cfg.roles.extractor} reconciler={cfg.roles.reconciler}")
        print("\n".join(c.line() for c in checks))
        print("GATE PASSED" if ok else "GATE FAILED")
        sys.exit(0 if ok else 1)
    elif args.cmd == "serve":
        import uvicorn

        from filings_agent.api.app import create_app

        if not cfg.api.tokens() and args.host not in LOOPBACK:
            sys.exit(
                f"refusing to serve on {args.host} without authentication: set "
                f"{cfg.api.tokens_env}=name:token[,name:token] (or use --host 127.0.0.1)"
            )
        _langsmith()  # traces for every run started through the API
        # One worker by default; more are safe (thread leases live in the database), but live
        # progress events stream only from the worker running the thread.
        uvicorn.run(create_app(cfg), host=args.host, port=args.port)
    elif args.cmd == "spend":
        from filings_agent.llm.spend_guard import get_guard

        print(json.dumps(get_guard(cfg).status(), indent=2))
    elif args.cmd == "sweep" and args.step == "research":
        from filings_agent.evals.research_experiments import run_research_experiment

        print(json.dumps(run_research_experiment(cfg, _langsmith(), args.split), indent=2))
    elif args.cmd == "sweep" and args.step == "reconcile":
        from filings_agent.evals.reconcile_experiments import run_reconcile_sweep

        if args.split == "test":
            print("NOTE: test split. Only run this for final confirmation, never while tuning.")
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        for m in models:
            cfg.model(m)
        for row in run_reconcile_sweep(
            cfg,
            _langsmith(),
            models,
            args.split,
            args.prompt,
            args.reps,
            args.concurrency,
            args.reconciler,
            args.retrieval,
        ):
            print(json.dumps(row))
    elif args.cmd == "sweep":
        from filings_agent.evals.extract_experiments import run_extract_sweep, write_sweep_table

        if args.split == "test":
            print("NOTE: test split. Only run this for final confirmation, never while tuning.")
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        for m in models:
            cfg.model(m)  # fail fast on a typo before spending anything
        rows = run_extract_sweep(
            cfg, _langsmith(), models, args.split, args.prompt, args.reps, args.concurrency
        )
        stamp = rows[0]["sweep"] if rows else "empty"
        print(write_sweep_table(rows, cfg.data_dir / "sweeps" / f"extract-{stamp}.md"))


if __name__ == "__main__":
    main()
