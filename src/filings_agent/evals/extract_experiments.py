# LangSmith client methods are partially untyped; this module is the boundary that accepts it.
# pyright: reportUnknownMemberType=false
"""LangSmith wiring for the extraction step: datasets, experiments, sweeps.

- Datasets are uploaded with deterministic example ids (uuid5 of our example_id), so re-uploading
  updates in place instead of duplicating: idempotent, like everything else here.
- One experiment per (model, prompt version, repetition), tagged with metadata so LangSmith's
  comparison view can slice by any of them.
- Every experiment row carries cost and latency next to quality, so a sweep produces a
  quality / cost / speed table, not just a leaderboard.
"""

from __future__ import annotations

import json
import statistics
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langsmith import Client
from langsmith.evaluation import EvaluationResult, EvaluationResults
from langsmith.schemas import Example, Run

from filings_agent.config import AppConfig
from filings_agent.datasets import ExtractExample
from filings_agent.evals.scoring import ExtractScore, micro, score_extraction
from filings_agent.llm.cache import StepCache
from filings_agent.llm.models import chat_model
from filings_agent.steps.extract import ExtractTask, LLMStatementExtractor

_NS = uuid.UUID("5b0e7f0a-5f0c-4b8e-9c55-1d0f3a8e2a11")


def dataset_name(step: str, split: str) -> str:
    return f"filings-agent/{step}-{split}"


def upload_extract_datasets(cfg: AppConfig, client: Client) -> dict[str, int]:
    rows = [
        ExtractExample.model_validate_json(line)
        for line in (cfg.data_dir / "datasets" / "extract.jsonl").read_text().splitlines()
    ]
    counts: dict[str, int] = {}
    for split in ("dev", "test"):
        name = dataset_name("extract", split)
        if client.has_dataset(dataset_name=name):
            ds = client.read_dataset(dataset_name=name)
        else:
            ds = client.create_dataset(
                name, description=f"10-K statement -> facts, graded vs SEC XBRL ({split} split)"
            )
        examples = [r for r in rows if r.split == split]
        existing = {str(e.id) for e in client.list_examples(dataset_id=ds.id)}
        new: list[dict[str, Any]] = []
        update: list[dict[str, Any]] = []
        for r in examples:
            ex_id = str(uuid.uuid5(_NS, r.example_id))
            payload: dict[str, Any] = {
                "id": ex_id,
                "inputs": {
                    "task_id": r.example_id,
                    "ticker": r.ticker,
                    "accession_no": r.accession_no,
                    "fiscal_year_end": r.fiscal_year_end.isoformat(),
                    "statement": r.statement.value,
                    "section_text": r.section_text,
                    "metrics": [m.value for m in r.metrics],
                },
                "outputs": {
                    "expected": [e.model_dump(mode="json") for e in r.expected],
                    "expect_absent": [m.value for m in r.expect_absent],
                },
                "metadata": {"ticker": r.ticker, "statement": r.statement.value, "split": split},
            }
            (update if ex_id in existing else new).append(payload)
        if new:
            client.create_examples(dataset_id=ds.id, examples=new)
        if update:
            client.update_examples(dataset_id=ds.id, updates=update)
        counts[name] = len(examples)
    return counts


def _task_from_inputs(inputs: dict[str, Any]) -> ExtractTask:
    from datetime import date

    from filings_agent.domain import Metric

    return ExtractTask(
        task_id=str(inputs["task_id"]),
        ticker=str(inputs["ticker"]),
        accession_no=str(inputs["accession_no"]),
        fiscal_year_end=date.fromisoformat(str(inputs["fiscal_year_end"])),
        statement=str(inputs["statement"]),
        section_text=str(inputs["section_text"]),
        metrics=tuple(Metric(m) for m in inputs["metrics"]),
    )


Target = Callable[[dict[str, Any]], dict[str, Any]]


def _outputs(obj: Run | Example) -> dict[str, Any]:
    """LangSmith types `outputs` as an untyped dict; this is the one place we accept that."""
    return cast(dict[str, Any], obj.outputs or {})


def _p95(xs: list[float]) -> float:
    return xs[min(len(xs) - 1, int(0.95 * len(xs)))]


def make_target(extractor: LLMStatementExtractor, rep: int) -> Target:
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        res = extractor.extract(_task_from_inputs(inputs), rep=rep)
        return {
            "facts": [
                {
                    "metric": f.metric.value,
                    "fiscal_year": f.fiscal_year,
                    "value": str(f.value),
                    "printed_value": f.printed_value,
                    "row_label": f.row_label,
                    "scale": f.scale,
                }
                for f in res.facts
            ],
            "not_reported": [m.value for m in res.not_reported],
            "rejected": [asdict(r) for r in res.rejected],
            "usage": asdict(res.usage) if res.usage else None,
            "error": res.error,
        }

    return target


def _score(run: Run, example: Example) -> ExtractScore:
    out = _outputs(run)
    ref = _outputs(example)
    return score_extraction(
        expected=cast(list[dict[str, Any]], ref.get("expected", [])),
        expect_absent=cast(list[str], ref.get("expect_absent", [])),
        predicted=cast(list[dict[str, Any]], out.get("facts", [])),
        rejected=len(cast(list[Any], out.get("rejected", []))),
        schema_error=out.get("error") is not None or not out,
    )


def extraction_evaluator(run: Run, example: Example) -> EvaluationResults:
    s = _score(run, example)
    usage = cast(dict[str, Any] | None, _outputs(run).get("usage"))
    results = [
        EvaluationResult(key="precision", score=s.precision),
        EvaluationResult(key="recall", score=s.recall),
        EvaluationResult(key="f1", score=s.f1),
        EvaluationResult(key="wrong_values", score=s.wrong_value),
        EvaluationResult(key="hallucinated", score=s.hallucinated),
        EvaluationResult(key="rejected_by_verifier", score=s.rejected),
        EvaluationResult(key="schema_error", score=int(s.schema_error)),
    ]
    if usage:
        results.append(EvaluationResult(key="cost_usd", score=float(usage["cost_usd"])))
        results.append(EvaluationResult(key="latency_s", score=float(usage["latency_s"])))
    return {"results": results}


def summary_evaluator(runs: list[Run], examples: list[Example]) -> EvaluationResults:
    scores = [_score(r, e) for r, e in zip(runs, examples, strict=True)]
    m = micro(scores)
    usages = [cast(dict[str, Any], _outputs(r).get("usage") or {}) for r in runs]
    lat = sorted(float(u.get("latency_s", 0.0)) for u in usages if u)
    results = [EvaluationResult(key=f"micro_{k}", score=v) for k, v in m.items()]
    total = sum(float(u.get("cost_usd", 0)) for u in usages)
    results.append(EvaluationResult(key="total_cost_usd", score=total))
    if lat:
        results.append(EvaluationResult(key="latency_p50_s", score=statistics.median(lat)))
        results.append(EvaluationResult(key="latency_p95_s", score=_p95(lat)))
    return {"results": results}


def run_extract_sweep(
    cfg: AppConfig,
    client: Client,
    models: list[str],
    split: str,
    prompt_version: str,
    reps: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    cache = StepCache(cfg.data_dir / "cache" / "llm")
    rows: list[dict[str, Any]] = []
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for model_name in models:
        extractor = LLMStatementExtractor(
            model_name, cfg.model(model_name), chat_model(cfg, model_name), cache, prompt_version
        )
        for rep in range(reps):
            results = client.evaluate(
                make_target(extractor, rep),
                data=dataset_name("extract", split),
                evaluators=[extraction_evaluator],
                summary_evaluators=[summary_evaluator],
                experiment_prefix=f"extract-{model_name}-{prompt_version}-r{rep}",
                description=f"extract step | {model_name} | prompt {prompt_version} | rep {rep}",
                metadata={
                    "step": "extract",
                    "strategy": "statement",
                    "model": model_name,
                    "model_id": cfg.model(model_name).id,
                    "prompt_version": prompt_version,
                    "rep": rep,
                    "split": split,
                    "sweep": stamp,
                },
                max_concurrency=concurrency,
            )
            per_example = [_score(r["run"], r["example"]) for r in results]
            usages = [cast(dict[str, Any], _outputs(r["run"]).get("usage") or {}) for r in results]
            lat = sorted(float(u.get("latency_s", 0.0)) for u in usages if u)
            row = {
                "sweep": stamp,
                "model": model_name,
                "prompt_version": prompt_version,
                "rep": rep,
                "split": split,
                "experiment": results.experiment_name,
                **micro(per_example),
                "cost_usd": sum(float(u.get("cost_usd", 0.0)) for u in usages),
                "latency_p50_s": statistics.median(lat) if lat else None,
                "latency_p95_s": _p95(lat) if lat else None,
                "examples": len(per_example),
            }
            rows.append(row)
            print(json.dumps(row))
    out = cfg.data_dir / "sweeps" / f"extract-{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def write_sweep_table(rows: list[dict[str, Any]], path: Path) -> str:
    cols = ["model", "prompt", "rep", "F1", "precision", "recall", "abstain", "schema err",
            "cost $", "p50 s", "p95 s"]  # fmt: skip
    header = "| " + " | ".join(cols) + " |\n" + "|---" * len(cols) + "|\n"
    body = "".join(
        f"| {r['model']} | {r['prompt_version']} | {r['rep']} | {r['f1']:.3f} "
        f"| {r['precision']:.3f} "
        f"| {r['recall']:.3f} | {r['abstain_accuracy']:.2f} | {r['schema_error_rate']:.2f} "
        f"| {r['cost_usd']:.4f} | {r['latency_p50_s'] or 0:.2f} | {r['latency_p95_s'] or 0:.2f} |\n"
        for r in sorted(rows, key=lambda r: (-r["f1"], r["cost_usd"]))
    )
    table = header + body
    path.parent.mkdir(parents=True, exist_ok=True)  # don't rely on the sweep having created it
    path.write_text(table)
    return table
