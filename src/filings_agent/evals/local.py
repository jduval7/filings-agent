"""Fast local scoring loop: same extractor, same scorer, no LangSmith experiment created.

For iterating (prompt tweaks, re-scoring after a scorer fix). Cached outputs make re-runs free;
new repetitions are real calls. Results that matter get recorded as LangSmith experiments via
`filings-agent sweep`.
"""

from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any

from filings_agent.config import AppConfig
from filings_agent.datasets import ExtractExample
from filings_agent.evals.scoring import ExtractScore, micro, score_extraction
from filings_agent.llm.cache import StepCache
from filings_agent.llm.models import chat_model
from filings_agent.steps.extract import ExtractResult, ExtractTask, LLMStatementExtractor


def load_examples(cfg: AppConfig, split: str) -> list[ExtractExample]:
    rows = (cfg.data_dir / "datasets" / "extract.jsonl").read_text().splitlines()
    return [e for e in (ExtractExample.model_validate_json(r) for r in rows) if e.split == split]


def task_of(e: ExtractExample) -> ExtractTask:
    return ExtractTask(
        task_id=e.example_id,
        ticker=e.ticker,
        accession_no=e.accession_no,
        fiscal_year_end=e.fiscal_year_end,
        statement=e.statement.value,
        section_text=e.section_text,
        metrics=tuple(e.metrics),
    )


def score_result(e: ExtractExample, res: ExtractResult) -> ExtractScore:
    return score_extraction(
        expected=[x.model_dump(mode="json") for x in e.expected],
        expect_absent=[m.value for m in e.expect_absent],
        predicted=[
            {"metric": f.metric.value, "fiscal_year": f.fiscal_year, "value": str(f.value)}
            for f in res.facts
        ],
        rejected=len(res.rejected),
        schema_error=res.error is not None,
    )


def _run_one(extractor: LLMStatementExtractor, rep: int, e: ExtractExample) -> ExtractResult:
    return extractor.extract(task_of(e), rep=rep)


@dataclass(frozen=True)
class RepScore:
    rep: int
    metrics: dict[str, float]
    cost_usd: float
    latency_p50_s: float
    latency_p95_s: float
    new_calls: int
    error_kinds: dict[str, int]  # e.g. {"model_error:ValidationException": 3, "parse": 2}


def _error_kinds(results: list[ExtractResult]) -> dict[str, int]:
    kinds: dict[str, int] = {}
    for r in results:
        if r.error:
            k = r.error if r.error.startswith("model_error:") else "parse"
            kinds[k] = kinds.get(k, 0) + 1
    return kinds


def score_model(
    cfg: AppConfig, model: str, prompt: str, split: str, reps: int, concurrency: int = 8
) -> list[RepScore]:
    examples = load_examples(cfg, split)
    extractor = LLMStatementExtractor(
        model,
        cfg.model(model),
        chat_model(cfg, model),
        StepCache(cfg.data_dir / "cache" / "llm"),
        prompt,
    )
    out: list[RepScore] = []
    for rep in range(reps):
        with ThreadPoolExecutor(concurrency) as pool:
            results = list(pool.map(partial(_run_one, extractor, rep), examples))
        scores = [score_result(e, r) for e, r in zip(examples, results, strict=True)]
        usages = [r.usage for r in results if r.usage]
        lat = sorted(u.latency_s for u in usages)
        out.append(
            RepScore(
                rep=rep,
                metrics=micro(scores),
                cost_usd=sum(u.cost_usd for u in usages),
                latency_p50_s=statistics.median(lat) if lat else 0.0,
                latency_p95_s=lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0,
                new_calls=sum(1 for u in usages if not u.cached),
                error_kinds=_error_kinds(results),
            )
        )
    return out


def summarize_reps(model: str, prompt: str, reps: list[RepScore]) -> dict[str, Any]:
    def mean_sd(xs: list[float]) -> tuple[float, float]:
        return statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0

    f1, f1_sd = mean_sd([r.metrics["f1"] for r in reps])
    ab, ab_sd = mean_sd([r.metrics["abstain_accuracy"] for r in reps])
    return {
        "model": model,
        "prompt": prompt,
        "reps": len(reps),
        "f1": f1,
        "f1_sd": f1_sd,
        "precision": statistics.fmean(r.metrics["precision"] for r in reps),
        "recall": statistics.fmean(r.metrics["recall"] for r in reps),
        "abstain": ab,
        "abstain_sd": ab_sd,
        "schema_err": statistics.fmean(r.metrics["schema_error_rate"] for r in reps),
        "cost_per_run_usd": statistics.fmean(r.cost_usd for r in reps),
        "p50_s": statistics.fmean(r.latency_p50_s for r in reps),
        "p95_s": statistics.fmean(r.latency_p95_s for r in reps),
        "new_calls": sum(r.new_calls for r in reps),
        "error_kinds": {
            k: sum(r.error_kinds.get(k, 0) for r in reps) for r0 in reps for k in r0.error_kinds
        },  # fmt: skip
    }


def table(rows: list[dict[str, Any]]) -> str:
    cols = ["model", "prompt", "reps", "F1 (±sd)", "P", "R", "not-reported (±sd)",
            "schema err", "$/run", "p95 s"]  # fmt: skip
    head = "| " + " | ".join(cols) + " |"
    sep = "|---" * len(cols) + "|"
    lines = [head, sep]
    for r in sorted(rows, key=lambda r: (-r["f1"], r["cost_per_run_usd"])):
        lines.append(
            f"| {r['model']} | {r['prompt']} | {r['reps']} | {r['f1']:.3f} ±{r['f1_sd']:.3f} "
            f"| {r['precision']:.3f} | {r['recall']:.3f} "
            f"| {r['abstain']:.2f} ±{r['abstain_sd']:.2f} "
            f"| {r['schema_err']:.2f} | {r['cost_per_run_usd']:.4f} | {r['p95_s']:.1f} |"
        )
    return "\n".join(lines)


# ---- baseline: one call over the whole 10-K ---------------------------------------------------


@dataclass(frozen=True)
class FilingExample:
    """All of one filing's graded facts, for strategies that read the whole document."""

    task: ExtractTask
    examples: list[ExtractExample]


def load_filing_examples(cfg: AppConfig, split: str) -> list[FilingExample]:
    by_filing: dict[str, list[ExtractExample]] = {}
    for e in load_examples(cfg, split):
        by_filing.setdefault(e.accession_no, []).append(e)
    out: list[FilingExample] = []
    for accn, exs in sorted(by_filing.items()):
        first = exs[0]
        metrics = tuple(dict.fromkeys(m for e in exs for m in e.metrics))
        md = (cfg.data_dir / "filings" / f"{accn}.md").read_text()
        out.append(
            FilingExample(
                task=ExtractTask(
                    task_id=f"{accn}:whole_filing",
                    ticker=first.ticker,
                    accession_no=accn,
                    fiscal_year_end=first.fiscal_year_end,
                    statement="entire 10-K filing (locate the primary financial statements)",
                    section_text=md,
                    metrics=metrics,
                ),
                examples=exs,
            )
        )
    return out


def score_whole_filing(
    cfg: AppConfig, model: str, prompt: str, split: str, reps: int, concurrency: int = 4
) -> list[RepScore]:
    filings = load_filing_examples(cfg, split)
    cache = StepCache(cfg.data_dir / "cache" / "llm")
    extractor = LLMStatementExtractor(
        model, cfg.model(model), chat_model(cfg, model), cache, prompt
    )
    out: list[RepScore] = []
    for rep in range(reps):
        with ThreadPoolExecutor(concurrency) as pool:
            results = list(pool.map(partial(_run_filing, extractor, rep), filings))
        # Score each filing's result against each of its statements' answer keys, so the
        # numbers are directly comparable with the per-statement strategy.
        scores = [
            score_result(e, res)
            for f, res in zip(filings, results, strict=True)
            for e in f.examples
        ]
        usages = [r.usage for r in results if r.usage]
        lat = sorted(u.latency_s for u in usages)
        out.append(
            RepScore(
                rep=rep,
                metrics=micro(scores),
                cost_usd=sum(u.cost_usd for u in usages),
                latency_p50_s=statistics.median(lat) if lat else 0.0,
                latency_p95_s=lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0,
                new_calls=sum(1 for u in usages if not u.cached),
                error_kinds=_error_kinds(results),
            )
        )
    return out


def _run_filing(extractor: LLMStatementExtractor, rep: int, f: FilingExample) -> ExtractResult:
    return extractor.extract(f.task, rep=rep)
