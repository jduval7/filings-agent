"""Memo step evaluation: deterministic checks first, then an LLM judge, then pairwise.

Inputs are built from the ANSWER KEY (correct figures, verified restatement causes), so the memo
step is tested in isolation: a bad extractor can't make a good memo look bad.

Three layers, cheapest and most trustworthy first:
  1. numbers_verified  code: every number in the memo is in the table (exact)
  2. judge criteria    LLM: faithful / caveats_correct / comparable / actionable (binary + reason)
  3. pairwise          LLM: memo A vs memo B on the same table, both orders, ties if inconsistent
"""

# LangSmith client methods are partially untyped; this module is a boundary that accepts it.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import json
import statistics
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from filings_agent.config import AppConfig
from filings_agent.domain import ExtractedFact, FilingRef, GoldFact, Metric
from filings_agent.evals.extract_experiments import dataset_name
from filings_agent.evals.reconcile_local import load_labels
from filings_agent.llm.cache import StepCache
from filings_agent.llm.models import chat_model
from filings_agent.sec.gold import find_discrepancies
from filings_agent.steps.judge import MEMO_RUBRIC, JudgeResult, LLMJudge
from filings_agent.steps.memo import Memo, MemoWriter, Row, build_table, table_text
from filings_agent.steps.reconcile import Cause, Resolution, auto_resolution, is_material

DEV_GROUPS: tuple[tuple[str, ...], ...] = (
    ("GE", "NVDA"),
    ("AAPL", "MSFT"),
    ("GE", "KO"),
    ("AAPL", "NVDA"),
    ("KO", "MSFT"),
)
TEST_GROUPS: tuple[tuple[str, ...], ...] = (("PFE", "WMT"), ("NKE", "WMT"), ("NKE", "PFE"))
_NS = uuid.UUID("0b7d1c55-3e2a-4f6b-9d81-6a4c2e9f7b10")


class MemoExample(BaseModel):
    example_id: str
    split: str
    group: list[str]
    rows: list[dict[str, Any]]
    changes: list[str]


def _row_to_dict(r: Row) -> dict[str, Any]:
    return {
        "ticker": r.ticker,
        "metric": r.metric.value,
        "fiscal_year": r.fiscal_year,
        "value": str(r.value),
        "display": r.display,
        "source": r.source,
        "changed_by": r.changed_by,
    }


def row_from_dict(d: dict[str, Any]) -> Row:
    return Row(
        str(d["ticker"]),
        Metric(d["metric"]),
        int(d["fiscal_year"]),
        Decimal(d["value"]),
        str(d["display"]),
        str(d["source"]),
        d.get("changed_by"),
    )


def _as_extracted(g: GoldFact) -> ExtractedFact:
    return ExtractedFact(
        ticker=g.ticker,
        metric=g.metric,
        fiscal_year=g.fiscal_year,
        period_end=g.period_end,
        value=g.value,
        accession_no=g.accession_no,
        statement="(answer key)",
        row_label="(answer key)",
        printed_value=str(g.value),
        scale="units",
        model="answer-key",
    )


def build_memo_dataset(cfg: AppConfig, labels_path: Path | None = None) -> list[MemoExample]:
    gold = [
        GoldFact.model_validate_json(x)
        for x in (cfg.data_dir / "gold" / "gold.jsonl").read_text().splitlines()
    ]
    filings = [
        FilingRef.model_validate_json(x)
        for x in (cfg.data_dir / "gold" / "filings.jsonl").read_text().splitlines()
    ]
    filed: dict[str, date] = {f.accession_no: f.filed for f in filings}
    labels = load_labels(labels_path) if labels_path else load_labels()
    out: list[MemoExample] = []
    for split, groups in (("dev", DEV_GROUPS), ("test", TEST_GROUPS)):
        for group in groups:
            facts = [g for g in gold if g.ticker in group]
            resolutions: dict[str, Resolution] = {}
            for d in find_discrepancies(facts, filed):
                if not is_material(d):
                    resolutions[d.id] = auto_resolution(d.id, d)
                    continue
                cause = labels.get(d.id) or Cause.UNKNOWN
                accn, value = d.values[-1]
                resolutions[d.id] = Resolution(
                    d.id, cause, "verified label", "", True, True, value, accn, False
                )
            rows, changes = build_table([_as_extracted(g) for g in facts], resolutions, filed)
            out.append(
                MemoExample(
                    example_id="+".join(group),
                    split=split,
                    group=list(group),
                    rows=[_row_to_dict(r) for r in rows],
                    changes=changes,
                )
            )
    return out


def write_memo_dataset(path: Path, rows: list[MemoExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows))


def load_memo_examples(cfg: AppConfig, split: str) -> list[MemoExample]:
    path = cfg.data_dir / "datasets" / "memo.jsonl"
    return [
        e
        for e in (MemoExample.model_validate_json(x) for x in path.read_text().splitlines())
        if e.split == split
    ]


def _material(e: MemoExample) -> tuple[list[Row], list[str], str]:
    rows = [row_from_dict(r) for r in e.rows]
    return rows, e.changes, table_text(rows, e.changes)


def make_writer(cfg: AppConfig, model: str) -> MemoWriter:
    cache = StepCache(cfg.data_dir / "cache" / "llm")
    return MemoWriter(model, cfg.model(model), chat_model(cfg, model), cache)


def make_judge(cfg: AppConfig, model: str | None = None) -> LLMJudge:
    name = model or cfg.roles.judge
    return LLMJudge(
        name, cfg.model(name), chat_model(cfg, name), StepCache(cfg.data_dir / "cache" / "llm")
    )


# ---- local scoring ----------------------------------------------------------------------------


def _write_and_judge(
    writer: MemoWriter, judge: LLMJudge, e: MemoExample
) -> tuple[Memo, JudgeResult]:
    rows, changes, material = _material(e)
    memo = writer.write(rows, changes)
    return memo, judge.grade(MEMO_RUBRIC, material, memo.markdown)


def score_memo(
    cfg: AppConfig,
    writer_model: str,
    split: str,
    judge_model: str | None = None,
    concurrency: int = 4,
) -> dict[str, Any]:
    if (judge_model or cfg.roles.judge) == writer_model:
        raise ValueError("the judge must not grade its own memos")
    examples = load_memo_examples(cfg, split)
    writer, judge = make_writer(cfg, writer_model), make_judge(cfg, judge_model)
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(partial(_write_and_judge, writer, judge), examples))
    # A judge error is no verdict, not a failed memo: score what was graded, count errors apart.
    graded = [j for _, j in results if not j.error]
    per_criterion = {
        c: statistics.fmean(j.passed.get(c, False) for j in graded) if graded else 0.0
        for c in MEMO_RUBRIC.names()
    }
    return {
        "writer": writer_model,
        "judge": judge.model_name,
        "split": split,
        "n": len(results),
        "numbers_verified": statistics.fmean(m.verified for m, _ in results),
        **{f"judge.{c}": v for c, v in per_criterion.items()},
        "judge_errors": sum(1 for _, j in results if j.error),
        "writer_cost_usd": sum(m.usage.cost_usd for m, _ in results),
        "judge_cost_usd": sum(j.usage.cost_usd for _, j in results),
        "verdicts": {
            e.example_id: {"verified": m.verified, "passed": j.passed, "reasons": j.reasons}
            for e, (m, j) in zip(examples, results, strict=True)
        },
    }


def compare_memos(
    cfg: AppConfig, model_a: str, model_b: str, split: str, judge_model: str | None = None
) -> dict[str, Any]:
    judge = make_judge(cfg, judge_model)
    if judge.model_name in (model_a, model_b):
        raise ValueError("the judge must not compare its own memos")
    wa, wb = make_writer(cfg, model_a), make_writer(cfg, model_b)
    tally = {"A": 0, "B": 0, "tie": 0}
    inconsistent = 0
    details: dict[str, Any] = {}
    for e in load_memo_examples(cfg, split):
        rows, changes, material = _material(e)
        ma, mb = wa.write(rows, changes), wb.write(rows, changes)
        res = judge.compare(material, ma.markdown, mb.markdown)
        tally[res.winner] += 1
        inconsistent += 0 if res.consistent else 1
        details[e.example_id] = {
            "winner": res.winner,
            "consistent": res.consistent,
            "reasons": res.reasons,
        }
    n = sum(tally.values()) or 1
    return {
        "a": model_a,
        "b": model_b,
        "judge": judge.model_name,
        "split": split,
        "a_wins": tally["A"],
        "b_wins": tally["B"],
        "ties": tally["tie"],
        "a_win_rate": tally["A"] / n,
        "position_bias_rate": inconsistent / n,
        "details": details,
    }


# ---- LangSmith --------------------------------------------------------------------------------


def upload_memo_datasets(cfg: AppConfig, client: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("dev", "test"):
        name = dataset_name("memo", split)
        if client.has_dataset(dataset_name=name):
            ds = client.read_dataset(dataset_name=name)
        else:
            ds = client.create_dataset(name, description=f"Answer-key tables -> memo ({split})")
        existing = {str(e.id) for e in client.list_examples(dataset_id=ds.id)}
        new: list[dict[str, Any]] = []
        update: list[dict[str, Any]] = []
        examples = load_memo_examples(cfg, split)
        for e in examples:
            ex_id = str(uuid.uuid5(_NS, e.example_id))
            payload = {
                "id": ex_id,
                "inputs": {"group": e.group, "rows": e.rows, "changes": e.changes},
                "outputs": {},
                "metadata": {"group": e.example_id, "split": split},
            }
            (update if ex_id in existing else new).append(payload)
        if new:
            client.create_examples(dataset_id=ds.id, examples=new)
        if update:
            client.update_examples(dataset_id=ds.id, updates=update)
        counts[name] = len(examples)
    return counts


def make_memo_target(writer: MemoWriter) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        rows = [row_from_dict(r) for r in inputs["rows"]]
        memo = writer.write(rows, list(inputs["changes"]))
        return {
            "memo": memo.markdown,
            "verified": memo.verified,
            "unverified": memo.unverified,
            "table": table_text(rows, list(inputs["changes"])),
            "cost_usd": memo.usage.cost_usd,
        }

    return target


def make_memo_evaluators(judge: LLMJudge) -> list[Callable[..., Any]]:
    from langsmith.evaluation import EvaluationResult

    def numbers_verified(run: Any, example: Any) -> Any:
        out = cast(dict[str, Any], run.outputs or {})
        return EvaluationResult(key="numbers_verified", score=int(bool(out.get("verified"))))

    def memo_judge(run: Any, example: Any) -> Any:
        out = cast(dict[str, Any], run.outputs or {})
        res = judge.grade(MEMO_RUBRIC, str(out.get("table", "")), str(out.get("memo", "")))
        return {
            "results": [
                EvaluationResult(
                    key=f"judge.{c}",
                    # no verdict when the judge failed: blank in LangSmith, not a 0
                    score=None if res.error else int(res.passed.get(c, False)),
                    comment=res.reasons.get(c, res.error or ""),
                )
                for c in MEMO_RUBRIC.names()
            ]
        }

    return [numbers_verified, memo_judge]


def run_memo_sweep(
    cfg: AppConfig, client: Any, models: list[str], split: str, concurrency: int = 4
) -> list[dict[str, Any]]:
    judge = make_judge(cfg)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict[str, Any]] = []
    for model in models:
        if model == judge.model_name:
            raise ValueError(f"{model} is the judge; it must not grade its own memos")
        results = client.evaluate(
            make_memo_target(make_writer(cfg, model)),
            data=dataset_name("memo", split),
            evaluators=make_memo_evaluators(judge),
            experiment_prefix=f"memo-{model}",
            description=f"memo step | writer {model} | judge {judge.model_name} | {split}",
            metadata={
                "step": "memo",
                "model": model,
                "judge": judge.model_name,
                "rubric": f"{MEMO_RUBRIC.name}/{MEMO_RUBRIC.version}",
                "split": split,
                "sweep": stamp,
            },
            max_concurrency=concurrency,
        )
        rows.append({"model": model, "split": split, "experiment": results.experiment_name})
    return rows


def run_pairwise_experiment(
    cfg: AppConfig, client: Any, experiment_a: str, experiment_b: str
) -> str:
    """LangSmith comparative experiment over two memo experiments on the same dataset."""
    from langsmith.evaluation import (
        evaluate_comparative,  # pyright: ignore[reportUnknownVariableType]
    )
    from langsmith.schemas import Example, Run

    judge = make_judge(cfg)

    def preference(runs: Sequence[Run], example: Example | None) -> dict[str, Any]:
        a, b = runs
        out_a = cast(dict[str, Any], a.outputs or {})
        out_b = cast(dict[str, Any], b.outputs or {})
        table = str(out_a.get("table", ""))
        res = judge.compare(table, str(out_a.get("memo", "")), str(out_b.get("memo", "")))
        scores = {"A": (1, 0), "B": (0, 1), "tie": (0.5, 0.5)}[res.winner]
        return {
            "key": "memo_preference",
            "scores": {a.id: scores[0], b.id: scores[1]},
            "comment": f"consistent={res.consistent}; {res.reasons[0]}",
        }

    result = evaluate_comparative(  # pyright: ignore[reportUnknownVariableType]
        (experiment_a, experiment_b),
        evaluators=[preference],
        experiment_prefix="memo-pairwise",
        description=f"pairwise memo preference, judge {judge.model_name}, both orders",
        client=client,
    )
    return str(getattr(result, "experiment_name", "") or "")


def save_report(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
