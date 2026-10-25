"""Memo eval: dataset from the answer key, no self-grading, report files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from filings_agent.config import AppConfig
from filings_agent.evals import memo_eval as me


def _gold_files(cfg: AppConfig) -> None:
    from datetime import date
    from decimal import Decimal

    from filings_agent.domain import FilingRef, GoldFact, Metric, Period

    filings = [
        FilingRef(
            ticker=t,
            cik=1,
            accession_no=f"{t}-{y}",
            form="10-K",
            filed=date(y + 1, 2, 1),
            period_of_report=date(y, 12, 31),
        )
        for t in ("GE", "NVDA")
        for y in (2023, 2024)
    ]
    gold: list[GoldFact] = []
    for t in ("GE", "NVDA"):
        for y, accn in ((2023, f"{t}-2023"), (2024, f"{t}-2024")):
            gold.append(
                GoldFact(
                    ticker=t,
                    metric=Metric.REVENUE,
                    period=Period(start=date(y, 1, 1), end=date(y, 12, 31)),
                    value=Decimal(100_000_000_000 + y),
                    accession_no=accn,
                    concept="x",
                )
            )
    # GE's FY2023 revenue restated in the 2024 filing (a spin-off)
    gold.append(
        GoldFact(
            ticker="GE",
            metric=Metric.REVENUE,
            period=Period(start=date(2023, 1, 1), end=date(2023, 12, 31)),
            value=Decimal(60_000_000_000),
            accession_no="GE-2024",
            concept="x",
        )
    )
    (cfg.data_dir / "gold").mkdir(parents=True)
    (cfg.data_dir / "gold" / "filings.jsonl").write_text(
        "".join(f.model_dump_json() + "\n" for f in filings)
    )
    (cfg.data_dir / "gold" / "gold.jsonl").write_text(
        "".join(g.model_dump_json() + "\n" for g in gold)
    )


def test_memo_dataset_built_from_answer_key_with_verified_causes(
    cfg: AppConfig, tmp_path: Path
) -> None:
    _gold_files(cfg)
    labels = tmp_path / "l.yaml"
    labels.write_text(yaml.safe_dump({"GE:revenue:FY2023": {"cause": "discontinued_operations"}}))
    examples = me.build_memo_dataset(cfg, labels)
    ge_nvda = next(e for e in examples if e.example_id == "GE+NVDA")
    ge_rows = [r for r in ge_nvda.rows if r["ticker"] == "GE"]
    assert {r["changed_by"] for r in ge_rows} == {None, "discontinued_operations"}
    me.write_memo_dataset(cfg.data_dir / "datasets" / "memo.jsonl", examples)
    assert [e.example_id for e in me.load_memo_examples(cfg, "dev")][0] == "GE+NVDA"
    row = me.row_from_dict(ge_rows[0])
    assert row.ticker == "GE"


def test_score_memo_refuses_self_grading(cfg: AppConfig) -> None:
    with pytest.raises(ValueError, match="own memos"):
        me.score_memo(cfg, cfg.roles.judge, "dev")


def test_report_files_are_json(tmp_path: Path) -> None:
    me.save_report(tmp_path / "x" / "r.json", {"a": 1})
    assert json.loads((tmp_path / "x" / "r.json").read_text()) == {"a": 1}
