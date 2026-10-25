"""Scoring helpers."""

from __future__ import annotations

from filings_agent.evals.scoring import micro, score_extraction


def test_micro_handles_empty_and_all_abstain() -> None:
    assert micro([])["f1"] == 0.0
    s = score_extraction([], ["operating_income"], [])
    assert s.precision == 1.0 and s.recall == 1.0 and s.abstain_correct == 1
