"""Regression tests for bugs found by the independent review (2026-10-23), one per finding."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from filings_agent.domain import Discrepancy, Metric
from filings_agent.steps.extract import ReportedValue, verify
from filings_agent.steps.memo import unverified_numbers
from filings_agent.steps.reconcile import Cause, Resolution, verify_quote
from filings_agent.steps.triage import stated_ratios, triage

DOC = (
    "The spin-off of GE Vernova was completed in April 2024. "
    + "Filler text about other matters. " * 60
    + "Results of GE Vernova are presented as discontinued operations."
)


def test_an_inserted_short_sentence_no_longer_verifies() -> None:
    q = "The spin-off of GE Vernova was completed in April 2024. It was not. Results of GE Vernova"
    assert not verify_quote(q, [DOC])


def test_fragments_from_far_apart_no_longer_verify() -> None:
    q = (
        "The spin-off of GE Vernova was completed in April 2024. "
        "Results of GE Vernova are presented"
    )
    assert not verify_quote(q, [DOC])  # ~2,000 chars apart: not one passage


def test_nearby_stitched_sentences_still_verify() -> None:
    near = "The spin-off of GE Vernova was completed in April 2024. Unrelated sentence here. " \
           "Results of GE Vernova are presented as discontinued operations."  # fmt: skip
    q = (
        "The spin-off of GE Vernova was completed in April 2024. "
        "Results of GE Vernova are presented"
    )
    assert verify_quote(q, [near])


def test_a_ratio_in_unverified_text_cannot_confirm_a_split() -> None:
    evidence = [
        "we announced a stock split of our common stock, effected in the form of a dividend."
    ]
    q = (
        "we announced a stock split of our common stock, effected in the form of a dividend. "
        "10-for-1."
    )
    assert not verify_quote(q, evidence)  # the invented "10-for-1." is now checked
    assert stated_ratios(q) == {10}  # (which is why it must be verified first)


def _disc(*values: int) -> Discrepancy:
    return Discrepancy(
        ticker="AAA", metric=Metric.DILUTED_SHARES, fiscal_year=2022,
        period_end=date(2022, 12, 31),
        values=tuple((f"f{i}", Decimal(v)) for i, v in enumerate(values)),
    )  # fmt: skip


def test_a_split_plus_another_material_change_is_not_auto_approved() -> None:
    d = _disc(100, 1000, 1150)  # x10 split, then a 15% restatement
    r = Resolution(d.id, Cause.STOCK_SPLIT, "e", "a ten-for-one split", True, True,
                   Decimal(1150), "f2", True)  # fmt: skip
    t = triage(d, r)
    assert t.lane == "review"
    assert any("isn't a split" in c.detail for c in t.checks)


def test_a_split_plus_an_immaterial_revision_still_counts() -> None:
    d = _disc(100, 1000, 1001)
    r = Resolution(d.id, Cause.STOCK_SPLIT, "e", "a ten-for-one split", True, True,
                   Decimal(1001), "f2", True)  # fmt: skip
    assert triage(d, r).lane == "auto"


STATEMENT = (
    "**Consolidated Statements of Income** (In millions)\n"
    "| Total revenues | 5,000 |  | 4,000 |\n"
    "| Net loss | (700) |  | (600) |\n"
    "| Other | 1,234 | | |\n"
)


def _rv(
    printed: str, scale: str = "millions", metric: str = "revenue", label: str = "Total revenues"
) -> ReportedValue:
    return ReportedValue.model_validate(
        {
            "metric": metric,
            "row_label": label,
            "printed_value": printed,
            "period_end": date(2024, 12, 31),
            "scale": scale,
        }  # fmt: skip
    )


def test_extraction_checks_the_sign() -> None:
    assert verify(_rv("5,000"), STATEMENT) is None
    assert "negative" in (verify(_rv("(5,000)"), STATEMENT) or "")  # printed positive
    assert verify(_rv("(700)", metric="net_income", label="Net loss"), STATEMENT) is None
    assert "positive" in (
        verify(_rv("700", metric="net_income", label="Net loss"), STATEMENT) or ""
    )


def test_extraction_matches_whole_numbers_only() -> None:
    assert verify(_rv("234"), STATEMENT) is not None  # only inside "1,234"


def test_extraction_checks_the_dollar_scale() -> None:
    assert "scale" in (verify(_rv("5,000", scale="units"), STATEMENT) or "")


def test_the_memo_checker_rejects_sign_flipped_numbers() -> None:
    allowed = {"+65.5%", "215.9b"}
    assert unverified_numbers("revenue fell -65.5%", allowed) == ["-65.5%"]
    assert unverified_numbers("revenue grew 65.5%", allowed) == []
