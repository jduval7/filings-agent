""""""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from filings_agent.domain import Discrepancy, Metric
from filings_agent.steps.reconcile import gather_evidence, is_material, verify_quote

EVIDENCE = [
    "On January 3, 2023, we completed the separation of the HealthCare business through the "
    "spin-off of GE HealthCare Technologies Inc\\. \\(GE HealthCare\\)\\. See Notes 2 and 3 for "
    "further information\\. The historical results of GE HealthCare are reported as discontinued "
    "operations."
]


def test_verbatim_quote_passes() -> None:
    assert verify_quote("we completed the separation of the HealthCare business", EVIDENCE)


def test_stitched_real_sentences_pass_in_order() -> None:
    q = (
        "spin-off of GE HealthCare Technologies Inc. (GE HealthCare). The historical results "
        "of GE HealthCare are reported as discontinued operations."
    )
    assert verify_quote(q, EVIDENCE)


def test_invented_or_reordered_text_fails() -> None:
    assert not verify_quote("we completed the sale of the Aviation business", EVIDENCE)
    reordered = (
        "The historical results of GE HealthCare are reported as discontinued operations. "
        "On January 3, 2023, we completed the separation of the HealthCare business"
    )
    assert not verify_quote(reordered, EVIDENCE)
    assert not verify_quote("", EVIDENCE)


def test_materiality_threshold() -> None:
    def d(a: int, b: int) -> Discrepancy:
        return Discrepancy(
            ticker="X", metric=Metric.REVENUE, fiscal_year=2023, period_end=date(2023, 12, 31),
            values=(("a", Decimal(a)), ("b", Decimal(b))),
        )  # fmt: skip

    assert not is_material(d(9481, 9482))  # GE FY2023 net income revision
    assert is_material(d(100, 60))


def test_evidence_prefers_split_language() -> None:
    md = ("filler " * 200) + "In June 2024, we executed a ten-for-one stock split." + (" x" * 50)
    ev = gather_evidence(md)
    assert ev and "ten-for-one stock split" in ev[0]


def test_unicode_hyphens_and_markdown_escapes_match() -> None:
    evidence = ["In May 2024, we announced a ten\\-for\\-one stock split, or the Stock Split\\."]
    assert verify_quote("we announced a ten‑for‑one stock split", evidence)
    assert not verify_quote("we announced a five‑for‑one stock split", evidence)


def test_split_evidence_found_through_markdown_escapes() -> None:
    # Real text from Amazon's and Chipotle's 10-Ks, as edgartools renders it.
    for md in (
        "**Common Stock Split** On May 27, 2022, we effected a 20 \\-for\\-1 stock split of our "
        "common stock and proportionately increased the number of authorized shares.",
        "We completed a ten\\-for\\-one forward stock split of our common stock\\.",
    ):
        ev = gather_evidence(("filler words here. " * 100) + md)
        assert ev and "split" in ev[0]


def test_spacing_around_hyphens_is_typography() -> None:
    evidence = ["On May 27, 2022, we effected a 20 \\-for\\-1 stock split of our common stock."]
    assert verify_quote("we effected a 20-for-1 stock split of our common stock", evidence)
    assert not verify_quote("we effected a 2-for-1 stock split of our common stock", evidence)


def test_stray_heading_markers_inside_sentences() -> None:
    # GE's 10-K as rendered: a "##" heading marker lands mid-sentence.
    evidence = [
        "we adopted Accounting Standards Update No\\. 2018\\-12, ## Financial Services – Insurance "
        "(Topic 944): Targeted Improvements to the Accounting for Long-Duration Contracts \\."
    ]
    quote = (
        "Update No. 2018-12, Financial Services – Insurance (Topic 944): Targeted Improvements"
        " to the Accounting for Long-Duration Contracts."
    )
    assert verify_quote(quote, evidence)
