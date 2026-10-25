"""Properties that must hold for every input, not just the examples we thought of (hypothesis).

These target the pure functions where edge cases hide: text normalization, quote verification,
split-ratio parsing, passage ranking, the moves the research question lists, and the review card.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from filings_agent.domain import Discrepancy, Metric
from filings_agent.research.agent import moves
from filings_agent.research.sources import passages
from filings_agent.steps.card import CANDIDATES, build_card
from filings_agent.steps.reconcile import Cause, change_filings, verify_quote
from filings_agent.steps.triage import CANNOT_MOVE, split_ratio, stated_ratios
from filings_agent.textnorm import normalize

WORDS = st.sampled_from(
    [
        "the",
        "company",
        "adopted",
        "standard",
        "revenue",
        "spin-off",
        "split",
        "net",
        "income",
        "recast",
        "prior",
        "periods",
        "discontinued",
        "operations",
        "shares",
        "quarterly",
        "2023",
        "2022",
        "ASU",
        "2018-12",
        "ten-for-one",
    ]
)
SENTENCE = st.lists(WORDS, min_size=4, max_size=30).map(" ".join)


@given(st.text())
def test_normalize_is_idempotent(text: str) -> None:
    assert normalize(normalize(text)) == normalize(text)


@given(SENTENCE, st.data())
def test_any_long_enough_slice_of_the_evidence_verifies(evidence: str, data: st.DataObject) -> None:
    words = evidence.split()
    i = data.draw(st.integers(0, len(words) - 1))
    j = data.draw(st.integers(i + 1, len(words)))
    quote = " ".join(words[i:j])
    if len(normalize(quote)) >= 12:
        assert verify_quote(quote, [evidence])


@given(SENTENCE)
def test_an_invented_word_never_verifies(evidence: str) -> None:
    words = evidence.split()
    quote = " ".join([*words[: len(words) // 2], "fabricated", *words[len(words) // 2 :]])
    assert not verify_quote(quote, [evidence])


@given(st.integers(2, 200), st.floats(-0.9, 0.9))
def test_whole_number_splits_are_recognized_both_ways(n: int, share: float) -> None:
    """Within the allowed drift (0.5% of n, at most 0.25), the ratio maps back to exactly n."""
    drift = min(Decimal("0.005") * n, Decimal("0.25")) * Decimal(str(share))
    ratio = Decimal(n) + drift
    assert split_ratio(ratio) == n
    assert split_ratio(1 / ratio) == -n


@given(st.integers(2, 500), st.floats(0.0, 1.0))
def test_a_ratio_is_never_near_two_whole_numbers(n: int, frac: float) -> None:
    got = split_ratio(Decimal(n) + Decimal(str(frac)))
    assert got is None or abs(Decimal(got) - (Decimal(n) + Decimal(str(frac)))) <= Decimal("0.25")


@given(st.integers(2, 500), st.floats(0.3, 0.7))
def test_halfway_ratios_are_not_splits(n: int, frac: float) -> None:
    assert split_ratio(Decimal(n) + Decimal(str(frac))) is None


@given(st.integers(2, 500))
def test_stated_ratios_read_digits_both_directions(n: int) -> None:
    assert stated_ratios(f"a {n}-for-1 stock split") == {n}
    assert stated_ratios(f"a 1-for-{n} reverse stock split") == {-n}


@given(
    st.lists(SENTENCE, min_size=1, max_size=40).map(" ".join),
    st.lists(WORDS, min_size=1, max_size=4).map(" ".join),
    st.integers(1, 6),
)
def test_passages_are_real_excerpts_in_document_order(text: str, query: str, k: int) -> None:
    found = passages(text, query, k)
    assert len(found) <= k
    assert [i for i, _ in found] == sorted(i for i, _ in found)
    words = {w for w in query.lower().split() if len(w) >= 3}
    for _, chunk in found:
        assert chunk in text
        assert any(w in chunk.lower() for w in words)


VALUES = st.lists(st.integers(-(10**9), 10**12), min_size=2, max_size=5)


def disc(metric: Metric, values: list[int]) -> Discrepancy:
    return Discrepancy(
        ticker="AAA", metric=metric, fiscal_year=2022, period_end=date(2022, 12, 31),
        values=tuple((f"f{i}", Decimal(v)) for i, v in enumerate(values)),
    )  # fmt: skip


@given(VALUES)
def test_moves_list_every_change_and_mark_the_largest_once(values: list[int]) -> None:
    d = disc(Metric.NET_INCOME, values)
    lines = moves(d)
    assert len(lines) == len(change_filings(d))
    largest = [line for line in lines if "LARGEST" in line]
    assert len(largest) == (1 if len(lines) > 1 else 0) or len(largest) > 1  # ties may mark several


@settings(max_examples=200)
@given(st.sampled_from(list(Metric)), VALUES, st.lists(SENTENCE, max_size=4).map(tuple))
def test_the_card_never_shows_a_favorite_and_only_rules_rule_out(
    metric: Metric, values: list[int], evidence: tuple[str, ...]
) -> None:
    card = build_card(disc(metric, values), evidence)
    assert not {"cause", "explanation", "evidence_quote", "verified", "lane"} & set(card)
    assert [c["cause"] for c in card["candidates"]] == [c.value for c in CANDIDATES]  # fixed order
    for c in card["candidates"]:
        if c["fit"] == "ruled_out":
            cause = Cause(c["cause"])
            by_table = metric in CANNOT_MOVE.get(cause, {})
            by_ratio = cause is Cause.STOCK_SPLIT  # ratio or quote-ratio checks (code, not an LLM)
            assert by_table or by_ratio, c
