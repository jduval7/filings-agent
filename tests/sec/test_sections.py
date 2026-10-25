"""Locator tests built from real formatting quirks seen in 10-K markdown."""

from __future__ import annotations

from decimal import Decimal

from filings_agent.datasets import is_answerable
from filings_agent.sec.sections import Scale, StatementKind, locate_statements

ROWS = "\n".join(f"| Line {i} | {1000 + i:,} |  | {900 + i:,} |  |" for i in range(8))


INCOME_ROWS = ROWS + "\n| Net income | $ | 5,000 | $ | 4,000 |"


BALANCE_ROWS = ROWS + "\n| Total assets | $ | 9,000 | $ | 8,000 |"


def test_standard_bold_headings() -> None:
    md = (
        "**Consolidated Statements of Income**\n\n**(In millions)**\n\n"
        f"|  | 2025 |  | 2024 |  |\n| --- | --- | --- | --- | --- |\n{INCOME_ROWS}\n\n"
        "**Consolidated Balance Sheets**\n\n(In millions)\n\n"
        f"|  | 2025 |  | 2024 |  |\n{BALANCE_ROWS}\n"
    )
    secs = locate_statements(md)
    assert secs[StatementKind.INCOME].scale is Scale.MILLIONS
    assert "Net income" in secs[StatementKind.INCOME].text
    assert "Total assets" in secs[StatementKind.BALANCE].text


def test_title_inside_first_table_row_ge_style() -> None:
    md = (
        "Report of auditors.\n\n"
        "| STATEMENT OF OPERATIONS |  |  |  |  |\n"
        "| (In millions; per-share amounts in dollars) | 2025 |  | 2024 |  |\n"
        f"{INCOME_ROWS}\n"
    )
    sec = locate_statements(md)[StatementKind.INCOME]
    assert sec.heading == "STATEMENT OF OPERATIONS"
    assert sec.scale is Scale.MILLIONS


def test_split_word_heading_microsoft_style() -> None:
    md = (
        "Item 8 **CASH FLOWS S** **TATEMENTS**\n\n"
        "| (In millions) Year Ended June 30, | 2025 |  | 2024 |  |\n"
        "| Net cash from operating activities | 136,162 |  | 118,548 |  |\n"
        f"{ROWS}\n"
    )
    assert StatementKind.CASH_FLOW in locate_statements(md)


def test_wrapped_header_row_apple_style() -> None:
    md = (
        "**CONSOLIDATED BALANCE SHEETS**\n\n(In millions)\n\n"
        "|  | September 27,\n\n\n2025 |  | September 28,\n\n\n2024 |  |\n"
        f"{BALANCE_ROWS}\n"
    )
    sec = locate_statements(md)[StatementKind.BALANCE]
    assert "Total assets" in sec.text


def test_index_decoy_loses_to_real_statement() -> None:
    md = (
        "## Statement of Operations for the years ended 2025, 2024\n\n"
        "## Statement of Financial Position at December 31\n\n"
        "| Exhibit | Description |\n" + ROWS + "\n\n"
        "| STATEMENT OF OPERATIONS |  |  |  |  |\n| (In millions) | 2025 |  | 2024 |  |\n"
        f"{INCOME_ROWS}\n"
    )
    sec = locate_statements(md)[StatementKind.INCOME]
    assert sec.heading == "STATEMENT OF OPERATIONS"


def test_comprehensive_income_is_not_the_income_statement() -> None:
    md = "**Consolidated Statements of Comprehensive Income**\n\n(In millions)\n\n" + INCOME_ROWS
    assert StatementKind.INCOME not in locate_statements(md)


def test_answerable_handles_mixed_scales_and_decimals() -> None:
    # Apple: shares in thousands inside a "millions" table
    assert is_answerable(Decimal(16_864_919_000), "| Diluted | 16,864,919 |")
    # Nike: shares in millions with one decimal
    assert is_answerable(Decimal(1_609_400_000), "| Diluted | 1,609.4 |")
    assert not is_answerable(Decimal(1_609_400_000), "| Diluted | 1,610.8 |")


def test_sections_detect_thousands_and_missing_scale() -> None:
    rows = "\n".join(f"| L{i} | {1000 + i:,} | {900 + i:,} |" for i in range(8))
    md = f"**Consolidated Balance Sheets**\n(Dollars in thousands)\n|  | 2024 | 2023 |\n{rows}\n" \
         "| Total assets | 9,000 | 8,000 |\n"  # fmt: skip
    sec = locate_statements(md)[StatementKind.BALANCE]
    assert sec.scale is Scale.THOUSANDS and sec.chars == len(sec.text)
    assert locate_statements("no statements here") == {}
