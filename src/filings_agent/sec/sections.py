"""Locate the three primary financial statements inside a 10-K's markdown.

Why not split by "Item 8"? Some filers (NVIDIA) put the statements under Item 15 and leave
Item 8 as a one-line pointer. Headings also vary: "Consolidated Statements of Operations",
"STATEMENT OF EARNINGS", "Statement of Financial Position" (GE). So we find candidate headings
by synonym, then keep the candidate followed by the biggest numeric table.

This is the default *locator strategy*. It is swappable (e.g. an LLM locator) and evaluated
like any other step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class StatementKind(StrEnum):
    INCOME = "income_statement"
    BALANCE = "balance_sheet"
    CASH_FLOW = "cash_flow_statement"


class Scale(StrEnum):
    UNITS = "units"
    THOUSANDS = "thousands"
    MILLIONS = "millions"
    BILLIONS = "billions"


SCALE_FACTOR: dict[Scale, int] = {
    Scale.UNITS: 1,
    Scale.THOUSANDS: 1_000,
    Scale.MILLIONS: 1_000_000,
    Scale.BILLIONS: 1_000_000_000,
}

# Matched against a *compacted* heading (lowercase, no spaces/markup), because HTML->markdown
# conversion splits words: Microsoft's title arrives as "**INC** **OME STATEMENTS**".
_HEADINGS: dict[StatementKind, re.Pattern[str]] = {
    StatementKind.INCOME: re.compile(
        r"statements?of(?:consolidated)?(?:income|operations|earnings)"
        r"|^(?:item\d+)?(?:consolidated)?incomestatements?"  # Microsoft: "INCOME STATEMENTS"
    ),
    StatementKind.BALANCE: re.compile(
        r"balancesheets?|statements?of(?:consolidated)?financialposition"
    ),
    StatementKind.CASH_FLOW: re.compile(
        r"statements?of(?:consolidated)?cashflows?|^(?:item\d+)?(?:consolidated)?cashflows?statements?"
    ),
}
_EXCLUDE = re.compile(r"comprehensive|notes?to|parenthetical|seeaccompanying|index")


def _compact(line: str) -> str:
    return re.sub(r"[\s*_#\\]+", "", line).lower()


# "(In millions)", "(MILLIONS, EXCEPT PER SHARE DATA)" (Pfizer), "December 31 (In millions)" (GE)
_SCALE = re.compile(r"(?:\bin\s+|\(\s*)(thousands|millions|billions)\b", re.I)
# A real statement contains its signature line; an MD&A table with the same title usually doesn't.
_SIGNATURE: dict[StatementKind, re.Pattern[str]] = {
    StatementKind.INCOME: re.compile(r"net\s+(?:income|earnings|loss)", re.I),
    StatementKind.BALANCE: re.compile(r"total\s+assets", re.I),
    StatementKind.CASH_FLOW: re.compile(r"operating\s+activities", re.I),
}
_NUMERIC_ROW = re.compile(r"^\|[^\n]*\d[\d,]{2,}", re.M)
_LOOKAHEAD = 600  # a heading's table must start within this many chars
_SCALE_WINDOW = 1_200
_MAX_SECTION = 16_000


@dataclass(frozen=True)
class Section:
    kind: StatementKind
    heading: str
    text: str
    start: int
    scale: Scale | None

    @property
    def chars(self) -> int:
        return len(self.text)


def _heading_lines(md: str) -> list[tuple[int, str]]:
    """Where statement titles live: short non-table lines ("**Consolidated Balance Sheets**"),
    or the first cell of an otherwise-empty table row ("| STATEMENT OF OPERATIONS |  |  |", GE)."""
    out: list[tuple[int, str]] = []
    pos = 0
    for line in md.splitlines(keepends=True):
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if cells and cells[0] and not any(cells[1:]) and len(cells[0]) <= 120:
                out.append((pos, cells[0].strip("*_ ")))
        else:
            stripped = s.strip("*#_ ").strip()
            if stripped and len(stripped) <= 120:
                out.append((pos, stripped))
        pos += len(line)
    return out


def _is_table_continuation(line: str) -> bool:
    """Table rows, blank lines, and wrapped header fragments like '2025 |' (Apple)."""
    s = line.strip()
    return not s or s.startswith("|") or (len(s) <= 40 and ("|" in s or s[0].isdigit()))


def _is_title_row(line: str) -> bool:
    """'| STATEMENT OF OPERATIONS |  |  |': first cell filled, the rest empty."""
    s = line.strip()
    if not s.startswith("|"):
        return False
    cells = [c.strip() for c in s.strip("|").split("|")]
    return bool(cells and cells[0]) and not any(cells[1:])


def _table_after(md: str, start: int) -> tuple[int, int, int] | None:
    """(table_start, table_end, numeric_rows) for the first table within the lookahead."""
    if md.startswith("|", start):
        t0 = start
    else:
        found = md.find("\n|", start, start + _LOOKAHEAD)
        if found < 0:
            return None
        t0 = found + 1
    t1 = t0
    after_blank = False
    while t1 < len(md):
        nl = md.find("\n", t1)
        nl = len(md) if nl < 0 else nl
        line = md[t1:nl]
        # A blank line then a title row starts a NEW table; don't merge it into this one.
        if not _is_table_continuation(line) or (after_blank and _is_title_row(line)):
            break
        after_blank = not line.strip()
        t1 = nl + 1
    rows = len(_NUMERIC_ROW.findall(md, t0, t1))
    return (t0, t1, rows)


def locate_statements(md: str) -> dict[StatementKind, Section]:
    # Rank by (has signature line, has scale marker, numeric rows). Real statements say
    # "(In millions)" and contain their signature line; indexes and MD&A excerpts usually don't.
    best: dict[StatementKind, tuple[tuple[bool, bool, int], Section]] = {}
    for pos, line in _heading_lines(md):
        compact = _compact(line)
        if _EXCLUDE.search(compact):
            continue
        for kind, pat in _HEADINGS.items():
            if not pat.search(compact):
                continue
            table = _table_after(md, pos)
            if table is None:
                continue
            _, t_end, rows = table
            if rows < 5:
                continue
            end = min(t_end, pos + _MAX_SECTION)
            text = md[pos:end]
            m = _SCALE.search(md, pos, min(pos + _SCALE_WINDOW, len(md)))
            scale = Scale(m.group(1).lower()) if m else None
            section = Section(kind=kind, heading=line, text=text, start=pos, scale=scale)
            score = (bool(_SIGNATURE[kind].search(text)), scale is not None, rows)
            if kind not in best or score > best[kind][0]:
                best[kind] = (score, section)
    return {k: s for k, (_, s) in best.items()}
