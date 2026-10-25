"""The facts ledger: the system's one real side effect, made idempotent.

- Facts are UPSERTed by their stable ids, so writing them twice changes nothing.
- Resolutions are an append-only history: every run's answer for a case is kept, with who
  decided, when and in which run. `resolutions` is a view over it, never a table that a later
  run overwrites. Precedence: a human pick beats an escalation, which beats an automatic answer;
  among equals the newest wins. So a stale or automatic run can never erase a reviewer's pick,
  and a later human pick supersedes an earlier one explicitly (found by review: the last thread
  to publish used to overwrite the row, whoever decided it).
- A case is the discrepancy AND the filings it compares: a new 10-K that restates the same year
  again is a new case, not an update of the old one.
- Each run's publication is recorded under its thread id in the SAME transaction as the writes,
  so a run publishes exactly once: a replay, a resume, or a client retry finds the record and
  returns the original publication instead of writing again.

SQLite here; the same SQL runs on Postgres (ON CONFLICT is standard), which is a config swap.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filings_agent.domain import ExtractedFact
from filings_agent.steps.reconcile import Resolution

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL, metric TEXT NOT NULL, fiscal_year INTEGER NOT NULL,
    period_end TEXT NOT NULL, value TEXT NOT NULL, accession_no TEXT NOT NULL,
    statement TEXT NOT NULL, row_label TEXT NOT NULL, printed_value TEXT NOT NULL,
    scale TEXT NOT NULL, model TEXT NOT NULL, thread_id TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS resolution_history (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    discrepancy_id TEXT NOT NULL, filing_set TEXT NOT NULL,
    cause TEXT NOT NULL, explanation TEXT NOT NULL, evidence_quote TEXT NOT NULL,
    material INTEGER NOT NULL, verified INTEGER NOT NULL,
    use_value TEXT NOT NULL, use_accession TEXT NOT NULL,
    decision TEXT NOT NULL, rank INTEGER NOT NULL, actor TEXT, decided_at TEXT,
    thread_id TEXT NOT NULL, recorded_at TEXT NOT NULL,
    UNIQUE (discrepancy_id, filing_set, thread_id)
);
CREATE INDEX IF NOT EXISTS resolution_history_case
    ON resolution_history (discrepancy_id, filing_set);
CREATE TABLE IF NOT EXISTS corrections (
    discrepancy_id TEXT NOT NULL, thread_id TEXT NOT NULL,
    model_cause TEXT NOT NULL, correct_cause TEXT NOT NULL, note TEXT NOT NULL,
    task_json TEXT NOT NULL, run_id TEXT, exported INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (discrepancy_id, thread_id)
);
CREATE TABLE IF NOT EXISTS memos (
    thread_id TEXT PRIMARY KEY, markdown TEXT NOT NULL, verified INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS publications (
    thread_id TEXT PRIMARY KEY,
    published_at TEXT NOT NULL, facts INTEGER NOT NULL, resolutions INTEGER NOT NULL
);
"""

# The current answer per case, derived from the history (never stored separately). The winner is
# the highest rank, then the newest; `later_escalations` says a reviewer escalated the case after
# the answer that stands (shown, not applied: only a human pick supersedes a human pick).
CURRENT_VIEW = """
CREATE VIEW IF NOT EXISTS resolutions AS
SELECT h.discrepancy_id, h.filing_set, h.cause, h.explanation, h.evidence_quote, h.material,
       h.verified, h.use_value, h.use_accession, h.decision, h.actor, h.decided_at, h.thread_id,
       h.recorded_at AS updated_at,
       (SELECT COUNT(*) FROM resolution_history l
         WHERE l.discrepancy_id = h.discrepancy_id AND l.filing_set = h.filing_set
           AND l.seq > h.seq AND l.decision = 'escalate') AS later_escalations
FROM resolution_history h
WHERE h.seq = (SELECT w.seq FROM resolution_history w
                WHERE w.discrepancy_id = h.discrepancy_id AND w.filing_set = h.filing_set
                ORDER BY w.rank DESC, w.seq DESC LIMIT 1);
"""

# Decision -> precedence. A human pick (a cause chosen on the card) outranks an escalation, which
# outranks an answer nobody looked at.
RANK = {"accept": 2, "use_original": 2, "escalate": 1, "auto": 0}


@dataclass(frozen=True)
class Decided:
    """One case's answer as a run publishes it."""

    resolution: Resolution
    decision: str  # accept | use_original | escalate | auto
    filing_set: str = ""  # the accession numbers the discrepancy compares, sorted, comma-joined
    actor: str | None = None  # the authenticated reviewer; None for "auto"
    decided_at: str | None = None  # when the reviewer submitted it (ISO 8601, UTC)


def filing_set(accessions: Iterable[str]) -> str:
    return ",".join(sorted(set(accessions)))


@dataclass(frozen=True)
class Correction:
    discrepancy_id: str
    model_cause: str
    correct_cause: str
    note: str
    task_json: str  # the reconcile input (discrepancy + evidence), replayable as a test
    run_id: str | None  # LangSmith root run of the invocation that published it


@dataclass(frozen=True)
class Publication:
    thread_id: str
    published_at: str
    facts: int
    resolutions: int
    newly_published: bool  # False when this call found an earlier publication and did nothing


class Ledger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._conn.executescript(SCHEMA)
        self._migrate_resolutions()
        self._conn.executescript(CURRENT_VIEW)
        self._lock = threading.Lock()

    def _migrate_resolutions(self) -> None:
        """Ledgers written before the history existed have a `resolutions` TABLE (one row per
        case, last writer wins). Move its rows into the history once and drop it, so the view
        can take its name. What the old table overwrote is gone; what it held is kept."""
        kind = self._conn.execute(
            "SELECT type FROM sqlite_master WHERE name = 'resolutions'"
        ).fetchone()
        if kind is None or kind[0] != "table":
            return
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO resolution_history (discrepancy_id, filing_set, cause,"
                " explanation, evidence_quote, material, verified, use_value, use_accession,"
                " decision, rank, actor, decided_at, thread_id, recorded_at)"
                " SELECT discrepancy_id, '', cause, explanation, evidence_quote, material,"
                " verified, use_value, use_accession, decision,"
                " CASE decision WHEN 'accept' THEN 2 WHEN 'use_original' THEN 2"
                " WHEN 'escalate' THEN 1 ELSE 0 END, NULL, NULL, thread_id, updated_at"
                " FROM resolutions"
            )
            self._conn.execute("DROP TABLE resolutions")

    def publish(
        self,
        thread_id: str,
        facts: Iterable[ExtractedFact],
        resolutions: Iterable[Decided],
        memo: tuple[str, bool] | None = None,
        corrections: Iterable[Correction] = (),
    ) -> Publication:
        """Write facts and append this run's answers, once per thread. Safe to call repeatedly."""
        now = datetime.now(UTC).isoformat()
        facts, resolutions = list(facts), list(resolutions)
        with self._lock, self._conn:  # one transaction: all of it or none of it
            # Take the write lock BEFORE checking: otherwise two processes can both see "not
            # published", and the second one fails on the publications key instead of returning
            # the first one's publication.
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT published_at, facts, resolutions FROM publications WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if row is not None:
                return Publication(thread_id, row[0], row[1], row[2], newly_published=False)
            self._conn.executemany(
                """INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(fact_id) DO UPDATE SET value=excluded.value,
                     row_label=excluded.row_label, printed_value=excluded.printed_value,
                     scale=excluded.scale, model=excluded.model,
                     thread_id=excluded.thread_id, updated_at=excluded.updated_at""",
                [
                    (
                        f.id,
                        f.ticker,
                        f.metric.value,
                        f.fiscal_year,
                        f.period_end.isoformat(),
                        str(f.value),
                        f.accession_no,
                        f.statement,
                        f.row_label,
                        f.printed_value,
                        f.scale,
                        f.model,
                        thread_id,
                        now,
                    )  # fmt: skip
                    for f in facts
                ],
            )
            self._conn.executemany(
                "INSERT OR IGNORE INTO resolution_history (discrepancy_id, filing_set, cause,"
                " explanation, evidence_quote, material, verified, use_value, use_accession,"
                " decision, rank, actor, decided_at, thread_id, recorded_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        d.resolution.discrepancy_id,
                        d.filing_set,
                        d.resolution.cause.value,
                        d.resolution.explanation,
                        d.resolution.evidence_quote,
                        int(d.resolution.material),
                        int(d.resolution.verified),
                        str(d.resolution.use_value),
                        d.resolution.use_accession,
                        d.decision,
                        RANK.get(d.decision, 0),
                        d.actor,
                        d.decided_at,
                        thread_id,
                        now,
                    )  # fmt: skip
                    for d in resolutions
                ],
            )
            self._conn.executemany(
                "INSERT OR IGNORE INTO corrections"
                " (discrepancy_id, thread_id, model_cause, correct_cause, note, task_json, run_id)"
                " VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        c.discrepancy_id,
                        thread_id,
                        c.model_cause,
                        c.correct_cause,
                        c.note,
                        c.task_json,
                        c.run_id,
                    )  # fmt: skip
                    for c in corrections
                ],
            )
            if memo is not None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO memos VALUES (?,?,?)",
                    (thread_id, memo[0], int(memo[1])),
                )
            self._conn.execute(
                "INSERT INTO publications VALUES (?,?,?,?)",
                (thread_id, now, len(facts), len(resolutions)),
            )
        return Publication(thread_id, now, len(facts), len(resolutions), newly_published=True)

    def history(self, discrepancy_id: str) -> list[dict[str, Any]]:
        """Every answer ever published for this discrepancy, oldest first (all filing sets)."""
        return self._rows(
            "SELECT * FROM resolution_history WHERE discrepancy_id = ? ORDER BY seq",
            (discrepancy_id,),
        )

    def current(self, discrepancy_id: str) -> list[dict[str, Any]]:
        """The answer that stands, one per filing set the discrepancy was seen with."""
        return self._rows(
            "SELECT * FROM resolutions WHERE discrepancy_id = ? ORDER BY filing_set",
            (discrepancy_id,),
        )

    def _rows(self, sql: str, args: tuple[Any, ...]) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(sql, args)
            names = [c[0] for c in cur.description]
            return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]

    def unexported_corrections(self) -> list[tuple[str, str, Correction]]:
        """(discrepancy_id, thread_id, correction) not yet pushed to LangSmith."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT discrepancy_id, thread_id, model_cause, correct_cause, note, task_json,"
                " run_id FROM corrections WHERE exported = 0"
            ).fetchall()
        return [(r[0], r[1], Correction(r[0], r[2], r[3], r[4], r[5], r[6])) for r in rows]

    def mark_exported(self, keys: list[tuple[str, str]]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                "UPDATE corrections SET exported = 1 WHERE discrepancy_id = ? AND thread_id = ?",
                keys,
            )

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                # table names come from the fixed tuple below, never from input
                t: int(self._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])  # nosec B608
                for t in (
                    "facts",
                    "resolutions",
                    "resolution_history",
                    "memos",
                    "corrections",
                    "publications",
                )
            }

    def close(self) -> None:
        self._conn.close()
