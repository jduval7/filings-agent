"""Ledger: upsert semantics, exactly-once publish and the decision history, directly."""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

from filings_agent.domain import ExtractedFact, Metric
from filings_agent.steps.reconcile import Cause, Resolution
from filings_agent.storage.ledger import Correction, Decided, Ledger


def _fact(value: int, label: str = "Revenue") -> ExtractedFact:
    return ExtractedFact(
        ticker="GE", metric=Metric.REVENUE, fiscal_year=2024, period_end=date(2024, 12, 31),
        value=Decimal(value), accession_no="A1", statement="income_statement", row_label=label,
        printed_value="x", scale="millions", model="m",
    )  # fmt: skip


RES = Resolution(
    "GE:revenue:FY2024", Cause.STOCK_SPLIT, "e", "q", True, True, Decimal(1), "A1", True
)


def test_publish_once_per_thread_even_with_different_payloads(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    first = ledger.publish("t1", [_fact(1)], [Decided(RES, "accept")], ("# memo", True))
    second = ledger.publish("t1", [_fact(2), _fact(3)], [], None)  # a replay with drifted state
    assert first.newly_published and not second.newly_published
    assert (second.facts, second.resolutions) == (1, 1)  # reports the ORIGINAL publication
    assert ledger.counts() == {"facts": 1, "resolutions": 1, "resolution_history": 1, "memos": 1,
                               "corrections": 0, "publications": 1}  # fmt: skip


def test_facts_upsert_across_threads_by_fact_id(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    ledger.publish("t1", [_fact(1, "Revenue")], [])
    ledger.publish("t2", [_fact(1, "Total revenues")], [])  # same fact id, newer run
    assert ledger.counts()["facts"] == 1 and ledger.counts()["publications"] == 2
    ledger.close()
    import sqlite3

    row = (
        sqlite3.connect(str(tmp_path / "l.sqlite"))
        .execute("SELECT row_label, thread_id FROM facts")
        .fetchone()
    )
    assert row == ("Total revenues", "t2")  # latest run's provenance wins


def test_corrections_export_bookkeeping(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    c = Correction("GE:revenue:FY2024", "stock_split", "accounting_change", "n", "{}", None)
    ledger.publish("t1", [], [], corrections=[c])
    ledger.publish("t1", [], [], corrections=[c])  # replay: no duplicate
    (pending,) = ledger.unexported_corrections()
    assert (
        pending[:2] == ("GE:revenue:FY2024", "t1")
        and pending[2].correct_cause == "accounting_change"
    )
    ledger.mark_exported([("GE:revenue:FY2024", "t1")])
    assert ledger.unexported_corrections() == []


def test_concurrent_publishers_on_one_file_publish_once_and_never_crash(tmp_path: Path) -> None:
    """Review finding: two processes could both see "not published"; the second then failed on
    the publications key. Separate connections stand in for separate processes."""
    path = tmp_path / "l.sqlite"
    ledgers = [Ledger(path) for _ in range(8)]
    start = threading.Barrier(len(ledgers))
    results: list[bool] = []
    errors: list[BaseException] = []

    def go(ledger: Ledger) -> None:
        start.wait()
        try:
            results.append(
                ledger.publish("t1", [_fact(1)], [Decided(RES, "accept")]).newly_published
            )
        except BaseException as e:  # noqa: BLE001 - recorded and asserted below
            errors.append(e)

    threads = [threading.Thread(target=go, args=(x,)) for x in ledgers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and sorted(results) == [False] * 7 + [True]
    assert ledgers[0].counts()["publications"] == 1


GE_NI = "GE:net_income:FY2022"
FILINGS = "0000040545-23-000012,0000040545-24-000014"
PICKED = Resolution(GE_NI, Cause.ACCOUNTING_CHANGE, "Cause picked by reviewer.", "q", True, True,
                    Decimal(1), "A2", False)  # fmt: skip
ESCALATED = replace(PICKED, cause=Cause.UNKNOWN, explanation="Escalated by reviewer.")


def test_a_later_run_never_erases_a_reviewers_pick(tmp_path: Path) -> None:
    """Review finding (reproduced): thread A's human pick was overwritten by thread B's
    escalation, and no history was kept. Now both are kept and the pick still stands."""
    ledger = Ledger(tmp_path / "l.sqlite")
    ledger.publish("A", [], [Decided(PICKED, "accept", FILINGS, "alice", "2026-10-01T00:00:00Z")])
    ledger.publish("B", [], [Decided(ESCALATED, "escalate", FILINGS, "bob", "2026-10-20T00:00Z")])
    ledger.publish("C", [], [Decided(replace(PICKED, cause=Cause.STOCK_SPLIT), "auto", FILINGS)])
    (now,) = ledger.current(GE_NI)
    assert (now["cause"], now["actor"], now["thread_id"]) == ("accounting_change", "alice", "A")
    assert now["later_escalations"] == 1  # bob's escalation is visible, not applied
    history = ledger.history(GE_NI)
    assert [(h["thread_id"], h["decision"], h["actor"]) for h in history] == [
        ("A", "accept", "alice"), ("B", "escalate", "bob"), ("C", "auto", None),
    ]  # fmt: skip


def test_a_later_human_pick_supersedes_an_earlier_one(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    ledger.publish("A", [], [Decided(PICKED, "accept", FILINGS, "alice")])
    ledger.publish("B", [], [Decided(replace(PICKED, cause=Cause.RECLASSIFICATION), "use_original",
                                     FILINGS, "bob")])  # fmt: skip
    (now,) = ledger.current(GE_NI)
    assert (now["cause"], now["actor"], now["later_escalations"]) == ("reclassification", "bob", 0)


def test_an_escalation_stands_until_a_human_picks(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    ledger.publish("A", [], [Decided(ESCALATED, "escalate", FILINGS, "bob")])
    ledger.publish("B", [], [Decided(PICKED, "auto", FILINGS)])  # an unreviewed run: below it
    assert ledger.current(GE_NI)[0]["decision"] == "escalate"
    ledger.publish("C", [], [Decided(PICKED, "accept", FILINGS, "alice")])
    assert ledger.current(GE_NI)[0]["actor"] == "alice"


def test_a_new_filing_makes_a_new_case(tmp_path: Path) -> None:
    """A new 10-K restating FY2022 again compares different filings: it must not be blocked by,
    or overwrite, the answer for the old comparison."""
    ledger = Ledger(tmp_path / "l.sqlite")
    ledger.publish("A", [], [Decided(PICKED, "accept", FILINGS, "alice")])
    newer = FILINGS + ",0000040545-25-000020"
    ledger.publish("B", [], [Decided(replace(PICKED, cause=Cause.STOCK_SPLIT), "auto", newer)])
    assert [(c["filing_set"], c["cause"]) for c in ledger.current(GE_NI)] == [
        (FILINGS, "accounting_change"), (newer, "stock_split"),
    ]  # fmt: skip


def test_an_old_ledger_is_migrated_into_the_history(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "l.sqlite"
    old = sqlite3.connect(str(path))
    old.execute(
        "CREATE TABLE resolutions (discrepancy_id TEXT PRIMARY KEY, cause TEXT NOT NULL,"
        " explanation TEXT NOT NULL, evidence_quote TEXT NOT NULL, material INTEGER NOT NULL,"
        " verified INTEGER NOT NULL, use_value TEXT NOT NULL, use_accession TEXT NOT NULL,"
        " decision TEXT NOT NULL, thread_id TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    old.execute(
        "INSERT INTO resolutions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (GE_NI, "accounting_change", "e", "q", 1, 1, "1", "A2", "accept", "t0", "then"),
    )
    old.commit()
    old.close()
    ledger = Ledger(path)
    (row,) = ledger.history(GE_NI)
    assert (row["decision"], row["rank"], row["thread_id"], row["recorded_at"]) == (
        "accept",
        2,
        "t0",
        "then",
    )
    assert ledger.current(GE_NI)[0]["cause"] == "accounting_change"
    Ledger(path).close()  # opening it again is a no-op
