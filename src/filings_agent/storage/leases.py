"""Thread leases: at most one process runs a thread at a time.

The API's per-thread asyncio locks only serialize requests inside one server process. With more
than one worker, or the CLI and the server on the same data dir, two processes could both see
"not started" and run the same thread: double spend and interleaved checkpoints (found by review).
A lease is a row in a shared database, taken in a write transaction, so the check and the claim
are one step for every process.

A lease expires: a process that dies mid-run frees its threads after `ttl_s` without anyone
cleaning up. A live run renews its lease from a heartbeat thread well before that.

SQLite here (BEGIN IMMEDIATE takes the database write lock); on Postgres the same table with
`INSERT ... ON CONFLICT ... WHERE expires_at < now()` does the same job.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


class ThreadBusy(RuntimeError):
    """Another process holds this thread's lease: it is running there right now."""


class Leases:
    def __init__(self, path: Path, ttl_s: float = 120.0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self.ttl_s = ttl_s
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS leases (thread_id TEXT PRIMARY KEY,"
                " owner TEXT NOT NULL, expires_at REAL NOT NULL)"
            )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection]:
        # One short-lived connection per operation: safe from any thread (the heartbeat's too).
        conn = sqlite3.connect(str(self._path), timeout=30, isolation_level=None)
        try:
            yield conn
        finally:
            conn.close()

    def acquire(self, thread_id: str, owner: str) -> bool:
        """Take the lease if it's free, expired, or already ours (re-entrant). False if held."""
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT owner, expires_at FROM leases WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            if row is not None and row[0] != owner and row[1] > now:
                conn.execute("ROLLBACK")
                return False
            conn.execute(
                "INSERT INTO leases VALUES (?, ?, ?) ON CONFLICT(thread_id) DO UPDATE SET"
                " owner = excluded.owner, expires_at = excluded.expires_at",
                (thread_id, owner, now + self.ttl_s),
            )
            conn.execute("COMMIT")
        return True

    def renew(self, thread_id: str, owner: str) -> bool:
        """Extend our lease. False if it was lost (expired and taken by someone else)."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE leases SET expires_at = ? WHERE thread_id = ? AND owner = ?",
                (time.time() + self.ttl_s, thread_id, owner),
            )
            return cur.rowcount == 1

    def release(self, thread_id: str, owner: str) -> None:
        """Give the lease back. Only its owner can; releasing twice is a no-op."""
        with self._connect() as conn:
            conn.execute("DELETE FROM leases WHERE thread_id = ? AND owner = ?", (thread_id, owner))

    def holder(self, thread_id: str) -> str | None:
        """Who is running this thread right now (an unexpired lease), if anyone."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT owner FROM leases WHERE thread_id = ? AND expires_at > ?",
                (thread_id, time.time()),
            ).fetchone()
        return str(row[0]) if row else None

    @contextmanager
    def hold(self, thread_id: str, owner: str) -> Generator[None]:
        """Hold the lease for the duration of a run, renewing it in the background."""
        if not self.acquire(thread_id, owner):
            raise ThreadBusy(f"thread {thread_id} is running in another process")
        stop = threading.Event()

        def heartbeat() -> None:
            while not stop.wait(self.ttl_s / 4):
                if not self.renew(thread_id, owner):
                    # Only after a stall longer than the ttl (a suspended process). The run can't
                    # be stopped from here; say so loudly instead of continuing silently.
                    print(f"[leases] lost the lease on {thread_id}", file=sys.stderr)
                    return

        beat = threading.Thread(target=heartbeat, name=f"lease-{thread_id}", daemon=True)
        beat.start()
        try:
            yield
        finally:
            stop.set()
            beat.join()
            self.release(thread_id, owner)
