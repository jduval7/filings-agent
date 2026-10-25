"""Thread leases: one process runs a thread at a time; dead holders expire; live ones renew."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from filings_agent.storage.leases import Leases, ThreadBusy


def test_one_holder_at_a_time_and_reentrant_for_the_owner(tmp_path: Path) -> None:
    a, b = Leases(tmp_path / "l.sqlite"), Leases(tmp_path / "l.sqlite")  # two "processes"
    assert a.acquire("t", "A") and a.acquire("t", "A")  # the owner re-enters
    assert not b.acquire("t", "B") and b.holder("t") == "A"
    b.release("t", "B")  # not the owner: no effect
    assert a.holder("t") == "A"
    a.release("t", "A")
    assert b.acquire("t", "B") and a.holder("t") == "B"


def test_concurrent_claims_have_exactly_one_winner(tmp_path: Path) -> None:
    path = tmp_path / "l.sqlite"
    Leases(path)
    start = threading.Barrier(8)
    won: list[bool] = []

    def claim(i: int) -> None:
        start.wait()
        won.append(Leases(path).acquire("t", f"worker-{i}"))

    threads = [threading.Thread(target=claim, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(won) == [False] * 7 + [True]


def test_a_dead_holder_expires(tmp_path: Path) -> None:
    dead = Leases(tmp_path / "l.sqlite", ttl_s=0.05)
    assert dead.acquire("t", "crashed")
    time.sleep(0.1)
    assert dead.holder("t") is None
    assert Leases(tmp_path / "l.sqlite").acquire("t", "next")
    assert not dead.renew("t", "crashed")  # it lost the lease and can tell


def test_hold_renews_while_running_and_releases_at_the_end(tmp_path: Path) -> None:
    leases = Leases(tmp_path / "l.sqlite", ttl_s=0.2)
    with leases.hold("t", "me"):
        time.sleep(0.5)  # longer than the ttl: only the heartbeat keeps it
        assert leases.holder("t") == "me"
        with pytest.raises(ThreadBusy), Leases(tmp_path / "l.sqlite").hold("t", "someone-else"):
            pass  # pragma: no cover
    assert leases.holder("t") is None


def test_a_lost_lease_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    leases = Leases(tmp_path / "l.sqlite", ttl_s=0.2)
    with leases.hold("t", "me"):
        leases.release("t", "me")  # stands in for: expired during a stall, taken by another
        assert Leases(tmp_path / "l.sqlite").acquire("t", "other")
        time.sleep(0.15)
    assert "lost the lease on t" in capsys.readouterr().err
