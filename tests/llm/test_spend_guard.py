"""Spend guard: caps refuse calls before they're sent, costs are recorded after, both ledgers."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.stub import Stubber

from filings_agent.llm.spend_guard import (
    UNKNOWN_PRICE,
    Ledger,
    SpendGuard,
    SpendLimitExceeded,
    _est_tokens,  # pyright: ignore[reportPrivateUsage]
    install,
)
from tests.conftest import make_cfg

NOVA = "us.amazon.nova-micro-v1:0"  # price (0.035, 0.14) per 1M tokens in configs/default.yaml


def _client() -> Any:
    return boto3.client(  # pyright: ignore[reportUnknownMemberType]
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


def _converse_response(in_tok: int, out_tok: int) -> dict[str, Any]:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": in_tok, "outputTokens": out_tok, "totalTokens": in_tok + out_tok},
        "metrics": {"latencyMs": 10},
    }


def _converse(client: Any, max_tokens: int = 100) -> None:
    client.converse(
        modelId=NOVA,
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        inferenceConfig={"maxTokens": max_tokens},
    )


def test_estimate_tokens_is_conservative() -> None:
    assert _est_tokens("x" * 300) == 100
    assert _est_tokens({"a": "b"}) >= 1


def test_ledger_accumulates_and_rolls_by_day_and_month(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.json")
    ledger.add("m1", 0.5)
    ledger.add("m1", 0.25)
    ledger.add("m2", 1.0)
    snap = ledger.snapshot()
    assert snap["total_usd"] == pytest.approx(1.75)
    assert snap["calls"] == 3
    assert snap["by_model_month"] == {"m1": pytest.approx(0.75), "m2": pytest.approx(1.0)}
    data = json.loads((tmp_path / "l.json").read_text())
    data["month"], data["day"] = "1999-01", "1999-01-01"  # pretend it's from long ago
    (tmp_path / "l.json").write_text(json.dumps(data))
    rolled = ledger.snapshot()
    assert rolled["month_usd"] == 0 and rolled["day_usd"] == 0
    assert rolled["total_usd"] == pytest.approx(1.75)  # lifetime total survives the roll


def test_a_missing_ledger_starts_clean(tmp_path: Path) -> None:
    assert Ledger(tmp_path / "l.json").snapshot()["total_usd"] == 0


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_a_corrupt_ledger_refuses_calls_and_is_not_overwritten(
    tmp_path: Path, content: str
) -> None:
    """Review finding: a corrupt ledger read as $0 spent (every cap passes) and the next write
    erased the real total. It now fails closed."""
    path = tmp_path / "l.json"
    path.write_text(content)
    with pytest.raises(SpendLimitExceeded, match="unreadable|not a JSON object"):
        Ledger(path).snapshot()
    with pytest.raises(SpendLimitExceeded):
        Ledger(path).add("m", 1.0)
    assert path.read_text() == content


def test_the_reservation_counts_the_system_prompt_and_tool_schemas(tmp_path: Path) -> None:
    """Review finding: only `messages` were estimated, so a big agent call (system prompt + tool
    schemas) could pass a cap it would exceed."""
    guard = SpendGuard(make_cfg(tmp_path))
    client = install(_client(), guard)
    seen: list[float] = []
    real = guard.reserve

    def spy(model_id: str, payload: Any, max_tokens: int | None) -> Any:
        r = real(model_id, payload, max_tokens)
        seen.append(r.usd)
        return r

    guard.reserve = spy  # type: ignore[method-assign]
    big = "x" * 30_000
    with Stubber(client) as stub:
        stub.add_response("converse", _converse_response(10, 10))
        client.converse(
            modelId=NOVA,
            system=[{"text": big}],
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            inferenceConfig={"maxTokens": 1},
        )
    assert seen[0] >= guard.cost(NOVA, 10_000, 1)  # ~30,000 chars of system prompt counted


def test_ledger_is_safe_under_concurrency(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.json")
    threads = [threading.Thread(target=ledger.add, args=("m", 0.01)) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ledger.snapshot()["calls"] == 50


def test_prices_come_from_config_and_unknown_models_cost_the_ceiling(tmp_path: Path) -> None:
    guard = SpendGuard(make_cfg(tmp_path))
    assert guard.price(NOVA) == (0.035, 0.14)
    assert guard.price("some.unknown-model") == UNKNOWN_PRICE
    assert guard.cost(NOVA, 1_000_000, 1_000_000) == pytest.approx(0.175)


def test_successful_call_records_actual_cost_in_both_ledgers(tmp_path: Path) -> None:
    guard = SpendGuard(make_cfg(tmp_path))
    client = install(_client(), guard)
    with Stubber(client) as stub:
        stub.add_response("converse", _converse_response(1000, 500))
        _converse(client)
    expected = (1000 * 0.035 + 500 * 0.14) / 1e6
    assert guard.project.snapshot()["total_usd"] == pytest.approx(expected)
    assert guard.shared is not None
    assert guard.shared.snapshot()["month_usd"] == pytest.approx(expected)
    assert guard.status()["project_calls"] == 1


def test_call_over_project_cap_is_refused_before_sending(tmp_path: Path) -> None:
    guard = SpendGuard(make_cfg(tmp_path, project_cap=0.0))
    client = install(_client(), guard)
    with Stubber(client) as stub:
        stub.add_response("converse", _converse_response(10, 10))
        with pytest.raises(SpendLimitExceeded, match="project cap"):
            _converse(client)
        # The queued response is still there: the request never reached the "network".
        with pytest.raises(AssertionError):
            stub.assert_no_pending_responses()
    assert guard.project.snapshot()["calls"] == 0


def test_monthly_cap_counts_spend_from_other_projects(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, monthly_cap=1.0)
    Ledger(tmp_path / "shared.json").add("memory-repo-model", 0.9999999)  # the Memory repo
    guard = SpendGuard(cfg)
    client = install(_client(), guard)
    with Stubber(client) as stub:
        stub.add_response("converse", _converse_response(10, 10))
        with pytest.raises(SpendLimitExceeded, match="monthly"):
            _converse(client, max_tokens=4096)


def test_spend_limit_is_not_an_exception_subclass() -> None:
    # Generic `except Exception` handlers (there are several) must not swallow a cap.
    assert not issubclass(SpendLimitExceeded, Exception)


def test_failed_call_releases_its_reservation_and_costs_nothing(tmp_path: Path) -> None:
    guard = SpendGuard(make_cfg(tmp_path))
    client = install(_client(), guard)
    with Stubber(client) as stub:
        stub.add_client_error("converse", service_error_code="ThrottlingException")
        with pytest.raises(Exception, match="ThrottlingException"):
            _converse(client)
    assert guard.project.snapshot()["total_usd"] == 0
    assert guard.reserved_usd() == 0


def test_install_is_idempotent(tmp_path: Path) -> None:
    guard = SpendGuard(make_cfg(tmp_path))
    client = install(install(_client(), guard), guard)
    with Stubber(client) as stub:
        stub.add_response("converse", _converse_response(10, 10))
        _converse(client)
    assert guard.status()["project_calls"] == 1  # charged once, not twice


def test_rate_limit_spaces_calls_to_the_same_model(tmp_path: Path) -> None:
    import time

    cfg = make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"spend": cfg.spend.model_copy(update={"max_rpm_per_model": 600})})
    guard = SpendGuard(cfg)  # 600/min -> 0.1s between calls
    t0 = time.monotonic()
    for _ in range(3):
        r = guard.reserve(NOVA, "hi", 10)
        guard.settle(r, 1, 1, ok=True)
    assert time.monotonic() - t0 >= 0.18


def test_a_timed_out_call_is_charged_its_worst_case(tmp_path: Path) -> None:
    """Review finding: a call that timed out on the client but ran on Bedrock was recorded as $0."""
    from botocore.exceptions import EndpointConnectionError, ReadTimeoutError

    guard = SpendGuard(make_cfg(tmp_path))
    client = install(_client(), guard)

    def fail_with(exc: BaseException) -> None:
        def raiser(**_: Any) -> None:
            raise exc

        client.meta.events.register_first("before-send.bedrock-runtime.Converse", raiser)
        try:
            with pytest.raises(type(exc)):
                _converse(client, max_tokens=100)
        finally:
            client.meta.events.unregister("before-send.bedrock-runtime.Converse", raiser)

    fail_with(EndpointConnectionError(endpoint_url="https://bedrock"))  # never sent: $0
    assert guard.project.snapshot()["total_usd"] == 0
    fail_with(ReadTimeoutError(endpoint_url="https://bedrock"))  # may have run: worst case
    assert guard.project.snapshot()["total_usd"] >= guard.cost(NOVA, 1, 100)
    assert guard.reserved_usd() == 0


def test_reservations_are_shared_by_every_process_on_the_ledger(tmp_path: Path) -> None:
    """Review finding: reservations were per process, so the CLI and the server together could
    overshoot a cap by whatever each had in flight. Two guards stand in for two processes."""
    cfg = make_cfg(tmp_path, project_cap=0.01)
    server, cli = SpendGuard(cfg), SpendGuard(cfg)
    held = server.reserve(NOVA, "x", 40_000)  # worst case ~$0.0056 in flight in the "server"
    assert cli.reserved_usd() == pytest.approx(held.usd)
    with pytest.raises(SpendLimitExceeded, match="in flight"):
        cli.reserve(NOVA, "x", 40_000)  # alone it fits; with the server's call it doesn't
    server.settle(held, 10, 10, ok=True)
    assert cli.reserved_usd() == 0
    cli.settle(cli.reserve(NOVA, "x", 40_000), 10, 10, ok=True)  # room again


def test_a_crashed_process_does_not_hold_a_cap_forever(tmp_path: Path) -> None:
    import json as _json

    cfg = make_cfg(tmp_path, project_cap=0.01)
    guard = SpendGuard(cfg)
    guard.reserve(NOVA, "x", 40_000)  # never settled: the process "died"
    d = _json.loads(cfg.spend.ledger.read_text())
    for r in d["reservations"].values():
        r["expires"] = 0  # ...long ago
    cfg.spend.ledger.write_text(_json.dumps(d))
    assert guard.reserved_usd() == 0
    guard.reserve(NOVA, "x", 40_000)  # not blocked by the dead reservation
