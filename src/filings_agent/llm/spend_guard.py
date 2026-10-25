"""Bedrock spend guard: every paid call is checked before it's sent and recorded after.

Ported from the Memory repo's guard. Before each call: per-model rate limit, then a worst-case
cost reservation checked against the project cap and the account-wide monthly cap. Over a cap ->
the call is refused before anything is sent. After each call: the actual cost (from the response's
token usage) goes into the project ledger and, if configured, the shared account ledger.

Reservations live in the project ledger file, under its lock, so every process on this machine
(server workers, the CLI) sees the others' calls in flight: per-process reservations let two
processes together overshoot a cap (found by review). Each expires, so a crashed process can't
hold a cap forever. A call that fails after it may have reached Bedrock (a read timeout) is
charged its worst case: Bedrock may have billed it, and $0 under-counted spend.

`SpendLimitExceeded` is a BaseException on purpose, so `except Exception` can't swallow it.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import threading
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from botocore.exceptions import ConnectTimeoutError, EndpointConnectionError

from filings_agent.config import AppConfig

UNKNOWN_PRICE = (15.0, 75.0)  # conservative ceiling for a model id not in the config
# Longer than any one call can take (read_timeout 120s, no botocore retries), so a reservation is
# only ever pruned after its process died without settling it.
RESERVATION_TTL_S = 600.0
# Errors raised before a request left this machine: nothing can have been billed.
NEVER_SENT = (ConnectTimeoutError, EndpointConnectionError)


class SpendLimitExceeded(BaseException):
    """A paid call was refused because it would exceed a spend cap."""


def _est_tokens(payload: Any) -> int:
    """Conservative input-token estimate (~3 chars/token)."""
    try:
        s = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    except (TypeError, ValueError):
        s = str(payload)
    return max(1, len(s) // 3)


class Ledger:
    """Persistent spend totals, shared safely across processes via flock.

    Same file format as the Memory repo's ledger, so both can write the shared one.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def _read(self) -> dict[str, Any]:
        """A missing ledger is a fresh one. An unreadable one refuses calls: read as $0 spent it
        would let every cap pass, and the next write would erase the real total."""
        try:
            d = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            raise SpendLimitExceeded(
                f"spend ledger {self.path} is unreadable ({type(e).__name__}); paid calls are "
                "refused until it is fixed or restored (`filings-agent spend` shows the totals)"
            ) from e
        if not isinstance(d, dict):
            raise SpendLimitExceeded(f"spend ledger {self.path} is not a JSON object")
        return cast(dict[str, Any], d)

    @staticmethod
    def _roll(d: dict[str, Any]) -> dict[str, Any]:
        today = dt.date.today()
        month, day = today.strftime("%Y-%m"), today.isoformat()
        if d.get("month") != month:
            d["month"], d["month_usd"], d["by_model_month"] = month, 0.0, {}
        if d.get("day") != day:
            d["day"], d["day_usd"] = day, 0.0
        d.setdefault("total_usd", 0.0)
        d.setdefault("calls", 0)
        return d

    def snapshot(self) -> dict[str, Any]:
        return self._roll(self._read())

    @contextmanager
    def locked(self) -> Generator[dict[str, Any]]:
        """Read-modify-write under the file lock (exclusive across processes)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                d = self._roll(self._read())
                yield d
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(d, indent=2))
                os.replace(tmp, self.path)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def charge(d: dict[str, Any], model_id: str, usd: float) -> None:
        d["month_usd"] += usd
        d["day_usd"] += usd
        d["total_usd"] += usd
        d["calls"] += 1
        by_model = cast(dict[str, float], d["by_model_month"])
        by_model[model_id] = by_model.get(model_id, 0.0) + usd

    @staticmethod
    def reservations(d: dict[str, Any]) -> dict[str, dict[str, float]]:
        """Unexpired reservations (expired ones are dropped from `d`)."""
        now = time.time()
        live = {
            k: v
            for k, v in cast(dict[str, dict[str, float]], d.get("reservations") or {}).items()
            if v.get("expires", 0.0) > now
        }
        d["reservations"] = live
        return live

    def add(self, model_id: str, usd: float) -> None:
        with self.locked() as d:
            self.charge(d, model_id, usd)

    def reserved_usd(self) -> float:
        return sum(r["usd"] for r in self.reservations(self.snapshot()).values())


@dataclass
class _Reservation:
    model_id: str
    usd: float
    payload: Any
    id: str = ""


class SpendGuard:
    def __init__(self, cfg: AppConfig) -> None:
        self._prices = {m.id: m.price for m in cfg.models.values()}
        self._prices[cfg.embeddings.id] = cfg.embeddings.price
        self._rpm_overrides = {cfg.embeddings.id: cfg.embeddings.max_rpm}
        self.project_cap = cfg.spend.project_cap_usd
        self.monthly_cap = cfg.spend.monthly_cap_usd
        self.max_rpm = cfg.spend.max_rpm_per_model
        self.project = Ledger(cfg.spend.ledger)
        shared = cfg.spend.shared_ledger or (
            Path(os.environ["SPEND_SHARED_LEDGER"])
            if os.environ.get("SPEND_SHARED_LEDGER")
            else None
        )
        self.shared = Ledger(shared.expanduser()) if shared else None
        self._lock = threading.Lock()
        self._last_call: dict[str, float] = {}

    def price(self, model_id: str) -> tuple[float, float]:
        return self._prices.get(model_id, UNKNOWN_PRICE)

    def cost(self, model_id: str, input_tokens: float, output_tokens: float) -> float:
        p_in, p_out = self.price(model_id)
        return (input_tokens * p_in + output_tokens * p_out) / 1e6

    def reserve(self, model_id: str, payload: Any, max_tokens: int | None) -> _Reservation:
        worst = self.cost(model_id, _est_tokens(payload), max_tokens or 4096)
        rpm = self._rpm_overrides.get(model_id, self.max_rpm)
        gap = 60.0 / rpm if rpm > 0 else 0.0
        with self._lock:
            now = time.monotonic()
            wait = self._last_call.get(model_id, 0.0) + gap - now
            self._last_call[model_id] = max(now, now + wait)
        if wait > 0:
            time.sleep(wait)
        rid = uuid.uuid4().hex
        # Check and reserve in one step under the project ledger's lock: every process's calls
        # in flight are in that file, so concurrent processes can't both squeeze under a cap.
        with self._lock, self.project.locked() as d:
            month_usd = float(self.shared.snapshot()["month_usd"]) if self.shared else 0.0
            in_flight = sum(r["usd"] for r in Ledger.reservations(d).values())
            checks = [
                ("project", float(d["total_usd"]), self.project_cap),
                ("monthly (account-wide)", month_usd, self.monthly_cap),
            ]
            for name, spent, cap in checks:
                would_be = spent + in_flight + worst
                if would_be > cap:
                    raise SpendLimitExceeded(
                        f"SPEND CAP HIT: {name} cap ${cap:.2f} would be exceeded "
                        f"(spent ${spent:.4f}, in flight ${in_flight:.4f}, this call worst-case "
                        f"${worst:.4f}, {model_id}). "
                        "Raise the cap in configs/*.yaml deliberately if this is expected."
                    )
            d["reservations"][rid] = {"usd": worst, "expires": time.time() + RESERVATION_TTL_S}
        return _Reservation(model_id, worst, payload, rid)

    def settle(
        self, r: _Reservation, input_tokens: int | None, output_tokens: int | None, ok: bool
    ) -> float:
        usd = 0.0
        if ok:
            in_tok = input_tokens if input_tokens is not None else _est_tokens(r.payload)
            usd = self.cost(r.model_id, in_tok, output_tokens or 0)
        return self._close(r, usd)

    def settle_unknown(self, r: _Reservation) -> float:
        """The call failed after it may have reached Bedrock (e.g. a read timeout): it may have
        been billed and we'll never see its usage. Charge the worst case we reserved."""
        return self._close(r, r.usd)

    def _close(self, r: _Reservation, usd: float) -> float:
        with self._lock, self.project.locked() as d:
            Ledger.reservations(d).pop(r.id, None)
            if usd:
                Ledger.charge(d, r.model_id, usd)
        if usd and self.shared:
            self.shared.add(r.model_id, usd)
        return usd

    def reserved_usd(self) -> float:
        """Worst-case cost of every call in flight, in every process using this ledger."""
        return self.project.reserved_usd()

    def status(self) -> dict[str, Any]:
        project = self.project.snapshot()
        out: dict[str, Any] = {
            "project_usd": round(float(project["total_usd"]), 4),
            "project_cap": self.project_cap,
            "project_calls": project["calls"],
            "by_model_month": project.get("by_model_month", {}),
        }
        if self.shared:
            out["account_month_usd"] = round(float(self.shared.snapshot()["month_usd"]), 4)
            out["monthly_cap"] = self.monthly_cap
        return out


_OPS = ("Converse", "ConverseStream", "InvokeModel", "InvokeModelWithResponseStream")


def install(client: Any, guard: SpendGuard) -> Any:
    """Attach the guard to a boto3 bedrock-runtime client via botocore events."""
    if getattr(client, "_spend_guard_installed", False):
        return client
    events = client.meta.events

    def _before(params: dict[str, Any], context: dict[str, Any], **_: Any) -> None:
        inf = cast(dict[str, Any], params.get("inferenceConfig") or {})
        # Everything sent is billed as input: the system prompt and tool schemas too, not just the
        # messages (agents send several KB of tool schemas on every call).
        payload = {k: v for k, v in params.items() if k != "modelId"}
        context["spend_guard"] = guard.reserve(
            str(params.get("modelId", "?")), payload, cast(int | None, inf.get("maxTokens"))
        )

    def _after(parsed: Any, context: dict[str, Any], **_: Any) -> None:
        r = cast(_Reservation | None, context.pop("spend_guard", None))
        if r is None:
            return
        ok = not (isinstance(parsed, dict) and "Error" in parsed)
        usage: dict[str, Any] = (
            cast(dict[str, Any], cast(dict[str, Any], parsed).get("usage") or {})
            if isinstance(parsed, dict)
            else {}
        )
        in_tok = (
            int(usage.get("inputTokens") or 0)
            + int(usage.get("cacheReadInputTokens") or 0)
            + int(usage.get("cacheWriteInputTokens") or 0)
            if usage
            else None
        )
        out_tok = int(usage["outputTokens"]) if "outputTokens" in usage else None
        guard.settle(r, in_tok, out_tok, ok)

    def _after_error(
        context: dict[str, Any], exception: BaseException | None = None, **_: Any
    ) -> None:
        r = cast(_Reservation | None, context.pop("spend_guard", None))
        if r is None:
            return
        if isinstance(exception, NEVER_SENT):
            guard.settle(r, None, None, ok=False)  # never left this machine: nothing billed
        else:
            guard.settle_unknown(r)  # may have run on Bedrock: charge the worst case

    for op in _OPS:
        # register_first: a refused call is stopped before botocore signs or sends anything.
        events.register_first(f"before-parameter-build.bedrock-runtime.{op}", _before)
        events.register(f"after-call.bedrock-runtime.{op}", _after)
        events.register(f"after-call-error.bedrock-runtime.{op}", _after_error)
    client._spend_guard_installed = True
    return client


_guards: dict[str, SpendGuard] = {}
_guards_lock = threading.Lock()


def get_guard(cfg: AppConfig) -> SpendGuard:
    """One guard per process (per ledger), so concurrent branches share one reservation pool."""
    with _guards_lock:
        key = str(cfg.spend.ledger)
        if key not in _guards:
            _guards[key] = SpendGuard(cfg)
        return _guards[key]
