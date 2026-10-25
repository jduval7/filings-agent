"""Eval regression gate: the check CI runs on every change.

Re-scores the configured extractor and reconciler on the dev split (cache-backed: an unchanged
prompt/model costs $0) and fails if quality is below the absolute bar or has regressed more than
the tolerance versus the committed baseline. A prompt edit, a model swap in config, or a code
change to parsing/verification all go through this.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from filings_agent.config import AppConfig
from filings_agent.evals.local import score_model, summarize_reps
from filings_agent.evals.reconcile_local import score_reconciler


@dataclass(frozen=True)
class Check:
    name: str
    value: float
    minimum: float
    baseline: float | None
    tolerance: float

    @property
    def passed(self) -> bool:
        regressed = self.baseline is not None and self.value < self.baseline - self.tolerance
        return self.value >= self.minimum and not regressed

    def line(self) -> str:
        base = f"baseline {self.baseline:.3f}" if self.baseline is not None else "no baseline"
        mark = "PASS" if self.passed else "FAIL"
        return f"{mark}  {self.name:28} {self.value:.3f}  (min {self.minimum:.3f}, {base})"


def run_gate(
    cfg: AppConfig, gate_path: Path, baseline_path: Path, update_baseline: bool = False
) -> tuple[bool, list[Check], dict[str, float]]:
    gate = cast(dict[str, Any], yaml.safe_load(gate_path.read_text()))
    baseline: dict[str, float] = (
        json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    )
    tol = float(gate.get("tolerance", 0.01))
    ex_cfg, rc_cfg = gate["extract"], gate["reconcile"]

    # Prompts come from the app config, never from the gate file: the gate tests what runs.
    ex = summarize_reps(
        cfg.roles.extractor,
        cfg.prompts.extract,
        score_model(cfg, cfg.roles.extractor, cfg.prompts.extract, "dev", reps=1),
    )
    # The reconciler that runs: the configured strategy and its version, not always the pipeline.
    strategy = cfg.reconcile.strategy
    version = cfg.prompts.reconcile if strategy == "pipeline" else cfg.reconcile.agent_version
    rc = score_reconciler(cfg, cfg.roles.reconciler, version, "dev", reps=1, strategy=strategy)
    # The easy lane skips human review, so one wrong answer there is published unchecked: the
    # gate holds it to 100% (an empty lane counts as correct).
    easy_wrong = len(rc.get("auto_wrong", []))
    easy = 1.0 if not easy_wrong else float(rc.get("auto_accuracy", 0.0))
    values = {
        "extract.f1": float(ex["f1"]),
        "extract.not_reported_accuracy": float(ex["abstain"]),
        "extract.schema_error_rate_inv": 1.0 - float(ex["schema_err"]),
        "reconcile.cause_accuracy": float(rc["cause_accuracy"]),
        "reconcile.quote_verified_rate": float(rc["verified_rate"]),
        "reconcile.easy_lane_accuracy": easy,
    }
    minimums = {
        "extract.f1": ex_cfg["f1_min"],
        "extract.not_reported_accuracy": ex_cfg["not_reported_min"],
        "extract.schema_error_rate_inv": 1.0 - ex_cfg["schema_error_max"],
        "reconcile.cause_accuracy": rc_cfg["cause_accuracy_min"],
        "reconcile.quote_verified_rate": rc_cfg["quote_verified_min"],
        "reconcile.easy_lane_accuracy": rc_cfg.get("easy_lane_accuracy_min", 1.0),
    }
    checks = [Check(k, v, float(minimums[k]), baseline.get(k), tol) for k, v in values.items()]
    ok = all(c.passed for c in checks)
    if update_baseline:
        baseline_path.write_text(json.dumps(values, indent=2) + "\n")
    return ok, checks, values
