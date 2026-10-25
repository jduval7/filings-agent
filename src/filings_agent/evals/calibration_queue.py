"""Label the judge-calibration items in LangSmith, then pull the labels back.

    uv run filings-agent calibrate queue   # the 20 items -> two annotation queues (one per rubric)
    #   label them in LangSmith: one pass/fail score per question, then "Next"
    uv run filings-agent calibrate pull    # your scores -> configs/calibration_labels.yaml
    uv run filings-agent calibrate score   # the judge against your labels

Each item is logged as a run in the project "filings-agent-calibration": its input is the material
the judge sees, its output is what's being graded. The queues show nothing the judge said, and not
which model wrote the output, so the labels are blind. The labels file stays the record (commit it);
LangSmith is where you fill it in.
"""

# LangSmith client methods are partially untyped; this module is a boundary that accepts it.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from filings_agent.config import AppConfig
from filings_agent.evals.calibration import LABELS_PATH, CalibrationItem, load_items, load_labels
from filings_agent.steps.judge import RUBRICS, Rubric

PROJECT = "filings-agent-calibration"
LABEL_PREFIX = "label."
_NS = uuid.UUID("7c4a2e90-3b1f-4d6e-8a25-e9f0c1d7b364")  # v2: without the answer key
INSTRUCTIONS = (
    "Judge calibration. Each item is one output to grade (Outputs) and the material it was based "
    "on (Inputs). Answer every question with 1 (pass) or 0 (fail), from the material alone. Strict "
    "is better: if in doubt, fail. You're not shown what the LLM judge said or which model wrote "
    "the output, on purpose: your answers are what the judge is measured against."
)


def queue_name(rubric: Rubric) -> str:
    return f"filings-agent: calibrate {rubric.name}"


# What the labeller sees besides the output: exactly what the judge sees (the material), plus the
# case's name. Never the answer key (true_cause), the model's label, or whether the code's checks
# passed: the judge doesn't get those, so a human who did would be grading something else.
SHOWN_CONTEXT = ("case", "group")


def shown_inputs(item: CalibrationItem) -> dict[str, str]:
    extra = {k: v for k, v in item.context.items() if k in SHOWN_CONTEXT}
    return {**extra, "material": item.material}


def run_id(item: CalibrationItem) -> str:
    return str(uuid.uuid5(_NS, item.item_id))


def rubric_items(rubric: Rubric) -> list[dict[str, Any]]:
    return [
        {
            "feedback_key": f"{LABEL_PREFIX}{c.name}",
            "description": c.question,
            "score_descriptions": {"0": "fail", "1": "pass"},
            "is_required": True,
        }
        for c in rubric.criteria
    ]


# A score type per question: LangSmith only offers a feedback key in the queue's form once it has
# a definition (found live: the questions without one were missing from the reviewer's options).
PASS_FAIL: dict[str, Any] = {
    "type": "categorical",
    "categories": [{"value": 1, "label": "pass"}, {"value": 0, "label": "fail"}],
}


def ensure_feedback_configs(client: Any, rubric: Rubric) -> None:
    existing = {str(f.feedback_key): f for f in client.list_feedback_configs()}
    for c in rubric.criteria:
        key = f"{LABEL_PREFIX}{c.name}"
        if key not in existing:
            client.create_feedback_config(key, feedback_config=PASS_FAIL)
        elif dict(existing[key].feedback_config).get("type") != "categorical":
            client.update_feedback_config(key, feedback_config=PASS_FAIL)


def ensure_queue(client: Any, rubric: Rubric) -> Any:
    ensure_feedback_configs(client, rubric)
    for q in client.list_annotation_queues(name=queue_name(rubric)):
        return q
    return client.create_annotation_queue(
        name=queue_name(rubric),
        description=f"Your labels for the {rubric.name} judge ({rubric.version})",
        rubric_instructions=INSTRUCTIONS,
        rubric_items=rubric_items(rubric),
    )


def _queued_path(cfg: AppConfig) -> Path:
    """Which items are already in a queue (so queueing twice adds nothing)."""
    return cfg.data_dir / "calibration" / "queued.json"


def queue_items(cfg: AppConfig, client: Any, wait_s: float = 2.0) -> dict[str, int]:
    """Log each item as a run (once) and put it in its rubric's queue (once)."""
    items = load_items(cfg)
    path = _queued_path(cfg)
    queued: set[str] = set(json.loads(path.read_text())) if path.exists() else set()
    created: list[CalibrationItem] = []
    now = datetime.now(UTC)
    for item in items:
        rid = run_id(item)
        if item.item_id in queued:
            continue
        client.create_run(
            name=f"{item.rubric}: {item.item_id}",
            inputs=shown_inputs(item),
            outputs={"output": item.output},
            run_type="chain",
            project_name=PROJECT,
            id=rid,
            start_time=now,
            end_time=now,
            extra={"metadata": {"item_id": item.item_id, "rubric": item.rubric}},
        )
        created.append(item)
    added: dict[str, int] = {}
    for name, rubric in RUBRICS.items():
        ids = [run_id(i) for i in created if i.rubric == name]
        if not ids:
            added[name] = 0
            continue
        queue = ensure_queue(client, rubric)
        for attempt in range(6):  # runs upload in the background: wait until LangSmith has them
            try:
                client.add_runs_to_annotation_queue(queue.id, run_ids=ids)
                break
            except Exception as e:
                if "not found" not in str(e).lower() or attempt == 5:
                    raise
                time.sleep(wait_s * (attempt + 1))
        added[name] = len(ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(queued | {i.item_id for i in created})))
    return added


def pull_labels(cfg: AppConfig, client: Any, path: Path = LABELS_PATH) -> dict[str, int]:
    """Your queue scores -> the labels file. A scored question overwrites the file's value; an
    unscored one keeps it. Returns how many questions are labelled per rubric."""
    labels = load_labels(path)
    for item in load_items(cfg):
        row = labels.setdefault(item.item_id, {})
        for name in RUBRICS[item.rubric].names():  # every question has a slot, scored or not
            row.setdefault(name, None)
        for fb in client.list_feedback(run_ids=[run_id(item)]):
            key = str(fb.key)
            if key.startswith(LABEL_PREFIX) and fb.score is not None:
                criterion = key.removeprefix(LABEL_PREFIX)
                if criterion in row:
                    row[criterion] = float(fb.score) >= 0.5
    header = (
        "# Your labels for judge calibration, pulled from the LangSmith queues\n"
        "# (`filings-agent calibrate pull`), or edited here.\n"
        "# true = pass, false = fail, null = skip.\n"
        "# These labels are yours: commit them.\n"
    )
    path.write_text(header + yaml.safe_dump(labels, sort_keys=True))
    counts: dict[str, int] = {}
    for item in load_items(cfg):
        done = sum(v is not None for v in labels[item.item_id].values())
        counts[item.rubric] = counts.get(item.rubric, 0) + done
    return counts
