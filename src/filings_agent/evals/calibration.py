"""Judge calibration: is the LLM judge's verdict worth trusting? Measured against YOUR labels.

  prepare  sample real outputs (reconcile explanations from several models, right and wrong; memos
           from several writers), write data/calibration/LABEL_ME.md to read and
           configs/calibration_labels.yaml to fill in (pass/fail per criterion). Existing labels
           are kept.
  score    run the judge on the same items; per criterion: agreement, Cohen's kappa (agreement
           corrected for chance), false passes (judge pass, human fail: the dangerous kind) and
           false fails. The judge is TRUSTED only if kappa >= 0.6 and false-pass rate <= 10%
           for every criterion with enough labels, including at least MIN_HUMAN_FAILS items the
           human failed (a judge that was never shown a failure hasn't shown it can catch one).
           The false-pass rate is per human fail: of the outputs you failed, how many it passed.

Why: a judge's score is an opinion until it's shown to agree with a careful human. In the Memory
project, swapping the judge model changed the ranking of the systems being judged.
"""

# LangSmith client methods are partially untyped; this module is a boundary that accepts it.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel

from filings_agent.config import AppConfig
from filings_agent.evals.memo_eval import _material, load_memo_examples, make_judge, make_writer  # pyright: ignore[reportPrivateUsage]
from filings_agent.evals.reconcile_local import load_tasks
from filings_agent.llm.cache import StepCache
from filings_agent.llm.models import chat_model
from filings_agent.steps.judge import MEMO_RUBRIC, RECONCILE_RUBRIC, RUBRICS, LLMJudge
from filings_agent.steps.reconcile import LLMReconciler, ReconcileTask

LABELS_PATH = Path("configs/calibration_labels.yaml")
RECONCILE_MODELS = ("claude-sonnet-4.6", "claude-haiku-4.5", "gpt-oss-120b", "nova-lite")
MEMO_MODELS = ("claude-haiku-4.5", "claude-sonnet-4.6", "nova-lite", "nova-micro")
MIN_LABELS = 6  # a criterion needs at least this many labels to be judged trustworthy
MIN_HUMAN_FAILS = 2  # ...and this many of them failed by the human
KAPPA_MIN = 0.6
FALSE_PASS_MAX = 0.10
_NS = uuid.UUID("6d2f8a1c-4b7e-4c39-a5e0-91f3b2c7d845")


class CalibrationItem(BaseModel):
    item_id: str
    rubric: str
    source_model: str
    material: str  # what the output was based on (the judge sees the same)
    output: str  # what's being graded
    context: dict[str, str]  # for the human: the label, the case, etc.


def _stable(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _reconcile_items(cfg: AppConfig, n: int) -> list[CalibrationItem]:
    tasks = load_tasks(cfg, "dev")
    cache = StepCache(cfg.data_dir / "cache" / "llm")
    right: list[CalibrationItem] = []
    wrong: list[CalibrationItem] = []
    for model in RECONCILE_MODELS:
        rec = LLMReconciler(
            model, cfg.model(model), chat_model(cfg, model), cache, cfg.prompts.reconcile
        )
        for t in tasks:
            r = rec.reconcile(t.task)
            item = CalibrationItem(
                item_id=f"rec-{_stable(model + t.task.discrepancy_id)[:10]}",
                rubric=RECONCILE_RUBRIC.name,
                source_model=model,
                material=_reconcile_material(t.task),
                output=f"cause: {r.cause.value}\nexplanation: {r.explanation}\n"
                f"evidence_quote: {r.evidence_quote}",
                context={
                    "case": t.task.discrepancy_id,
                    "true_cause": t.label.value,
                    "model_cause": r.cause.value,
                },
            )
            (right if r.cause is t.label else wrong).append(item)
    # Half right, half wrong, deterministic: the judge must be tested on both kinds.
    pick = sorted(right, key=lambda i: _stable(i.item_id))[: n // 2]
    pick += sorted(wrong, key=lambda i: _stable(i.item_id))[: n - len(pick)]
    return pick


def _reconcile_material(task: ReconcileTask) -> str:
    d = task.discrepancy
    vals = "; ".join(f"{a}: {v:,}" for a, v in d.values)
    ev = "\n".join(f"[{i + 1}] {e}" for i, e in enumerate(task.evidence))
    return f"{d.id} reported values (oldest first): {vals}\nEvidence excerpts:\n{ev}"


def _memo_items(cfg: AppConfig, n: int) -> list[CalibrationItem]:
    items: list[CalibrationItem] = []
    for model in MEMO_MODELS:
        writer = make_writer(cfg, model)
        for e in load_memo_examples(cfg, "dev"):
            rows, changes, material = _material(e)
            memo = writer.write(rows, changes)
            items.append(
                CalibrationItem(
                    item_id=f"memo-{_stable(model + e.example_id)[:10]}",
                    rubric=MEMO_RUBRIC.name,
                    source_model=model,
                    material=material,
                    output=memo.markdown,
                    context={"group": e.example_id, "numbers_verified": str(memo.verified)},
                )
            )
    return sorted(items, key=lambda i: _stable(i.item_id))[:n]


def prepare(
    cfg: AppConfig, n_reconcile: int = 12, n_memo: int = 8, labels_path: Path = LABELS_PATH
) -> list[CalibrationItem]:
    items = _reconcile_items(cfg, n_reconcile) + _memo_items(cfg, n_memo)
    out_dir = cfg.data_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "items.jsonl").write_text("".join(i.model_dump_json() + "\n" for i in items))
    (out_dir / "LABEL_ME.md").write_text(_label_me(items))
    existing = load_labels(labels_path)
    merged: dict[str, dict[str, bool | None]] = {}
    for i in items:
        prior = existing.get(i.item_id, {})
        merged[i.item_id] = {c: prior.get(c) for c in RUBRICS[i.rubric].names()}
    header = (
        "# Your labels for judge calibration. For each item in data/calibration/LABEL_ME.md,\n"
        "# set each criterion to true (pass) or false (fail). Leave null to skip.\n"
        "# Rubric questions are in LABEL_ME.md. These labels are yours: commit them.\n"
    )
    labels_path.write_text(header + yaml.safe_dump(merged, sort_keys=True))
    return items


def _label_me(items: list[CalibrationItem]) -> str:
    parts = [
        "# Judge calibration: label these\n",
        "For each item, read the material and the output, then answer each criterion in "
        "`configs/calibration_labels.yaml` (true = pass, false = fail). Strict is better: "
        "if in doubt, fail.\n",
    ]
    for rubric in (RECONCILE_RUBRIC, MEMO_RUBRIC):
        parts.append(f"\n## Rubric: {rubric.name}\n")
        parts += [f"- **{c.name}**: {c.question}" for c in rubric.criteria]
    for i in items:
        ctx = ", ".join(f"{k}={v}" for k, v in i.context.items())
        parts.append(
            f"\n---\n\n## {i.item_id} ({i.rubric}, by {i.source_model})\n\n_{ctx}_\n\n"
            f"### Material\n\n```\n{i.material}\n```\n\n### Output\n\n{i.output}\n"
        )
    return "\n".join(parts)


def load_labels(path: Path = LABELS_PATH) -> dict[str, dict[str, bool | None]]:
    if not path.exists():
        return {}
    return cast(dict[str, dict[str, bool | None]], yaml.safe_load(path.read_text()) or {})


def load_items(cfg: AppConfig) -> list[CalibrationItem]:
    path = cfg.data_dir / "calibration" / "items.jsonl"
    return [CalibrationItem.model_validate_json(x) for x in path.read_text().splitlines()]


# ---- agreement --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Agreement:
    criterion: str
    n: int
    agreement: float
    kappa: float
    false_pass: int  # judge said pass, human said fail
    false_fail: int
    human_fails: int = 0

    @property
    def false_pass_rate(self) -> float:
        """Of the outputs the human failed, the share the judge passed (1.0 if none to test)."""
        return self.false_pass / self.human_fails if self.human_fails else 1.0

    @property
    def trusted(self) -> bool:
        return (
            self.n >= MIN_LABELS
            and self.human_fails >= MIN_HUMAN_FAILS
            and self.kappa >= KAPPA_MIN
            and self.false_pass_rate <= FALSE_PASS_MAX
        )


def cohen_kappa(human: list[bool], judge: list[bool]) -> float:
    n = len(human)
    if n == 0:
        return 0.0
    po = sum(h == j for h, j in zip(human, judge, strict=True)) / n
    ph, pj = sum(human) / n, sum(judge) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def agreement_for(criterion: str, pairs: list[tuple[bool, bool]]) -> Agreement:
    human = [h for h, _ in pairs]
    judge = [j for _, j in pairs]
    n = len(pairs)
    return Agreement(
        criterion=criterion,
        n=n,
        agreement=sum(h == j for h, j in pairs) / n if n else 0.0,
        kappa=cohen_kappa(human, judge),
        false_pass=sum(1 for h, j in pairs if j and not h),
        false_fail=sum(1 for h, j in pairs if h and not j),
        human_fails=sum(1 for h in human if not h),
    )


def score(
    cfg: AppConfig, judge_model: str | None = None, labels_path: Path = LABELS_PATH
) -> dict[str, Any]:
    items = load_items(cfg)
    labels = load_labels(labels_path)
    judge: Any
    if judge_model and judge_model.startswith("openai:"):  # LangSmith's online evaluator
        from filings_agent.evals.online_judge import OnlineEvaluatorJudge

        judge = OnlineEvaluatorJudge(judge_model.removeprefix("openai:"))
    else:
        judge = make_judge(cfg, judge_model)
    only = getattr(judge, "rubric", None)  # a judge for one rubric grades only those items
    pairs: dict[str, list[tuple[bool, bool]]] = {}
    disagreements: list[dict[str, Any]] = []
    graded = 0
    errors: list[str] = []
    for i in items:
        human = {c: v for c, v in labels.get(i.item_id, {}).items() if v is not None}
        if not human:
            continue
        if i.source_model == judge.model_name:
            continue  # never let the judge grade its own output, even in calibration
        if only and i.rubric != only:
            continue
        res = judge.grade(RUBRICS[i.rubric], i.material, i.output)
        if res.error:
            errors.append(f"{i.item_id}: {res.error}")  # no verdict: reported, never a pass/fail
            continue
        graded += 1
        for c, h in human.items():
            if c not in res.passed:
                continue
            j = res.passed[c]
            pairs.setdefault(f"{i.rubric}.{c}", []).append((bool(h), j))
            if bool(h) != j:
                disagreements.append(
                    {
                        "item": i.item_id,
                        "criterion": c,
                        "human": h,
                        "judge": j,
                        "judge_reason": res.reasons.get(c, ""),
                    }
                )
    results = [agreement_for(c, p) for c, p in sorted(pairs.items())]
    usable = [a for a in results if a.n >= MIN_LABELS]
    return {
        "judge": judge.model_name,
        "items_graded": graded,
        "judge_errors": errors,
        "criteria": [
            asdict(a) | {"trusted": a.trusted, "false_pass_rate": a.false_pass_rate}
            for a in results
        ],
        "trusted": bool(usable) and all(a.trusted for a in usable),
        "disagreements": disagreements,
    }


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# Judge calibration: {result['judge']}",
        "",
        f"Items graded: {result['items_graded']}. "
        f"Verdict: **{'TRUSTED' if result['trusted'] else 'NOT TRUSTED'}** "
        f"(needs kappa >= {KAPPA_MIN}, false-pass rate <= {FALSE_PASS_MAX:.0%} of the items you "
        f"failed, >= {MIN_LABELS} labels and >= {MIN_HUMAN_FAILS} fails per criterion).",
        "",
        "| criterion | n | you failed | agreement | kappa | false pass | false fail | trusted |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in result["criteria"]:
        lines.append(
            f"| {a['criterion']} | {a['n']} | {a.get('human_fails', 0)} | {a['agreement']:.2f} "
            f"| {a['kappa']:.2f} | {a['false_pass']} | {a['false_fail']} "
            f"| {'yes' if a['trusted'] else 'no'} |"
        )
    if result.get("judge_errors"):
        lines += ["", f"Judge errors (no verdict, not counted): {len(result['judge_errors'])}"]
    if result["disagreements"]:
        lines += [
            "",
            "## Disagreements (read these: they show where the rubric or judge is off)",
            "",
        ]
        lines += [
            f"- `{d['item']}` {d['criterion']}: you={d['human']}, judge={d['judge']}. "
            f"Judge's reason: {d['judge_reason']}"
            for d in result["disagreements"]
        ]
    return "\n".join(lines) + "\n"


# ---- LangSmith: the calibration as a dataset + experiment -------------------------------------


def upload_calibration(cfg: AppConfig, client: Any, labels_path: Path = LABELS_PATH) -> int:
    name = "filings-agent/judge-calibration"
    if client.has_dataset(dataset_name=name):
        ds = client.read_dataset(dataset_name=name)
    else:
        ds = client.create_dataset(name, description="Human pass/fail labels for the LLM judge")
    labels = load_labels(labels_path)
    existing = {str(e.id) for e in client.list_examples(dataset_id=ds.id)}
    new: list[dict[str, Any]] = []
    update: list[dict[str, Any]] = []
    for i in load_items(cfg):
        human = {c: v for c, v in labels.get(i.item_id, {}).items() if v is not None}
        if not human:
            continue
        ex_id = str(uuid.uuid5(_NS, i.item_id))
        payload = {
            "id": ex_id,
            "inputs": {
                "rubric": i.rubric,
                "material": i.material,
                "output": i.output,
                "source_model": i.source_model,
            },
            "outputs": {"human": human},
            "metadata": {"item_id": i.item_id, **i.context},
        }
        (update if ex_id in existing else new).append(payload)
    if new:
        client.create_examples(dataset_id=ds.id, examples=new)
    if update:
        client.update_examples(dataset_id=ds.id, updates=update)
    return len(new) + len(update)


def run_calibration_experiment(cfg: AppConfig, client: Any, judge_model: str | None = None) -> str:
    from langsmith.evaluation import EvaluationResult

    judge: LLMJudge = make_judge(cfg, judge_model)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        if inputs["source_model"] == judge.model_name:
            return {"skipped": "judge's own output"}
        res = judge.grade(RUBRICS[inputs["rubric"]], inputs["material"], inputs["output"])
        return {"passed": res.passed, "reasons": res.reasons, "error": res.error}

    def agrees_with_human(run: Any, example: Any) -> dict[str, Any]:
        judged = cast(dict[str, bool], (run.outputs or {}).get("passed") or {})
        human = cast(dict[str, bool], (example.outputs or {}).get("human") or {})
        return {
            "results": [
                EvaluationResult(key=f"agree.{c}", score=int(judged.get(c) == h))
                for c, h in human.items()
                if c in judged
            ]
        }

    results = client.evaluate(
        target,
        data="filings-agent/judge-calibration",
        evaluators=[agrees_with_human],
        experiment_prefix=f"judge-calibration-{judge.model_name}",
        description=f"Does judge {judge.model_name} agree with the human labels?",
        metadata={"judge": judge.model_name, "rubrics": "reconcile_explanation/v1,memo/v1"},
    )
    return str(results.experiment_name)


def save(cfg: AppConfig, result: dict[str, Any]) -> Path:
    out = cfg.data_dir / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"report-{result['judge']}.json").write_text(json.dumps(result, indent=2))
    path = out / f"REPORT-{result['judge']}.md"
    path.write_text(report_markdown(result))
    return path
