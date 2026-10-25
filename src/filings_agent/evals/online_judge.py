"""The LangSmith online evaluator for memos: defined here, pasted into LangSmith, calibrated here.

LangSmith runs this evaluator itself, on a sample of new traces, and posts one pass/fail score per
rubric question onto each trace (judge_faithful, judge_caveats_correct, ...).
`online-evaluator --create` posts it as a rule through the API; without `--create` it prints
what to paste into the UI (Tracing project -> Evaluators -> + Evaluator):

    uv run filings-agent online-evaluator          # prints the prompt, criteria, filter, mapping

and of the same judge, run locally, to calibrate it against your labels before trusting it:

    uv run filings-agent calibrate score --judge openai:gpt-4.1-mini

Same prompt, same rubric, same model, temperature 0. The one difference is LangSmith's own output
parsing around the prompt, so re-check a few traces by hand after switching it on. The model is
an OpenAI one: a different family from the Claude memo writer (no self-grading), and the key lives
in LangSmith's workspace secrets (and OPENAI_API_KEY locally, for calibration only).
"""

from __future__ import annotations

import time
from typing import Any, cast

from pydantic import BaseModel, Field, create_model

from filings_agent.steps.extract import Usage
from filings_agent.steps.judge import MEMO_RUBRIC, JudgeResult, Rubric

DEFAULT_MODEL = "gpt-4.1-mini"
FEEDBACK_PREFIX = "judge_"  # judge_faithful, ...: OpenAI's structured output wants plain names
SAMPLING = {
    "filings-agent-demo": "100% (few runs; every one is worth showing)",
    "filings-agent": "10%",
}
FILTER = (
    "Root runs named filings-analysis with feedback online.memo_verified = 1 "
    "(runs that published a memo whose numbers passed the code check)"
)
FILTER_EXPR = (
    'and(eq(is_root, true), eq(name, "filings-analysis"), '
    'and(eq(feedback_key, "online.memo_verified"), eq(feedback_score, 1)))'
)
# the workspace secret LangSmith reads the key from (Settings -> Secrets)
OPENAI_SECRET = {"lc": 1, "type": "secret", "id": ["OPENAI_API_KEY"]}
WEEKLY_LIMIT_USD = 1.0  # LangSmith calls OpenAI outside our spend guard: this caps it
VARIABLES = {"table": "outputs.memo.table", "memo": "outputs.memo.markdown"}

SYSTEM = (
    "You grade a short analyst memo about restated figures in SEC 10-K filings. You see the table "
    "the memo was written from, and the memo. Judge the memo against the table only: anything not "
    "supported by the table counts against it, including background knowledge. Code has already "
    "checked that every number in the memo appears in the table, so judge the words, not the "
    "arithmetic. Be strict: if in doubt, fail. Answer each question pass or fail, with one "
    "sentence of reasoning."
)
HUMAN = "TABLE (the memo's only source):\n{table}\n\nMEMO:\n{memo}"


def criteria(rubric: Rubric = MEMO_RUBRIC) -> list[tuple[str, str]]:
    """(feedback key, the question) per criterion: what LangSmith's feedback configuration lists."""
    return [(f"{FEEDBACK_PREFIX}{c.name}", c.question) for c in rubric.criteria]


def setup_text(model: str = DEFAULT_MODEL) -> str:
    """Everything to paste into the LangSmith Evaluators tab, in the order the form asks for it."""
    lines = [
        "LangSmith online evaluator: memo quality",
        "",
        "Where: Tracing -> project -> Evaluators tab -> + Evaluator -> LLM-as-a-judge",
        f"Name: memo quality ({MEMO_RUBRIC.name} {MEMO_RUBRIC.version})",
        f"Filter: {FILTER}",
        "Sampling rate: " + "; ".join(f"{k}: {v}" for k, v in SAMPLING.items()),
        f"Model: {model} (OpenAI), temperature 0. Key: Settings -> Secrets -> OPENAI_API_KEY",
        "",
        "Prompt, system message:",
        SYSTEM,
        "",
        "Prompt, human message (use the editor's variable syntax: {table} or {{table}}):",
        HUMAN,
        "",
        "Map the variables to the run:",
        *[f"  {var} -> {path}" for var, path in VARIABLES.items()],
        "",
        "Feedback configuration: one Boolean per criterion, with reasoning on:",
        *[f"  {key}: {question}" for key, question in criteria()],
        "",
        "Then: in the evaluator's settings, turn on few-shot corrections, so scores you correct",
        "on traces become examples in its prompt.",
    ]
    return "\n".join(lines)


def rule_payload(
    project_id: str,
    sampling_rate: float,
    model: str = DEFAULT_MODEL,
    backfill_from: str | None = None,
) -> dict[str, Any]:
    """The evaluator as LangSmith stores it (POST /runs/rules): the same prompt, rubric and model
    as the local judge used for calibration, so what's calibrated is what runs."""
    props = {f"{FEEDBACK_PREFIX}{c.name}": {"type": "boolean", "description": c.question}
             for c in MEMO_RUBRIC.criteria}  # fmt: skip
    return {
        "display_name": f"memo quality ({MEMO_RUBRIC.name} {MEMO_RUBRIC.version})",
        "session_id": project_id,
        "sampling_rate": sampling_rate,
        "filter": FILTER_EXPR,
        "backfill_from": backfill_from,
        # Few-shot corrections (a person's corrected scores become examples in the prompt) are left
        # off: LangSmith's API refuses them for this rule ("Invalid rule for corrections dataset",
        # 2026-10-24). Try the toggle in the evaluator's settings in the UI.
        "spend_limit": {"limit_usd": WEEKLY_LIMIT_USD, "window": "weekly"},
        "evaluators": [{"structured": {
            "prompt": [["system", SYSTEM], ["human", HUMAN]],
            "template_format": "f-string",
            "schema": {"title": "memo_quality", "type": "object",
                       "description": "Pass or fail for each rubric question.",
                       "properties": props, "required": list(props)},
            "variable_mapping": {"table": "output.memo.table", "memo": "output.memo.markdown"},
            "model": {"lc": 1, "type": "constructor",
                      "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
                      "kwargs": {"model": model, "temperature": 0,
                                 "openai_api_key": OPENAI_SECRET}},
        }}],
    }  # fmt: skip


def create_rule(api: Any, project_id: str, sampling_rate: float, backfill_from: str | None) -> str:
    """Create the evaluator on a project, or replace one we created before. Returns its id."""
    payload = rule_payload(project_id, sampling_rate, backfill_from=backfill_from)
    for rule in api.get("/runs/rules").json():
        if (
            rule.get("session_id") == project_id
            and rule.get("display_name") == payload["display_name"]
        ):
            api.delete(f"/runs/rules/{rule['id']}").raise_for_status()
    resp = api.post("/runs/rules", json=payload)
    resp.raise_for_status()
    return str(resp.json()["id"])


def _schema(rubric: Rubric) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for c in rubric.criteria:
        fields[f"{c.name}_reasoning"] = (str, Field(description="one sentence"))
        fields[c.name] = (bool, Field(description=c.question))
    return create_model("MemoGrades", **fields)


class OnlineEvaluatorJudge:
    """The online evaluator's prompt and model, run locally (for calibration). Grades memos only."""

    rubric = MEMO_RUBRIC.name

    def __init__(self, model: str = DEFAULT_MODEL, chat: Any | None = None) -> None:
        self.model_name = f"openai:{model}"
        if chat is None:
            from langchain_openai import ChatOpenAI

            chat = ChatOpenAI(model=model, temperature=0)
        self._structured = chat.with_structured_output(_schema(MEMO_RUBRIC))

    def grade(self, rubric: Rubric, material: str, output: str) -> JudgeResult:
        names = rubric.names()
        t0 = time.monotonic()
        try:
            out = cast(
                BaseModel,
                self._structured.invoke(
                    [("system", SYSTEM), ("human", HUMAN.format(table=material, memo=output))]
                ),
            )
            data = out.model_dump()
            passed = {n: bool(data[n]) for n in names}
            reasons = {n: str(data.get(f"{n}_reasoning", "")) for n in names}
            error = None
        except Exception as e:  # no verdict is reported as an error, never as a pass or fail
            passed, reasons, error = {}, {}, f"{type(e).__name__}: {e}"
        usage = Usage(self.model_name, 0, 0, 0.0, time.monotonic() - t0, cached=False)
        return JudgeResult(rubric.name, passed, reasons, usage, error)
