"""LangSmith's online evaluator: what's pasted into LangSmith is what gets calibrated here."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from filings_agent.evals import online_judge as oj
from filings_agent.steps.judge import MEMO_RUBRIC


class FakeChat:
    """Stands in for ChatOpenAI: structured output that passes everything but `faithful`."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.seen: list[Any] = []

    def with_structured_output(self, schema: Any) -> Any:
        chat = self

        class Runner:
            def invoke(self, messages: Any) -> Any:
                chat.seen.append(messages)
                if chat.fail:
                    raise TimeoutError("no answer")
                grades: dict[str, Any] = {}
                for c in MEMO_RUBRIC.criteria:
                    grades[c.name] = c.name != "faithful"
                    grades[f"{c.name}_reasoning"] = f"because {c.name}"
                return schema(**grades)

        return Runner()


def test_the_setup_lists_everything_the_langsmith_form_asks_for() -> None:
    text = oj.setup_text()
    for key, question in oj.criteria():
        assert key in text and question in text
    assert oj.FILTER in text and "outputs.memo.table" in text and oj.SYSTEM in text
    assert "few-shot corrections" in text


def test_it_grades_the_memo_against_its_table() -> None:
    chat = FakeChat()
    judge = oj.OnlineEvaluatorJudge("gpt-test", chat=chat)
    res = judge.grade(MEMO_RUBRIC, "the table", "the memo")
    assert res.error is None and res.passed["faithful"] is False and res.passed["actionable"]
    assert res.reasons["comparable"] == "because comparable"
    human = chat.seen[0][1][1]
    assert "the table" in human and "the memo" in human  # the variables the evaluator maps
    assert judge.model_name == "openai:gpt-test" and judge.rubric == MEMO_RUBRIC.name


def test_no_answer_is_an_error_not_a_verdict() -> None:
    res = oj.OnlineEvaluatorJudge("gpt-test", chat=FakeChat(fail=True)).grade(MEMO_RUBRIC, "t", "m")
    assert res.error and res.passed == {}


def test_calibration_with_the_online_evaluator_grades_memo_items_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import yaml

    from filings_agent.config import load_config
    from filings_agent.evals import calibration
    from filings_agent.evals.calibration import CalibrationItem

    items = [
        CalibrationItem(item_id="memo-1", rubric="memo", source_model="claude-haiku-4.5",
                        material="table", output="memo", context={}),
        CalibrationItem(item_id="rec-1", rubric="reconcile_explanation", source_model="nova-lite",
                        material="evidence", output="why", context={}),
    ]  # fmt: skip

    def load_items(cfg: Any) -> list[CalibrationItem]:
        return items

    class WithFakeChat(oj.OnlineEvaluatorJudge):
        def __init__(self, model: str) -> None:
            super().__init__(model, chat=FakeChat())

    monkeypatch.setattr(calibration, "load_items", load_items)
    monkeypatch.setattr(oj, "OnlineEvaluatorJudge", WithFakeChat)
    memo = {"faithful": False, "caveats_correct": True, "comparable": True, "actionable": True}
    rec = {"supported": True, "specific": True, "consistent": True}
    labels = tmp_path / "labels.yaml"
    labels.write_text(yaml.safe_dump({"memo-1": memo, "rec-1": rec}))
    result = calibration.score(load_config(), "openai:gpt-test", labels)
    assert result["judge"] == "openai:gpt-test" and result["items_graded"] == 1  # memo only
    assert {c["criterion"].split(".")[0] for c in result["criteria"]} == {"memo"}
    assert result["disagreements"] == []  # the fake agrees with the labels


class FakeRulesAPI:
    """The three /runs/rules calls create_rule makes, against an in-memory list."""

    def __init__(self, rules: list[dict[str, Any]]) -> None:
        self.rules = rules
        self.deleted: list[str] = []
        self.posted: list[dict[str, Any]] = []

    def get(self, path: str) -> Any:
        return SimpleNamespace(json=lambda: self.rules)

    def delete(self, path: str) -> Any:
        self.deleted.append(path.rsplit("/", 1)[-1])
        return SimpleNamespace(raise_for_status=lambda: None)

    def post(self, path: str, json: dict[str, Any]) -> Any:
        self.posted.append(json)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"id": "new"})


def test_rule_payload_scores_published_memos_with_one_boolean_per_question() -> None:
    p = oj.rule_payload("proj", 0.1, backfill_from="2026-10-24T00:00:00Z")
    assert p["session_id"] == "proj" and p["sampling_rate"] == 0.1
    assert p["filter"] == oj.FILTER_EXPR and "eq(feedback_score, 1)" in p["filter"]
    assert p["spend_limit"]["limit_usd"] == oj.WEEKLY_LIMIT_USD
    structured = p["evaluators"][0]["structured"]
    props = structured["schema"]["properties"]
    assert set(props) == {f"judge_{c.name}" for c in MEMO_RUBRIC.criteria}  # no dots: API rule
    assert all(v["type"] == "boolean" for v in props.values())
    assert structured["variable_mapping"] == {
        "table": "output.memo.table",
        "memo": "output.memo.markdown",
    }
    model = structured["model"]["kwargs"]
    assert model["temperature"] == 0 and model["openai_api_key"] == oj.OPENAI_SECRET


def test_create_rule_replaces_only_its_own_rule_on_that_project() -> None:
    name = oj.rule_payload("proj", 1.0)["display_name"]
    api = FakeRulesAPI(
        [
            {"id": "old", "session_id": "proj", "display_name": name},
            {"id": "other-project", "session_id": "prod", "display_name": name},
            {"id": "someone-elses", "session_id": "proj", "display_name": "toxicity"},
        ]
    )
    assert oj.create_rule(api, "proj", 1.0, None) == "new"
    assert api.deleted == ["old"]
    assert api.posted[0]["session_id"] == "proj"
