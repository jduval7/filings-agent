# langchain create_agent is only partly typed (see research/agent.py).
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false

"""One definition of 'worth retrying', shared by the graph's RetryPolicy and agent middleware."""

from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from filings_agent.llm.transient import is_transient, model_retry

THROTTLED = ClientError({"Error": {"Code": "ThrottlingException"}}, "Converse")
BAD_REQUEST = ClientError({"Error": {"Code": "ValidationException"}}, "Converse")


class Flaky(BaseChatModel):
    """Raises `error` for the first `failures` calls, then answers."""

    failures: int = 1
    error: Exception = THROTTLED
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "flaky"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return ChatResult(generations=[ChatGeneration(message=AIMessage("answer"))])

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def _agent(model: Flaky) -> Any:
    return create_agent(model=model, tools=[], middleware=[model_retry(initial_delay=0.0)])


def test_a_throttled_model_call_is_retried_inside_the_agent() -> None:
    model = Flaky(failures=2)  # two throttles, then an answer: within the 2 retries
    out = _agent(model).invoke({"messages": [("user", "why?")]})
    assert out["messages"][-1].content == "answer" and model.calls == 3


def test_a_persistent_outage_is_raised_not_turned_into_a_message() -> None:
    # on_failure="error": the graph's RetryPolicy and the run's error handling must see it
    model = Flaky(failures=10)
    with pytest.raises(ClientError):
        _agent(model).invoke({"messages": [("user", "why?")]})
    assert model.calls == 3  # the first try plus 2 retries


def test_a_bad_request_is_never_retried() -> None:
    model = Flaky(failures=1, error=BAD_REQUEST)
    with pytest.raises(ClientError):
        _agent(model).invoke({"messages": [("user", "why?")]})
    assert model.calls == 1


def test_what_counts_as_transient() -> None:
    assert is_transient(THROTTLED) and is_transient(TimeoutError())
    assert not is_transient(BAD_REQUEST) and not is_transient(ValueError("bad JSON"))
