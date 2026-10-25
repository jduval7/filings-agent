"""Reusable fake chat models that behave like `with_structured_output(..., include_raw=True)`."""

from __future__ import annotations

from typing import Any

from botocore.exceptions import ClientError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel


class ScriptedChat(BaseChatModel):
    """Returns scripted outputs in order (the last one repeats). An item can be:
    a pydantic model (parsed OK), None (schema failure), or an Exception (raised)."""

    script: list[Any]
    calls: int = 0
    seen_configs: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        chat = self

        class Structured:
            """Records the config exactly as the step passed it to invoke()."""

            def invoke(self, messages: Any, config: Any = None) -> dict[str, Any]:
                chat.seen_configs.append(config or {})
                return run(messages)

        def run(_: Any) -> dict[str, Any]:
            item = self.script[min(self.calls, len(self.script) - 1)]
            self.calls += 1
            if isinstance(item, BaseException):
                raise item
            usage = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}
            parsed = item if isinstance(item, BaseModel) else None
            return {
                "raw": AIMessage(content="", usage_metadata=usage),
                "parsed": parsed,
                "parsing_error": None if parsed else ValueError("schema mismatch"),
            }

        return Structured()


def client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "Converse")
