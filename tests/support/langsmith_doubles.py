"""LangSmith client doubles: datasets kept in memory, and an `evaluate` that runs the target over
the dataset's examples locally, like LangSmith does."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any


class FakeClient:
    def __init__(self) -> None:
        self.datasets: dict[str, dict[str, dict[str, Any]]] = {}
        self.created: list[str] = []
        self.updated: list[str] = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        return dataset_name in self.datasets

    def create_dataset(self, dataset_name: str, *, description: str | None = None) -> Any:
        self.datasets[dataset_name] = {}
        return SimpleNamespace(id=dataset_name)

    def read_dataset(self, *, dataset_name: str) -> Any:
        return SimpleNamespace(id=dataset_name)

    def list_examples(self, *, dataset_id: str) -> list[Any]:
        return [SimpleNamespace(id=i) for i in self.datasets[dataset_id]]

    def create_examples(self, *, dataset_id: str, examples: list[dict[str, Any]]) -> None:
        for ex in examples:
            self.datasets[dataset_id][ex["id"]] = ex
            self.created.append(ex["id"])

    def update_examples(self, *, dataset_id: str, updates: list[dict[str, Any]]) -> None:
        for ex in updates:
            self.datasets[dataset_id][ex["id"]] = ex
            self.updated.append(ex["id"])


class EvalResults(list[dict[str, Any]]):
    experiment_name = "exp-123"


class EvaluatingClient:
    """`evaluate` runs the target over the dataset's examples like LangSmith does, locally."""

    def __init__(self, examples: dict[str, list[dict[str, Any]]]) -> None:
        self.examples = examples
        self.calls: list[dict[str, Any]] = []

    def evaluate(self, target: Callable[[dict[str, Any]], dict[str, Any]], *, data: str,
                 **kwargs: Any) -> EvalResults:  # fmt: skip
        self.calls.append({"data": data, **kwargs})
        out = EvalResults()
        for ex in self.examples[data]:
            run = SimpleNamespace(outputs=target(ex["inputs"]))
            example = SimpleNamespace(
                inputs=ex["inputs"], outputs=ex["outputs"], metadata=ex.get("metadata", {})
            )
            out.append({"run": run, "example": example})
        return out


def check_feedback(kwargs: dict[str, Any]) -> None:
    """Validate feedback values the way the real client does. The real `create_feedback` takes
    **kwargs, so a signature check can't catch a bad value ("app" was rejected in production)."""
    from langsmith import schemas

    if "feedback_source_type" in kwargs:
        schemas.FeedbackSourceType(kwargs["feedback_source_type"])
    score = kwargs.get("score")
    if score is not None and not isinstance(score, (int, float, bool)):
        raise TypeError(f"score must be a number, got {type(score).__name__}")
