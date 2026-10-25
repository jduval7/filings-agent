"""The review UI's fixtures are what the API really returns (see tests/support/api_fixtures.py)."""

from __future__ import annotations

from tests.support.api_fixtures import OUT, build, render

REGENERATE = "uv run python -m tests.support.api_fixtures --write"


def test_ui_fixtures_match_what_the_api_returns() -> None:
    for name, value in build().items():
        path = OUT / f"analysis-{name}.json"
        assert path.exists(), f"missing {path}: run `{REGENERATE}`"
        assert path.read_text() == render(name, value), (
            f"{path.name} is stale: the API's response changed. Regenerate with `{REGENERATE}`, "
            "then fix the web types and tests."
        )
