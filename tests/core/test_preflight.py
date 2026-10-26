"""A fresh clone gets one clear line, not a traceback, when a command needs something it lacks."""

from __future__ import annotations

from pathlib import Path

import pytest

from filings_agent import preflight


def test_missing_data_is_explained_not_crashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FILINGS_DATA_DIR", str(tmp_path / "data"))
    assert preflight.main(["make receipt", "data", "cache"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("make receipt: it needs the eval data in data/")
    assert "make demo" in err and "Traceback" not in err


def test_present_data_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "datasets").mkdir(parents=True)
    (tmp_path / "datasets" / "extract.jsonl").write_text("")
    (tmp_path / "cache" / "llm").mkdir(parents=True)
    monkeypatch.setenv("FILINGS_DATA_DIR", str(tmp_path))
    assert preflight.main(["make receipt", "data", "cache"]) == 0


def _no_dotenv(*_: object, **__: object) -> bool:
    return False


def test_missing_keys_are_named(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("dotenv.load_dotenv", _no_dotenv)  # no .env fills them in
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert preflight.main(["make data", "sec"]) == 2
    assert "SEC_USER_AGENT" in capsys.readouterr().err
    assert preflight.main(["make import-qa", "langsmith"]) == 2
    assert "LANGSMITH_API_KEY" in capsys.readouterr().err


def test_friendly_turns_setup_errors_into_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FILINGS_DATA_DIR", "data")

    @preflight.friendly("filings-agent")
    def reads_missing_data() -> None:
        Path("data/datasets/extract.jsonl").read_text()

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stop:
        reads_missing_data()
    assert stop.value.code == 2
    assert "make data" in capsys.readouterr().err


def test_real_bugs_still_raise() -> None:
    @preflight.friendly("filings-agent")
    def broken() -> None:
        raise ValueError("a real bug")

    with pytest.raises(ValueError, match="a real bug"):
        broken()
