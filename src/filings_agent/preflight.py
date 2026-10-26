"""What a command needs before it starts, checked up front so a fresh clone gets one clear line
instead of a traceback.

    uv run python -m filings_agent.preflight "make receipt" data cache   # used by the Makefile

Needs: `data` (the downloaded filings and eval datasets), `cache` (the cached model answers the
scores are recomputed from), `aws` (credentials for Bedrock), `langsmith` (an API key), `sec`
(the SEC contact the SEC requires). None of these are in the repo: `data/` is about 360 MB of
downloads and paid model outputs, and credentials are yours.
"""

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

NO_CREDENTIALS = "No credentials? `make demo` shows everything, and `make demo-check` checks it."


class SetupError(RuntimeError):
    """Something the user has to provide (data, credentials, a setting), not a bug."""


def _data_dir() -> Path:
    return Path(os.environ.get("FILINGS_DATA_DIR", "data"))


def need_sec() -> None:
    if not os.environ.get("SEC_USER_AGENT", "").strip():
        raise SetupError(
            "it needs SEC_USER_AGENT='Your Name you@example.com' in .env (the SEC requires a "
            "contact on every request). See .env.example."
        )


def need_data() -> None:
    if not (_data_dir() / "datasets" / "extract.jsonl").exists():
        raise SetupError(
            "it needs the eval data in data/, which isn't in the repo (downloaded filings and "
            "datasets). Build it with `make data` (needs SEC_USER_AGENT)."
        )


def need_cache() -> None:
    if not (_data_dir() / "cache" / "llm").is_dir():
        raise SetupError(
            "it recomputes results from the cached model answers in data/cache/, which aren't in "
            "the repo (they're paid Bedrock outputs). They're created by running the sweeps with "
            "AWS credentials."
        )


def need_aws() -> None:
    try:
        import boto3

        creds = boto3.Session(profile_name=os.environ.get("AWS_PROFILE") or None).get_credentials()
    except Exception as e:  # e.g. a named profile that doesn't exist
        raise SetupError(
            f"it needs AWS credentials for Bedrock ({e}). Set AWS_PROFILE in .env."
        ) from e
    if creds is None:
        raise SetupError("it needs AWS credentials for Bedrock. Set AWS_PROFILE in .env.")


def need_langsmith() -> None:
    if not os.environ.get("LANGSMITH_API_KEY", "").strip():
        raise SetupError(
            "it needs LANGSMITH_API_KEY in .env (smith.langchain.com, Settings, API Keys)."
        )


CHECKS: dict[str, Callable[[], None]] = {
    "sec": need_sec, "data": need_data, "cache": need_cache, "aws": need_aws,
    "langsmith": need_langsmith,
}  # fmt: skip


def explain(exc: BaseException) -> str | None:
    """A setup problem in plain words, or None if it's a real error that should surface."""
    if isinstance(exc, SetupError):
        return str(exc)
    if (
        isinstance(exc, FileNotFoundError)
        and exc.filename
        and str(exc.filename).startswith(str(_data_dir()))
    ):
        if "calibration" in str(exc.filename):
            return (
                f"{exc.filename} doesn't exist: prepare the calibration items first with "
                "`filings-agent calibrate prepare` (it needs the eval data from `make data`)."
            )
        return (
            f"{exc.filename} doesn't exist: the eval data in data/ isn't in the repo. Build it "
            "with `make data` (needs SEC_USER_AGENT)."
        )
    name = type(exc).__name__
    if name in ("NoCredentialsError", "ProfileNotFound", "PartialCredentialsError"):
        return "it needs AWS credentials for Bedrock. Set AWS_PROFILE in .env."
    if name in ("UnauthorizedSSOTokenError", "SSOTokenLoadError", "TokenRetrievalError"):
        return "your AWS SSO session has expired. Run `aws sso login`."
    if name == "LangSmithAuthError":
        return "LangSmith refused the API key. Set a valid LANGSMITH_API_KEY in .env."
    return None


def friendly(what: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a command's main: a setup problem prints one line and exits 2; anything else raises."""

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def run(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = explain(e)
                if msg is None:
                    raise
                print(f"{what}: {msg}\n{NO_CREDENTIALS}", file=sys.stderr)
                sys.exit(2)

        return run

    return wrap


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    args = argv if argv is not None else sys.argv[1:]
    what, needs = args[0], args[1:]
    for need in needs:
        try:
            CHECKS[need]()
        except SetupError as e:
            print(f"{what}: {e}\n{NO_CREDENTIALS}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
