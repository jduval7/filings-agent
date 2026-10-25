"""Which failures are worth retrying, and the retry layers built on that one definition.

Transient means throttling or an outage: retry with backoff. Anything else (the model's own bad
output, a parse error, a spend cap) is never retried. The graph's RetryPolicy (graph/build.py)
and the agents' LangChain retry middleware (reconcile_agent/agent.py, research/agent.py) all use
`is_transient`, so a throttle is treated the same wherever it happens.
"""

from __future__ import annotations

from botocore.exceptions import ClientError, HTTPClientError
from botocore.exceptions import ConnectionError as BotoConnectionError
from langchain.agents.middleware import ModelRetryMiddleware

TRANSIENT_CODES = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelNotReadyException",
        "ModelTimeoutException",
    }
)


def is_transient(exc: BaseException) -> bool:
    """Retry throttling and outages; never retry the model's own bad output or a spend cap."""
    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code", "") in TRANSIENT_CODES
    # botocore's network errors (read timeout, connect timeout, endpoint unreachable, connection
    # closed) are not the builtin TimeoutError/ConnectionError, so they used to count as bad model
    # output and fail the branch; botocore's own one retry hid it. It now makes no retries.
    return isinstance(exc, (TimeoutError, ConnectionError, HTTPClientError, BotoConnectionError))


def model_retry(initial_delay: float = 1.0) -> ModelRetryMiddleware:
    """LangChain's retry middleware for an agent's model calls: a throttle on one call retries
    that call, instead of failing the whole agent run (and the graph node re-running all of it).
    `on_failure="error"`: once the retries are used up the error is raised, so the node's
    RetryPolicy and the run's error handling still see a persistent outage. (The default,
    "continue", would hand the model an error message and carry on.) Every attempt goes through
    the spend guard, so retries are counted and capped like any other call."""
    return ModelRetryMiddleware(
        max_retries=2, retry_on=is_transient, on_failure="error", initial_delay=initial_delay
    )
