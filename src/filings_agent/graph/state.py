"""Graph state. Every accumulating field is a dict keyed by a stable id with a merge reducer.

Why not list-append (`operator.add`)? If a branch's write is applied twice (a retry, a replay
after resume), a list duplicates it. A keyed merge is idempotent: applying the same write twice
leaves the same state. f(f(s, x), x) == f(s, x).
"""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired, TypedDict

from filings_agent.domain import Discrepancy, ExtractedFact
from filings_agent.research.findings import ResearchResult
from filings_agent.steps.extract import ExtractTask, Rejection, Usage
from filings_agent.steps.reconcile import ReconcileTask, Resolution


def merge[V](left: dict[str, V] | None, right: dict[str, V] | None) -> dict[str, V]:
    """Idempotent keyed merge: right wins per key, nothing is ever appended twice."""
    return {**(left or {}), **(right or {})}


class AnalysisInput(TypedDict):
    tickers: list[str]


class AnalysisState(TypedDict, total=False):
    tickers: list[str]
    tasks: list[ExtractTask]
    facts: Annotated[dict[str, ExtractedFact], merge]  # fact_id -> fact
    rejected: Annotated[dict[str, list[Rejection]], merge]  # task_id -> rejections
    usage: Annotated[dict[str, Usage], merge]  # task_id -> cost/latency
    errors: Annotated[dict[str, str], merge]  # task_id -> why the branch failed
    # wave 2
    discrepancies: Annotated[dict[str, Discrepancy], merge]  # discrepancy_id -> discrepancy
    reconcile_tasks: list[ReconcileTask]
    resolutions: Annotated[dict[str, Resolution], merge]  # discrepancy_id -> resolution
    reconcile_usage: Annotated[dict[str, Usage], merge]
    # human review + publish
    decisions: Annotated[dict[str, Decision], merge]  # discrepancy_id -> human decision
    # reviewer-triggered research, keyed "<discrepancy_id>#r<round>" (a round runs once)
    research_requests: Annotated[dict[str, ResearchRequest], merge]
    research: Annotated[dict[str, ResearchResult], merge]
    research_usage: Annotated[dict[str, Usage], merge]
    memo: dict[str, object]  # markdown, verified, unverified numbers, attempts
    memo_usage: Usage
    publication: dict[str, object]


class Decision(TypedDict):
    """A human's call on one material change."""

    # accept = use the newest (recast) value; escalate = no candidate explains it: published
    # now, flagged as unresolved
    # research = not enough to decide: run the research agent on this case (note = optional
    # hint for it) and show the card again
    action: Literal["accept", "use_original", "escalate", "research"]
    note: NotRequired[str]
    # The cause the reviewer picked on the card. When it differs from the agent's (hidden) answer
    # it becomes a regression test (see corrections).
    correct_cause: NotRequired[str]
    # Who decided and when, stamped by the server (API: the authenticated user; CLI: --reviewer),
    # never taken from the client's payload. Kept in the ledger's resolution history.
    actor: NotRequired[str]
    decided_at: NotRequired[str]


class ResearchRequest(TypedDict):
    discrepancy_id: str
    hint: str
    round: int


class ResearchInput(TypedDict):
    task: ReconcileTask
    request: ResearchRequest


class ReconcileInput(TypedDict):
    task: ReconcileTask


class ExtractInput(TypedDict):
    """Private input for one fan-out branch: just its statement, not the whole state."""

    task: ExtractTask
