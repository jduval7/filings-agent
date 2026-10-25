"""What a research round produces. Plain data, kept apart from the agent so the review card and
the checkpoint serializer don't import the agent stack."""

from __future__ import annotations

from dataclasses import dataclass

from filings_agent.steps.extract import Usage


@dataclass(frozen=True)
class VerifiedFinding:
    cause: str  # one of the four causes, or "other"
    other_cause: str
    stance: str  # supports | contradicts
    quote: str
    source_url: str
    note: str
    fetched_at: str


@dataclass(frozen=True)
class ResearchResult:
    discrepancy_id: str
    round: int
    hint: str
    findings: tuple[VerifiedFinding, ...]
    dropped: int  # findings whose quote wasn't in the cited source
    looked_at: str
    open_questions: str
    usage: Usage | None
    error: str | None = None
