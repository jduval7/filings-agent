# langchain's create_agent generics don't line up for middleware with different context types;
# this module is the typing boundary that accepts it (behaviour is covered by tests).
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportCallIssue=false, reportArgumentType=false
"""The reconcile agent: a LangChain `create_agent` that investigates WHY a reported number changed.

Where it sits: the graph's `reconcile` node calls a `Reconciler`. With `reconcile.strategy: agent`
that's `AgentReconciler` (this module); with `pipeline` it's the fixed `LLMReconciler`
(steps/reconcile.py), kept as the baseline. The rest of the graph is unchanged: orchestration
stays in code, the agent is one bounded step inside it.

Harness (per the LangChain docs, "agent = model + harness"):
  tools       fact_history (where did it change?) and search_filing (RAG: why?)
  context     ReconcileContext, passed at invoke time, read by tools via ToolRuntime
  output      response_format=AgentAnswer (structured: cause, explanation, exact quote)
  middleware  model/tool call limits (bounded cost), tool errors -> messages the model can act on
  after       code verifies the quote is in a passage the agent actually retrieved, else the
              answer is downgraded to "unknown" and goes to a human (same rule as the pipeline)

Versions are explicit (`reconcile.agent_version`) so evals can compare them, and every version is
part of the cache key.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from botocore.exceptions import ClientError
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    wrap_tool_call,
)
from langchain.agents.structured_output import StructuredOutputError, ToolStrategy
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from filings_agent.config import ModelSpec
from filings_agent.domain import FilingRef
from filings_agent.llm.cache import StepCache
from filings_agent.reconcile_agent.retrieval import FilingIndex, Passage
from filings_agent.reconcile_agent.tools import TOOLS, ReconcileContext
from filings_agent.steps.extract import MODEL_FAILURE_CODES, Usage
from filings_agent.steps.reconcile import Cause, ReconcileTask, Resolution, verify_quote

# Part of the cache key. Bump when the harness changes how results are produced or collected,
# so stale results can't be reused. "artifact-1": evidence read from search_filing artifacts.
HARNESS = "toolstrategy-1"


def retrieved_passages(messages: list[Any]) -> list[dict[str, Any]]:
    """What the agent actually saw: the artifacts of its search_filing calls, de-duplicated."""
    seen: dict[str, dict[str, Any]] = {}
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "search_filing" and m.artifact:
            for p in cast(list[dict[str, Any]], m.artifact):
                seen[str(p["passage_id"])] = p
    return list(seen.values())


class AgentAnswer(BaseModel):
    cause: Cause
    explanation: str = Field(description="One or two sentences, plain English.")
    evidence_quote: str = Field(
        description="An exact quote (one contiguous passage, under 300 characters) copied from a "
        "search_filing result. Empty only if cause is 'unknown'."
    )
    passage_id: str = Field(default="", description="The [id] of the passage quoted.")


CAUSES = (
    "Causes:\n"
    "- stock_split: share counts changed by a split ratio (e.g. x10, x3); dollar amounts don't.\n"
    "- discontinued_operations: a business was spun off or sold and prior years were recast to "
    "exclude it.\n"
    "- accounting_change: a new accounting standard was adopted retrospectively.\n"
    "- reclassification: amounts moved between lines or segments to conform to the current "
    "presentation.\n"
    "- unknown: the filings don't explain the change. Prefer unknown over guessing."
)

PROMPTS: dict[str, str] = {
    "v1": (
        "You investigate why a company reported the same figure differently in two annual "
        "reports.\n"
        "1. Call fact_history to see which report changed the number.\n"
        "2. Call search_filing on the report(s) where it changed to find the explanation.\n"
        "3. Answer with the cause, a short explanation, and an exact quote from a search result.\n"
        + CAUSES
    ),
}


@wrap_tool_call
def tool_errors_to_messages(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]
) -> ToolMessage | Command[Any]:
    """An unexpected tool failure becomes a message the model can react to, not a crash."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: {type(e).__name__}: {e}. Check the arguments and try again.",
            tool_call_id=request.tool_call["id"],
        )


def build_reconcile_agent(model: BaseChatModel, version: str = "v1") -> Any:
    """The compiled agent graph (a LangGraph subgraph). Bounded: at most 8 model calls and
    6 tool calls per discrepancy, so a confused agent can't loop up the bill."""
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=PROMPTS[version],
        # ToolStrategy, not the bare schema: a bare schema lets create_agent pick native structured
        # output per provider (Sonnet on Bedrock), where one malformed JSON reply raised and killed
        # the run. ToolStrategy is the same for every model, and handle_errors feeds a parse error
        # back to the model to retry (bounded by the model-call limit).
        response_format=ToolStrategy(AgentAnswer, handle_errors=True),
        context_schema=ReconcileContext,
        middleware=[
            ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=6, exit_behavior="continue"),
            tool_errors_to_messages,
        ],
        name="reconcile_agent",
    )


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    accession_no: str  # "" for tools without a filing argument
    query: str


@dataclass(frozen=True)
class AgentOutcome:
    """Everything an eval needs: the answer, the path, and what retrieval returned."""

    resolution: Resolution
    tool_calls: tuple[ToolCallRecord, ...]
    passages: tuple[Passage, ...]


def question(task: ReconcileTask) -> str:
    d = task.discrepancy
    first, last = d.values[0][1], d.values[-1][1]
    return (
        f"{d.ticker} reported {d.metric.value} for fiscal year {d.fiscal_year} as {first:,} in an "
        f"earlier annual report and {last:,} in a later one. Why did it change?"
    )


class AgentReconciler:
    """Implements the graph's `Reconciler` protocol with the tool-using agent."""

    def __init__(
        self,
        model_name: str,
        spec: ModelSpec,
        chat: BaseChatModel,
        cache: StepCache,
        index: FilingIndex,
        filings_of: Callable[[str], dict[str, FilingRef]],
        version: str = "v1",
        k: int = 5,
    ) -> None:
        self.name = f"reconcile-agent/{model_name}/{version}"
        self._model_name = model_name
        self._spec = spec
        self._agent = build_reconcile_agent(chat, version)
        self._cache = cache
        self._index = index
        self._filings_of = filings_of
        self._version = version
        self._k = k

    def reconcile(self, task: ReconcileTask, rep: int = 0) -> Resolution:
        return self.investigate(task, rep).resolution

    def investigate(self, task: ReconcileTask, rep: int = 0) -> AgentOutcome:
        d = task.discrepancy
        key = StepCache.key(
            step="reconcile-agent",
            harness=HARNESS,
            model=self._spec.id,
            version=self._version,
            prompt=PROMPTS[self._version],
            discrepancy=d.model_dump(mode="json"),
            filings=sorted(self._filings_of(d.ticker)),  # what the tools can search is an input
            index=self._index.version,
            k=self._k,
            rep=rep,
        )
        record = self._cache.get(key)
        cached = record is not None
        if record is None:
            record = self._run(task)
            if record.get("error") is None or not str(record["error"]).startswith("model_error"):
                self._cache.put(key, record)
        return self._outcome(task, record, cached)

    def _run(self, task: ReconcileTask) -> dict[str, Any]:
        d = task.discrepancy
        ctx = ReconcileContext(
            discrepancy=d, filings=self._filings_of(d.ticker), index=self._index, k=self._k
        )
        t0 = time.monotonic()
        error: str | None = None
        messages: list[Any] = []
        answer: dict[str, Any] | None = None
        try:
            result = cast(
                dict[str, Any],
                self._agent.invoke(
                    {"messages": [HumanMessage(question(task))]},
                    config={
                        "run_name": f"reconcile-agent {d.id}",
                        "tags": [
                            "step:reconcile",
                            "strategy:agent",
                            f"model:{self._model_name}",
                            f"ticker:{d.ticker}",
                        ],
                        "metadata": {
                            "discrepancy_id": d.id,
                            "agent_version": self._version,
                            "retrieval": self._index.version,
                        },
                    },
                    context=ctx,
                ),
            )
            messages = list(result.get("messages", []))
            structured = result.get("structured_response")
            if isinstance(structured, AgentAnswer):
                answer = structured.model_dump(mode="json")
            else:
                error = "no structured answer (limit reached or model stopped early)"
        except StructuredOutputError as e:  # retries exhausted: unknown, and don't cache it
            error = f"model_error:structured_output:{type(e).__name__}"
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in MODEL_FAILURE_CODES:
                raise  # throttling/outage: the graph's RetryPolicy handles it
            error = f"model_error:{code}"
        calls = [
            {
                "name": tc["name"],
                "accession_no": str(tc["args"].get("accession_no", "")),
                "query": str(tc["args"].get("query", "")),
            }
            for m in messages
            if isinstance(m, AIMessage)
            for tc in m.tool_calls
        ]
        in_tok = sum(
            (m.usage_metadata or {}).get("input_tokens", 0)
            for m in messages
            if isinstance(m, AIMessage)
        )
        out_tok = sum(
            (m.usage_metadata or {}).get("output_tokens", 0)
            for m in messages
            if isinstance(m, AIMessage)
        )
        return {
            "answer": answer,
            "error": error,
            "tool_calls": calls,
            "passages": retrieved_passages(messages),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_s": time.monotonic() - t0,
        }

    def _outcome(self, task: ReconcileTask, rec: dict[str, Any], cached: bool) -> AgentOutcome:
        d = task.discrepancy
        usage = Usage(
            self._model_name,
            int(rec["input_tokens"]),
            int(rec["output_tokens"]),
            self._spec.cost(int(rec["input_tokens"]), int(rec["output_tokens"])),
            float(rec["latency_s"]),
            cached,
        )
        passages = tuple(Passage(**p) for p in rec["passages"])
        calls = tuple(ToolCallRecord(**c) for c in rec["tool_calls"])
        accn, value = d.values[-1]
        if rec["answer"] is None:
            res = Resolution(
                task.discrepancy_id,
                Cause.UNKNOWN,
                "Agent returned no answer.",
                "",
                True,
                False,
                value,
                accn,
                True,
                usage,
                rec["error"],
            )
            return AgentOutcome(res, calls, passages)
        ans = AgentAnswer.model_validate(rec["answer"])
        # The quote must come from text the agent actually retrieved: retrieval is the evidence.
        quoted = verify_quote(ans.evidence_quote, [p.text for p in passages])
        verified = ans.cause is Cause.UNKNOWN or quoted
        res = Resolution(
            discrepancy_id=task.discrepancy_id,
            cause=ans.cause if verified else Cause.UNKNOWN,
            explanation=ans.explanation if verified else f"Unverified claim: {ans.explanation}",
            evidence_quote=ans.evidence_quote,
            material=True,
            verified=verified,
            use_value=value,
            use_accession=accn,
            needs_review=True,
            usage=usage,
        )
        return AgentOutcome(res, calls, passages)


def record_json(outcome: AgentOutcome) -> str:
    """Compact, human-readable trajectory for reports."""
    return json.dumps(
        [
            f"{c.name}({c.accession_no or ''}{': ' + c.query if c.query else ''})"
            for c in outcome.tool_calls
        ]
    )
