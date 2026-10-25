# Contributing / handoff guide

For the engineer who picks this up next. Read this, then `README.md` (what it does), `EVALS.md`
(how we know it works), `RUNBOOK.md` (how to operate it), `MAINTAINING.md` (how to keep improving
it) and `DECISIONS.md` (why it is the way it is).

## Mental model in one paragraph

A LangGraph `StateGraph` orchestrates everything in code: it locates statements, fans out one
extraction branch per statement (`Send`), detects restated figures in plain Python, fans out one
reconcile branch per material restatement, **triages** each answer with accounting rules in code,
pauses for a human (`interrupt()`) on everything code can't confirm, writes a memo and publishes
once. The human sees a **review card** per change (every candidate cause with its evidence and fit,
never the agent's answer) and can pick a cause, escalate, or ask for **research**: the graph sends the
case to one bounded investigation per cause over live, allowlisted sources, then shows the card
again. LLMs do bounded jobs inside nodes (read a table, explain a restatement, investigate one cause,
write prose); **code checks every LLM output against its source** (numbers in the statement, quotes
in the evidence or the cited page, memo numbers in the table) and **code rules decide what's ruled
out**. The reconcile step can be a fixed pipeline (production) or a tool-using `create_agent` (the
measured comparison). Every step has its own eval dataset and is scored in LangSmith.

## Where things live

```
src/filings_agent/
  config.py              typed config (configs/default.yaml): models, roles, prompts, strategies
  domain.py              the core types: Metric, FilingRef, GoldFact, ExtractedFact, Discrepancy
  sec/                   SEC data: throttled+cached client, statement locator, XBRL answer key
  steps/                 one module per LLM step, each = strategy + deterministic checks
    extract.py           statement -> facts (LLM reads, code parses/verifies)
    reconcile.py         restatement -> cause (pipeline strategy: keyword evidence + 1 call)
    triage.py            does the cause fit the numbers? accounting rules -> auto | review lane
    card.py              the review card: every cause, evidence, fit; never the agent's answer
    memo.py              table -> memo (code builds numbers, LLM writes prose, checker)
    judge.py             LLM-as-judge rubrics + pairwise
  reconcile_agent/       the agent strategy for reconcile (LangChain create_agent)
    retrieval.py         chunk -> embed (cached) -> per-filing vector store; vector | hybrid
    tools.py             @tool fact_history, @tool search_filing; ReconcileContext
    agent.py             create_agent + middleware + versioned prompts; AgentReconciler
    factory.py           builds the configured reconciler (pipeline | agent) for everyone
  research/              reviewer-triggered research (one create_agent per cause, fanned out by code)
    sources.py           live sources behind an allowlist: SEC EDGAR + PwC Viewpoint; cache
    tools.py             read_changed_filing, search_sec_filings, read_source, list_guidance
    agent.py             CAUSE_BRIEFS, investigators, quote verification; Researcher
    findings.py          plain result types (shared by the card and the checkpoint serializer)
  graph/
    state.py             graph state (keyed, idempotent reducers)
    build.py             the graph: nodes, Send fan-outs, triage, review <-> research loop, publish
    runner.py            checkpoints (SQLite), streaming, run/resume, status
    studio.py            entry point for LangGraph Studio (langgraph.json)
  evals/                 everything that measures: scorers, LangSmith experiments, gate, judge
                         calibration, trajectories, online checks
  llm/                   model factory (config name -> chat model), spend guard, step cache
  storage/ledger.py      the one side effect: idempotent publish, append-only decision history
  storage/leases.py      one process per thread (a lease row; API workers and the CLI share it)
  api/app.py             FastAPI + SSE
web/                     the review UI: React + TypeScript (Vite), served by FastAPI at / once built
  src/decisions.ts       what a complete decision is, and its API payload (pure, tested)
  src/components/        ReviewCard (one card), AnalysisView (a run: progress, cards, result)
tests/                   mirrors src/ (tests/steps/test_card.py tests steps/card.py); no network, no LLM
  support/               shared test doubles: graph_doubles (fake store, extractors, reconciler,
                         review_env), cases (GE net income), research_doubles, langsmith_doubles,
                         eval_data, fakes (scripted chat models). Tests never import other tests.
```

## Invariants (break these and the results stop meaning anything)

1. **The agent never reads the answer key.** XBRL (`data/gold/`) is for grading only.
2. **Every LLM output is checked by code against its source.** Unverified = rejected or "unknown",
   never silently accepted.
3. **What's evaluated is what runs.** Models and prompt/agent versions come from config only, via
   one factory per step. No defaults in classes (the type checker enforces the required arguments).
4. **The test split is run once, at the end.** Tune on dev. The **fresh** split holds companies
   added after choices were frozen: never tune on it, keep it out of the gate, report it separately.
   When it's been used to make a decision, it's no longer fresh: add new companies for the next check.
5. **The judge never grades its own model's output** (config validation), and judge scores are
   indicative until `calibrate score` says TRUSTED.
6. **No side effects before `interrupt()`.** The review node re-runs from the top on resume.
7. **State reducers are keyed merges.** Never `operator.add` on accumulating state: replays and
   retries must not duplicate.
8. **Every paid call goes through the spend guard** (`llm/models.py` installs it on the client).
   `SpendLimitExceeded` is a BaseException: never catch it.
9. **Caches are content-addressed** (model, prompt/version, input, repetition). If you change
   anything that changes the output, it must be part of the key (research also keys on `HARNESS` and
   `SOURCES_VERSION`: bump them when the harness or the text extraction changes).
10. **Only code rules rule a cause out.** An LLM's "contradicts" or a pile of research findings is
    shown as evidence; it never changes a cause's fit on the card.
11. **Nothing marks a favorite.** The card lists causes in a fixed order and never shows the agent's
    answer or its hand-picked quote; research output has no "most likely" field. Tests guard both.
12. **Auto-approval needs every link confirmed by code** (`triage.py`): "not covered" means a human
    looks, never "probably fine". New rules must be general accounting, not per-company.
13. **Research reads only allowlisted sources** (`research/sources.py`), the SEC contact header goes
    to sec.gov only, and fetched text is data, never instructions.

## Recipes

**Try a different model for a step**
1. Add it to `models:` in `configs/default.yaml` with its price (AWS Pricing API).
2. `uv run filings-agent score <step> --models <name>`; then `sweep` to record it in LangSmith.
3. If it's better, change `roles:`; run `make gate`; record the decision in DECISIONS.md.

**Change a prompt**
Add a new version (`"v3"`) next to the old one in the step's `PROMPTS` dict. Never edit a version
in place: cached results and past experiments refer to it. Score, compare, switch `prompts:`.

**Add a tool to the reconcile agent**
1. Write it in `reconcile_agent/tools.py`: `@tool`, typed args, a docstring written for the model,
   read per-run data from `runtime.context` (`ReconcileContext`), return errors as text.
2. Add it to `TOOLS`; bump the agent version (`PROMPTS["v3"]`) if the prompt mentions it.
3. Unit-test it with the fake tool-calling model (`tests/test_reconcile_agent.py`).
4. Re-run `score reconcile --reconciler agent` and check the trajectory and retrieval metrics.

**Add a company**
Add the ticker to `companies:` (or `reconcile_companies:`) and to a split, `make data`. For
reconcile cases, label each material discrepancy in `configs/reconcile_labels.yaml` with the
cause, the citing passage, and an `anchors:` phrase; the labelling script verifies anchors exist
in the filing where the number changed.

**Add a triage rule** (e.g. "a reclassification can't change total net income")
1. Confirm it against the codification or a Big 4 guide, and cite it in the reason text.
2. Add it to `CANNOT_MOVE` (rules out) or a fingerprint check (confirms) in `steps/triage.py`.
3. Unit-test the rule, then run `filings-agent cards --eval`: `true_cause_hidden` must stay empty.
4. `score reconcile` (and `--reconciler agent`) to see the lanes move; DECISIONS entry.

**Add a cause** (e.g. error correction)
Add it to `Cause`, an evidence pattern family in `steps/reconcile.py`, `CANDIDATES` in `card.py`,
a brief in `research/agent.py` (`CAUSE_BRIEFS`), and labels in `configs/reconcile_labels.yaml`.
New prompt versions for anything that lists causes. Re-run every reconcile eval.

**Add a research source**
Add the host to `GUIDANCE_HOSTS` and starting pages to `GUIDANCE_START` in `research/sources.py`,
after checking the page renders without JavaScript (DART doesn't) and allows automated fetching
(FASB doesn't). Bump `SOURCES_VERSION` if extraction changes. Test the allowlist.

**Change the research budget** (`TOOL_BUDGET`, `MODEL_CAP` in `research/agent.py`; rounds in
`research.max_rounds`): bump `HARNESS`, then `sweep research` and compare cost and
`true_cause_supported` against the last experiment. Keep the tool budget below the model cap, or
investigations hit the cap before writing a report (DECISIONS: fanout-1).

**Add an eval**
Deterministic first (`evals/scoring.py` style: a pure function + a unit test). LLM judge only for
judgment calls, with a binary rubric in `steps/judge.py`, then calibrate it (RUNBOOK section 4).

**Add a test**
Put it in the file that mirrors the module under test. Reusable doubles go in `tests/support/`;
never import from another test file. `make check` must stay offline and fast.

## Checks, from cheapest to most real

| Command | What | Cost |
|---|---|---|
| `make check` | lint, strict types, unit + property + contract tests (offline) | $0 |
| `make web-check` | the review UI: strict TypeScript, Vitest (incl. contract + properties) | $0 |
| `make fresh-check` | everything above on a copy without gitignored files (what CI sees) | $0 |
| `make audit` | known CVEs (Python, npm) and a security lint of `src/` | $0 |
| `make receipt` | the docs' headline numbers recomputed from the cache (paid calls blocked) | $0 |
| `make smoke` | the real server, graph and cached models over HTTP, incl. LangSmith feedback | ~$0.01 |
| `make web-e2e` | the review UI in Chromium against the real server; accessibility | ~$0.01 |

Changed what the API returns? Regenerate the UI fixtures (`uv run python -m tests.support.api_fixtures
--write`), then update `web/src/types.ts`: the contract tests on both sides tell you what's off.
Changed a headline number? `make receipt` tells you which sentence to update.

## Conventions

- Python 3.12, `uv`, ruff (100 cols), **pyright strict** on `src/` and `tests/`. Third-party
  libraries without types are confined to boundary modules that say so in a header comment.
- Tests: `make check` must stay offline and fast. Fakes: `tests/support/fakes.py` (scripted chat
  models), the fake tool-calling model in `tests/reconcile_agent/test_agent.py` (per the LangChain
  testing docs).
- Coverage floor 95% (`make cov`, CI). The thinnest modules are the LangSmith experiment runners
  (`evals/calibration.py`, `evals/agent_eval.py`): add tests there before adding features there.
- Write down non-obvious decisions in DECISIONS.md with the evidence (numbers, experiment names).
