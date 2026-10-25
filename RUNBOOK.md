# Runbook

For whoever operates or changes filings-agent: how to run it, review runs, evaluate a change before
shipping it, keep the LLM judge honest, triage production runs, and handle incidents.
Every command here is real; results land in `data/` and in LangSmith project `filings-agent`.

## 0. Setup (once)

```bash
make setup
cp .env.example .env        # SEC_USER_AGENT, LANGSMITH_API_KEY, LANGSMITH_TRACING=true, AWS_PROFILE
make data                   # filings + answer key + all eval datasets (cached; re-runs are free)
uv run filings-agent upload-datasets   # pushes them to LangSmith (idempotent)
uv run filings-agent queue setup       # creates the "filings-agent: review" annotation queue
make check && make gate     # tests + types green; eval gate PASSED
```

## 1. Running an analysis

| Way | Command | Use when |
|---|---|---|
| CLI | `uv run filings-agent run --tickers GE,NVDA` | scripting, demos in a terminal |
| API | `make serve`, then `POST /analyses` with an `Idempotency-Key` | integration; retries are safe |
| Review UI | `make web && make serve`, open `http://localhost:8000` | the human in the loop: start, watch, decide cards, see the result |
| Studio | `make studio`, open `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024` | seeing the graph itself run |

A run either **publishes** (nothing material, or only changes triage confirmed) or stops at
**awaiting_review**.

**Same request, same run.** Asking again with the same inputs replays the earlier analysis
(`replayed` / `created: false`) and pays for nothing. The inputs are the tickers, the filings on
disk for them, every model, prompt and setting that shapes the result, and `GRAPH_VERSION`
(`graph/build.py`). So after `make data` picks up a new 10-K, or a role or prompt changes, the same
request is a **new** analysis. To re-run on purpose, use `--fresh` (CLI), `fresh: true` (API) or
the UI's *Re-analyze* box. Bump `GRAPH_VERSION` when a code change alters what a run publishes.

**Who can call the API.** Set `FILINGS_API_TOKENS="alice:<token>,bob:<token>"` (each token at least
16 characters; e.g. `python -c "import secrets; print(secrets.token_urlsafe(24))"`). Every API call
then needs `Authorization: Bearer <token>`; the UI asks for the token once and keeps a session
cookie. The name before the colon is recorded as the reviewer of every decision. Without tokens the
server runs in local mode (reviewer `local`) and `serve` refuses any `--host` but loopback. The CLI
records `--reviewer` (default `$USER`).

**One process per thread.** A run holds a lease on its thread in `data/leases.sqlite` while it
executes (renewed every 30 s, expires 2 minutes after its process dies). Several server workers, or
the CLI next to the server, can't run one thread twice: the second gets `running` (API) or
`ThreadBusy` (CLI). Live progress events stream only from the worker running the thread; other
workers report the status.

**Reading the cards:** `uv run filings-agent cards --thread <id>` (also in Studio and in
`GET /analyses/{id}`). Each card shows the old and new value, which filing changed it, and every
candidate cause marked *fits / possible / ruled out / no evidence* by code rules, with the filing's
sentences for each. **The agent's own answer is deliberately not shown.** Research findings, when
requested, appear under each cause with their source; the investigator's note is an unverified
reading, so judge from the quote.

**Deciding** (`filings-agent review --thread <id> --tickers ... --decisions decisions.json`, or
`POST /analyses/{id}/review`). One decision per pending item; partial answers are rejected:

```json
{"GE:net_income:FY2022": {"action": "accept", "correct_cause": "accounting_change",
                           "note": "ASU 2018-12 adopted retrospectively"},
 "GE:revenue:FY2022":    {"action": "research", "note": "check the 8-K for the Vernova spin-off"},
 "X:cash:FY2023":        {"action": "escalate", "note": "none of these fit"}}
```
- `accept` uses the restated (newest) figure; `use_original` keeps the originally reported one.
- **Always set `correct_cause` to the cause you pick.** It's published with the card's evidence for
  that cause. When it differs from the agent's hidden answer, the case becomes a regression test
  (section 6).
- `escalate`: no candidate explains it. The number is published now, cause "unknown", with your note.
  To resolve it later, decide the case in a new run (`--fresh` / *Re-analyze*): a human pick
  supersedes the escalation.

**The decision history.** Nothing a run publishes overwrites an earlier answer. Every answer is a
row in `resolution_history` (cause, decision, reviewer, when, which run, and the filings compared);
the `resolutions` view picks the one that stands: a human pick (`accept`/`use_original`) beats an
escalation, which beats an automatic answer; among equals the newest wins. `later_escalations` > 0
means someone escalated after the pick that stands: look at it. A new 10-K that restates the same
year again compares different filings, so it's a new case:

```bash
sqlite3 data/ledger.sqlite "SELECT seq, thread_id, decision, actor, decided_at, cause
  FROM resolution_history WHERE discrepancy_id = 'GE:net_income:FY2022' ORDER BY seq"
```
- `research`: not enough to decide. One investigation per cause runs (about $0.75, 30 s), then the
  card comes back with findings. The note is an optional hint about where to look. At most
  `research.max_rounds` (2) per case; then decide or escalate.

After every real run the CLI/API posts **online checks** to the trace and routes it to the review
queue if a human is needed (section 7).

## 2. Where everything is in LangSmith

| What | Where | What to look at |
|---|---|---|
| Every run | Project `filings-agent` | One trace per invocation. Spans are named `extract GE balance_sheet`, `reconcile NVDA:diluted_shares:FY2024`, `judge memo`. Filter by tag `ticker:GE`, `step:extract`, `model:...` |
| Test sets | Datasets `filings-agent/{extract,reconcile,memo}-{dev,test}`, `e2e-dev` | Inputs + correct answers (SEC XBRL for numbers, verified labels for causes) |
| Production mistakes | Dataset `filings-agent/reconcile-corrections` | Cases a reviewer corrected; included in every reconcile eval and the gate |
| Judge trust | Dataset `filings-agent/judge-calibration` + experiments `judge-calibration-*` | `agree.*` per criterion: how often the judge matches your labels |
| Model/prompt choices | Experiments on each dataset; use **Compare** | Per-example scores, cost, latency; sort by regressions |
| Pairwise memo | Comparative experiments `memo-pairwise-*` | Which writer won each table, with the judge's reason |
| Runs needing a human | Annotation queue `filings-agent: review` | Rubric: `human.causes_correct`, `human.memo_acceptable`, `human.root_cause` |
| Health of production | Feedback keys `online.*` on traces | Filter/chart: errors, unknown causes, memo verified, cost |
| What reviewers did | Feedback keys `review.*` on finished traces | `review.agent_agreement` (reviewer's pick vs the agent's hidden answer), `review.escalated`, `review.research_rounds`, `review.auto_approved`; each decision in the comment |
| Research quality | Dataset `filings-agent/research-hard` + experiments `research-*` | `true_cause_supported`, `true_cause_contradicted`, `discarded_rate`, cost per case |

## 3. Evaluating a change (model swap, prompt edit, code change)

Never ship a change on vibes. The loop, cheapest first:

1. **Local score (cached, cheap):**
   `uv run filings-agent score extract --models nova-lite --prompt v2 --reps 3`
   (also `score reconcile`, `score memo`). Read F1 ± sd, "not reported", cost, p95.
2. **Record it as an experiment:** `uv run filings-agent sweep extract --models nova-lite --prompt v2`.
   In LangSmith, **Compare** against the current choice. Open the worst examples and click into
   their traces: *why* did it fail? (Usually one of the known traps: share scale, which net income
   line, a missing column.)
3. **Change config** (`roles:` / `prompts:` in `configs/default.yaml`), never code defaults.
4. **Gate:** `make gate`. Fails below the bar in `configs/eval_gate.yaml` or on a regression vs
   `evals/baseline.json`. If the change is a deliberate trade-off, re-baseline with
   `uv run filings-agent gate --update-baseline` and say why in DECISIONS.md.
5. **Test split, once, at the end:** `sweep ... --split test`. Never tune against it.
6. **Write the decision** in DECISIONS.md with the numbers and experiment names.

Re-evaluate prompts after any model swap: the same prompt rule helped gpt-oss and hurt nova-lite.

## 4. The LLM judge

**Used only where code can't grade:** memo prose (faithful, caveats_correct, comparable,
actionable) and reconcile explanations (supported, specific, consistent). Numbers, quotes and
causes are graded by code.

- Rubrics are binary questions with a reason each (`steps/judge.py`).
- The judge model (`roles.judge`) may not be the reconciler or memo writer (config validation).
- Pairwise comparisons run in both orders; inconsistent = tie, and the rate is reported as
  `position_bias_rate`.

**Calibrate before trusting it, and again whenever the judge model or a rubric changes:**

```bash
uv run filings-agent calibrate prepare         # 20 real outputs, right and wrong, from several models
uv run filings-agent calibrate queue           # into two LangSmith annotation queues, one per rubric
#   label in LangSmith: 1 (pass) or 0 (fail) per question; blind to the judge and the model
uv run filings-agent calibrate pull            # your scores -> configs/calibration_labels.yaml (commit it)
uv run filings-agent calibrate score --upload   # agreement, Cohen's kappa, false passes; LangSmith experiment
```
Trusted only if every criterion with ≥6 labels has **kappa ≥ 0.6 and false-pass rate ≤ 10%**.
Read the "Disagreements" section of `data/calibration/REPORT-<judge>.md`: each one is either a
rubric question that's ambiguous (rewrite it, bump the rubric version) or a judge that's wrong
(try another judge model: `--judge nova-pro`). Until trusted, report judge scores as indicative only.

**The online evaluator (LangSmith runs it on live traces).** One definition, in
`evals/online_judge.py`: an OpenAI judge (a different model family from the Claude memo writer)
with the memo rubric, scoring published runs and posting `judge_faithful`, `judge_caveats_correct`,
`judge_comparable` and `judge_actionable` onto each trace (underscores: LangSmith's schema keys
can't contain dots).

1. Add `OPENAI_API_KEY` in LangSmith (Settings -> Secrets), and in `.env` for calibration.
2. Calibrate it before trusting it, against the same labels:
   `uv run filings-agent calibrate score --judge openai:gpt-4.1-mini` (memo items only).
3. Create it: `uv run filings-agent online-evaluator --create filings-agent-demo --sampling 1`
   (and `--create filings-agent --sampling 0.1` for production). It posts the rule through the
   API, replacing any earlier copy, with a 1 dollar weekly spend cap. Without `--create` it prints
   the same settings for pasting into the UI (Tracing -> project -> Evaluators -> + Evaluator).
4. Few-shot corrections (scores you correct on traces become examples in its prompt) are a UI
   toggle; the API refused it on this rule, so turn it on in the Evaluators tab if you want it.

It runs outside the spend guard (LangSmith calls OpenAI with its own key), so keep the sampling
rate low in production and watch the OpenAI bill.

## 5. Pairwise memo comparison

```bash
uv run filings-agent compare memo --models claude-haiku-4.5,nova-lite          # local
uv run filings-agent compare memo --experiments memo-claude-haiku-4.5-xxxx,memo-nova-lite-yyyy  # LangSmith
```
Use pairwise for prose (more reliable than absolute scores). Look at `a_win_rate` and
`position_bias_rate`; a high bias rate means the judge isn't discriminating. Don't use it for
anything with a right answer.

## 6. Production mistakes become tests

1. Reviewer sets `correct_cause` on a wrong item (section 1).
2. `uv run filings-agent export-corrections`: adds it to `filings-agent/reconcile-corrections`,
   puts `reviewer_cause_correction` feedback (with LangSmith's `correction` field) on the trace.
3. From then on it's in `score reconcile`, the reconcile experiments, and **the gate**. If the gate
   now fails, fix the step (prompt, evidence, verifier) or consciously re-baseline. Don't silence it.

**The second door: QA's catches.** A QA reviewer working the annotation queue finds a wrong cause
the analyst didn't. In the trace, open that change's `reconcile` span, **Add to Dataset** ->
`filings-agent/reconcile-qa`, and set the reference output to `{"cause": "<the right cause>"}`.
Then `make import-qa` checks each example (it must be a reconcile step, with one of the candidate
causes) and appends the new ones to `configs/reconcile_corrections.jsonl`, the same file the gate
reads. Commit it: the diff is where someone checks what became a test. Refused examples are listed
with the reason, and importing twice adds nothing.

## 7. Online checks and the review queue

Posted on every finished run's trace: the **reviewer feedback** `review.*` (section 2). Falling
agreement or rising escalations is the first sign the agent or its evidence got worse.

Posted on every real run's trace (feedback, deterministic, no LLM): `online.branch_errors`,
`online.unknown_causes`, `online.unverified_quotes`, `online.memo_verified` (−1 = no memo),
`online.awaiting_review`, `online.cost_usd`. A run is added to the queue
**filings-agent: review** if it's awaiting review, had branch errors, has an unknown cause, or its
memo failed verification; `online.routed_to_review` carries the reason.

**Triage (in LangSmith, Annotation queues → filings-agent: review):**
1. Open the trace. For reconcile spans, compare the cause to the quoted evidence. For the memo, check
   the caveats against `changed_by` in the table.
2. Score the rubric (`human.causes_correct`, `human.memo_acceptable`, `human.root_cause`).
3. Wrong cause? Also submit the review decision with `correct_cause` (section 1) so it becomes a test.
4. Spot a pattern (same root cause 3+ times)? That's a ticket: evaluate a fix with section 3.

Optional: in the LangSmith UI you can add an **online evaluator** rule on project `filings-agent`
(e.g. an LLM judge on the memo span, sampled at 10%). The SDK has no API for rules, so the
deterministic checks above run in code.

## 8. Trajectory and end-to-end evaluation

`uv run filings-agent sweep e2e` runs whole scenarios (GE+NVDA, AAPL+MSFT, KO, NVDA) and checks:
- `trajectory_match`: the path through the graph. A restated company must go
  reconcile → interrupt → review; a clean one must skip both;
- `review_correct`: the human gate fired exactly when it should;
- `restatement_recall/precision`: restatements caught vs the answer key ("extra" ones are usually
  extraction errors masquerading as restatements).

Unit-level version with fakes: `tests/evals/test_trajectory.py` (no LLM).

## 9. Incidents

| Symptom | Likely cause | Do this |
|---|---|---|
| `SpendLimitExceeded` | Project cap ($30) or account month cap ($100) reached | `uv run filings-agent spend`. Raise `spend.project_cap_usd` deliberately or wait. Never catch it in code |
| Run stopped mid-fan-out with `ThrottlingException` | Bedrock throttling past 4 retries | Re-run the same request / same `Idempotency-Key`: it resumes; finished LLM calls come from cache |
| Server killed mid-run | Crash | Client retries with the same key → `resuming`. Branch nodes re-run but cost nothing (cache). See `scripts/crash_rehearsal.py` |
| Memo shows **NOT VERIFIED** | Writer used a number not in the table | The table is right, the prose isn't. Check the queue; if it recurs, re-evaluate the writer (`score memo`) |
| Spike in `online.unknown_causes` | New restatement type, or evidence not found | Open the reconcile spans; check `evidence` contains the explaining passage; add a pattern family if not |
| Gate fails after a correction was exported | A production mistake is now a test | Expected. Fix the step or re-baseline consciously (section 6) |
| Judge scores look off | Judge not calibrated for this rubric/model | Re-run section 4; don't report judge numbers until trusted |
| Studio: "upgrade your agent server" | `langgraph-api` too old or tracing off | `make studio` (sets tracing); `langgraph-api` ≥ 0.11 is pinned in dev deps |
| LangSmith: "legacy API usage" | `read_run`/`list_runs` used in an ad-hoc script | Use `client.runs.retrieve()/query()`; `tests/core/test_conventions.py` guards the repo |
| Card says "Research failed: <cause>: no report" | An investigation hit its call cap before reporting | The other causes still count. If it recurs, check `TOOL_BUDGET < MODEL_CAP` (`research/agent.py`) and the trace for that cause |
| Research findings "discarded as unverifiable" rising | Investigators paraphrasing, or a source's text extraction changed | Open the trace; compare the quote with the page. If a site changed its HTML, fix `html_to_text` and bump `SOURCES_VERSION` |
| Research refused a URL | Not on the allowlist (by design) | Add a source only via the CONTRIBUTING recipe |
| SEC 403 / 429 | Missing `SEC_USER_AGENT` or rate | Set it in `.env`; the client throttles to 5 req/s and caches everything |
| API answers 401 | `FILINGS_API_TOKENS` is set and the call has no or an unknown token | Send `Authorization: Bearer <token>`; the UI prompts for it |
| `serve` refuses to start: "without authentication" | A non-loopback `--host` and no tokens | Set `FILINGS_API_TOKENS`, or serve on 127.0.0.1 |
| `ThreadBusy` (CLI) / `running` while nothing runs here (API) / event `running_elsewhere` | Another process holds the thread's lease | Wait for it; a dead process's lease expires in 2 minutes (`data/leases.sqlite`) |
| `[leases] lost the lease on <thread>` | The process stalled past the lease (e.g. a suspended laptop) | Another process may have taken over; check `GET /analyses/{id}` before resuming |
| "Analyse GE" shows an old result | Same tickers, same filings, same settings: a replay by design | *Re-analyze* (`fresh: true`), or `make data` if a new 10-K should be there |

## 10. Costs (measured)

| Activity | Cost |
|---|---|
| Analysis of 2 companies (24 extractions, ~12 reconciles, memo), fresh | ~$0.14 |
| Same, cached | $0 |
| Extraction sweep, 1 model, dev (57 statements) | $0.01 (nova-lite) to $0.79 (sonnet) |
| Reconcile sweep, 1 model, dev (23 cases) | $0.003 to $0.21 |
| Memo + judge, 1 writer, dev (5 tables) | a few cents |
| `make gate` with nothing changed | $0 (cached) |
| One research round (5 investigations, Sonnet 4.6) | ~$0.74, ~27 s (runs only when a reviewer asks) |
| Research eval, 11 hard cases | ~$8 fresh, $0 cached |

Spend guard: every Bedrock call reserves its worst case before sending; `uv run filings-agent spend`.
Reservations are in the project ledger file, so every process on the machine counts the others'
calls in flight. A call that times out after it may have reached Bedrock is charged its worst case
(the guard never sees its usage). botocore makes one attempt; LangGraph's retry policy does the
retries. Compare `spend` with Cost Explorer now and then: spend made elsewhere is invisible to it.

## 11. Reference

| Command | Does |
|---|---|
| `make check` / `make cov` / `make gate` | lint+types+tests / coverage ≥95% / eval regression gate |
| `filings-agent score {extract,reconcile,memo}` | local, cached scoring (`--reps`, `--prompt`, `--judge`) |
| `filings-agent sweep {extract,reconcile,memo,e2e,research}` | LangSmith experiments (`--split test` once only) |
| `filings-agent cards --thread <id> \| --case <ids> \| --eval` | review cards for a paused run / for dataset cases / scored against labels (no LLM) |
| `filings-agent compare memo` | pairwise memo judge (local or comparative experiment) |
| `filings-agent calibrate {prepare,score}` | judge calibration (`--upload` records in LangSmith) |
| `filings-agent export-corrections` | reviewer corrections → dataset + trace feedback |
| `filings-agent queue setup` | create/verify the annotation queue |
| `filings-agent run / review / serve` | operate the agent |
| `make studio` | LangGraph Studio server with tracing on |
