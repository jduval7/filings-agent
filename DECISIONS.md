# Decisions

Each entry: what we decided, why, and the evidence. Model choices link to LangSmith experiments.

## 2026-10-17: Data scope

- **Extraction/end-to-end corpus:** 8 companies, 10-Ks filed since 2023 (31 filings).
  GE, NVDA, WMT, PFE have real restatements; AAPL, MSFT, NKE, KO are clean controls with
  unusual fiscal year-ends.
- **Reconcile-only companies:** AMZN, TSLA, CMG, AVGO (stock splits), JNJ, MMM, DHR (spin-offs).
  Without them the reconcile set was 20 discrepancies, mostly GE: too few to compare models.
  With them: 40 (dev 28, test 12). They are never run through extraction, so cost is unchanged.
- **Splits by company**, never by filing: test companies are never seen while tuning.
- **Answer key = SEC XBRL company facts**, matched by accession number. The agent never reads it.
- **Graded facts must be answerable**: the value has to appear in the statement text the extractor
  is given (538 of 553). The rest (e.g. GE diluted shares, which only appear in a note) become
  "should say not reported" cases instead of unfair misses.

## 2026-10-17: Baseline for "is fan-out worth it?"

The baseline is **one LLM call over the whole 10-K** (~90k tokens), since that's what a naive build does.
Comparing against a per-filing call over the three located statements would mostly test the
locator, not the fan-out. Fan-out per statement is justified by cost, latency (parallel),
retrying a single failed branch, and resuming after a crash. We'll report all of these, not just accuracy.

## 2026-10-17: Go/no-go on extraction difficulty

The first extraction sweep (day 2) checks whether models actually differ. If every model scores
about the same on located statements, that's a real finding (use the cheapest model), and we add a
harder variant (raw pages instead of located statements) so the sweep has signal.

## 2026-10-17: Materiality

A discrepancy with a relative spread under 0.5% is "immaterial": auto-accepted, no human review.
Example: GE FY2023 net income 9,481 -> 9,482 ($M).

## 2026-10-17: Quality bar for the model ladder

Extraction: **F1 >= 0.97 AND "not reported" accuracy >= 0.9** on the dev split. The ladder picks
the cheapest model that clears both. Two parts on purpose: a model that extracts well but
invents numbers for metrics that aren't on the statement is not acceptable for financial data.
Caveat: "not reported" has only 8 dev examples (one example = 12.5 points), so it's read
together with repetitions, never from one run.

## 2026-10-17: Extraction ladder, prompt v1 (dev, 1 rep)

| model | F1 | not reported | $ / 57 statements |
|---|---|---|---|
| claude-sonnet-4.6 | 0.994 | 0.75 | 0.789 |
| gpt-oss-120b | 0.991 | 1.00 | 0.042 |
| claude-haiku-4.5 | 0.968 | 0.75 | 0.259 |
| nova-lite | 0.899 | 0.38 | 0.013 |
| nova-pro | 0.855 | 0.50 | 0.168 |
| nova-micro | 0.628 | 0.50 | 0.006 |

Failure analysis (from cached outputs, $0):
- nova-micro: mostly returned only the current year's column (instruction following).
- nova-lite: every miss was one of two traps. (1) Which net income line: KO "Consolidated Net
  Income" includes noncontrolling interests; GE "attributable to common shareholders" is after
  preferred dividends. (2) Share scale: read "(In millions, except per share data)" as "shares
  aren't in millions" (NVDA 2,510 -> 2,510 shares instead of 2.51B).
- Nova Pro scored below the cheaper Nova Lite: bigger isn't automatically better.
- Ministral 8B: high precision but 21% schema failures.

## 2026-10-17: Prompt v2

Written from the v1 dev failure analysis, as **general rules only** (every column; heading scale
applies to share counts unless stated; "net income attributable to the company"), with no
company-specific hints, so it can't overfit dev companies. The test split stays untouched until
final confirmation.

v1 -> v2 F1 (dev, 1 rep): gpt-oss-120b 0.991 -> 1.000, haiku-4.5 0.968 -> 1.000,
nova-2-lite 0.883 -> 0.978, nova-pro 0.855 -> 0.974, nova-lite 0.899 -> 0.932,
nova-micro 0.628 -> 0.698. Every model improved. The Nova family still fails "not reported"
(0.25-0.62): it fills in values for metrics that aren't on the statement.

## 2026-10-17: Failure handling in the fan-out

LangGraph 1.2's node `error_handler` would be the natural tool, but it does **not fire for
`Send` branches** (verified with a minimal repro: fires for a normal node, never for a pushed
task). Design used instead:
- transient errors (throttling/outage) are re-raised so `RetryPolicy` retries with backoff; if
  they persist the run stops, and a resume re-runs only unfinished branches (finished ones are
  checkpointed). Covered by `test_resume_after_crash_reruns_only_unfinished_branches`.
- anything else (bad model output, parse failure) is recorded for that branch; siblings carry on.
- `SpendLimitExceeded` is a BaseException: never caught, always stops the run.

Also: node `timeout` is async-only in LangGraph; the Bedrock client's read timeout bounds calls.

## 2026-10-17: Context limits argue for locate-then-fan-out

10-K markdown is 220k-720k characters (median ~116k tokens, max ~206k). gpt-oss-120b's context
is 128k tokens, so the whole-filing approach can't even fit about half the filings for the
best-value model, and Haiku's 200k overflows on the largest. Measured in the baseline run.

## 2026-10-17: Extraction model choice (dev, prompt v2, 3 reps)

| model | F1 (±sd) | not reported (±sd) | $ / 57 statements | p95 s |
|---|---|---|---|---|
| claude-haiku-4.5 | 0.999 ±0.003 | 0.96 ±0.07 | 0.269 | 4.3 |
| gpt-oss-120b | 0.995 ±0.009 | 1.00 ±0.00 | 0.042 | 9.0 |
| nova-2-lite | 0.975 ±0.005 | 0.38 ±0.00 | 0.089 | 2.2 |

Both top models clear the bar on every repetition; **gpt-oss-120b is the extractor** (cheapest that
clears it, 6.4x cheaper than Haiku). Trade-off accepted: higher tail latency. In a fan-out the
run's wall-clock is set by the slowest branch, so p95 matters more than p50; still ~10s per run.
Pending: confirmation on the test split (run once, at the end).

## 2026-10-17: Reconcile step: the ranking flips

First reconcile ladder (dev, 10 labelled cases) exposed three problems, two of them mine:
1. **Evidence came only from the newest filing.** GE FY2022 net income went 225 -> 339 in the 2024
   10-K and 339 -> 336 in the 2025 one; the model only saw the 2025 filing. Fix: gather evidence
   from every filing where the value changed (`change_filings`).
2. **Evidence was biased to one cause.** Slots filled from the first keyword family; GE says
   "discontinued operations" everywhere, so its ASU 2018-12 adoption never reached the model.
   Fix: round-robin across cause families, skip table-of-contents fragments.
3. **The verifier was measuring typography.** gpt-oss writes "ten‑for‑one" with U+2011 hyphens;
   correct answers were downgraded to unknown. Fix: shared `textnorm.normalize` (NFKC, dashes,
   quotes, markdown escapes) used by both verifiers. Still strict about words: a stitched quote
   must be made of verbatim fragments, in order, from one passage.

| model (prompt v2) | cause accuracy before -> after fixes | quote verified | $ / 10 cases |
|---|---|---|---|
| claude-sonnet-4.6 | 0.90 -> 0.90 | 0.90 | 0.098 |
| nova-lite | 0.60 -> 0.80 | 1.00 | 0.0015 |
| qwen3-32b | 0.70 -> 0.80 | 1.00 | 0.0033 |
| claude-haiku-4.5 | 0.80 -> 0.80 | 1.00 | 0.034 |
| gpt-oss-120b | 0.30 -> 0.60 | 0.80 | 0.0047 |

**The extraction winner is the worst reconciler.** Per-step model choice is the point.
Remaining hard case: GE net income (accounting change vs spin-off). Telling them apart needs
domain knowledge: net income attributable to the company includes discontinued operations, so a
spin-off alone doesn't change it. Only Sonnet avoided the wrong cause (it said unknown).
**Reconciler: claude-sonnet-4.6** for now; reconcile volume is small (~15 calls per analysis), so
the cost is cents. n=10 is small; the 7 reconcile-only companies add ~21 cases once downloaded.
Test split has no accounting-change cases, so a prompt rule for that case couldn't be verified on
test; not added yet.

## 2026-10-17: Baseline result: one call per whole 10-K vs locate + fan-out (dev, prompt v2)

| strategy | F1 | recall | failed calls | $ / run | p95 s |
|---|---|---|---|---|---|
| locate + fan-out, gpt-oss-120b (3 reps) | 0.995 | 0.993 | 1% | 0.042 | 9.0 |
| whole filing, gpt-oss-120b | 0.403 | 0.252 | 68% | 0.206 | 25.5 |
| whole filing, nova-lite | 0.188 | 0.104 | 84% | 0.015 | 41.0 |

gpt-oss-120b whole-filing failures: 4 filings over its 128k-token context (~160k tokens), and 9
calls returned no usable structured output on long inputs; 6 of 19 worked. Nova-lite failures
were intermittent: the smallest filing failed during the concurrent run but succeeded when
re-run alone, and error codes weren't recorded at the time (the scorer now records error kinds).
So nova-lite's baseline is reported as "unreliable, cause unconfirmed", not explained.
Conclusion: locating the statements first is what makes cheap models viable. It's 5x cheaper,
~3x faster at p95, and the only variant that works on most filings.

## 2026-10-17: Reconcile set expanded to 21 dev / 12 test verified labels

With SEC_USER_AGENT set, downloaded the 7 reconcile-only companies (34 filings) and labelled every
material discrepancy from the filing text where the number changed (configs/reconcile_labels.yaml
cites the passage for each). Two more of my bugs surfaced, both markdown typography:
- the split pattern `\w+[- ]for[- ]\w+` never matched `20 \-for\-1` (escaped hyphen + space), so
  split evidence only reached the model through the generic "stock split" fallback;
- the verifier treated "20 \-for\-1" (filing) and "20-for-1" (model quote) as different text, so
  correct Amazon/Chipotle answers were downgraded to unknown. Fixed in `textnorm.normalize`.

| model (prompt v2, 2 reps) | cause accuracy | quote verified | $ / 21 cases |
|---|---|---|---|
| claude-sonnet-4.6 | 0.95 | 0.98 | 0.187 |
| claude-haiku-4.5 | 0.90 | 1.00 | 0.064 |
| nova-lite | 0.83 ±0.03 | 0.98 | 0.003 |
| gpt-oss-120b | 0.83 ±0.03 | 0.95 | 0.009 |
| qwen3-32b | 0.74 ±0.10 | 0.88 | 0.006 |

Sonnet's only miss is GE FY2022 net income (accounting change read as spin-off). Haiku misses
both GE net-income cases. **Reconciler stays claude-sonnet-4.6.** At ~15 reconcile calls per
analysis it costs about $0.13, and it's the step where errors are hardest for a human to spot.
Haiku (0.90) is the budget option.

## 2026-10-17: Memo step: numbers from code, prose from the model, comparisons on one basis

Code builds the table and every derived number (YoY %); the model writes prose; a deterministic
checker extracts every number from the memo and rejects any not in the table (one retry with the
failures fed back, else a NOT VERIFIED banner). Same idea as the Memory project's receipt check.

The first real memo passed the number check and was still wrong: "NVDA diluted shares +889%"
(FY2023 from the post-split filing vs FY2022 from a pre-split one) and "GE debt fell from $30.8B
to $18.8B" (pre- vs post-spin-off basis). Every number was real; the comparisons weren't.
Fix: the memo table uses each company's **latest filing only** (all its years share one basis),
YoY changes are computed only between rows from the same filing, and restatements go to caveats.
Lesson for the presentation: a numeric receipt check is necessary but not sufficient; the
comparisons themselves need a consistency rule, enforced in code.

## 2026-10-17: Reviewer corrections -> regression tests (the loop, end to end on real data)

1. Real run (GE + NVDA). The reconciler said GE FY2021/22 net income changed because of the
   spin-off. Acting as reviewer, I corrected both to accounting_change (ASU 2018-12, verified in the
   2024 10-K) via `correct_cause` in the review decision.
2. Publish stored the corrections with the exact reconcile input and the run's LangSmith id, in
   the same transaction as the publication (exactly once).
3. `export-corrections` added them to the LangSmith dataset `filings-agent/reconcile-corrections`
   (deterministic ids) and put `reviewer_cause_correction` feedback (score 0, native `correction`
   field, deterministic feedback id) on the run's trace.
4. The reconcile eval now includes them, so **the CI gate failed** (0.913 vs baseline 0.952): a
   deploy would be blocked until the reconciler handles the case or someone re-baselines on purpose.
5. Investigating, I found the real causes were mine, not the model's:
   - **prompt drift**: the production runner built the reconciler with the class default prompt
     (v1) while every eval measured v2. Fix: `prompts:` in config is the single source for runner,
     API and gate; class defaults removed (the type checker then flagged the remaining call site).
   - two more verifier typography bugs: a stray `##` heading marker mid-sentence in GE's 10-K, and
     "Contracts \." (space before the period). Sonnet's quotes were faithful; the verifier
     rejected them.
6. After the fixes: Sonnet with prompt v2 scores **1.00 on all 23 dev cases**, the gate passes, baseline
   refreshed.

Reconcile prompt v3 (adds "net income includes discontinued operations, so a spin-off alone doesn't
change it") was tested and **not adopted**: no-op for Sonnet (1.00 -> 1.00), helps gpt-oss-120b
(0.78 -> 0.91), and hurts nova-lite (0.78 -> 0.57). The same rule helps one model and hurts another,
so prompts are pinned per configuration and always re-evaluated after a model swap.

## 2026-10-17: Final confirmation on the test split (run once)

Test companies: WMT, PFE, NKE (extraction); WMT, PFE, TSLA, AVGO, MMM (reconcile). Never used while
tuning prompts, verifiers or model choice. Recorded as LangSmith experiments.

| step | model | dev | test | $ (test) |
|---|---|---|---|---|
| extract | gpt-oss-120b | F1 0.995 ±0.009 | **F1 0.998**, not reported 1.00 | 0.031 |
| extract | claude-haiku-4.5 | F1 0.999 ±0.003 | F1 0.928, schema err 8%, not reported 0.67 | 0.165 |
| extract | nova-lite | F1 0.932 | F1 0.875 | 0.008 |
| reconcile | claude-sonnet-4.6 | 23/23 | **12/12** (incl. PFE reclassification, unseen cause) | 0.108 |
| reconcile | claude-haiku-4.5 | 0.83 | 0.92 | 0.036 |
| reconcile | gpt-oss-120b | 0.78 | 0.67 | 0.005 |
| reconcile | nova-lite | 0.78 | 0.67 | 0.002 |

Both choices hold. Notable: Haiku's near-perfect dev extraction didn't generalize (schema failures
and invented values on test companies) while gpt-oss-120b held, which is the reason to keep a
held-out split and run it only once. Caveats: test sets are small (36 statements, 12 discrepancies),
single repetition.

## 2026-10-17: Crash rehearsal found a gap the unit test couldn't

`scripts/crash_rehearsal.py` starts the API, begins a fresh run (12 branches), SIGKILLs the server
after 4 branches finish, restarts it, and retries the same request with the same Idempotency-Key.
First result: the resume re-ran all 12 branches, not 8.

Investigation (deterministic repro with a fake extractor + SQLite checkpointer):
- Under SIGKILL, finished branches' writes were missing from the checkpoint DB in both
  `durability="async"` and `"sync"`.
- Watching the DB during a normal run: branch writes for a fan-out step only appear when the whole
  step finishes (0 rows while 2..10 branches have finished, then 12 at once). Graceful failures
  flush them, which is why `test_resume_after_crash_reruns_only_unfinished_branches` passes.

Resolution: the guarantee is "never pay twice", delivered by two layers. LangGraph checkpoints
cover graceful failures; the content-addressed StepCache, written atomically per LLM call, covers
hard kills. Re-run with the rehearsal: 12 nodes re-ran, **4 served from cache, 8 new calls, 0 paid
twice**, complete result, published once. Runner uses `durability="sync"` (step checkpoint persisted
before the next step). Claim in the README updated to say exactly this, not more.
Option not taken: split the fan-out into several smaller steps (more checkpoints, more
latency) to make individual branches durable.

## 2026-10-17: LangGraph Studio

Studio needed (1) an entry point that builds Deps from config, since the server passes JSON context
(`graph/studio.py`, `typed_context=False`, nodes fall back to `default_deps`), (2) `langgraph-api`
>= 0.11 for in-Studio traces. The newest FastAPI (0.143.3) required OpenTelemetry >= 1.44, while
langgraph-api 0.16.1 pins an older one; the FastAPI minimum was an over-pin, relaxed to >=0.115
(resolved 0.141.1, API tests green). (3) Tracing on in the server's environment
(`LANGSMITH_TRACING=true`): `make studio` sets it.

## 2026-10-17: LangSmith deprecations

LangSmith flagged `GET /runs/{run_id}` and `POST /runs/query` (removal Jan 31, 2027) from
langsmith-py 0.14.x. Source: my one-off verification scripts (`read_run`/`list_runs` to fetch trace
links), not the repo; 0.14.11 is the latest SDK, whose replacements are `client.runs.retrieve()` /
`client.runs.query()`. `evaluate()` uses `list_runs` internally on older backends (warning
suppressed by the SDK, switches to v2 automatically). Reading the SDK also surfaced a deprecation
that DID affect our code: feedback without `session_id` "will stop working". `export-corrections`
now resolves the tracing project's id and passes it. `tests/test_no_deprecated_apis.py` fails if
`read_run`/`list_runs` appear in `src/` or `scripts/`, or if feedback loses its `session_id`.

## 2026-10-17: LLM judge, calibration, pairwise, trajectories, online checks

- **Judge only where code can't grade** (memo prose; reconcile explanations). Binary rubric
  questions with a reason each; judge = gpt-oss-120b (different family from the Claude writer and
  reconciler; config enforces it). One retry on schema failure (gpt-oss skips the tool call ~1%:
  5 incomplete verdicts in the first memo run, all succeeded on retry).
- **First memo scores (dev, 5 tables)**, numbers verified by code vs judge criteria:
  sonnet 1.00 verified / faithful 0.40 / caveats 1.00 / comparable 0.40 / actionable 0.80;
  haiku 0.80 / 0.40 / 0.60 / 0.20 / 0.40; nova-lite 0.20 / 0.20 / 0.60 / 0.80 / 0.40;
  nova-micro 0.80 verified. The code check discriminates clearly; the judge is harsh across the
  board (even Sonnet "faithful" 0.40). Strict or wrong? Unknown until calibrated.
- **Pairwise haiku vs sonnet memos: 5/5 ties, position-bias rate 0.40.** The judge preferred
  whichever memo came first in 2 of 5 tables; running both orders turned those into ties instead of
  fake wins. So this judge can't yet separate these writers. Judge scores are reported as
  **indicative only** until `calibrate score` says TRUSTED.
- **Calibration set prepared:** 20 real outputs (12 reconcile explanations from 4 models, half
  right/half wrong; 8 memos from 4 writers), for human labels in `configs/calibration_labels.yaml`.
  The judge never grades its own outputs, even in calibration (gpt-oss items are skipped when
  gpt-oss judges).
- **End-to-end trajectory experiment (4 scenarios): trajectory_match 1.00, review_correct 1.00,
  restatement recall 1.00, precision 1.00.** GE+NVDA went reconcile → interrupt → review; AAPL+MSFT
  and KO skipped both.
- **Online checks + annotation queue:** every real run posts `online.*` feedback; runs needing a
  human go to queue `filings-agent: review` (rubric attached). First live run hit a 404 adding to
  the queue: traces upload in the background, so the run didn't exist server-side yet. Fix: flush
  tracers before monitoring + retry with backoff on "not found". Verified: run routed, 7 checks posted.
- Found while wiring: the API status endpoint mutated the dict returned by the status function
  (harmless in production, which returns fresh dicts; corrupting for any shared/cached source).
  Now returns a copy.

## 2026-10-21: Where LLMs are used and why

Rule: **if code can compute the right answer, code does it.** An LLM is used only where the input is
human language too varied for rules, and every LLM output is checked by code against its source.

| Step | Built with | Why |
|---|---|---|
| plan (find the statements) | code | Heading synonyms + "the biggest numeric table after it". Headings vary (NVIDIA files statements under Item 15; GE's balance sheet is "Statement of Financial Position") but rules hold. An LLM locator is a swappable strategy, not needed so far. |
| extract (read the figures) | LLM, code-verified | Every company labels and lays out tables differently ("Net sales" / "Total revenues" / "Revenues", column order, where "in millions" sits, parentheses for negatives). Rules would be a pile of per-company exceptions. The LLM reports figures exactly as printed; code does the arithmetic and rejects anything not literally in the statement. For SEC filers XBRL makes this step optional: see the next entry. |
| find_discrepancies | code | Subtraction against a threshold. An LLM would only add error. |
| reconcile (explain why) | LLM, code around it | The reason is prose, worded differently every time, and often sits beside other events (GE 2023: a spin-off *and* an insurance accounting rule). Choosing which event moved *this* number is reading comprehension. Code gathers candidate evidence and verifies the quote. Some causes leave a numeric fingerprint code could detect alone (an exact 10x share count = split); not used yet. |
| review | human | By design (`interrupt()`). |
| memo | LLM prose, code numbers | Weakest LLM case: a fill-in-the-blanks template would be cheaper and never wrong. The LLM adds readability only; the checker rejects any number not in the table. |
| grading memos | LLM judge | "Clear and faithful?" is a judgment call. Every other grade is deterministic. Judge scores stay indicative until calibrated. |

## 2026-10-21: XBRL vs reading the filing

The SEC's XBRL data contains every reported figure, including prior years as repeated in each later
filing, so **changes can be detected from XBRL alone**. Our answer key does exactly that. What XBRL
does not give is the **reason**: it lives in the notes and MD&A prose (XBRL tags the notes only as
large text blocks, which still have to be read).

- **Production recommendation for SEC filers:** numbers and change detection from XBRL; spend the
  LLM only on the explanation (reconcile). That's the step that needs one.
- **Why extraction from the document is still here:**
  1. Most financial documents have no XBRL: private companies, loan packages, PDFs, earnings decks,
     non-US filings. Extraction is the part that transfers.
  2. XBRL has documented quality problems: custom extension tags instead of standard ones (~30% of
     elements in one analysis), scale errors (millions vs billions), sign errors. A cross-check
     against the human-readable document catches some.
  3. For this project, XBRL gives extraction an exact answer key (F1 0.995 dev / 0.998 test), i.e.
     an eval with ground truth.

Sources: [CFA Institute on XBRL quality](https://blogs.cfainstitute.org/?p=23699),
[XBRL International on extensions](https://xbrl.org/?p=4462),
[SEC DERA custom tag rates](https://www.sec.gov/dera/reportspubs/assessment-custom-tag-rates-xbrl.html),
[XBRL US on restated values](https://xbrl.us/guidance/reporting-restated-values-in-the-face-of-the-financials/).

## 2026-10-21: Measuring the value

**Who needs this.** Quant researchers (backtests on restated data have look-ahead bias; vendors
sell "as first reported" vs restated databases, e.g. Compustat Point-in-Time, Calcbench), credit
and equity analysts maintaining multi-year spreads, audit and governance teams. Quiet in-filing
corrections ("little r" revisions, no 8-K) were ~76% of error corrections in 2021 and ~62% in 2022.

**Terminology:** most of our cases (splits, spin-offs, accounting changes) are **recasts**, not error
**restatements**. Same effect on the reported history, different cause. The code says "restatement"
for both; say "recast" when it matters.

**What we don't know yet: the time saved.** No credible published figure exists for "minutes to
explain one changed number". Broad surveys say analysts spend 75-80% of their time gathering and
reconciling data, which supports "real time sink" but doesn't measure this task. A vendor claims
2-4 hours to re-key one set of statements (marketing, order of magnitude only). So we measure our own:

1. **Human baseline:** time 5 dev cases by hand (find the change, find the reason, copy the quote).
   Estimate 5-20 minutes each, mostly searching the filing. Unmeasured until timed.
2. **System + review:** system latency (agent v1 p95 13.8 s, ~$0.04/case) plus the time to review an
   answer that arrives with a quote (expected under a minute: check the quote).
3. **Confident-wrong rate:** the saving only holds if confident answers can be skimmed. "Unknown"
   costs today's time, no more; a confident wrong answer is the real cost.

Result table to fill: minutes per case (human alone vs human + system), $ per case, confident-wrong
rate. **Status: pending the timed baseline.**

Sources: [Audit Analytics restatement types](https://auditanalytics.help.ideagen.com/hc/en-gb/articles/35432233953554/),
[Society for Corporate Governance statistics](https://connect.societycorpgov.org/blogs/randi-morrison/2022/06/01/financial-restatement-statistics-trends),
[Compustat Point-in-Time](https://www.globalcustodian.com/?p=26067),
[as-reported vs restated and look-ahead bias](https://dev.to/tradevodata/as-reported-vs-restated-financial-data-why-the-difference-matters-for-backtesting-1big),
[SAS on analysts' time](https://www.sas.com/en_us/insights/articles/business-intelligence/become-a-strategic-business-partner-using-analytics-in-finance.html),
[V7 on spreading time](https://www.v7labs.com/automations/financial-statement-spreading).

## 2026-10-21: Reconcile agent v1: four harness bugs, then an honest baseline

The take-home asks for a tool-using agent with retrieval, so reconcile gained a second strategy
(`reconcile.strategy: agent`): a LangChain `create_agent` with `fact_history` and `search_filing`
tools over a per-filing vector index (Titan v2 embeddings, cached). The first run scored cause
accuracy 0.30 / retrieval hit 0.52. Reading the traces found **bugs in the harness, not the model**:

1. **Mutated the runtime context.** Retrieved passages were written into `ReconcileContext`, which
   the docs treat as immutable per run. Fix: `search_filing` returns `content_and_artifact`, and the
   evaluator reads passages from the ToolMessage artifacts. Context is now a frozen dataclass.
2. **Silent fallback to the wrong filing list.** `filings_all.jsonl` had never been regenerated, so
   the factory fell back to the 8 main companies and every search for AMZN/CMG/DHR/JNJ returned
   "unknown filing" (all 11 "missed" cases). Fix: regenerate; the factory now raises instead of
   falling back.
3. **Cache key incomplete.** The harness version and the set of searchable filings weren't in the
   agent's cache key, so stale results could be reused. Both are in the key now, with a test.
4. **One malformed JSON answer crashed the run.** A bare schema as `response_format` let
   `create_agent` pick native structured output for Sonnet on Bedrock, where a parse failure raises.
   Fix: `ToolStrategy(AgentAnswer, handle_errors=True)`: identical for every model, and the parse
   error goes back to the model to retry (bounded by the model-call limit). Exhausted retries become
   "unknown", uncached.

Also: embeddings were throttled by the 300 rpm chat limit (per-model override 1200 rpm, per-filing
lock, warm-up before the run) and concurrent cache writes collided (uuid temp file + `os.replace`).

**v1 baseline** (claude-sonnet-4.6, vector retrieval, dev, 23 cases, $0.99, p95 13.8 s):
cause accuracy **0.83**, quote verified 0.91, retrieval hit 1.00, searched the changed filing 1.00,
trajectory superset 1.00, 3.2 tool calls/case. Injection (3 synthetic cases, reported separately):
followed 0/3, resisted 2/3. The pipeline strategy scores 23/23 on the same cases.

**Every failure is GE**, and the traces show two different problems:

- **Right answer, quoted from memory** (GE cash FY2022, revenue FY2021 → "unknown"). Sonnet quoted a
  real sentence from GE's 10-K word for word (it is in the filing), but no search had returned it,
  so the verifier rejected it. Correct behaviour under "evidence must be retrieved".
- **Confident and wrong** (GE net income FY2022: answered discontinued operations; label is the
  ASU 2018-12 insurance accounting change). It took the famous spin-off and never searched for
  another cause. But moving a business to discontinued operations doesn't change *total* net income,
  so the spin-off can't explain this change. **The verifier didn't catch it:** the quote was real.
  The check asks "does the evidence exist?", not "does it explain this number?"

**Underlying problem:** the agent decides the story before looking, then searches only to confirm
it, because the same model chooses the searches and the answer. The pipeline already separates
these (evidence for every cause gathered up front, one decision), which is likely why it wins here.

**Open, to decide with the author:** (a) split the agent into a looker that must cover every cause
and a decider that sees only the gathered evidence; (b) keep both strategies and report the
comparison (the structured version wins on this task); (c) add a code check that the claimed cause
*can* move this kind of metric; (d) treat "unknown" as success and headline the confident-wrong
rate. Hybrid retrieval is not motivated by this evidence (retrieval hit already 1.00); that metric
is also loose (any anchor anywhere counts), a known limitation.

## 2026-10-21: Triage: an easy lane now, a hard lane planned

**Problem.** Every live check confirms that things **exist**, not that they're **connected**.
Reconcile's `verify_quote` proves the quote is in the evidence, not that it explains this number:
agent v1's GE FY2022 net income answer (spin-off) passed with a real quote, though a spin-off
can't move total net income. Extraction has the same gap: the printed value and the row label are
each found somewhere in the statement, but not checked to be in the same row and year column, and
the match is a substring ("234" matches inside "1,234"). The extraction gap is **not addressed
here** (offline F1 0.995 against XBRL says it's rare, but it's a hole).

**Decision (with the author).** Route answers by how well code can confirm them, instead of
reviewing everything or rejecting anything unproven:

- **Easy lane (built):** `steps/triage.py`, plain code after every reconcile answer (pipeline and
  agent). Each link is *confirmed*, *contradicted* or *not covered*. Skip review only if every
  link is confirmed, including a numeric fingerprint. Rules are general accounting, not per
  company: a split can't move dollar amounts; a spin-off can't move net income attributable or
  the share count; a split must show a whole-number ratio (±0.5%, weighted-average shares drift:
  AVGO x10.0047) and the quote must name that ratio. Today only stock splits can be confirmed.
  Anything not confirmed goes to review: "not covered" means "a human looks", never "probably
  fine". Switch: `reconcile.auto_approve` (on in config, off by default in code). The reviewer
  now sees the checks that failed.
- **Hard lane:** see the review card entry below (the author's later decisions replace the
  "propose an answer" plan first written here).

**Results** (claude-sonnet-4.6, all from cached answers, $0):

| strategy | split | n | cause acc | easy lane | easy-lane wrong | review-lane acc | wrong answers contradicted by a rule |
|---|---|---|---|---|---|---|---|
| pipeline v2 | dev | 23 | 1.00 | 0.26 (6) | 0 | 1.00 | none to catch |
| pipeline v2 | test | 12 | 1.00 | 0.42 (5) | 0 | 1.00 | none to catch |
| agent v1 | dev | 23 | 0.83 | 0.26 (6) | 0 | 0.76 | GE net income FY2022 (+ its correction replay) |

- The confident-and-wrong GE answer now goes to review with the reason attached. The other GE
  failures (quoted from memory) were already "unknown".
- AVGO FY2023 (test) was answered correctly but quoted a sentence without the ratio, so it went to
  review: the rule costs one extra review, by design.
- **Caveats:** the split tolerance was set after seeing the dev ratios (AMZN, AVGO); the easy
  lane holds 11 split cases in total, so "0 wrong" is a small sample. dev n=23 includes the
  reviewer-correction replay of GE net income.

**Open, for the author:** (a) easy lane is ~26-42% of material changes, splits only;
spin-offs have no numeric fingerprint we compute, so they all go to review. A candidate (not
built): confirm with the discontinued-operations amounts printed in the note. (b) Relax "quote names
the ratio" to "the passage the quote came from names it" (would move AVGO FY2023 to the easy lane)?
(c) Extraction binding (cell check, whole-number match) is still open.

## 2026-10-21: Review card v1: every cause, no favorite

Built for the review lane, from deep research into how reviewers handle AI proposals
(`reports/Restatement review card design.md`). Independent studies found that explanations make
people more likely to accept an AI answer whether it is right or wrong. The one reliable lever is
making the check cheaper than working the answer out. A real quote that made a wrong cause look
verified (GE) is the "convincingly wrong" case in that literature.

**Decisions (with the author):**
- **The agent never marks a favorite.** The card lists every candidate cause in a fixed order, with
  the sentences from the filing that mention it and whether it fits the numbers (*fits / possible /
  ruled out / no evidence*). The agent's own answer and its hand-picked quote are **not shown**:
  placing that quote under one cause would make it the favorite. The agent's answer still drives
  the easy lane and the evals.
- **Teach while checking:** a ruled-out cause carries its one-line accounting reason.
- **Numbers first:** old and new value, the change, which filing changed it.
- **The reviewer picks the cause, or escalates.** The pick is published with the card's evidence
  for *that* cause. Picking differently from the hidden agent answer becomes a regression test as
  before. **Escalate** publishes the number now, with cause "unknown" and the reviewer's note.
  (Waiting for escalated cases instead is a one-line change; the author hasn't chosen yet.)
- Pure code from the evidence the pipeline already gathers (`steps/card.py`). No LLM call.

**Result** (`filings-agent cards --eval`, all labelled cases, no LLM):

| split | n | true cause shown as | true cause ruled out or hidden | causes left open (of 4) |
|---|---|---|---|---|
| dev | 21 | fits 6, possible 15 | 0 | 2.5 avg |
| test | 12 | fits 4, possible 8 | 0 | 2.2 avg |

The card never steered a reviewer away from the right answer, and it rules out about half the
candidates on average. For GE FY2022 net income, the spin-off appears with its real quote **and**
is ruled out with the reason; the ASU 2018-12 adoption is shown alongside.

**Known limits:** evidence is keyword sentences, so some quotes are boilerplate ("certain prior year
amounts have been reclassified") and keep "reclassification" open; no cause outside the four
(error correction, segment change, acquisition) can be shown; the card's effect on reviewer speed or
accuracy is unmeasured.

**Planned, not built:**
- **Keep researching:** the reviewer declines with an optional hint ("check the segment note"), and
  a research step (deep agent) looks somewhere new and returns a new card. The hint changes where it
  looks, never the bar: quotes are still verified and causes still checked against the numbers.
  A round limit per case escalates automatically.
- **"None of these" as a signal:** repeated declines point to a cause missing from the list, which
  becomes a new cause type and a new test.
- **Measuring reviewers and the easy lane** (from the research): a random audit sample of
  auto-approved items (only flagged items get labelled otherwise), known-wrong cards mixed into the
  queue (people miss errors that are rare), reason codes on every decision, and time per card.

## 2026-10-21: Reviewer-triggered research: deepagents tried, code fan-out kept

**Decision (with the author):** the reviewer can answer a card with **research** (optionally with a
hint, e.g. "check for an 8-K"). Only then does research run, for that one case, so the cost lands
on the cases a human judged hard. The card comes back with the findings; at most
`research.max_rounds` (2) rounds per case, then decide or escalate.

**Sources: live and allowlisted.** SEC EDGAR full-text search and filing documents (primary
source), PwC Viewpoint (interpretive guidance), plus the changed filings we already downloaded.
Checked 2026-10-21: fasb.org refuses automated fetching (403) and Deloitte DART renders its text
with JavaScript, so neither is usable. The SEC contact header goes to sec.gov only. Code re-reads
every cited source and drops quotes that aren't in it verbatim. Research findings are shown on the
card but **never override a code rule**. Nothing in the output can mark a favorite.

**Why not deepagents.** v1 used LangChain `deepagents` (planner + `cause-investigator` subagent).
Its trace showed the only feature doing work was the subagents (4 `task` calls); no planning, no
scratch files. Whether every cause got looked at was the model's choice. Our checklist is known
(four causes + "anything else") and coverage is the point, so **code fans out one bounded
`create_agent` investigation per cause**, in parallel, each in its own context. Rule of thumb:
deepagents when you can't write the plan down (e.g. "a company's full restatement history over ten
years"); a code fan-out when you can.

**Live runs on GE net income FY2022** (Sonnet 4.6):

| harness | cost | time | investigations reporting | verified findings | what went wrong |
|---|---|---|---|---|---|
| deepagents | $1.80 | 127 s | (one agent) | 13 | read XBRL tag soup (reader bug, fixed); favored the spin-off |
| fan-out, caps 6 model / 8 tool | $0.43 | — | 1 of 5 | 2 | model cap hit before any report (harness bug, fixed) |
| fan-out, tool budget 5 first | $0.31 | 19 s | 5 of 5 | 0 | spent the budget re-finding filings by search (fixed: changed filings handed over) |
| fan-out, changed filings in hand | $0.83 | 25 s | 5 of 5 | 18 (6 discarded) | **all investigators read the 2025 filing (339 -> 336, Vernova) instead of the 2024 one (225 -> 339, ASU 2018-12)** |
| fan-out, told where the largest move is | $0.89 | 26 s | 5 of 5 | 21 (2 discarded) | none blocking: see below |

The code rule kept the spin-off "ruled out" on the card throughout. But the research findings
argue for the spin-off, and the accounting-change investigator "contradicted" its own (true) cause.

**Diagnosis of run 4 (harness, not model):** the question listed both changed filings without
saying where the big move happened. Code knows it: 225 -> 339 (+50.7%) in the 2024 10-K, 339 -> 336
(-0.9%) in the 2025 10-K. (Materiality doesn't separate them: -0.9% is above our 0.5% threshold.)
The question now lists every move with its size and marks the largest.

**Run 5:** every investigator read the 2024 filing. The accounting-change investigator found and
quoted GE's ASU 2018-12 adoption and tied it to the 225 -> 339 move; its one "contradicts" finding
is correctly scoped to the small 2025 move. The code rule still rules out the spin-off. The
reclassification and "other" investigators still argue the spin-off, and one note misattributes
Vernova to the 2024 filing: the existence-vs-binding problem persists in the notes, which is why
the card shows them as unverified.

**Decided with the author (card display):**
- Quotes and sources first; the investigator's note is labelled "unverified reading" and collapsed.
- At most 2 findings per cause, alternating for/against, with a count of the rest: a pile of
  findings for one cause would read as a favorite.
- An investigator's "contradicts" is shown as evidence and never changes a cause's fit; only code
  rules rule a cause out.

**Cost:** ~$0.90 and ~26 s per research round with Sonnet 4.6 (5 parallel investigations). Research
runs only when a reviewer asks; total spent on these five development runs: ~$4.30.

**Still open:** one case (GE) is not an eval. A research eval over the labelled hard cases (does the
true cause get at least one verified supporting finding; does any investigator contradict the true
cause) is the next step before trusting it more widely.

## 2026-10-21: Research eval (LangSmith experiment) and reviewer feedback loop

**Research eval** (`filings-agent sweep research`, experiment
`research-claude-sonnet-4.6-fanout-4-8dc6d335`, dataset `filings-agent/research-hard`): 11 hard cases,
one per company and metric (non-splits, dev + test; GE revenue FY2021-23 etc. not repeated), no hint,
round 1.

| metric | result |
|---|---|
| true cause has a verified supporting finding | **11/11** |
| a verified finding contradicts the true cause | 1/11 (GE net income: the finding is scoped to the small 2025 move, see above) |
| true cause still open on the card after research | 11/11 (code rules decide fits) |
| other causes with supporting findings, avg | 0.9 (noise the reviewer reads past) |
| quotes discarded as not in the cited source | 9.9% |
| cost / latency per case | $0.74 avg ($8.16 total, GE reused from cache) / 27.5 s avg |

**Caveats:** n=11 and 9 of the 11 are spin-offs [corrected 2026-10-23: was "10"]; the harness was developed on GE net income
(5 runs), so that case isn't independent; the 10 others were never seen during development.
"Supporting finding" means a verified quote the investigator says supports the cause; whether a
reviewer reaches the right decision faster with it is not measured. Most of the "wrong causes
supported" is reclassification on spin-off cases (moving a business to discontinued operations is
literally a reclassification between lines), so the cause list overlaps there.

**Reviewer feedback loop:** when a run finishes, the reviewers' decisions are posted as human
(`app`) feedback on its LangSmith trace: `review.agent_agreement` (reviewer's pick vs the agent's
hidden answer, over cases where they picked: human-measured accuracy on exactly the hard cases),
`review.escalated`, `review.research_rounds`, `review.auto_approved`, `review.decisions`, with every
decision in the comment. Charted over time, these are the production signals: agreement falling or
escalations rising means the agent or the evidence got worse. Corrected causes still become
regression tests through `export-corrections`, as before.

## 2026-10-22: Fresh split: does it generalize beyond GE?

**Why:** the review card, triage and research were all shaped by one case (GE net income) and the
research eval was 10/11 spin-offs. A fair criticism: "you fixed the case you studied".

**What:** a third split, `fresh`, for companies added after every choice was frozen: never tuned
on, not in the CI gate (which scores dev), reported separately. Six life insurers that adopted
ASU 2018-12 (LDTI) on 2023-01-01 and restated 2021-22: MET, PRU, AFL, LNC, GL, UNM. The XBRL
answer key found 24 material restatements. **12 are labelled accounting_change** (net income at
AFL, GL, MET, PRU, UNM; GL operating income), each citing a passage verified verbatim in the filing
where the number changed. 12 are left unlabelled: revenue and cash-flow changes the text doesn't tie
to one cause (LDTI can't change cash), and all of LNC, which adopted LDTI **and** restated 2021-22
for an error (2022 Form 10-K/A): two causes at once, one outside our cause list. Labels were drafted
by the coding assistant from the filing text; to be spot-checked by the author (done 2026-10-24, see below). Dev and test datasets
are byte-for-byte unchanged.

| on the 12 fresh accounting-change cases | result |
|---|---|
| review card (no LLM) | true cause open 12/12, never ruled out; 2.0 causes left open of 4 |
| production pipeline (Sonnet 4.6, prompt v2, unchanged) | **12/12**, quotes verified 12/12, $0.11 |
| triage | 0 auto-approved (correct: code can't confirm an accounting change) |
| agent v1 (create_agent + vector RAG) | 0.83 (10/12); both misses "unknown" (unverifiable quote), **0 confident-wrong** |
| research, one case per company and metric (6) | true cause supported **6/6**, contradicted 0/6; other causes supported 0.33 avg; 19.5% of findings discarded as unverifiable (vs 9.9% on the first set: insurer 10-Ks are huge and table-heavy); $0.69, 25 s per case, $4.14 total |

**Combined research result:** true cause supported on 17/17 hard cases (10 companies, three causes [corrected 2026-10-23: was "11 companies, two causes"]),
contradicted on 1/17 (GE, scoped to the small 2025 move).

**Caveats:** still small and still one standard (all fresh cases are ASU 2018-12); research ran one
fiscal year per company (both years cite the same filing). Run locally, not as a LangSmith
experiment: the LangSmith monthly trace quota was exhausted, mostly by test runs that had tracing on
(fixed: `tests/conftest.py` forces tracing off; `test_tests_never_send_traces` guards it). Outputs
are cached and saved in `data/research_eval/fresh.json`. After billing was updated (2026-10-22) they
were recorded from the cache at no model cost: experiment `research-claude-sonnet-4.6-fanout-4-2e153916`
on dataset `filings-agent/research-fresh`, same results.

**Also fixed:** `split_of` returned "test" for any ticker not in dev; it now refuses unknown tickers.

## 2026-10-22: Live check of the reviewer feedback loop: two bugs the fakes couldn't see

A real run (GE, thread `live-ge-review-1`, $0.11, mostly cached) paused with 10 cards; decided with
the cited labels (8 picks, 2 escalations: GE long-term debt FY2023 and operating cash flow FY2021,
the cases nobody could label) and published once. The `online.*` checks landed on the trace, but
**no `review.*` feedback did**:

1. **Wrong status.** Review feedback posted only when the run's status was `"done"`; a run that
   publishes reports `"published"` (`"done"` is only for runs without a ledger). The unit test used
   a made-up `"done"` summary, so it passed while production never posted. Fix: post for either
   finished status; the test now uses `"published"`, plus a test for `"done"`.
2. **Invalid source type.** `feedback_source_type="app"` is rejected by the SDK (only `"api"` and
   `"model"` exist). The fake client accepted anything. Fix: `"api"`, and the fake now validates
   against the SDK's real enum.

After both fixes, re-posting (idempotent ids) put the expected feedback on the trace
(`90a24716-...`): `review.decisions` 10, `review.escalated` 2, `review.agent_agreement` 1.0 (the
pipeline's hidden answers matched all 8 picks), `review.research_rounds` 0, each decision in the
comment. Lesson for the runbook: anything that talks to an external service gets one live check
before it's called done; a fake only proves what the fake knows.

## 2026-10-23: Review UI (React) for the human in the loop

The card existed as JSON and terminal text; reviewers needed a page. `web/`: React 19 + TypeScript
(strict) + Vite, served by FastAPI at `/` once built (`make web`), talking only to the existing API.
One page per run: live progress (SSE), the changes code decided by itself, the cards with decision
controls (pick the cause, keep the original figure, escalate, or research with a hint), a submit bar
that stays disabled until every card is decided (the API rejects partial reviews), then the decisions
and the memo. The agent's cause is never rendered, including in the progress stream, which carries
it; the UI shows only the lane. Picking a cause code ruled out shows a warning, not a block: the
reviewer can overrule a rule, and that override is visible. Decision logic is a pure module with its
own tests (Vitest + Testing Library, 12 tests); CI runs typecheck, tests and build.

Building it found three backend bugs, fixed with tests:
1. After a review, the events stream replayed the paused run's "end" first, so a client following
   the resumed run stopped at once. Each launch now gets a fresh event log.
2. A run stayed "running" for seconds after "end" while monitoring posted to LangSmith, so a client
   reloading on "end" looped. Monitoring now runs in the background after the run counts as finished.
3. (Not a bug: added) `GET /analyses` lists recent runs for the UI.

Verified end to end on the real server: NVDA published without review (both splits confirmed by
code); GE paused with 10 cards, 9 decided and 1 sent to research, the card came back with 21
verified findings, then decided and published once with a verified memo. Cost ~$0.005 (cached).

## 2026-10-23: Review UI design: dark by default, built to read as an audit tool

Researched how financial products signal trust before restyling (sources below). What we applied:
- **Color only for state.** Neutral layered surfaces (dark: Carbon-style gray 100/90 layers, not
  pure black); color appears only on status: fits / possible / ruled out / no evidence, awaiting
  review, published. Stripe's dashboard is the reference for "state-only color" in financial tables.
- **Status never relies on color alone:** every badge pairs an icon shape with a text label (IBM
  Carbon's status-indicator pattern; Bloomberg ships alternate schemes for its color-vision-deficient
  Terminal users).
- **Tabular figures** for every number; filing accession numbers in IBM Plex Mono; amounts the way
  analysts read them ($225.0M -> $339.0M, +50.7%).
- **Provenance visible:** each excerpt is a citation labelled with its source (SEC 10-K excerpt from
  the filing that changed the number, or the research source's host and read date) and a verified
  mark; the investigator's reading stays collapsed and labelled unverified.
- **One look, two themes:** dark by default, light available, the toggle remembers the choice. Fonts
  (Inter, IBM Plex Mono) are bundled, so the demo works offline.

Checked in a real browser (both themes, a run awaiting review, a published run). Found and fixed:
the cost tile divided cached calls by statements ("22/12"), and code's reasons showed identifiers
("discontinued_operations can move cash").

Sources: [fintech dark design kit](https://cdn.jsdelivr.net/npm/@wrongstack/core@0.295.1/design-kits/fintech-dark/KIT.md),
[Lazarev: financial dashboard design](https://www.lazarev.agency/articles/financial-dashboard-design),
[Carbon status indicators](https://www.carbondesignsystem.com/patterns/status-indicator-pattern),
[Carbon color](https://carbondesignsystem.com/elements/color/overview/),
[Bloomberg: designing the Terminal for color accessibility](https://www.techatbloomberg.com/blog/designing-the-terminal-for-color-accessibility/).

## 2026-10-23: Shoring up before submission: cited rules, CI on a fresh clone, guardrails

- **Triage rules cite their sources** (checked against Deloitte's DART roadmaps): ASC 260-10-55-12
  (a stock split restates share counts and per-share amounts only) and ASC 205-20-45-3A (results of
  discontinued operations are "a separate component of income", i.e. inside net income, so moving a
  business there can't change the total). The spin-off/share-count rule has no codification
  paragraph and is labelled as reasoning rather than given an invented citation. The source shows on
  the review card; a test fails if a new rule has neither.
- **CI on a fresh clone.** The eval gate needs AWS, SEC and LangSmith secrets; a new repository won't
  have them, so the first push would have shown a red build. A `gate-config` job now skips the gate
  with a notice when they're missing. Running the suite the way CI would (no `data/`, no `.env`)
  found one more failure: building the app read `SEC_USER_AGENT` eagerly because the research
  fetcher wanted it, so a server or Studio without it would have crashed at startup. The SEC
  contact is now read when research first reaches sec.gov.
- **Guardrails and data handling** documented in one place (README): what personal data exists
  (reviewer notes, the SEC contact), and each risk with its guardrail and the code that enforces it.

## 2026-10-23: Confidence campaign: test the seams, not just the units

**Why:** every bug found by hand so far sat at a seam a unit test assumed away (a run status the
runner never returns, a feedback value the SDK rejects, a stream replaying an old "end", a crash
without `SEC_USER_AGENT`, tests sending traces, a deleted test, wrong numbers in the docs). More unit
tests of the same kind wouldn't have caught them; tests against reality would.

**Built (all runnable by anyone, almost all $0):**
| Check | Command | Catches |
|---|---|---|
| fakes match the real classes (methods, parameter names) and real values (feedback enums) | `make test` | fakes looser than reality |
| API contract: UI fixtures generated through the real FastAPI app; a Python test fails when stale; the UI's types must cover them exactly (compile-time + test) and the dashboard renders them | `make test`, `make web-check` | renamed/removed fields |
| end-to-end smoke: real server, graph and cached models over HTTP; 29 checks incl. every decision type, refusals, idempotency, research round-trip, publish once, feedback in LangSmith, cost | `make smoke` | seams between server, graph, client and LangSmith |
| browser end-to-end: Chromium clicks a full GE review; theme persistence; axe accessibility in both themes | `make web-e2e` | UI bugs, contrast |
| docs receipt: every headline number recomputed from cache (paid calls blocked) and matched against the exact sentence | `make receipt` | wrong or stale numbers in prose |
| fresh clone: copy without gitignored files, run every check | `make fresh-check` | environment assumptions |
| property-based tests (hypothesis, fast-check) on parsing/verification/decision logic | `make test`, `make web-check` | edge cases |
| dependency and code security scans | `make audit` | known CVEs, risky patterns |

**Found and fixed by these, today:**
1. `TestClient` used without `with` cancelled background runs between requests; the API tests passed
   only because the fakes were fast. Fixed: one event loop per test, like the real server.
2. Property test: a split ratio of 129.5 was within 0.5% of both 129 and 130. Fixed: drift is also
   capped at 0.25, so a ratio is near at most one whole number (all real splits unchanged).
3. The docs claimed 47 labelled restatements; the real count is 45 (the two GE correction replays
   were counted twice). Test counts in the README were stale twice; the receipt now checks them.
4. A fake (`create_dataset(name=...)`) didn't match the real signature (`dataset_name`).
5. axe: light-theme secondary text at 4.46:1, below WCAG AA's 4.5:1. Darkened to pass.
6. Two dependencies with known CVEs (certifi; opentelemetry-api, pinned by the dev-only Studio server)
   upgraded; bandit: a silent `except: pass` now logs, an `assert` used as a runtime check is now an
   explicit error, one SQL string annotated (table names from a fixed tuple).
7. New: the API records each run's LangSmith trace id; the dashboard links "Open trace in LangSmith".

## 2026-10-23: Independent review: four reviewers, ~40 findings, fixed with a test each

**Why:** the sweeps kept finding bugs, so four independent reviewers (each with fresh context) read
the code by area: graph/steps/storage, research/LLM/spend, API/UI, and evals/doc claims. Each
finding below is fixed with a regression test that fails on the old code (the publish race was
checked: 3/3 failures before, 5/5 passes after).

**Correctness (the guarantees the docs make):**
- Quote verification could be fooled: fragments under 12 characters were skipped and the pieces
  only had to appear somewhere, in order. Now every fragment counts and all must fall within one
  800-character passage.
- Auto-approval could pass an unexplained change (a 10:1 split, then ×1.15 a year later). Every
  material step must now be a split.
- Extraction never checked the sign or the scale, and "234" matched inside "1,234". Now: whole
  number, sign (parentheses), dollar scale. The memo check also requires the sign.
- A rejected review answer locked the run (replayed on every resume), and an unknown action was
  treated as "keep the original". The review node now asks again with the problems shown.
- Re-running a finished request re-paid the work, kept the old decisions and then published
  nothing (silent data loss). It now returns the stored result (`replayed`); an unfinished thread
  asks for `--resume`/`review`; `--fresh` gets its own thread.
- Two processes could both see "not published"; the second crashed. `BEGIN IMMEDIATE` first.
- API: a review of cards that changed since (another tab, a research round) is refused (409,
  `review_token`); accept without a cause, research with no rounds left and blank tickers are
  rejected; pending cards no longer leak the agent's cause through the status endpoint.
- Dashboard: errors shown inline with the decisions kept, the event stream reconnects after a
  blip, late responses for another run are dropped, accessible progress bars.

**Safety (research and spend):**
- A research finding could cite any allowed page, including one never read or another company's
  filing. It must now cite a page the investigation read, and an SEC filing must be this company's.
- A network outage reached the model as "tool error", ended as "no evidence" and was cached, so
  the round could never be retried. Outages now fail that investigation (shown, not cached).
- `allowed()` accepted credentials and other ports and raised on malformed URLs; the fetcher held
  one lock across each request (five investigations read one page at a time).
- **A corrupt spend ledger read as $0 spent** and was then overwritten (the cap failed open). This
  reverses the earlier "corrupt ledger starts clean" choice: it now refuses paid calls. The cost
  reservation also counted only `messages`; it now counts the system prompt and tool schemas.

**Found while re-verifying the numbers:** research looked up company ids only among the 8
extraction companies, so DHR, JNJ, MMM and all six insurers ran with none: "this company's
filings" searched every company's, and changed filings were cited as `/edgar/data/0/`. The stricter
checks showed it (GL's real filing, `/data/320335/`, couldn't be matched to an unknown company:
research fresh 6/6 → 5/6). Fixed (ids from every company's filing list, the same bug and fix as the
reconcile agent's earlier; a search with no id now refuses), and changed filings are matched by
accession number, so cached citations to `/data/0/` still point at our copy. With the fix: 17/17
again. The cached investigations still ran with the wide search; a clean re-run is ~$12, over this
project's remaining budget, so it's flagged in EVALS.md (known failure 9) for the next owner.

**Evals and claims:**
- Reviewer corrections lived in gitignored `data/`, so the CI gate never had them (docs said it
  did). Moved to `configs/reconcile_corrections.jsonl` (committed).
- The gate didn't check the easy lane (the #1 priority) and always scored the pipeline. It now
  holds easy-lane accuracy at 100% and scores the configured reconciler.
- Judge errors counted as failed memos; calibration divided false passes by all labels (not by the
  human fails) and trusted a criterion with no human fails.
- `review.picked` posted next to `review.agent_agreement`, which leaves escalations out.
- Doc claims corrected, each now checked by the receipt where it's a number: card 27% ruled out /
  14% no evidence / 59% open (not "about half"); research's wrong causes also get supporting
  findings (0.9 per hard case, 0.3 per fresh case); easy lane 0 of 11 (95% upper bound ~27%);
  "anchor hit", not "retrieval hit"; the injection eval covers only agent v1; the fresh split is new
  companies but GE's standard, one class, labels drafted by the assistant and not yet spot-checked (done 2026-10-24)
  (the labels file said they were); the test split was run once for model and prompt choices, but
  triage, the card and research came later; counts (9 spin-offs, 10 companies, five insurers).

**Result:** 369 backend tests, 28 UI tests, smoke 31 checks, browser e2e 4/4, receipt OK,
fresh-clone check (95.2% coverage), audit clean (setuptools upgraded for GHSA-xmj3-9gf7-h52w).
Spend for all of this: ~$0.02 (cached runs).

## 2026-10-23: Between runs: decision history, run identity, leases, authentication

**Why:** an external review found that idempotency held within a run but not between runs. Each
finding was checked against the code first (the overwrite was reproduced against the real
`Ledger`), and each fix has a test that fails on the old code.

**A later run erased earlier human decisions.** `resolutions` had one row per discrepancy, and
every publish overwrote it: thread A's reviewer picked `accounting_change` for GE net income FY2022,
thread B escalated it, and the ledger showed only B's escalation. Worse than reported: the case key
(`ticker:metric:FY`) left out the filings, so a new 10-K's restatement of the same year collided
with the old case. **Decision:** an append-only `resolution_history` (who, when, which run, which
filings compared) with `resolutions` as a view derived from it. Precedence: a human pick beats an
escalation, which beats an automatic answer; among equals, the newest. An escalation after a pick
is shown (`later_escalations`), not applied: only a human pick supersedes a human pick. The case
is the discrepancy plus its filing set. Existing ledgers are migrated on open (their rows become
history; what they overwrote is gone). Still open, for the author: waiting on escalations instead
of publishing them (the 2026-10-21 card entry). Today an escalation is resolved by deciding the case
in a new run, which is now safe.

**Run identity ignored the data and most of the config.** The thread was a hash of the tickers,
the extraction model and the extraction prompt: after a new 10-K, or a new reconcile model or
prompt, the same request replayed the old result, and the UI (tickers only, no key) could only
ever show the first run. The API's `fresh` never started anything once the thread existed.
**Decision:** `run_identity` hashes the tickers, the accession numbers on disk, every model,
prompt and setting that shapes the result, and `GRAPH_VERSION`. With an `Idempotency-Key`, the key
stays the identity (a retry must reach the same run even if the filings changed between attempts);
the body fingerprint is what the client sent, and a mismatch is now **422**, as the IETF
idempotency-key draft says (409 stays for "still in progress"). `fresh` is a new analysis: a new
thread without a key; with one, a rep derived from the key, so a retry still dedupes. The UI has a
*Re-analyze* box.

**Locks lived in one process's memory.** Two workers (or the CLI and the server) could both see
"not started" and run one thread: double spend, interleaved checkpoints. **Decision:** a lease row
per thread in SQLite (`storage/leases.py`), taken under `BEGIN IMMEDIATE`, renewed by a heartbeat,
expiring 2 minutes after its process dies. The API takes it before answering (so the loser of a
race answers `running`, not 202) and passes it to the run, which re-enters it; the CLI takes its
own. Why a lease over a transaction-scoped lock (`FOR UPDATE`, `pg_advisory_xact_lock`): a run
lasts minutes and pauses for review, so no transaction can span it. On Postgres the same table
works. The spend guard's reservations had the same flaw (per process); they now live in the
project ledger file under its flock, each with an expiry.

**No authentication; decisions didn't say who.** **Decision:** bearer tokens from
`FILINGS_API_TOKENS` (`name:token`), compared in constant time; the name is stamped on every
decision by the server (never from the payload), as is the time. The UI trades the token for an
HttpOnly, SameSite=Strict cookie, because EventSource can't send headers. Without tokens the server
runs in local mode and `serve` refuses a non-loopback host. Considered: LangGraph Agent Server
custom auth (`@auth.on`) does the same if this moves onto LangGraph Platform; per-user thread
ownership isn't needed for one review team.

**Smaller:**
- Spend: a call that timed out on the client but ran on Bedrock was recorded as $0. It's now charged
  its worst case (only errors raised before sending cost $0). botocore retried twice per call,
  invisibly to the guard (its hooks fire once per call, not per attempt), times LangGraph's four:
  up to 8 sends. botocore now makes one attempt. **Found while doing it:** botocore's network
  errors (`ReadTimeoutError`, `EndpointConnectionError`, ...) are not the builtin
  `TimeoutError`/`ConnectionError`, so `is_transient` never retried them; botocore's own retry hid
  it. They're transient now.
- The memo call wasn't cached (the cache existed but wasn't wired), and a fresh run reused the memo
  key; both fixed.
- Every run opened a ledger connection and never closed it; runs now share one per file.
- The review token was optional; it's required.
- Re-exporting corrections after a partial failure re-sent existing example ids (LangSmith lists
  409 Conflict for that), so a half-finished export could never finish. Existing ids are skipped.
  Not yet confirmed against live LangSmith.

## 2026-10-24: Fresh labels checked against the companies' own tables

**Why:** the 12 fresh labels (accounting change, ASU 2018-12, at five insurers) were drafted by the
coding assistant and marked as not yet checked by the author.

**What the check found.** Each insurer that adopted the standard printed a table in its FY2023 10-K
with the amount as first reported, the effect of the standard, and the amount as reported now. For
10 labels the table adds up exactly to the change in our dataset: GL net income and operating income
(its pre-tax total; the adoption impact on income before taxes equals the change), MET, PRU and UNM
net income, for 2021 and 2022. Aflac prints no such table for its income statement. Its 10-K says
"All relevant prior-year amounts have been adjusted for the adoption of ASU 2018-12" and explains
which 2021 lines moved, and it names no other cause.

**Decision (with the author): keep Aflac's 2 labels, record how each label was verified, report
both ways.** `verified_by: tie_out | narrative` (with a note of the figures) is in
`configs/reconcile_labels.yaml`, and the receipt reports the fresh split on all 12 labels and on the
10 that tie out: pipeline 12/12 and 10/10, agent 10/12 and 9/10, research 6/6 and 5/5. The criterion
is the strength of the filing's evidence, decided without looking at model results. Dropping Aflac
would have raised the agent's score (one of its two fresh misses is AFL), which would look like
removing a hard case after seeing the results.

