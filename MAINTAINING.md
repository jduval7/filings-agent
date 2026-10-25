# Maintaining filings-agent

For the team that owns this after handoff: how to get fluent, the routine that keeps it improving,
what to do when a new failure shows up, and how to train the next person. Operating commands live in
[RUNBOOK.md](RUNBOOK.md); code structure and invariants in [CONTRIBUTING.md](CONTRIBUTING.md).

## What you're taking over

A LangGraph agent whose quality is measured, not asserted. Three things keep it honest, and your job
is to keep all three running:

1. **Code checks every LLM output** against its source, and code rules (not LLMs) decide which
   explanations are ruled out.
2. **Humans decide what code can't confirm**, from a review card that shows evidence for every
   cause and never the agent's guess.
3. **Every mistake becomes a test.** Reviewer corrections become dataset rows, the CI gate blocks
   regressions, and reviewer decisions are charted in LangSmith.

If one of these stops (corrections not exported, the gate skipped, feedback not posted), quality
drifts silently. That's the main risk of a handoff.

## Who owns what

| Role | Owns | Typical time |
|---|---|---|
| **Engineer (owner)** | code, prompts, harness, evals, the gate, the weekly loop below | ~1 h/week, plus fixes |
| **Domain expert** (an accountant) | cause labels, triage rules, the cause list, judge calibration labels | ~1 h/month, plus rule reviews |
| **Reviewers** (analysts) | decisions on cards; `correct_cause`; notes on escalations | per filing season |

Engineers don't write accounting rules alone: every rule in `steps/triage.py` is a claim about
accounting, and a wrong rule rules out the right answer with confidence. A domain expert signs off,
and the rule cites its source.

## Getting fluent (one week)

**Day 1: run it.** `make setup && make data && make check`. Then `make studio`, run
`{"tickers": ["NVDA"]}`, watch the fan-out, and stop at review. Run `filings-agent cards --thread
<id>` and decide. Find the run in LangSmith and click through its spans.

**Day 2: follow one case end to end.** GE net income FY2022 is the whole system in one case:
- `filings-agent cards --case GE:net_income:FY2022` (the card; why the spin-off is ruled out);
- `configs/reconcile_labels.yaml` (the label and its citing passage);
- the `reconcile-corrections` dataset (the reviewer correction that became a test);
- `score reconcile --reconciler agent` (the agent's wrong answer, and triage catching it);
- the `research-*` experiment in LangSmith (evidence per cause);
- DECISIONS.md entries from 2026-10-17 (corrections loop) and 2026-10-21 (triage, card, research).

**Days 3-4: read DECISIONS.md.** Not for the history: for the pattern. Most "model failures" were
harness bugs (evidence from the wrong filing, a verifier failing on typography, prompt drift, an
incomplete cache key, call caps in the wrong order). That's the instinct to build.

**Day 5: ship one change through the loop.** Pick something small from EVALS.md's known failures
(e.g. the extraction same-row check, EVALS.md known failure 2). Score, experiment, gate, DECISIONS entry, PR. Then break
the gate on purpose (swap in a cheaper reconcile model and run `make gate`) to see it block a bad change.

## The weekly improvement loop (~1 hour)

1. **Read the signals** in LangSmith, project `filings-agent`:
   - `review.agent_agreement` (reviewer's pick vs the agent's hidden answer): falling means the
     agent got worse on hard cases;
   - `review.escalated`, `online.unknown_causes`: rising means new kinds of restatement;
   - `review.auto_approved`: how much review triage is saving;
   - `online.cost_usd`, branch errors.
2. **Work the annotation queue** `filings-agent: review`: score the rubric; for wrong causes make
   sure a `correct_cause` decision exists.
3. **Export corrections:** `filings-agent export-corrections`. They join every eval and the gate.
4. **Pick the top failure** (most frequent, or most expensive when wrong) and run the playbook below.
5. **Write it down:** a DECISIONS.md entry with the numbers and experiment names, even when the
   answer is "not fixing, here's why".

Monthly, with the domain expert: review escalations ("none of these fit" is a missing cause), new
rule proposals, and the calibration labels if the judge or a rubric changed.

## A new failure case arrives: the playbook

Use this for a reviewer's report, a red gate, or an interviewer's "here's a case, what do you do?".

1. **Reproduce it without spending.** Find the trace (filter by tag `ticker:<X>`, `step:reconcile`).
   Look at the card: `filings-agent cards --case <id>` if it's in the labelled set; otherwise run
   the analysis on that ticker (cached calls are free).
2. **Find the step.** Wrong number → extract. Wrong or missing restatement → find_discrepancies
   (often an extraction error that looks like a restatement). Wrong cause → reconcile. Wrong
   memo → memo. Wrong route (skipped review) → triage.
3. **Harness or model?** Check these before blaming the model. Each one has happened here:
   - Did the step see the right evidence? (evidence from the wrong filing; one cause crowding out
     the others; research reading the filing with the small move)
   - Did the verifier reject a correct answer? (Unicode hyphens, markdown escapes, a stray `##`)
   - Is the running prompt or version the one that was evaluated? (prompt drift)
   - Could a stale cache answer? (is everything that changes the output in the key?)
   - Did a limit cut the work short? (call caps; "no report")
   - Did a check pass something wrong because it tests existence, not the link? (GE)
4. **Make it a test before fixing it.** Reviewer decision with `correct_cause` + `export-corrections`,
   or a label in `configs/reconcile_labels.yaml` citing the passage, or a unit test for a harness
   bug. The gate should now fail; that's the point.
5. **Fix at the right layer,** cheapest and most general first: evidence gathering, a verifier, a
   code rule (with the domain expert), a new prompt *version* (never edit one in place), then the
   model choice. Never a per-company special case.
6. **Prove it:** `score` locally, `sweep` for a LangSmith experiment, **Compare** against the current
   one, check the slices didn't regress elsewhere, `make gate`.
7. **Write the DECISIONS.md entry** and open the PR.

## Changing things safely

| Change | Evidence required before merge |
|---|---|
| Prompt | new version next to the old; `score` + `sweep` on dev; re-check after any model swap (the same rule helped one model and hurt another) |
| Model for a step | `sweep` over candidates; cheapest that clears the bar on every repetition; gate |
| Triage rule | cited source + domain expert; `cards --eval` keeps `true_cause_hidden` empty; reconcile scores per lane |
| Research harness or sources | bump `HARNESS` / `SOURCES_VERSION`; `sweep research`; compare cost and `true_cause_supported` |
| Judge model or rubric | recalibrate against human labels before reporting judge numbers |
| Test split | run once, at the end, after choices are frozen |

## What's next (in order)

1. Random audit sample of auto-approved changes (the only honest estimate of the easy lane's error
   rate) and known-wrong "catch trial" cards in the queue (EVALS.md, production plan).
2. Extend the triage rules with the domain expert (e.g. reclassifications and total net income),
   each citing its source (a test enforces it).
3. Extraction: the number and its label in the same row and year column (whole-number, sign and
   scale checks are done).
4. Re-run research with company ids (EVALS.md known failure 9; ~$12) and add an injection eval for
   the pipeline and research (known failure 10).
5. Calibrate the judge (20 labels in `configs/calibration_labels.yaml`).
6. A research eval with more than spin-offs; measure time per card with real reviewers.
7. A long-horizon "restatement history for a company over ten years" feature: open-ended, where a
   deep agent (planning, notes, subagents) would fit, unlike per-case research
   ([why](DECISIONS.md#2026-10-21-reviewer-triggered-research-deepagents-tried-code-fan-out-kept)).

## Training the next person

- **Pair on cards first.** Review five real cards together before touching code. The domain is the
  hard part; the code follows from it.
- **Walk GE end to end** (day 2 above) and have them explain it back in plain words.
- **Have them break the gate** with a cheaper model, and read why it failed.
- **Their first PR goes through the full loop**, with a DECISIONS entry, even if it's small.
- **Review their DECISIONS entries, not just their code.** The habit to teach is "every claim has a
  number, every number has a script, and failures are written down".
