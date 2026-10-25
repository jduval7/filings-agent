# Feeding human review decisions back into LLM agents (LangChain / LangGraph / LangSmith focus)

Researched 2026-10-21. Source legend: **[OFFICIAL]** = LangChain/LangGraph/LangSmith docs or API reference; **[VENDOR-GUIDE]** = Anthropic engineering guidance; **[PAPER]** = peer-reviewed or arXiv; **[PRACTITIONER]** = opinion/blog by a known practitioner; **[WEAK]** = low-authority blog, use with caution.

## Q1. LangGraph / LangChain v1 human-in-the-loop: review payloads and allowed actions

### Takeaway
In the LangChain v1 / LangGraph 1.x era there are two layers. The low-level one is `interrupt()` plus `Command(resume=...)` (a checkpointer and a `thread_id` are required). On top of that, the LangChain v1 `HumanInTheLoopMiddleware` standardizes the payload as `action_requests` plus `review_configs`, with four decision types: approve, edit, reject (with a message) and respond. The older `HumanInterrupt` / Agent Inbox schema from `langgraph.prebuilt` is deprecated.

### Cited Findings
- **[OFFICIAL]** `HumanInTheLoopMiddleware` takes `interrupt_on`, a dict that maps tool names to `True` (interrupt with defaults), `False` (auto-approve) or an `InterruptOnConfig` with `allowed_decisions` (`'approve'`, `'edit'`, `'reject'`, `'respond'`), `description` (a static string or a callable) and an optional `when` predicate that gates the interrupt on the tool's arguments. Other options: `description_prefix` (default "Tool execution requires approval"), `interrupt_mode` (`"batched"` default, or `"per_call"`) and `edit_notice`. A checkpointer is required (InMemorySaver for dev; AsyncPostgresSaver or MongoDBSaver for prod). — [LangChain docs: Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- **[OFFICIAL]** Interrupt payload shape: `{'action_requests': [{'name', 'args', 'description'}], 'review_configs': [{'action_name', 'allowed_decisions'}]}`. Resume in batched mode with `Command(resume={"decisions": [{"type": "approve"}, {"type": "reject", "message": "..."}]})`. An edit is `{"type": "edit", "edited_action": {"name": ..., "args": {...}}}`. Per-call mode is keyed by `interrupt.id`. — [LangChain docs: Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- **[OFFICIAL]** Decision semantics: `approve` runs the tool with the original arguments; `edit` changes the arguments before execution; `reject` skips execution and can include a feedback `message` that goes back to the model; `respond` returns the human's message as the tool result. The middleware runs as an `after_model` hook, so it intercepts tool calls after the model generates them and before they execute. — [LangChain docs: Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- **[OFFICIAL]** LangGraph `interrupt()` patterns: approve/reject (resume with `Command(resume=True/False)` and route on the value), review-and-edit state (the resume value becomes the return value of `interrupt()`), tool-call review (`interrupt()` inside the tool), and input validation (one `interrupt()` per node invocation, looping via conditional edges rather than `while True`). — [LangGraph docs: Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- **[OFFICIAL]** LangGraph rules:
  - Do not wrap `interrupt()` in a bare try/except, because it works by raising.
  - Keep the order of interrupts deterministic, because resume values are matched by index.
  - Code before `interrupt()` re-executes on resume, so side effects must be idempotent or moved to a later node.
  - The checkpointer and `thread_id` act as the resume cursor.
  
  — [LangGraph docs: Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- **[OFFICIAL] Deprecation:** `langgraph.prebuilt` `HumanInterrupt` is deprecated with the message "HumanInterrupt has been moved to `langchain.agents.interrupt`." Its schema has `action_request`, `config` (`HumanInterruptConfig`: allow_ignore / allow_respond / allow_edit / allow_accept) and `description` (a Markdown string). — [LangChain reference: HumanInterrupt](https://reference.langchain.com/python/langgraph.prebuilt/prebuilt/interrupt/HumanInterrupt)
- Agent Inbox is a LangChain UI for viewing, editing, accepting or responding to interrupts, built on the `HumanInterrupt` schema. — [Agent Inbox repo index (algolia docsearch of langchain-ai/agent-inbox)](https://docsearch.algolia.com/mcp/docs/repo/langchain-ai/agent-inbox)
- The older static `interrupt_before` compile-time breakpoint approach has been superseded by calling `interrupt()` inside nodes or tools. — [search summary; secondary](https://medium.com/@areebahmed575/langgraphs-interrupt-function-the-simpler-way-to-build-human-in-the-loop-agents-faef98891a92) (the current official interrupts page no longer presents `interrupt_before` or `NodeInterrupt` as the primary pattern: [LangGraph docs: Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts))

### Inferences
- The official payload carries the action (name and args) and a human-readable `description`, which can be a callable and so can be built from the tool arguments. It has no field for the agent's rationale, evidence or confidence. To show evidence on a review card, put it into the `description` callable or into a custom `interrupt()` value.
- The official decision vocabulary has nothing that records why a reviewer rejected or edited something. A reject `message` goes to the model, not to a durable store. To capture reasons for the improvement loop, also log the decision and its reason as LangSmith feedback on the run (see Q2).
- An edit (`edited_action`) is a natural (agent proposal, human-corrected) pair. It maps directly to a dataset example with corrected reference outputs.

### Gaps
- I did not find official guidance on what evidence to show reviewers (for example, sources or confidence) beyond `description`.
- I did not verify whether Agent Inbox is still actively maintained or has been superseded in the v1 era.

## Q2. LangSmith: feedback API, annotation queues, datasets, online evaluators, automation rules, Align Evals

### Takeaway
LangSmith has the full set of loop primitives:
- `create_feedback` (with `score`, `value`, `comment`, a `correction` ground-truth field and a source type) attached to any run or trace.
- Annotation queues: single-run or pairwise, with rubrics, instructions, reservations, a reviewers-per-run setting and assigned reviewers. From a queue you can add corrected examples to datasets.
- Automation rules with filters, a 0–100% sampling rate and backfill. Rule actions: add to queue, add to dataset, run an online or code evaluator, webhook, alert, extend retention.
- Align Evaluator: calibrates an LLM-as-judge against human labels using an "alignment score".

### Cited Findings
- **[OFFICIAL]** Example call: `client.create_feedback(key="user_feedback", score=1, trace_id=trace_id, session_id=session_id, comment="...")`. Feedback can be attached to any child run of a trace (for example, the retrieval step versus the generation step), not only the root. For browser clients, presigned feedback tokens give a URL scoped to one run and one feedback key, so the API key is not exposed. — [LangSmith docs: Attach user feedback](https://docs.langchain.com/langsmith/attach-user-feedback)
- **[OFFICIAL]** The `correction` parameter of `create_feedback` is "the proper ground truth for this run" (a dict). `feedback_source_type` records whether the feedback came from a model or the API (default API). — [LangSmith SDK reference: create_feedback](https://langsmith-sdk.readthedocs.io/en/latest/client/langsmith.client.Client.create_feedback.html); also [reference.langchain.com](https://reference.langchain.com/python/langsmith/client/Client/create_feedback)
- **[OFFICIAL]** Annotation queues come in two kinds. Single-run queues show one run or thread at a time against a configurable rubric of feedback keys, with categorical descriptions recommended. Pairwise annotation queues (PAQs) show two runs side by side for baseline-versus-candidate comparison, and need only a feedback key and an optional description. Reviewer instructions appear in a sidebar. Other settings:
  - **Reservations** lock an item so two people don't review it at once; the lock expires automatically.
  - **Number of reviewers per run** sets how many reviewers must finish an item before it leaves the queue.
  - **Assigned reviewers**: an item is "Completed" only when every assigned reviewer has submitted.
  
  From a run item you can add a corrected example to a dataset, and "assertions can replace manual output editing" when you create expected outputs. Automation rules can enqueue matching runs, or whole threads once they go idle. — [LangSmith docs: Annotation queues](https://docs.langchain.com/langsmith/annotation-queues)
- **[OFFICIAL]** Automation rule actions: Add to Annotation Queue, Add to Dataset, Trigger Webhook, Run Online Evaluator, Run Custom Code Evaluator, Extend Data Retention, Trigger Alert. Filters can use feedback (for example, negative feedback) and evaluator scores. A sampling rate (0–100%) controls what fraction of matching items trigger the action. "Apply to Past Runs" backfills from a date. Thread rules wait for an idle period (minimum 2 minutes), process at most 500 threads per execution, and require runs under 7 days old. Rules run independently; to chain them, filter one rule on the output of another (for example, on feedback an evaluator wrote). — [LangSmith docs: Automation rules](https://docs.langchain.com/langsmith/rules)
- **[OFFICIAL]** "Improve LLM-as-judge evaluators using human feedback" (Align Evaluator) has four steps: (1) select experiments or runs; (2) add them to an annotation queue for expert labeling; (3) test the evaluator prompt in the Evaluator Playground against the labels; (4) refine and repeat. The alignment score is "the percentage of examples where the evaluator's judgment matches that of the human expert". The UI compares human and LLM scores side by side, sorts to show unaligned cases, and saves a baseline score. Recommendations: start with about 20 labeled examples, keep labels balanced between positive and negative, turn on judge reasoning, and expand the labeled set to avoid overfitting. It works for both offline (dataset) and online evaluators. — [LangSmith docs: Improve judge evaluator](https://docs.langchain.com/langsmith/improve-judge-evaluator-feedback)
- **[OFFICIAL / VENDOR BLOG]** Align Evals was announced on the LangChain blog. — [Introducing Align Evals](https://www.langchain.com/blog/introducing-align-evals) (I did not fetch the full text; the page redirected.)

### Inferences
- A clean mapping for a review card:
  - approve → `create_feedback(key="review_decision", score=1)`
  - reject → `score=0`, with a `comment` and a categorical `value` for the reason
  - edit or correct → `correction={...}` on the specific child run that was wrong
  
  Then an automation rule on `review_decision=0` or on corrections adds the item to a dataset (the regression set) and/or a queue for root-cause labeling.
- Raw "alignment score" is percent agreement. On imbalanced labels (most items pass), it can look high even when the judge misses failures. Pair it with TPR/TNR (see Q3).
- The automation rule sampling rate is the built-in way to send a random audit sample of auto-approved runs to a queue (see Q4).

### Gaps
- I did not find official LangSmith documentation of a built-in inter-annotator agreement metric (for example, Cohen's kappa) across multiple reviewers on a queue. The "reviewers per run" setting collects multiple labels, but I found no documented agreement report.
- I did not confirm whether `correction` values appear automatically as reference outputs when a run is added to a dataset from a queue.

## Q3. Industry practice for closing the loop (error analysis, validating validators, production failures → evals, explicit vs implicit feedback, capturing reasons)

### Takeaway
Practitioners and vendors agree on the loop:
1. Read real traces.
2. Write free-text critiques that give the reason, not just a label.
3. Cluster the critiques into a failure taxonomy.
4. Turn failures into a small eval set of 20–50 items, split into regression and capability evals.
5. Build binary pass/fail judges and validate them against held-out expert labels using TPR/TNR.

Grading criteria drift as reviewers see outputs, so the rubric itself has to be iterated.

### Cited Findings
- **[PAPER]** Shankar et al. 2024, "Who Validates the Validators?", introduces EvalGen, a mixed-initiative tool that generates candidate assertions (Python or LLM-grader prompts) and asks the user to grade sample outputs to align them. Key finding, "criteria drift": "users need criteria to grade outputs, but grading outputs helps users define criteria". Some criteria depend on the specific outputs observed. — [arXiv 2404.12272](https://arxiv.org/abs/2404.12272)
- **[PRACTITIONER]** Hamel Husain, "Critique Shadowing":
  - Use one principal domain expert, not a proxy.
  - Build a diverse dataset across features, scenarios and personas.
  - Get pass/fail judgments with detailed critiques, written clearly enough that a new employee could follow them.
  - Fix obvious errors before building a judge.
  - Build the judge iteratively against the expert labels.
  - Do error analysis by root cause.
  - Build specialized judges only after that.
  
  He calls 1–5 scales "doing it wrong" and says to measure the judge by TPR/TNR rather than raw agreement, which "can be misleading with imbalanced datasets". — [hamel.dev: Using LLM-as-a-Judge](https://hamel.dev/blog/posts/llm-judge/)
- **[PRACTITIONER]** Hamel Husain and Shreya Shankar, Evals FAQ:
  - Error analysis is "the most important activity in evals". Its steps are open coding (free-text notes on traces, at least 30 traces by domain experts before automating), then axial coding into a failure taxonomy, then iterating to "theoretical saturation", with "roughly 100 diverse traces" as a working pool.
  - Use a "benevolent dictator" single annotator for small and medium companies. With multiple annotators, measure Cohen's kappa and hold alignment sessions to refine the rubric.
  - Prefer binary labels over Likert.
  - Split judge labels into train (few-shot examples in the prompt), dev (iterate) and test (held out), and report TPR/TNR on test.
  - Sampling strategies: random, clustering, classifier-flagged and negative-feedback-driven. "Keep some random traces in every batch" so you can find failure modes your current signals do not describe.
  
  — [hamel.dev: Evals FAQ](https://hamel.dev/blog/posts/evals-faq/)
- **[VENDOR-GUIDE]** Anthropic, "Demystifying evals for AI agents":
  - Source tasks from your bug tracker, your support queue and your pre-release manual checks. "20-50 simple tasks drawn from real failures is a great start."
  - LLM-as-judge graders "should be closely calibrated with human experts".
  - "You won't know if your graders are working well unless you read the transcripts and grades from many trials."
  - Capability evals start at a low pass rate. Regression evals should hold near 100%. Saturated capability evals "graduate" into regression suites.
  
  — [Anthropic Engineering](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- **[PRACTITIONER]** Eugene Yan, "Patterns for building LLM-based systems", says to collect user feedback:
  - Explicit feedback is a deliberate response (thumbs up/down, regenerate). Implicit feedback comes from behavior: Midjourney's variation, upscale and download actions, and Copilot's accept, edit or ignore.
  - Feedback feeds eval datasets, fine-tuning and guardrails. Data is "one of the few moats for LLM products".
  - Embed feedback collection in the normal UX rather than a separate workflow.
  
  — [eugeneyan.com: LLM patterns](https://eugeneyan.com/writing/llm-patterns/)

### Inferences
- For a review card, the reviewer's correction plus reason (a free-text critique and a categorical failure code) is worth more than the approve/reject bit. It feeds open and axial coding and becomes the train/dev/test labels for a judge.
- Edits are a strong implicit signal (like Copilot's "accepted with edits"). The size of the edit (a diff between the agent's output and the corrected output) can be logged as its own metric.
- Overfitting to corrections: use corrections as few-shot examples only from the train split, and measure improvement on a held-out test split. A correction-driven prompt change counts as a win only if regression evals stay near 100%.
- Criteria drift means the rubric shown on the card should be versioned. Labels made under rubric v1 may not be comparable with labels made under v2.

### Gaps
- I could not locate a Chip Huyen source on feedback design. Her "Building A Generative AI Platform" post has no feedback section, and the relevant material appears to be in her book *AI Engineering* (2025), which I did not access.
- I did not fetch OpenAI's eval guidance, so OpenAI's view is not represented.
- The EvalGen user-study size (number of participants) was not in the abstract I fetched.

## Q4. Active learning / selective labeling and the selection-bias problem (random audits of auto-approved items)

### Takeaway
If humans only label what the system flags, the labeled data is not a random sample. You cannot estimate the error rate of auto-approved items from it, and models or judges trained on it inherit the bias. This is the "selective labels problem". The standard fix is to send a small random sample of auto-approved or high-confidence items to review. In LangSmith, that is an automation rule with a sampling rate.

### Cited Findings
- **[PAPER]** Lakkaraju, Kleinberg, Leskovec, Ludwig and Mullainathan (KDD 2017), "The Selective Labels Problem": when observed outcomes depend on prior human decisions (for example, outcomes are observed only for defendants released on bail), "the instances for which outcomes are observed do not represent a random sample of the population". The paper proposes "contraction" to compare models with humans without counterfactual inference. — [KDD'17 paper PDF](https://www.cs.cornell.edu/home/kleinber/kdd17-selective.pdf); [PMC version](https://pmc.ncbi.nlm.nih.gov/articles/PMC5958915)
- **[PRACTITIONER]** "Keep some random traces in every batch". Random sampling is the hedge against failure modes your current flags do not describe. Classifier-flagged and negative-feedback sampling are complementary to it, not substitutes. — [hamel.dev: Evals FAQ](https://hamel.dev/blog/posts/evals-faq/)
- **[OFFICIAL]** LangSmith automation rules have a configurable sampling rate (0–100%) and can filter on feedback or evaluator scores, then add the matching runs to an annotation queue. — [LangSmith docs: Automation rules](https://docs.langchain.com/langsmith/rules)
- **[WEAK]** Practitioner writing describes randomly routing a small percentage of AI-approved, high-confidence items to humans to catch confident-but-wrong cases and detect drift. — [search result summary; e.g. dev.to automation bias post](https://dev.to/brennhill/automation-bias-why-people-rubber-stamp-ai-and-how-to-fix-it-2587)

### Inferences
- A concrete design: rule A sends 100% of flagged items to the review queue. Rule B sends a fixed small percentage of auto-approved items to a separate "audit" queue, or to the same queue with an `audit=true` tag. The auto-approved error rate is estimated from rule B only, with a binomial confidence interval. Detecting rare errors needs enough audits; for example, to bound an error rate near 1% you need hundreds of audited items. This is standard statistics, not taken from a source.
- Keep audit samples tagged so they are never mixed into the "flagged" population when error rates are computed. The flagged population will always show a higher error rate by construction.
- Active learning, which picks the most informative (uncertain) items, improves label efficiency but makes the selection bias worse for measurement. Keep uncertainty-sampled labels for improvement and random labels for measurement.

### Gaps
- I did not fetch a dedicated active-learning survey (for example, Settles), and found no official Google or Microsoft production-ML guide with explicit audit-sampling rates. I found no sourced recommended audit percentage.

## Q5. Measuring reviewer agreement and quality; reviewer anchoring on the agent's answer

### Takeaway
Use a single expert "benevolent dictator" where you can. With multiple reviewers, measure Cohen's kappa on overlapping items, which LangSmith's reviewers-per-run setting supports, and hold calibration sessions. Showing reviewers the agent's answer creates a real automation-bias risk of omission and commission errors. Seeded known-bad items (gold questions) and blind or independent labeling are the usual countermeasures.

### Cited Findings
- **[PRACTITIONER]** With multiple annotators, measure agreement with metrics like Cohen's kappa and hold alignment sessions to resolve disagreements and refine rubrics. Otherwise, use a single domain expert as "benevolent dictator". — [hamel.dev: Evals FAQ](https://hamel.dev/blog/posts/evals-faq/)
- **[OFFICIAL]** LangSmith queues can require N reviewers per run and assigned reviewers, which produces overlapping labels for agreement analysis. — [LangSmith docs: Annotation queues](https://docs.langchain.com/langsmith/annotation-queues)
- **[PAPER]** Skitka, Mosier and Burdick (1999), "Does automation bias decision-making?" (*Int. J. Human-Computer Studies*), studied 80 students on a simulated flight task. Participants with an automated aid made omission errors (missing events the aid didn't flag) and commission errors (following the aid against other 100%-valid indicators). The non-automated group outperformed the group with a "very but not perfectly reliable" aid. The authors define automation bias as using automated cues "as a heuristic replacement for vigilant information seeking". — [NASA NTRS record](https://ntrs.nasa.gov/citations/19980048379) (I found the record via a search summary; I did not verify the exact NTRS-to-paper mapping); [Aviation Today summary](https://aviationtoday.com/2000/03/01/safety-in-avionics-blindly-following-the-computer)
- **[WEAK]** Secondary claims that people with an automated aid missed 41% of critical events the aid failed to flag (control group: 3%), and that human intervention success was only 9–26% across oversight strategies. — [dev.to automation bias post](https://dev.to/brennhill/automation-bias-why-people-rubber-stamp-ai-and-how-to-fix-it-2587). These numbers are unverified against the primary papers; do not quote them without checking.
- **[PRACTITIONER]** Reviewing LLM reasoning helped domain experts standardize their own criteria. — [hamel.dev: Using LLM-as-a-Judge](https://hamel.dev/blog/posts/llm-judge/)

### Inferences
- Countermeasures against anchoring, derived from the automation-bias findings rather than prescribed by LangChain docs:
  - Show evidence before the agent's verdict, or have the reviewer commit to a judgment first on a random subset.
  - Seed known-bad items (gold questions) into the queue to measure each reviewer's catch rate.
  - Track approval rate and time spent per reviewer. A near-100% approval rate with low dwell time suggests rubber-stamping.
- Treat a correction as a label with noise. Where two reviewers disagree, send the item to adjudication rather than straight into the regression dataset.

### Gaps
- I found no official LangSmith feature for gold-question seeding or per-reviewer accuracy reporting.
- I found no primary study quantifying anchoring specifically for LLM-output review (as opposed to aviation or decision-support automation bias).
