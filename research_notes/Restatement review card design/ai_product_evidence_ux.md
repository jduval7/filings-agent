# How AI products for financial documents present evidence for human review

Scope note: about 20 search/fetch calls. Several primary help pages returned 404/403/SSL errors (Daloopa docs, Extend docs, Instabase docs/blog, MindBridge support), so some findings rest on search-result snippets of those pages. Those are marked "[snippet]". Labels: **[VENDOR]** = vendor marketing or docs; **[INDEPENDENT]** = third-party review, aggregator or academic work. Aggregator sites (rfp.wiki, tooldirectory, financialfilings.com) are weak independent evidence. They often paraphrase vendor copy.

## Q1. Research and analyst tools (Hebbia, AlphaSense, Daloopa, Bloomberg, Fiscal.ai, Rogo, etc.): what does a user see to verify an answer?

### Takeaway
The pattern every major product shares is click-to-source: each claim or cell links to the exact passage, which opens highlighted in the source document. Hebbia also shows the step-by-step reasoning per cell. None of the research tools I found publicly shows numeric confidence scores, competing interpretations, or an old-vs-new value diff. Daloopa is the closest to "why did this historical number change": it stores as-reported values by default and handles restated values separately, in a beta.

### Cited Findings
- **Hebbia Matrix [VENDOR, via aggregator]:** results appear in a spreadsheet-style grid, with documents as rows and questions as columns. "Users can explore any cell to see the AI's step-by-step reasoning chain and the precise source passage." — [vantaige.io](https://vantaige.io/fr/ai-tool/hebbia)
- **Hebbia [VENDOR, via aggregator]:** "sentence-level citations, linking every output to the precise source it came from." It also "displays each intermediate reasoning step so analysts can verify outputs rather than treating the AI as a black box." — [rfp.wiki](https://www.rfp.wiki/vendors/hebbia); [neurons-lab](https://neurons-lab.com/articles/hebbia-vs-rogo/)
- **Hebbia [VENDOR, Anthropic customer article]:** "each answer lands in its own cell on a grid" for "full transparency, traceability, and steerability." Work is decomposed into steps because "firms want control over which documents feed the analysis and how each step is built." The article does not mention confidence scores or a human review workflow. — [claude.com article](https://claude.com/resources/articles/working-at-the-frontier-how-hebbia-builds-ai-for-financial-diligence-that-cant-miss-a-detail)
- **Hebbia [INDEPENDENT, weak]:** G2 reviewers praise "compressing multi-day due diligence into hours with verifiable citations." Cons include a learning curve and difficulty staying organized across many files. There are only about 11 G2 ratings, so statistical confidence is low. — [rfp.wiki G2 summary](https://www.rfp.wiki/artificial-intelligence/ai-data-agents/hebbia/v7-go)
- **AlphaSense Generative Search [VENDOR help center]:** "Select any summary to open the source document, where the relevant passage is highlighted." Users can "highlight any text in the response to quickly ask a follow-up question or verify a specific claim." Citations survive copy and PDF export. A visible "Research Plan" lists the steps taken. When no relevant content exists, "Gen Search will say so, instead of fabricating a response." The page mentions no confidence scores and no handling of conflicting sources. — [AlphaSense help](https://help.alpha-sense.com/hc/en-us/articles/41666587181203-Interacting-with-Generative-Search)
- **AlphaSense [VENDOR]:** "citations to the exact snippet of text from where the information was sourced." — [AlphaSense buyers guide](https://www.alpha-sense.com/blog/product/generative-ai-tools-for-market-research-buyers-guide/)
- **Daloopa [VENDOR] [snippet]:** "Click-Through Audit." Clicking any data point in Excel takes the user to "the exact location that a particular data point was obtained from." All data points are hyperlinked to the source filing, with a claimed accuracy above 99%. The fetched doc page returned 404. — [Daloopa docs](https://docs.daloopa.com/docs/one-click-audit); [tooldirectory.ai](https://tooldirectory.ai/tools/daloopa)
- **Daloopa Stitcher [VENDOR]:** pulls "the source snippet for every instance of a tagged datapoint." The user sees the context of the same line item across all periods. — [Daloopa video](https://www.daloopa.com/videos/daloopas-stitcher-helps-contextualize-faster)
- **Daloopa restatements [VENDOR] [snippet]:** Daloopa extracts data "from the point it is first disclosed, not restated numbers," meaning as-reported. Restated values are covered "for a growing set of companies," and a private beta offers a restated version of its models. Daloopa also has a separate "resegmentations" doc. — [Daloopa data sheets](https://docs.daloopa.com/docs/data-sheets); [Daloopa resegmentations](https://docs.daloopa.com/docs/resegmentations); [Daloopa FAQ](https://docs.daloopa.com/docs/faqs)
- **Bloomberg AI earnings call summaries (Jan 2024) [VENDOR press]:** users click points in the summary sidebar to jump to the relevant transcript section. "From the summary, you can click through to the source transcript to see exactly what was said, then you can click through to the earnings call itself to listen to the tone." "Context links" go to Company Financials, Supply Chain and other Terminal functions. — [Bloomberg press](https://www.bloomberg.com/company/press/bloomberg-launches-ai-powered-earnings-call-summaries); [fintechnews.sg](https://fintechnews.sg/84034/ai/bloomberg-introduces-ai-powered-earnings-call-summaries-for-enhanced-financial-analysis/)
- **Fiscal.ai [aggregator]:** "click-thru auditability from fundamental to source table" and source links back to filings. — [financialfilings.com](https://financialfilings.com/alternatives/fiscal-ai/)
- **Industry norm [aggregator, weak]:** research briefs that cite company data without the source filing date and accession number "are not auditable." — [allmind.ai](https://allmind.ai/research/ai-sec-filing-analysis)

### Inferences
- Linking to the exact passage or table and highlighting it is now standard, not a differentiator. A restatement review card should at least deep-link both the original filing's figure and the later filing's figure, each highlighted, with filing date and accession number.
- Research tools build trust through provenance (citations, a visible plan or reasoning chain), not through calibrated confidence. Showing no confidence score at all is the industry norm on the research side.
- Daloopa keeping as-reported and restated values separate is a direct precedent for a card that shows the as-reported value, the restated value, and the source for each.
- Bloomberg's chain from summary to transcript to audio suggests layered drill-down: a short claim, then the passage, then the full raw source.

### Gaps
- Rogo, Brightwave, Toggle, Tegus, Canalyst/Visible Alpha, S&P Kensho Extract/Link, V7 Go, and Bloomberg Document Insights / BloombergGPT features: no verification-UX details found in the time budget. Rogo results covered funding only.
- I found no public evidence that any of these research tools shows alternative interpretations, "why not X" reasoning, or an explicit diff between old and new values of a restated figure.
- No vendor publishes how they measure their citation precision. Daloopa's ">99%" is unaudited by any third party I found.

## Q2. Credit spreading and statement-extraction tools: how do they show low-confidence extractions and route them to analysts?

### Takeaway
Document-automation and spreading tools use per-field confidence scores with thresholds. Low-confidence fields are color-flagged (yellow is common) and sent to exception or review queues, while high-confidence items pass straight through. Validation rules, meaning business-logic checks, are used alongside model confidence. I found little public UI detail from the bank-spreading vendors (Moody's, nCino, etc.).

### Cited Findings
- **Generic extraction pattern [VENDOR] [snippet; Extend docs page 404'd]:** confidence is a value from 0 to 1. "Low confidence values are highlighted yellow." Low-confidence data points "might need manual verification." — [Extend docs](https://docs.extend.ai/product/review-experience/confidence-scores); [Heron glossary](https://www.herondata.io/glossary/confidence-score)
- **Routing [VENDOR] [snippet]:** high-confidence submissions "move forward automatically, while edge cases are routed into exception queues for human review." — [Heron glossary](https://www.herondata.io/glossary/confidence-score)
- **Instabase [VENDOR] [snippet]:** confidence scores are "the primary trigger for human review." The Human Review Suite defines review queues with escalation paths by document type. Admins add a "Confidence rule > Field confidence" threshold in a Validations tab. Reviewer efficiency features include multi-monitor setups, keyboard shortcuts and auto-suggestions. — [Instabase blog](https://instabase.com/blog/overcoming-the-limitations-of-llms-data-validation-and-human-review); [Instabase docs](https://docs.instabase.com/automate/validating-documents)
- **Ocrolus [aggregator]:** combines automated extraction with manual review "to validate fields and resolve low-confidence results," with exception handling and review queues. — [fitgap](https://us.fitgap.com/products/007555/ocrolus)
- **Moody's QUIQspread [VENDOR / awards press]:** an AI/ML spreading tool trained on the financials of more than 22 million private firms. It is described as making spreading "more efficient, accurate, consistent, and traceable." I found no detail on its UI for low confidence. — [BusinessWire](https://www.businesswire.com/news/home/20201207005633/en); [Moody's spreading page](https://www.moodys.com/web/en/us/solutions/lending/loan-origination/spreading-scoring.html)
- **Spreading for monitoring [VENDOR]:** periodic re-spreads give "automatic trend lines, and early warnings that fire when the trend turns." — [botminds.ai](https://botminds.ai/solutions/financial-spreading-performance-tracking)

### Inferences
- Routing by confidence threshold is the established pattern for extraction. Applied to restatement explanations, it suggests auto-accepting only when deterministic checks pass (for example, the restated figure ties to the new filing and the delta ties to a disclosed reclassification). Everything else goes to a queue with the failing check named.
- Rule failures are more actionable than a bare probability, because they tell the reviewer what to look at.

### Gaps
- nCino spreading, Evolve/FlashSpread, Abrigo, S&P, and Moody's CreditLens: I found no public screenshots or docs describing their exception UI.
- I found no evidence on whether LLM-derived confidence scores in these tools are calibrated.

## Q3. Audit and accounting AI tools: how do they present flagged anomalies and evidence, and how do they limit alert fatigue?

### Takeaway
MindBridge is the clearest example: every transaction gets a risk score, and the reviewer sees which named "control points" (rules, statistical and ML tests) fired. Reviewers can re-weight or remove control points. That gives each flag a stated reason and a tunable threshold, which is the main lever against alert fatigue. DataSnipper's pattern is a persistent two-way link between a workpaper cell and the highlighted region of the source document.

### Cited Findings
- **MindBridge [VENDOR]:** it scores every transaction with an "Ensemble AI" of rules-based tests, statistical methods and ML. About 28 control points carry individual weights. It "clearly communicates which control points have triggered for each transaction, making it easy to understand why a transaction has been flagged as high risk." Users can add or remove control points per assertion and adjust their weights. — [MindBridge whitepaper](https://info.mindbridge.ai/hubfs/docs-website/Whitepaper/MindBridgeAnalytics-AI-Audit-Methodology-Whitepaper.pdf); [MindBridge news](https://www.mindbridge.ai/news/mindbridge-delivers-another-major-update-to-its-leading-audit-and-financial-risk-analytics-platform/)
- **MindBridge [VENDOR]:** it has enhanced "the clarity and explainability of its proprietary financial Expert Score control point." The Outlier Anomaly control point flags "an unusual combination of date, amount, nominal account, and monetary flow." — [MindBridge news](https://www.mindbridge.ai/news/mindbridge-delivers-another-major-update-to-its-leading-audit-and-financial-risk-analytics-platform/); [MindBridge Aug 2026 release notes](https://support.mindbridge.ai/hc/en-us/articles/42089153896983-August-2026-Release-Notes)
- **DataSnipper [VENDOR]:** a "Snip extracts selected data from your source document, and then automatically creates a reference between the two." Document Matching finds the matching text, date or number in a source document and links it to sample data. Each extraction keeps a reference so the audit trail is "easy to review without the need for manual cross-referencing." — [DataSnipper test of control](https://www.datasnipper.com/external-audit/test-of-control); [DataSnipper ADKL case](https://www.datasnipper.com/resources/adkl-germany)

### Inferences
- For a review card, MindBridge suggests a pattern: list the named checks that fired or passed (for example "segment total ties," "8-K Item 4.02 present," "note mentions reclassification"), not just one score, and let the reviewer see each reason.
- DataSnipper suggests that every number on the card should be a "snip" linked to its highlighted location, both the old filing cell and the new filing cell.

### Gaps
- KPMG Clara, EY Helix, Deloitte Argus/Omnia, PwC Halo, Trullion and Fieldguide: not researched in budget. Big Four tools are mostly internal, with little public UI documentation.
- I found no published false-positive rates or alert-fatigue metrics from MindBridge or other vendors.

## Q4. Published user research and independent evidence on what helps reviewers verify, trust and click to source

### Takeaway
Academic evidence is a caution. Citation-style interfaces look trustworthy, but in one study only about half of generated sentences were fully supported and about a quarter of citations did not support their sentence. Explanations raise acceptance of AI answers whether or not the AI is right. Explanations reduce overreliance only when they make verification cheaper than doing the task yourself. I found no independent user studies of the finance vendors' review UIs.

### Cited Findings
- **Liu et al. 2023, "Evaluating Verifiability in Generative Search Engines" [INDEPENDENT, Findings of EMNLP 2023]:** this audited Bing Chat, NeevaAI, Perplexity and YouChat. Only 51.5% of generated sentences were fully supported by citations (citation recall), and only 74.5% of citations supported their sentence (citation precision). The authors warn of a "facade of trustworthiness." — [arXiv 2304.09848](https://arxiv.org/abs/2304.09848v2); [Stanford HAI](https://hai.stanford.edu/news/generative-search-engines-beware-facade-trustworthiness)
- **Bansal et al., CHI 2021 [INDEPENDENT]:** "Explanations increase the chance that humans will accept the AI's recommendation regardless of whether the AI is correct." Complementary team performance was not improved by state-of-the-art explanations compared with simply showing the AI's confidence. — [arXiv 2006.14779](https://arxiv.org/abs/2006.14779v1); [UW IDL](https://idl.uw.edu/papers/ai-explanations-team-performance)
- **Vasconcelos et al. 2023 [INDEPENDENT]:** overreliance does not generally fall when explanations are added. It does fall when explanations are simpler than the task, making verification cheap, and when the stakes for a correct answer are higher. Null effects in earlier work may come from explanations that do not "sufficiently reduc[e] the costs of verifying the AI's prediction." — [arXiv 2212.06823](https://arxiv.org/abs/2212.06823); [Stanford HAI](https://hai.stanford.edu/news/ai-overreliance-problem-are-explanations-solution)
- **Practitioner reviews [INDEPENDENT, weak]:** G2 reviewers value Hebbia's "verifiable citations" as the reason it saves time. There are few reviews (about 11). — [rfp.wiki](https://www.rfp.wiki/vendors/hebbia)

### Inferences
- A citation link alone does not prove anything. The card should make the reviewer check the claim against the highlighted source, for example by showing the old and new numbers side by side with the passage, rather than offering a prose explanation they can rubber-stamp.
- Following Vasconcelos, the explanation must be quicker to verify than re-deriving it. Show the two numbers, the delta, and the one disclosure sentence that accounts for it.
- Following Bansal, a plain confidence or check status may help as much as a long rationale. Fluent prose rationales may increase wrong approvals.
- The citation precision/recall figures from Liu et al. give a measurable target: each claim on the card should be checked automatically, by entailment or number matching, against its cited span before a human sees it.

### Gaps
- I found no independent user studies, usability tests or third-party evaluations of review UIs from Hebbia, AlphaSense, Daloopa, Bloomberg or the spreading tools. Evidence on those products is vendor or aggregator only.
- I found no finance-specific data on how often analysts actually click through to the source.

## Q5. Does any product show alternative explanations or "why not X" reasoning?

### Takeaway
I found no evidence that any of the surveyed financial-document AI products shows competing interpretations or "why not X" reasoning to the reviewer. The closest patterns are AlphaSense's visible research plan, Hebbia's per-cell reasoning chain, AlphaSense's explicit "no answer" behavior, and MindBridge's list of which tests fired.

### Cited Findings
- AlphaSense shows a Research Plan and declines to answer when content is missing. Its help page does not describe handling of conflicting sources. — [AlphaSense help](https://help.alpha-sense.com/hc/en-us/articles/41666587181203-Interacting-with-Generative-Search)
- Hebbia exposes a step-by-step reasoning chain per cell. — [vantaige.io](https://vantaige.io/fr/ai-tool/hebbia)
- MindBridge exposes the triggered control points per flagged item. — [MindBridge whitepaper](https://info.mindbridge.ai/hubfs/docs-website/Whitepaper/MindBridgeAnalytics-AI-Audit-Methodology-Whitepaper.pdf)

### Inferences
- A restatement card that lists candidate causes, such as error correction (Item 4.02 / "Big R"), revision ("little r"), reclassification, resegmentation, discontinued operations, or an accounting-standard adoption, would be differentiated. It would mark each cause as supported or ruled out, with its evidence or the absence of it. This is my design inference, not an observed market pattern.
- Draft design patterns for the review card, synthesized from the above:
  1. Side-by-side old and new values with the computed delta, each deep-linked to a highlighted cell in its filing (Daloopa, DataSnipper).
  2. One short cited explanation, with each sentence linked to a passage (Hebbia, AlphaSense), checked automatically for citation support before display (Liu et al.).
  3. Named checks that passed or failed, instead of a single opaque score (MindBridge, Instabase validation rules).
  4. Route by deterministic checks: auto-pass clean items and queue exceptions with the failing reason (Instabase, Heron).
  5. An explicit "insufficient evidence" state, never a guessed explanation (AlphaSense).
  6. Optional drill-down layers: claim, then passage, then full filing (Bloomberg).
  7. Alternative causes considered and ruled out, which is a gap in the market.

### Gaps
- I could not confirm whether any product (including ones not reviewed, such as Brightwave, Rogo or Kensho) shows alternative hypotheses. Absence of evidence here reflects limited search depth, not proof that none exists.
