# Human-in-the-loop review of AI recommendations: automation bias, anchoring, reviewer fatigue, and evidence designs that counter them

Scope note: research notes for the "restatement review card", where a human reviewer approves an LLM's explanation of why a financial figure changed, and the LLM is sometimes confidently, plausibly and quote-backed wrong. Every claim below has an inline source. Items I could not verify within the research budget are listed under Gaps rather than stated as fact.

## 1. Automation bias and complacency: what the core findings are (classic and LLM era)

### Takeaway
Automation bias is robust. It shows up in experts as well as novices, in teams as well as individuals, and training or instructions alone do not remove it. Plain explanations do not reliably reduce it and often increase acceptance whether the AI is right or wrong. What helps is lowering the cost of verifying the AI (evidence that is easy to check), making the reviewer accountable, expressing uncertainty in the first person, and forcing engagement (at a cost in user satisfaction).

### Cited Findings
**Classic human factors / medicine**
- Skitka, Mosier & Burdick (1999), simulated flight task: participants without an automated aid did better on monitoring than those with a highly (but not perfectly) reliable aid. Aided participants made omission errors (missed events the aid did not flag) and commission errors (followed the aid even when it contradicted their training and other 100%-valid indicators that were available). — [Wikipedia summary of Skitka et al. 1999, Int. J. Human-Computer Studies](https://en.wikipedia.org/wiki/Automation_bias)
- Participants who saw themselves as accountable for how they used the automation were significantly more likely to verify it and made significantly fewer automation-related errors (Skitka/Mosier accountability work). — [Wikipedia: Automation bias](https://en.wikipedia.org/wiki/Automation_bias)
- Parasuraman & Manzey (2010, *Human Factors* 52(3):381-410), integrative review: complacency and automation bias are overlapping phenomena with attention at the center. Complacency appears under multi-task load, in both naive and expert operators, and "cannot be overcome with simple practice". Automation bias produces omission and commission errors with imperfect aids, affects individuals and teams, and "cannot be prevented by training or instructions". — [TRID abstract](https://trid.trb.org/View/1085175); [SAGE DOI](https://journals.sagepub.com/doi/10.1177/0018720810376055)
- Goddard, Roudsari & Wyatt (2012, JAMIA), systematic review of 74 studies across healthcare, aviation, military and HCI: — [PMC3240751](https://pmc.ncbi.nlm.nih.gov/articles/PMC3240751)
  - Meta-analysis of 4 healthcare studies: erroneous advice was 26% more likely to be followed when a decision support system (CDSS) was present (RR 1.26, 95% CI 1.11-1.44).
  - "Negative consultations", where a correct decision made before the advice was switched to an incorrect one after seeing it: 6-11% across four studies.
  - Omission rates in aviation studies varied from 55% to 0%, so results are highly heterogeneous.
  - Mediators: trust in the system increases reliance and self-confidence decreases it ("trust is possibly the strongest driving factor"). Inexperience, high workload, time pressure and task complexity all increase reliance. The review reports a reliability threshold of about 70% below which aid use hurts performance.
  - Mitigators: training, greater user accountability, showing the system's confidence alongside each piece of advice ("improved the appropriateness of user reliance"), making the advice less prominent on screen, presenting it as information rather than a command (under time pressure), and adaptive task allocation.
  - Note the tension: Goddard lists training as a mitigator, while Parasuraman & Manzey say training or instructions cannot prevent the bias. A reasonable reading is that training lowers the bias but does not remove it.

**LLM / XAI era**
- Bansal et al. (CHI 2021), "Does the Whole Exceed its Parts?": with an AI about as accurate as humans, AI assistance gave some complementary gains, but explanations did not add to them. Explanations "increased the chance that humans will accept the AI's recommendation, regardless of its correctness." — [arXiv 2006.14779](https://arxiv.org/abs/2006.14779)
- Vaccaro, Almaatouq & Malone (2024, *Nature Human Behaviour*), pre-registered meta-analysis of 106 experiments and 370 effect sizes (2020-2023): — [PMC11659167](https://pmc.ncbi.nlm.nih.gov/articles/PMC11659167/)
  - Human+AI was worse than the best of human or AI alone (g = -0.23, 95% CI -0.39 to -0.07) but better than humans alone (g = 0.64).
  - Decision tasks showed losses (g = -0.27, p = .002). Creation tasks showed non-significant gains (g = 0.19).
  - When the AI alone beat humans alone, combining them produced losses.
  - **Neither AI explanations nor AI confidence displays significantly moderated performance.**
- Buçinca, Malaya & Gajos (CSCW 2021), "To Trust or to Think", N = 199: three cognitive-forcing designs significantly reduced overreliance compared with simple XAI designs. However, participants gave the least favorable ratings to the designs that reduced overreliance most, and the benefit was larger for people high in Need for Cognition. Explanations alone did not reduce overreliance. — [Harvard page](https://www.eecs.harvard.edu/~kgajos/papers/2021/bucinca2021trust.shtml); [arXiv 2102.09692](https://arxiv.org/abs/2102.09692)
- Vasconcelos et al. (CSCW 2023), "Explanations Can Reduce Overreliance on AI Systems During Decision-Making", 5 studies, N = 731, maze task:
  - People strategically choose whether to engage with an explanation, using a cost-benefit calculus.
  - Explanations reduced overreliance when the task was harder, when the explanation was easier to verify, and when there was monetary reward for being correct.
  - The authors argue that earlier null results may come from explanations that "[do not] sufficiently reduc[e] the costs of verifying the AI's prediction."
  - [arXiv 2212.06823](https://arxiv.org/abs/2212.06823)
- Kim, Liao, Vorvoreanu, Ballard & Wortman Vaughan (FAccT 2024), pre-registered, N = 404, medical Q&A with a fictional LLM search engine: first-person uncertainty ("I'm not sure, but...") lowered agreement with the system and raised accuracy, through reduced (but not eliminated) overreliance on incorrect answers. — [arXiv 2405.00623](https://www.arxiv.org/abs/2405.00623)
- Si et al. (NAACL 2024), "LLMs Help Humans Verify Truthfulness, Except When They Are Convincingly Wrong", 80 crowdworkers doing fact-checking:
  - Readers of LLM explanations were more efficient than search-engine users with similar accuracy, but they over-relied when the explanation was wrong.
  - Contrastive explanations (why the claim may be true and why it may be false) reduced overreliance but did not significantly beat search engines.
  - Showing LLM explanations alongside retrieved passages gave no complementary benefit over the passages alone.
  - [ACL Anthology](https://aclanthology.org/2024.naacl-long.81); [arXiv 2310.12558](https://arxiv.org/abs/2310.12558v2)
- Passi & Vorvoreanu (Microsoft, MSR-TR-2022-12): a review of about 60 papers defining overreliance as accepting incorrect AI outputs, covering antecedents, mechanisms, measurement and mitigations. A 2024 follow-up synthesizes appropriate reliance on generative AI (MSR-TR-2024-7). — [Microsoft Research](https://www.microsoft.com/en-us/research/?p=854967); [GenAI follow-up](https://www.microsoft.com/en-us/research/?p=1133469)
- Schemmer et al. (IUI 2023, N = 200) propose "Appropriateness of Reliance" (AoR), a two-dimensional measure that separates relying on correct advice from rejecting incorrect advice, and test the effect of explanations on it. — [arXiv 2302.02187](https://arxiv.org/abs/2302.02187v3)

### Inferences
- Robust (multiple reviews or meta-analyses): automation bias exists in experts, training does not cure it, and explanations as such do not improve team accuracy. Bansal, Buçinca and the Vaccaro meta-analysis agree on this.
- The reviewer's situation is close to Si et al.'s "convincingly wrong" condition. A fluent, quote-backed causal story is exactly the kind of explanation that raises acceptance regardless of correctness. The quotes help only if checking them is cheap and the check actually tests the causal claim, not just whether the quote exists. A quote can be real while the cause attributed to it is wrong.
- Vasconcelos' cost-of-verification framing is the most useful design lever. The review card should make the decisive check (does this source passage actually state the cause, and does the arithmetic reconcile?) a one-glance operation. Otherwise reviewers will rationally skip it.
- Approving a cause from a fixed set of candidates is a "decision task", the category where the meta-analysis found human+AI losses. Expect rubber-stamping unless the design counteracts it.
- Measure with an AoR-style split: (a) the rate of correct overrides on seeded wrong explanations and (b) the rate of wrongly rejecting correct explanations. Approval rate alone is not enough.

### Gaps
- Effect sizes for the individual cognitive-forcing designs in Buçinca 2021 (named in the paper as on-demand, update and wait designs, but the abstract does not give the numbers) were not retrieved.
- I did not retrieve the Skitka 1999 paper directly. Findings come from a secondary summary, and the exact error-rate percentages are not stated here.

## 2. Anchoring: AI-first vs evidence-first vs human-first

### Takeaway
Having the reviewer commit to a provisional judgment before seeing the AI's answer reduces alignment with the AI. That cuts both ways: it lowers agreement with wrong AI answers and with right ones. In one expert study it did not add time. Evidence that it improves overall accuracy is mixed or null.

### Cited Findings
- Fogliato et al. (FAccT 2022), "Who Goes First?", 19 veterinary radiologists:
  - Those who registered provisional answers before seeing the AI agreed with it less, whether the AI was correct or not.
  - They also rated the AI as less useful and were less willing to seek a colleague's opinion when they disagreed with it.
  - Requiring provisional decisions "did not lengthen the time participants spent on the task."
  - [arXiv 2205.09696](https://arxiv.org/abs/2205.09696)
- A secondary description of the literature says Fogliato et al. found no significant accuracy difference between the AI-first and human-first workflows. It also says human-first reduced over-reliance on wrong AI recommendations, and calls anchoring more frequent in AI-first designs. — [arXiv 2402.01292, "From Evidence to Decision: Exploring Evaluative AI"](https://arxiv.org/pdf/2402.01292) (secondary; not verified against Fogliato's own full text)
- Buçinca et al.'s cognitive-forcing conditions also included asking people to decide before seeing the AI. This reduced overreliance but was disliked. — [arXiv 2102.09692](https://arxiv.org/abs/2102.09692)
- Goddard et al.: lowering the prominence of the advice on screen reduced automation bias. — [PMC3240751](https://pmc.ncbi.nlm.nih.gov/articles/PMC3240751)
- The "Evaluative AI" proposal (Miller and coauthors, arXiv 2402.01292) argues for decision support that shows evidence for and against each hypothesis instead of a recommendation. This is a position/design paper, not a controlled outcome trial. — [arXiv 2402.01292](https://arxiv.org/pdf/2402.01292)

### Inferences
- A pure "human-first" workflow (reviewer independently works out the cause before seeing the LLM's answer) is likely too costly for financial-restatement review at volume. Evidence also says it lowers agreement with correct AI answers too.
- A cheaper middle ground the evidence supports: show the numbers and the source evidence first, with the claimed cause collapsed or de-emphasized. Ask a lightweight question (for example "which line items drive the delta?") before revealing the AI's narrative. Use this for high-risk or sampled items only.
- Based on Fogliato's timing result, adding a provisional-judgment step does not necessarily cost time for experts. That is one small study (n = 19), though.

### Gaps
- No large controlled trial found comparing evidence-first with AI-first specifically for text/causal explanations (as opposed to classification or imaging).
- Fogliato's accuracy result was found only through a secondary source.

## 3. Contrastive / "why not" explanations, alternatives, counter-evidence, confidence displays

### Takeaway
Contrastive (both-sides) LLM explanations reduce overreliance on convincingly wrong explanations, but they do not beat reading the primary evidence. In aggregate, confidence displays are not a reliable lever (no significant effect in the 2024 meta-analysis). First-person verbal uncertainty did help in one large pre-registered study. Effects on time per decision are under-reported. Plain LLM explanations save time over search, and forcing designs likely cost time and satisfaction.

### Cited Findings
- Contrastive explanations (why true and why false) mitigated over-reliance on wrong LLM explanations but did not significantly outperform a search-engine (raw evidence) condition. Combining LLM explanation and retrieved evidence gave no complementary benefit. — [Si et al., NAACL 2024](https://aclanthology.org/2024.naacl-long.81)
- LLM explanations made users significantly more efficient than search at similar accuracy, and that efficiency included accepting wrong answers. — [Si et al.](https://arxiv.org/abs/2310.12558v2)
- First-person uncertainty phrasing raised accuracy and reduced (but did not eliminate) overreliance. — [Kim et al., FAccT 2024](https://www.arxiv.org/abs/2405.00623)
- Showing an updated confidence level with each piece of advice improved appropriateness of reliance (in studies Goddard reviewed). — [Goddard et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3240751). Contrasted by: confidence displays did not significantly moderate human+AI performance across 106 experiments. — [Vaccaro et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11659167/)
- Explanations that are easier to verify reduce overreliance. Hard-to-process ones do not. — [Vasconcelos et al.](https://arxiv.org/abs/2212.06823)
- Forcing designs: less overreliance but the lowest subjective ratings. — [Buçinca et al.](https://www.eecs.harvard.edu/~kgajos/papers/2021/bucinca2021trust.shtml)

### Inferences
- For the review card, showing the strongest competing explanation (for example "could also be FX translation / reclassification / acquisition") plus counter-evidence is supported as an overreliance reducer. Expect it to cost some speed.
- The bigger lever is putting the primary source (the filing passage plus a reconciliation of the numbers) where the reviewer verifies directly. LLM prose added on top of evidence gave no complementary gain in Si et al., so the narrative should be secondary to the evidence.
- A numeric confidence score from an LLM should not be relied on to prevent rubber-stamping. If used, prefer verbal first-person hedges tied to specific weaknesses ("the source does not state the cause explicitly"). Those have the better evidence.

### Gaps
- No quantitative time-per-decision figures were retrieved for contrastive designs.
- No study found that tests contrastive explanations specifically for financial causal attribution.

## 4. Alert fatigue and reviewer fatigue: what reduces it

### Takeaway
When most alerts are low value, overrides become the default. Clinical drug alerts are overridden in roughly half to almost all cases, and SOC analysts admit to ignoring alerts. The best quantitative evidence for a fix is severity tiering: interrupt only for the most severe items, which then get near-total compliance.

### Cited Findings
- Drug safety alerts in computerized physician order entry (CPOE) are overridden in 49%-96% of cases (17-paper review). Some overriding is justified, and the authors distinguish "appropriate" from "useful" alerts. — [van der Sijs et al. 2006, JAMIA (PMC1447540)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1447540)
- Paterno et al. (2009, JAMIA): the hospital that tiered drug-drug interaction (DDI) alerts by severity accepted 100% of the most severe alerts, against 34% at the non-tiered hospital. Moderate alerts were accepted at 29% vs 10%. Lack of tiering was associated with high override of severe alerts. Observational two-site comparison, not randomized. — [JAMIA 16(1):40-46](https://academic.oup.com/jamia/article/16/1/40/865495)
- An industry blog cites a 2024 meta-analysis putting physician override rates at about 90% (95% CI 85-95%), with only 0.5% of overrides of the highest-severity DDIs judged appropriate, and 100-200 alerts per clinician per day. — [Mindbowser blog](https://www.mindbowser.com/alert-fatigue-healthcare/). CAUTION: secondary vendor source. I could not identify or verify the underlying 2024 meta-analysis, so treat these numbers as unverified.
- SOC: in a FireEye/IDC survey, in-house analysts said 45% of alerts are false positives and 35% ignore alerts to manage overload. MSSP analysts reported 53% false positives and 44% ignoring alerts when queues are full. Vendor-sponsored survey, not peer-reviewed. — [Security Sales](https://www.securitysales.com/business/industry/fireeye-soc-report-alert-fatigue/); [Cybersecurity Dive](https://www.cybersecuritydive.com/news/security-alert-analyst-SOC-idc-fireeye/595111/)
- Goddard et al.: higher workload and time pressure increase reliance on automation. — [PMC3240751](https://pmc.ncbi.nlm.nih.gov/articles/PMC3240751)

### Inferences
- Tiering is directly transferable. Interrupt with a full review card only for changes with high materiality or high model risk. Handle low-risk explanations through a lighter pass or sampled audit. Do not present all items with equal weight.
- "Batching related items" (for example, several line items that share one root cause such as a segment reorganization) is plausible but I found no controlled evidence for it.

### Gaps
- No peer-reviewed controlled studies found for content-moderation reviewer fatigue or for batching interventions within the budget.
- The "~90% override" 2024 meta-analysis could not be verified at the primary source.

## 5. Selective prediction / deferral: routing easy items to automation, hard items to humans

### Takeaway
Routing is theoretically attractive, but humans struggle to know when to defer. When AI is better than the human on a task, human+AI tends to lose to AI alone. The key failure mode is vigilance: when targets (here, wrong explanations) are rare in what reaches the human, miss rates jump sharply (low-prevalence effect). That is a strong caution for queues where most items are pre-cleared.

### Cited Findings
- Low-prevalence effect (Wolfe, Horowitz & Kenner 2005): miss rate was 0.30 when targets appeared on 1% of trials vs 0.07 at 50% prevalence. Observers search more briefly when targets are rare. In a clinical-style setting, participants missed 30% of cancers at low prevalence vs 12% at high prevalence. The effect "is easy to produce and surprisingly hard to cure" and appears in real-world screening. — [PMC review on prevalence effects (PMC3069706)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3069706); [Cognitive Research: Principles and Implications](https://link.springer.com/10.1186/s41235-016-0042-3)
- Parasuraman & Manzey: complacency is driven by attention allocation under load, and practice does not overcome it. — [TRID](https://trid.trb.org/View/1085175)
- Vaccaro et al.: when AI alone outperformed humans, human+AI combinations showed losses. Only 3 experiments tested predetermined division of labor (g = 0.22, not significant), so the evidence on routing is thin. — [PMC11659167](https://pmc.ncbi.nlm.nih.gov/articles/PMC11659167/)
- Mozannar, Satyanarayan & Sontag: humans need an accurate mental model of when to rely on the AI. Teaching with selected exemplars improved crowdworker performance when workers drew the right lessons. — [vis.mit.edu PDF](https://vis.mit.edu/pubs/teaching-defer-exemplars.pdf)
- The "Beyond Learning to Defer" line of work notes humans over-rely on AI when it is wrong, and such systems rarely beat the human or AI alone. — [arXiv 2206.13202](https://arxiv.org/pdf/2206.13202)
- Goddard: adaptive task allocation increased vigilant behavior. — [PMC3240751](https://pmc.ncbi.nlm.nih.gov/articles/PMC3240751)

### Inferences
- If routing sends only "hard" items to reviewers, the error rate in that queue is higher, which helps against the low-prevalence effect. If instead reviewers see a stream of mostly correct pre-cleared items, expect misses well above what lab studies suggest.
- Mitigations the evidence supports: (1) seed known-wrong explanations (catch trials) to keep prevalence up and to measure the correct-override rate; (2) give per-item feedback on whether the reviewer's call was right, which helps build an accurate mental model per Mozannar; (3) route by model risk rather than at random.
- Seeded catch trials are standard practice in airport baggage screening (threat-image projection). I did not retrieve a source for that here; see Gaps.

### Gaps
- No primary source retrieved on threat-image projection or catch-trial effectiveness.
- No field study found on LLM-explanation review queues specifically.

## 6. Practical guidance: PAIR, HAX, NIST AI RMF, EU AI Act Art. 14

### Takeaway
Regulators and industry guidelines now explicitly expect designs that counter automation bias. The EU AI Act names it in the text of Article 14. HAX provides 18 validated guidelines on making capabilities, reasons and corrections clear.

### Cited Findings
- EU AI Act Art. 14(4)(b): oversight measures must enable overseers "to remain aware of the possible tendency of automatically relying or over-relying on the output" (automation bias). Art. 14(4) also requires that overseers can understand capacities and limitations, correctly interpret output, and decide to disregard, override or reverse it. — [Presencis Art. 14 text](https://cdn.presencis.com/regulations/eu-ai-act/article-14/); [SSRN analysis](https://papers.ssrn.com/abstract=5147196)
- Microsoft's Guidelines for Human-AI Interaction (Amershi et al., CHI 2019): 18 guidelines distilled from more than 20 years of HCI guidance and validated in a 4-step process. They are grouped into four phases: initially, during interaction, when wrong, and over time. — [Microsoft Research project page](https://www.microsoft.com/research/project/guidelines-for-human-ai-interaction/)

### Inferences
- If the product ever serves EU-regulated high-risk uses, Art. 14 makes "how do you counter automation bias?" a compliance question, not just a UX one. Catch-trial override rates are a natural way to show it is being met.

### Gaps
- Exact wording of individual HAX guidelines (for example the ones on making clear why the system did what it did and supporting efficient correction) was not retrieved; the project page renders them as an image. Verify against the CHI 2019 paper before quoting.
- Google PAIR People + AI Guidebook and NIST AI RMF (Govern/Map/Measure/Manage human-oversight subcategories) were not retrieved within the research budget, so no claims are made about their content.

## Synthesis for the restatement review card (inference, grounded in sections above)
1. Make the evidence the main thing, not the narrative. Show the source passage(s) and a reconciliation of the numbers (old -> new, which components moved) first. Show the LLM's causal claim as a hypothesis to check. (Si et al.; Vasconcelos; Goddard on advice prominence)
2. Make the decisive check cheap. Highlight the exact span that supposedly states the cause, and flag when the quote mentions the item but not the cause. This targets the "quote-backed but wrong" failure. (Vasconcelos)
3. Show the best competing cause and any counter-evidence (contrastive). Expect fewer wrong approvals and a modest slowdown. (Si et al.)
4. Use specific first-person hedges instead of a bare confidence number. (Kim et al. vs Vaccaro null result on confidence displays)
5. Tier by materiality and model risk. Reserve full cards and a lightweight "predict first" step for high-tier items. (Paterno; Fogliato; Buçinca)
6. Keep wrong items common enough to stay vigilant, and measure. Seed known-wrong explanations, report correct-override and wrong-rejection rates separately (AoR), and give reviewers feedback. (Wolfe prevalence effect; Schemmer; Mozannar)
7. Make reviewers accountable. Record who approved what with a required short rationale for high-tier approvals. (Skitka/Mosier; Goddard)

Robustness grading:
- **Robust** (multiple reviews or meta-analyses): automation bias in experts; training alone insufficient; explanations alone do not improve team accuracy; low-prevalence miss inflation; alert override is common.
- **Moderate** (single controlled studies, decent n): cognitive forcing (N = 199); verification-cost framing (N = 731); first-person uncertainty (N = 404); contrastive LLM explanations (N = 80).
- **Weak / small**: human-first timing (n = 19); tiering (two-site observational); SOC statistics (vendor survey).
- **Conflicts**: confidence displays help (Goddard) vs no significant effect (Vaccaro); training mitigates (Goddard) vs cannot prevent (Parasuraman & Manzey).
