# How practitioners review restated / recast historical figures in 10-Ks (US SEC filers, as of Oct 2026)

Research note scope: ~20 tool calls. Several primary sources returned HTTP 403 (Ideagen/Audit Analytics help center FAQ, PCAOB Data Points page, FMP article) or unreadable PDFs (Lyle/Siano/Yohn 2024 paper, KPMG LDTI analyst survey); where I only saw a search snippet, that is flagged. Vendor content is marked **[VENDOR]**.

## 1. Equity/credit analysts maintaining multi-year spreads/models: what they do when a new 10-K recasts prior years

### Takeaway
The public, citable evidence on analyst workflow is thin and mostly vendor-authored. What exists says retail/mid-tier models typically overwrite history with the latest (recast) figures, while institutional workflows keep as-reported values, restated values and filing-level revision history side by side. Even a structured vendor like Daloopa keeps "original" and "restated" as two separate records and picks the canonical one by a mechanical "latest capture wins" rule, not by analyst judgment.

### Cited Findings
- **[VENDOR, FMP]** "Institutional workflows explicitly track as-reported financials, restated data, and filing-level revision histories"; "Retail and mid-tier models typically overwrite historical data with the latest available figures, erasing the context of what the market knew at a specific point in time." (search snippet; page returned 403 on fetch) — [Financial Modeling Prep](https://site.financialmodelingprep.com/education/financial-analysis/restated-financials-footnotes-and-data-quality-apis-how-to-track-revisions-validate-fundamentals-and-build-trustworthy-financial-models)
- **[VENDOR, FMP]** Defines as-reported ("original or first-filed") as the value in the filing that first made it public, stamped with that filing's date. Restated ("latest") is the figure after later amendments, comparative restatements, reclassifications or M&A reallocations. (search snippet) — [FMP insights](https://site.financialmodelingprep.com/insights/enterprise/how-financial-data-revisions-and-restatements-flow-through-financial)
- **[VENDOR, Daloopa]** Daloopa keeps two records per data point when a restatement happens: an "original fundamental" ("the value as first reported for that period") and a "restated fundamental" ("the revised value for that same period"). A `restated` boolean is part of the uniqueness key. The canonical record is chosen by taking "the metadata from the maximum `fundamental_id`", so the most recent capture wins. Captures come from an "Autotagger" (automated, fast) and from "Analyst fundamentals" (manual review, arrives later). The docs do not describe how a restatement is detected or verified. — [Daloopa docs](https://docs.daloopa.com/docs/fundamental-uniqueness-restatements)
- **[VENDOR, Wall Street Prep]** Their modeling best-practices guide gives no specific guidance on which filing to take historicals from or how to handle recasts. Its only relevant advice is to "err on the side of over-commenting" and to cite sources in cell comments. — [Wall Street Prep](https://www.wallstreetprep.com/knowledge/financial-modeling-best-practices-and-conventions/)
- **[VENDOR, Moody's]** Moody's QUIQspread / CreditLens marketing describes automated mapping of uploaded statements to a standard chart of accounts. I found no public material on how prior-period restatements are handled in a spread. — [BIIA on Moody's QUIQspread](https://www.biia.com/moodys-analytics-introduces-an-automated-spreading-solution-for-financial-statements/)
- In 10-Ks, "all referenced amounts for prior periods and prior period comparisons" typically reflect restated balances throughout the filing. This means a single new 10-K silently changes every comparative number an analyst copies from it. (search snippet from an EDGAR 10-K, ARCP FY2014) — [ARCP 10-K FY2014](https://www.sec.gov/Archives/edgar/data/0001507385/000150738515000030/arcp1231201410-k.htm)

### Inferences
- The information an analyst needs to accept or adjust a recast figure maps directly onto what the vendors store: old value with its filing and date, new value with its filing and date, a restated flag, and the cause. Nobody publicly documents attaching "the note that explains it" or "which other line items moved with it". That linkage looks like the gap.
- Daloopa's "latest capture wins" rule suggests vendors handle the overwrite-vs-keep choice mechanically. The judgment calls (is it comparable, should comps be adjusted) stay with the analyst.

### Gaps
- No independent (non-vendor) source found on sell-side or buy-side analyst procedure: re-spread vs overwrite, footnoting conventions, adjusting comparables. BIWS, Macabacus and CFA Institute material on this was not found in the searches I ran.
- No source found on bank credit-spreading practice for restated prior years (CreditLens, nCino, etc. docs not public).

## 2. Point-in-time / as-reported data vendors: how restated values are recorded, flagged and explained

### Takeaway
Compustat Snapshot/PIT appends rather than overwrites, keeping original values and every later change at the data-element level. FactSet offers restated vs non-restated choices plus a PIT product. XBRL itself marks the prior value with `ScenarioPreviouslyReportedMember` and assumes the undimensioned default is the restated amount. Academic work shows Compustat's routine re-standardization materially changes history, which makes replication with non-PIT files "nearly impossible".

### Cited Findings
- **[VENDOR, S&P]** Compustat Snapshot "provides historically reported preliminary and final data while retaining original values and all succeeding changes from 1988 to present", and "identifies, manages and tracks changes at the data element level on a daily basis". Compustat PIT is described as combining originally reported and restated data in annual, quarterly and YTD periodicities, with data "known to the investing public at observation dates from December 1986 forward" for ~29,000 companies. (search snippets) — [UNC Kenan-Flagler: Compustat Snapshot](https://kenanflaglerresearchtools.web.unc.edu/research-resource/compustat-snapshot/); [Global Custodian](https://www.globalcustodian.com/?p=26067); [S&P Marketplace query library: "Returns all changes to a Compustat item over time"](https://www.marketplace.spglobal.com/en/support/query-library/query-(317))
- **[ACADEMIC]** Lyle, Siano & Yohn (July 2024), "Re-Standardized Financial Statement Data". In Snapshot, newly standardized data "are appended to the prior standardized data" rather than overwritten. Standardizations "significantly alter key financial figures such as sales and earnings", leading to material differences in research findings, and precise replication with common Compustat products is "nearly impossible". (search snippet; PDF text not extractable, so I did not verify specific magnitudes) — [Yale SOM PDF](https://som.yale.edu/sites/default/files/2024-07/Re-Standardized%20Financial%20Statement%20Data.pdf); [Columbia copy](https://business.columbia.edu/sites/default/files-efs/imce-uploads/ADP/Fall%202024/Lyle%20Siano%20Yohn%202024.pdf)
- Related academic title: "Lost in standardization: Effects of financial statement database discrepancies on inference" (Penn State) — [PSU Pure](https://pure.psu.edu/en/publications/lost-in-standardization-effects-of-financial-statement-database-d/) (title only; not read)
- **[VENDOR, FactSet via ICE/library guides]** FactSet Fundamentals offers standardized and as-reported bases, with "a choice of restated data (from 2004) and non-restated data". A ratio summary shows "Original and Restated" side by side. FactSet Fundamentals Point-in-Time "incorporates known restatements". (search snippets) — [ICE MarketQ: FactSet Fundamental Data](https://download.dataservices.theice.com/products/marketq/help/factset_fundamental_data.htm); [Uni Hamburg FactSet guide](https://www.wiso.uni-hamburg.de/bibliothek/recherche/datenbanken/unternehmensdaten/factset-fundamentals.pdf)
- **[STANDARDS BODY, XBRL US, 2011]** When statements include a "Restated" column, filers should "footnote those items that are actually restated with an XBRL footnote". Prior numbers "should be dimensionally qualified with the ScenarioPreviouslyReportedMember". `RestatementAdjustmentMember` shows the adjustment between previously reported and restated amounts. `RestatedMember` "should generally not appear… as it represents the defaulted value". "It is assumed that the default value is the restated amount if it differs from a previously reported amount for the same period." (Older guidance, issued Dec 2, 2011.) — [XBRL US guidance](https://xbrl.us/guidance/reporting-restated-values-in-the-face-of-the-financials/)
- **[VENDOR, Tradevo Data; founder-authored]** Claims "18,723 restatements across 312,751 point-in-time rows" (~6%). Schema: `original_value` + `first_filed`, `latest_value` + `restated` flag, with a 0.5% same-tag threshold to flag a meaningful restatement. — [DEV Community / Tradevo](https://dev.to/tradevodata/as-reported-vs-restated-financial-data-why-the-difference-matters-for-backtesting-1big)
- **[VENDOR, Calcbench]** Calcbench provides as-reported, line-item-level statements and as-filed detail via WRDS. I found no public Calcbench methodology doc on restatement flagging. — [WRDS Calcbench overview](https://wrds-www.wharton.upenn.edu/documents/1144/Calcbench-1.pdf) (search snippet)

### Inferences
- The industry-standard representation is the pair (first-filed value + first-filed date, latest value + latest filing), plus a flag. XBRL's `ScenarioPreviouslyReportedMember` lets a machine see old vs new in a single filing, but only when the filer tags it, and many recasts (disc-ops, splits) arrive with no previously-reported column at all.
- None of the vendor docs I found expose why a value changed (error vs disc-ops vs new standard vs split) as a structured field. Cause attribution still appears to be manual reading.

### Gaps
- No Bloomberg, LSEG/Refinitiv, or Compustat analyst-QA procedure docs found publicly (likely behind client login).
- No public Calcbench help article on restatement tracking found.

## 3. Auditors, audit committees and governance: what they look at for error corrections

### Takeaway
The governing frame is ASC 250 together with SAB 99/108 (SAB Topic 1.N). Materiality decides Big R (restate, reissue, 8-K Item 4.02, amend filings, ICFR reassessment) vs little r (revise comparatives in the next periodic filing, no 4.02) vs out-of-period adjustment. Little r has become the dominant form, and SEC staff (Munter, March 2022) has warned that materiality analyses are biased toward "not material".

### Cited Findings
- **Definitions (PwC Viewpoint FSP 30.7):** Big R is when prior statements are materially misstated; they are restated and columns labeled "as restated". Little r is when the error is immaterial to prior periods but correcting it in the current period would be material; prior statements are revised at next presentation without the "as restated" label. An out-of-period adjustment applies when the error is immaterial to both periods and is corrected in the current period with disclosure. — [PwC Viewpoint 30.7](https://viewpoint.pwc.com/dt/us/en/pwc/accounting_guides/financial_statement_/financial_statement___18_US/chapter_30_accountin_US/307_correction_of_an_US.html)
- **Materiality method:** SAB Topic 1.N (ASC 250-10-S99-2) requires both the "rollover" (income statement) and "iron curtain" (balance sheet accumulated) methods. — [PwC Viewpoint 30.7](https://viewpoint.pwc.com/dt/us/en/pwc/accounting_guides/financial_statement_/financial_statement___18_US/chapter_30_accountin_US/307_correction_of_an_US.html)
- **Required disclosures (ASC 250-10-50-7 to 50-9):** nature of the error; effect on each financial-statement line item and per-share amount for each prior period; cumulative effect on retained earnings/equity at the earliest period presented; gross and net tax effects. — [PwC Viewpoint 30.7](https://viewpoint.pwc.com/dt/us/en/pwc/accounting_guides/financial_statement_/financial_statement___18_US/chapter_30_accountin_US/307_correction_of_an_US.html)
- **SEC process for Big R:** 8-K Item 4.02 non-reliance, amend prior filings (10-K/A, 10-Q/A), and reassess ICFR and disclosure-controls conclusions. — [PwC Viewpoint 30.7](https://viewpoint.pwc.com/dt/us/en/pwc/accounting_guides/financial_statement_/financial_statement___18_US/chapter_30_accountin_US/307_correction_of_an_US.html)
- **Audit Analytics taxonomy:** three error-correction types: non-reliance (Big R, with 8-K 4.02), revision (little r, prior periods revised within a periodic filing, no 4.02), and out-of-period adjustments. (search snippet; help center page 403) — [Ideagen Audit Analytics: Understanding restatement types](https://auditanalytics.help.ideagen.com/hc/en-gb/articles/35432233953554/)
- **Trend data:**
  - Little r rose to ~76% of restatements in 2020 vs ~35% in 2005 (Audit Analytics data, 2005-2020) — [Cooley PubCo](https://cooleypubco.com/?p=23453)
  - 2019: 383 revisions vs 83 restatements (15-year low for Big R); since 2014 about three-quarters of corrections were revisions — [Bloomberg Law](https://news.bloomberglaw.com/financial-accounting/companies-fixing-more-errors-by-revisions-than-restatements)
  - For 2024, total restatements were up 7% YoY and Big R was at a 9-year high (Ideagen Audit Analytics, via search snippet) — [Deep Quarry](https://deepquarry.substack.com/p/the-resurgence-of-restatements-is)
  - The PCAOB published 2025 Data Points on restatements and auditor turnover (PDF; not readable, 403) — [PCAOB DP04 2025](https://pcaobus.org/economicandriskanalysis/data-points/dp04-2025-financial-restatements-auditor-turnover.pdf)
- **Thompson research:** 45% of revisions (1,500 corrections, 2004-2015) met at least one SEC materiality criterion, which suggests discretion is being used to avoid Big R. — [Bloomberg Law](https://news.bloomberglaw.com/financial-accounting/companies-fixing-more-errors-by-revisions-than-restatements); related: [CLS Blue Sky Blog 2019](https://clsbluesky.law.columbia.edu/2019/10/15/do-firms-conceal-material-misstatements-by-reporting-revisions-rather-than-restatements/?amp=1)
- **SEC Acting Chief Accountant Paul Munter (March 2022):** materiality must be "an objective assessment focused on whether there is a substantial likelihood it is important to the reasonable investor". Staff observed "some materiality analyses appear to be biased toward supporting an outcome that an error is not material". Staff rejected arguments that statements are "too old", that errors are industry-wide, or that offsetting errors cancel out. — [Cooley PubCo](https://cooleypubco.com/?p=23453)
- **Language obscures corrections (Audit Analytics, March 2014; older):** in 2006 over 93% of corrections used "restate" or a derivative; by 2012 only 44% did. Firms instead say "revise prior periods", "correct immaterial errors", "prior period adjustments", etc. 65% of 2012 restatements had no Item 4.02. — [Audit Analytics blog](https://blog.auditanalytics.com/restatements-whats-in-a-name/)

### Inferences
- For a reviewer, the cause label (Big R / little r / out-of-period) is itself a risk signal. Little r corrections are buried in a periodic filing with no 8-K trigger and often avoid the word "restate". Detecting them means finding the revision note and its "as previously reported / adjustment / as revised" table, which is exactly the hunting work a tool could do.
- The ASC 250 disclosure list (line item, per-share, cumulative retained earnings, tax) is a ready-made checklist of fields a review card should show for an error correction.

### Gaps
- Could not read the Audit Analytics FAQ or PCAOB 2025 Data Points (403). 2025 counts are unverified.
- No CAQ or Society for Corporate Governance audit-committee checklist retrieved.

## 4. Concrete pain points / time costs

### Takeaway
I found no independent, quantified time-cost data (hours per filing, etc.). The documented pain points are qualitative: vague language hides corrections; recasts silently change every comparative; vendor re-standardization alters history; look-ahead bias in backtests.

### Cited Findings
- Corrections are hard to find because firms avoid the word "restate" (93% to 44% from 2006 to 2012), using "more than one way to explain why financial statements were wrong". — [Audit Analytics blog](https://blog.auditanalytics.com/restatements-whats-in-a-name/)
- Revisions avoid 8-K filings, "keeping corrections hidden longer" and delaying investor awareness. — [Bloomberg Law](https://news.bloomberglaw.com/financial-accounting/companies-fixing-more-errors-by-revisions-than-restatements)
- **Backtest look-ahead bias:** joining today's restated fundamentals to historical prices assumes knowledge the market didn't have. **[VENDOR]** — [Tradevo/DEV](https://dev.to/tradevodata/as-reported-vs-restated-financial-data-why-the-difference-matters-for-backtesting-1big); academic confirmation that re-standardization alters sales/earnings — [Lyle, Siano & Yohn 2024](https://som.yale.edu/sites/default/files/2024-07/Re-Standardized%20Financial%20Statement%20Data.pdf)
- **[VENDOR, MightyBot blog]** Title asserts spreading templates "break". Not read; marketing. — [MightyBot](https://mightybot.ai/blog/how-financial-spreading-works-and-why-templates-break/)

### Inferences
- Likely drudgery (inferred from the disclosure structure, not measured): locating which filing first showed the new value; diffing old vs new across every line; finding the explaining note (which may use soft language); confirming the change ties to the note's reconciliation table; checking that totals and EPS moved consistently.
- Likely genuine judgment: whether a recast series is comparable; whether to adjust peers/comps; whether a little r looks like a disguised Big R (Munter/Thompson concerns); whether the change alters a thesis.

### Gaps
- No survey or time study found quantifying analyst hours spent on restatement reconciliation. Any "X hours per filing" claim would need a new source.

## 5. Accounting rules that bound what each cause can change (checklist material)

### Takeaway
Each cause has a predictable footprint, documented in ASC 250 / 260 / 205-20 and Big 4 guides. A reviewer can use the footprint to sanity-check a change: a figure moving outside its cause's footprint is a red flag.

### Cited Findings
- **Stock splits / stock dividends (ASC 260-10-55):** accounted for retroactively, including splits after period end but before statements are issued, "by considering them outstanding for the entirety of each period presented". Only share counts and per-share data change. — [Deloitte DART EPS Roadmap ch. 8](https://dart.deloitte.com/USDART/home/codification/presentation/asc260-10/roadmap-earnings-per-share/chapter-8-other-considerations?combine) (via search snippet)
- **Retrospective adjustment of income** (accounting change or error): prior basic and diluted EPS must be restated "as if the restated or retrospectively adjusted income or loss had been originally reported in the prior period(s)", with the per-share effect disclosed. — [Deloitte DART EPS Roadmap ch. 8](https://dart.deloitte.com/USDART/home/codification/presentation/asc260-10/roadmap-earnings-per-share/chapter-8-other-considerations?combine)
- **Discontinued operations:** companies "recast audited financial statements for prior years to reflect operations as discontinued", and similarly for stock splits in all periods (example 8-K recast filings). — [EDGAR 8-K example](https://www.sec.gov/Archives/edgar/data/773910/000095012906008335/h39387e8vk.htm) (search snippet)
- **Error correction:** cumulative effect on periods before those presented goes into opening carrying amounts and retained earnings at the earliest period presented; line-item and per-share effects are shown for each prior period. — [PwC Viewpoint 30.7](https://viewpoint.pwc.com/dt/us/en/pwc/accounting_guides/financial_statement_/financial_statement___18_US/chapter_30_accountin_US/307_correction_of_an_US.html); Deloitte ASC 250 overview — [DART](https://dart.deloitte.com/USDART/tree/vsid/490038)
- **LDTI (ASU 2018-12):** effective for SEC filers (non-SRC) for fiscal years beginning after Dec 15, 2022. Adopters restated prior comparative periods, generally with a modified retrospective approach and a transition date of Jan 1, 2021. (search snippets) — [Cherry Bekaert](https://www.cbh.com/insights/articles/what-ldti-means-for-insurance-accounting/); [Forvis Mazars](https://forvismazars.us/getmedia/1c360e7b-7088-4b0c-9258-153c0525c4d5/relief-on-long-duration-insurance-transition.pdf); KPMG LDTI analyst survey (2022; PDF unreadable) — [KPMG](https://kpmg.com/kpmg-us/content/dam/kpmg/pdf/2022/ldti-analyst-survey.pdf)

### Inferences
These footprints are inferred from the standards above and should be checked against KPMG/Deloitte ASC 205-20 text before encoding:

| Cause | What can change | What should not change |
|---|---|---|
| Split | Shares outstanding, EPS, DPS, per-share data | Dollar totals (revenue, net income, assets) |
| Discontinued operations reclass | Continuing-ops lines (revenue, opex, income from continuing ops), with offsetting movement to "income from discontinued ops" | Net income attributable to the company for the period; total equity |
| Retrospective standard adoption | Prior-period line items and opening retained earnings (and AOCI for LDTI) | Cash |
| Error correction | Any line, but must reconcile to the ASC 250 "as previously reported / adjustment / as restated" table | — |

### Gaps
- Did not retrieve the ASC 205-20 text or a Big 4 statement confirming that a disc-ops reclass leaves net income unchanged. This is standard, but unsourced here.
- ASC 606 full-retrospective vs modified-retrospective transition specifics were not retrieved.
- The LDTI AOCI and retained-earnings transition mechanics are not sourced in detail.
