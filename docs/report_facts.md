# Facts for the project report (all verified by running the code on 2026-11-20)

## Cover details
- Title: AI Decision Auditor
- Subtitle: Auditing the decisions of an AI system for fairness, transparency, robustness and accountability
- Course: Trustworthy AI (TAI)
- Team name: GodL
- Members: Anant Sharma (2024BCS-007), Piyush Agrawal (2024BCS-047), Vishal Yadav (2024BCS-083)
- Submitted to: Sir Narinder Singh Punn
- Programme: B.Tech Computer Science and Engineering, ABV-IIITM Gwalior
- Submitted: November 2026
- Code: github.com/itsvishalyadav/ai-decision-auditor

## What the project is
A Python tool that audits the decisions of an AI decision-maker and returns a scored audit report, a verdict and tested fixes. Two modes:
1. Model audit: a loan-approval model is trained inside the tool and audited (we can see its predictions and query it).
2. Black-box audit: upload a CSV of any AI system's decisions (decision column, true outcome column, protected attributes); the tool audits fairness from decisions alone, without access to the model.
Interfaces: a 9-page Streamlit dashboard (app.py), a command-line audit (run_audit.py) that writes HTML reports, and evaluate.py which prints every number in this report. Core methods are implemented from scratch with scikit-learn, pandas, numpy; dashboard uses Streamlit and Plotly. 10 automated tests (pytest).
Dashboard pages: Overview; 1 Model under audit; 2 Trust scorecard; 3 Fairness; 4 Explain a decision; 5 Robustness; 6 Bias mitigation; 7 Audit trail; 8 Audit an external AI.

## Why this is a trustworthy-AI problem
- AI already makes or shapes decisions about credit, hiring, insurance and bail. Known failures: ProPublica's 2016 "Machine Bias" analysis of COMPAS (higher false-positive rate for Black defendants); Amazon's experimental recruiting tool (reported 2018) downgraded CVs containing "women's"; Apple Card (2019) allegations of lower credit limits for women.
- In each case, accuracy was not the problem being checked. Nobody independently checked the decisions.
- EU AI Act (Regulation (EU) 2024/1689) lists credit scoring and recruitment as high-risk. Relevant articles: Art. 10 data governance and bias examination; Art. 12 record-keeping; Art. 13 transparency; Art. 14 human oversight; Art. 15 accuracy and robustness.
- NIST AI Risk Management Framework 1.0 (2023) trustworthiness characteristics: valid and reliable; safe; secure and resilient; accountable and transparent; explainable and interpretable; privacy-enhanced; fair with harmful bias managed.
- EU High-Level Expert Group, Ethics Guidelines for Trustworthy AI (2019): seven requirements incl. human agency and oversight, technical robustness, transparency, diversity/non-discrimination/fairness, accountability.
- Our angle: an auditor is a separate party from the model builder. It should be able to work on decisions alone, should not trust the model's own claims, and should leave a record that cannot be quietly edited.

Mapping of requirements to what the tool does (use as a table):
- Fairness / non-discrimination (EU AI Act Art. 10; NIST "fair"): 5 group-fairness metrics per protected attribute, a flip test, proxy detection, intersectional view, three mitigations with re-audit.
- Transparency / explainability (Art. 13; NIST "explainable"): global permutation importance, per-decision Shapley values with a plain-language reason, recourse (what change would flip the decision), explanation-fidelity check.
- Robustness (Art. 15; NIST "secure and resilient"): noise-stability test at 4 noise levels.
- Human oversight (Art. 14): decisions within 5 percentage points of the threshold flagged for a person.
- Accuracy (Art. 15; NIST "valid and reliable"): accuracy, F1, ROC-AUC.
- Accountability / record-keeping (Art. 12; NIST "accountable"): hash-chained audit log with tamper detection; every model gets a 12-character fingerprint.
- Privacy: out of scope. The tool does no membership-inference or data-protection checks.

## Demo data (synthetic, generated in code; seed 42)
- 4,000 loan applications. Features: age, gender, age_group, education, part_time, income, employment_years, credit_score, loan_amount, debt_to_income, late_payments. Target: approved.
- The historical labels were made biased on purpose: past decision makers penalised women, applicants under 25 and part-time workers, on top of a "merit" score built from income, credit score, loan amount, debt-to-income, late payments and employment years.
- part_time is a proxy for gender: 45.2% of women vs 9.4% of men work part time.
- Historical approval rate: Male 68.8%, Female 39.9%; age 25-59: 58.4%, 60+: 60.2%, Under 25: 35.1%.
- Bias strength is a parameter (0 = fair history, 1 = default, 2 = very biased; slider in the dashboard).
- Why synthetic: we planted the bias, so we know the right answer and can check the auditor finds it. A real dataset's true bias is unknown.
- 70/30 stratified train/test split; all audits run on the 1,200 test decisions. Decision threshold 0.5.

## Design
### Architecture (Figure 1, drawn separately)
Historical data -> AI model -> decisions -> AUDITOR (five checks: fairness, transparency, robustness, performance, accountability) -> trust score and verdict -> mitigation -> re-audit. The black-box mode skips the model and feeds a CSV of decisions straight to the fairness checks.

### Fairness checks (per protected attribute: gender, age_group)
| Metric | Question | Pass if |
| Demographic parity difference | gap in approval rate between groups | <= 0.10 |
| Disparate impact ratio | lowest group's approval rate / highest group's | >= 0.80 (US EEOC four-fifths rule, 1978) |
| Equal opportunity difference | gap in true-positive rate (qualified people approved) | <= 0.10 |
| Equalized odds difference | larger of the TPR gap and the FPR gap | <= 0.10 |
| Predictive parity difference | gap in precision | <= 0.10 |
Plus:
- Flip test (counterfactual fairness idea, Kusner et al. 2017): change only the protected attribute for every test applicant, re-predict, count changed decisions. Only for binary attributes.
- Proxy detection: Cramer's V (categorical) or correlation ratio eta (numeric) between each feature and the protected attribute; >= 0.2 flagged.
- Intersectional view: approval rate for gender x age group.

### Transparency
- Global: permutation importance = drop in ROC-AUC when a feature is shuffled (Breiman 2001), 5 repeats.
- Local: Monte-Carlo Shapley values (Strumbelj and Kononenko 2014; the idea behind SHAP, Lundberg and Lee 2017). 150 random feature orderings against 150 background applicants. Shown as a waterfall from "average applicant" to "this applicant", plus a reason sentence such as "Main reasons for REJECTION: credit_score = 620 (-21%); income = 28,600 (-18.5%); gender = Female (-11%)".
- Fidelity check: Shapley contributions must add up to (this prediction - average prediction). Mean error is reported.
- Recourse (Wachter et al. 2017): greedy search over actionable features only (income, credit_score, loan_amount, debt_to_income, late_payments, employment_years), trying steps of 1x, 3x and 6x a base step and taking the best gain per unit of effort, until the decision flips. Gender, age and education are never suggested.
- Transparency score is penalised if a protected attribute (25 points each) or a flagged proxy (10 each) is among the top 3 features, and by fidelity error.

### Robustness and human oversight
- Gaussian noise of 1%, 2%, 5% and 10% of each numeric feature's standard deviation, 5 trials each; measure the share of decisions that flip. Pass limit: at most 3% flips at the 2% level.
- Borderline decisions (probability within 0.05 of the threshold) are listed for human review.

### Accountability
- Every recorded decision (inputs, decision, probability, explanation, human-review flag, model fingerprint, UTC timestamp) is appended to a JSON-lines log. Each record stores the SHA-256 hash of the previous record. Verification recomputes the chain and reports the first broken record and why ("record content was modified" or "link to previous record is broken").
- A "Simulate tampering" button flips a past decision in the file to demonstrate detection. In the demo: "TAMPERING DETECTED at record #3: record content was modified".
- Model fingerprint: SHA-256 of model name, parameters, features and thresholds, first 12 hex characters.

### Trust score and verdict
- Overall = weighted average: Fairness 35%, Transparency 20%, Robustness 15%, Performance 15%, Accountability 15%.
- Performance score = (ROC-AUC - 0.5) / 0.4 x 100, capped at 100. Accountability = 100 if the log verifies, else 0.
- Verdict: >= 80 and no critical issue -> TRUSTWORTHY; >= 60 -> CONDITIONAL (fix before deployment); below 60 -> NOT TRUSTWORTHY.
- Critical issues block TRUSTWORTHY whatever the score: disparate impact below 0.80, more than 5% of decisions flipping on the protected attribute alone, or a tampered log.
- The weights and thresholds are our design choice, stated openly. They are not a legal standard.

### Mitigations (tested, then re-audited)
1. Reweighing (Kamiran and Calders 2012): training weights that make group and label statistically independent; gender removed from the inputs.
2. Remove gender and its proxy part_time from the inputs.
3. Group-specific thresholds (post-processing in the style of Hardt, Price and Srebro 2016): a separate decision threshold per gender so approval rates match.

## Evaluation results (command: python evaluate.py; tests: python -m pytest)
### Three model types, default bias, gender used as an input
| Model | Accuracy | ROC-AUC | Gender DI | Gender checks failed | Flip test | Overall | Verdict |
| Logistic Regression | 87.7% | 0.954 | 0.547 | 5 of 5 | 23.7% | 54.7 | NOT TRUSTWORTHY |
| Random Forest | 85.2% | 0.936 | 0.519 | 4 of 5 | 26.3% | 60.7 | CONDITIONAL (critical issue present) |
| Gradient Boosting | 87.0% | 0.948 | 0.568 | 5 of 5 | 21.8% | 56.7 | NOT TRUSTWORTHY |
All three: part_time flagged as a gender proxy (Cramer's V 0.374); age (0.745) and employment_years (0.365) flagged for age group; gender is in the top 3 most important features. Least favoured groups: Female and Under 25.

### Gradient Boosting baseline in detail (the model used in the demo)
Gender: demographic parity 0.304 FAIL; disparate impact 0.568 FAIL; equal opportunity 0.132 FAIL; equalized odds 0.132 FAIL; predictive parity 0.116 FAIL.
Age group: demographic parity 0.278 FAIL; disparate impact 0.557 FAIL; equal opportunity 0.062 PASS; equalized odds 0.129 FAIL; predictive parity 0.159 FAIL.
Pillar scores: Fairness 15.5, Transparency 61.0, Robustness 60.6, Performance 100, Accountability 100. Overall 56.7.
Example decision: applicant age 30, female, income 28,600, credit score 620: approval probability 4.3%, rejected. Shapley: credit_score -21%, income -18.5%, gender = Female -11%. Same applicant as male: 16.7%. Recourse: credit score 620 -> 725 and late payments 1 -> 0 gives APPROVED at 55.7%.
Recourse was found for 8 of 8 rejected test applicants tried, for each of the three model types.

### Fairness through unawareness (Gradient Boosting, gender removed from inputs, nothing else changed)
Accuracy 84.2%. Gender DI 0.804 (just passes), but demographic parity 0.125, equalized odds 0.139 and predictive parity 0.248 still fail. Flip test 0% (gender is not an input) yet the gap remains through part_time. Overall 76.5, CONDITIONAL.

### Mitigations (applied to the Gradient Boosting baseline, same train/test split)
| | Baseline | Reweighing | Drop gender + part_time | Group thresholds |
| Accuracy | 87.0% | 84.9% | 83.3% | 81.4% |
| Demographic parity (<=0.10) | 0.304 | 0.110 | 0.062 | 0.031 |
| Disparate impact (>=0.80) | 0.568 | 0.825 | 0.896 | 0.948 |
| Equal opportunity (<=0.10) | 0.132 | 0.043 | 0.078 | 0.192 |
| Equalized odds (<=0.10) | 0.132 | 0.160 | 0.212 | 0.272 |
| Predictive parity (<=0.10) | 0.116 | 0.252 | 0.284 | 0.307 |
| Flip test | 21.8% | 0% | 0% | 11.8% |
| Gender checks passed | 0 of 5 | 2 of 5 | 3 of 5 | 2 of 5 |
| Overall score | 56.7 | 74.4 | 71.9 | 60.8 |
Notes: the group thresholds were Male 0.79 and Female 0.18. That method equalises approval rates but uses gender directly in the decision, so the flip test comes back at 11.8%, and equal opportunity and predictive parity get worse. No mitigation fixed age-group bias, because all three only target gender (age-group disparate impact stays about 0.59-0.60). Equalized odds and predictive parity get worse under every mitigation. This is the trade-off predicted by Kleinberg, Mullainathan and Raghavan (2016) and Chouldechova (2017): when base rates differ, these criteria cannot all hold at once.

### Control: fair history (bias strength 0, Gradient Boosting, gender not an input)
All 5 gender checks pass (DI 0.967, demographic parity 0.024). Age group: 3 of 5 pass; equalized odds 0.119 and predictive parity 0.118 fail narrowly. Overall 86.5, TRUSTWORTHY. So the auditor does not raise a gender alarm when there is no planted gender bias. (Same model fingerprint as the gender-blind model because the fingerprint covers the model, not the training data. Note this as a limitation.)

### Robustness (share of test decisions that flip under noise)
Gradient Boosting baseline: 3.5% at 1% noise, 3.6% at 2%, 4.2% at 5%, 5.2% at 10%. Logistic Regression: 2.8%, 3.0%, 3.4%, 4.2%. Random Forest: 3.7%, 3.7%, 4.2%, 4.9%. Fair-history model: 3.7% at 2% noise.
Every model, including the fair one, is at or just above the 3% limit, and the flip rate barely changes between 1% and 2% noise. This suggests the flips come from decisions sitting right at the threshold (46 borderline decisions for Gradient Boosting, 38 for Logistic Regression, 93 for Random Forest), not from instability. The 3% limit is probably too strict for this data.

### Explanation fidelity
Mean additivity error of the Shapley explanations: 0.023 to 0.042 probability points across the models audited (60 orderings per check). The unit test with 200 orderings over 100 background rows passes with tolerance 0.03.

### Black-box audit of an external AI hiring tool (sample CSV, 1,500 candidates)
Gender: all 5 checks fail (demographic parity 0.200, DI 0.632, equal opportunity 0.196, equalized odds 0.223, predictive parity 0.136); fairness score 27.3/100.
Ethnicity (3 groups): 4 of 5 fail (DI 0.755, demographic parity 0.122, equal opportunity 0.133, equalized odds 0.133); predictive parity 0.088 passes; score 51.8/100. Total 9 of 10 checks fail. The tool used no model internals.

### Speed
Training plus full audit takes about 1.6 s (Logistic Regression), 2.0 s (Gradient Boosting) and 6.8 s (Random Forest) on a laptop.

### Tests (10, all pass, about 6 s)
Known metric values on a hand-made example; perfectly fair decisions pass; the injected bias is detected and part_time flagged; no disparate-impact alarm on fair data; Shapley values add up; recourse flips a rejection without touching gender or age; the flip test is non-zero when gender is an input and zero when it is not; reweighing makes group and label independent; reweighing improves disparate impact; tampering is detected at the exact record.

## Problems found while building it, then fixed (true, from development)
1. Shapley values did not add up. Each sampled ordering started from a random background applicant, so the sum of contributions missed (prediction - average) by about 0.05 (0.495 vs 0.447 in the failing test). Fix: cycle through the background applicants evenly. The additivity test now passes.
2. Recourse search got stuck on tree models. Tree models are flat over small ranges, so a single small step (for example +5,000 income) showed no gain and the greedy search gave up without flipping the decision. Fix: try steps of 1x, 3x and 6x and rank by gain per unit of effort. Recourse is now found for 8 of 8 rejected applicants for each model type.
3. Proxy detection crashed with pandas 3, which stores text columns as a string type rather than object, so the check for "is this column categorical" was wrong. Fix: test for numeric types instead.
4. The explanation chart was misleading: the final bar was drawn from the average down to the prediction, so a 4% approval chance looked like a long 60% bar. Fix: the chart now reads average applicant -> each factor -> this applicant, with the approval threshold marked.
5. Single-applicant predictions failed when a row's columns lost their types (one row from a mixed table becomes all "object"). Fix: the model casts inputs back to the training types in one place.

## Related work and what we take from each
- IBM AI Fairness 360 (Bellamy et al. 2018, arXiv:1810.01943) and Microsoft Fairlearn (Bird et al. 2020): the fairness metrics and the reweighing / threshold mitigations. We reimplement them so every number can be traced.
- SHAP (Lundberg and Lee 2017, arXiv:1705.07874) and Strumbelj and Kononenko (2014): Shapley-value explanations.
- LIME (Ribeiro, Singh and Guestrin 2016, arXiv:1602.04938): local explanation, which we do not use; Shapley values were chosen because they add up exactly to the prediction.
- Wachter, Mittelstadt and Russell (2017, arXiv:1711.00399): counterfactual explanations / recourse.
- Kusner et al. (2017, arXiv:1703.06856): counterfactual fairness, the idea behind the flip test.
- Hardt, Price and Srebro (2016, arXiv:1610.02413): equal opportunity and threshold post-processing.
- Kamiran and Calders (2012), Knowledge and Information Systems 33(1): reweighing.
- Kleinberg, Mullainathan and Raghavan (2016, arXiv:1609.05807) and Chouldechova (2017, Big Data 5(2)): impossibility of satisfying all fairness criteria together.
- Raji et al. (2020, arXiv:2001.00873), Closing the AI Accountability Gap: internal algorithmic auditing as a process; our audit trail and report follow that idea.
- Breiman (2001), Random Forests, Machine Learning 45: permutation importance.

## Limitations (honest)
- Synthetic data. Real datasets (UCI Adult, German Credit) were not audited.
- The thresholds (0.10, 0.80, 3%) and the pillar weights are conventions and design choices, not law. The score can rank models but does not certify one.
- The robustness test is dominated by borderline decisions; every model including the fair one fails it narrowly. The criterion needs rethinking (for example, ignore decisions already flagged as borderline).
- Shapley values are approximated by sampling (error up to about 0.04); recourse search is greedy and not guaranteed to find the smallest change.
- The flip test only works for binary attributes, so it does not run for age group (three groups).
- Single random split and seed; no confidence intervals on the metrics.
- The tests were written by us, the builders, so they show the checks work on cases we thought of.
- The audit log is a local file. The hash chain detects edits but not deletion of the whole file; a real deployment would anchor hashes externally or use write-once storage. Anyone with file access can also rewrite the whole chain consistently.
- The model fingerprint covers the model, not the training data, so two models trained on different data can share a fingerprint.
- Group-specific thresholds use the protected attribute directly; this may be unlawful in some jurisdictions.
- Only binary classification decisions are audited. No privacy checks.

## Reproducing (Windows PowerShell, from the project folder)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest            # 10 passed
.\.venv\Scripts\python.exe evaluate.py          # every number in section 5
.\.venv\Scripts\python.exe run_audit.py         # before/after HTML reports in reports/
.\.venv\Scripts\python.exe -m streamlit run app.py   # dashboard at http://localhost:8501
Everything runs offline with no keys or accounts.
