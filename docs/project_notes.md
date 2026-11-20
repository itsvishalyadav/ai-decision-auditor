# Facts for writing: AI Decision Auditor (Trustworthy AI course project)

## What it is
A Python tool (Streamlit web dashboard + command-line script) that audits the decisions of an AI system against the pillars of Trustworthy AI and produces a scored audit report with recommendations and tested fixes. It works in two ways:
1. White-box/grey-box: audit a model trained inside the tool (a loan-approval AI).
2. Black-box: upload a CSV of any AI system's decisions (decision column, true outcome column, protected attributes) and get a fairness audit. Sample provided: decisions of an AI CV-screening/hiring tool (1,500 candidates, attributes gender and ethnicity).

## Why it matters (background research)
- AI now decides loans, hiring shortlists, insurance, bail. Known failures: Amazon's experimental recruiting tool (reported 2018) downgraded CVs containing the word "women's"; ProPublica's 2016 analysis of the COMPAS recidivism tool found higher false-positive rates for Black defendants; Apple Card (2019) faced allegations of giving women lower credit limits.
- EU AI Act (Regulation (EU) 2024/1689): credit scoring and recruitment are high-risk systems. Relevant articles: Art. 10 data governance & bias examination; Art. 12 record-keeping/logging; Art. 13 transparency; Art. 14 human oversight; Art. 15 accuracy, robustness.
- NIST AI Risk Management Framework (AI RMF 1.0, 2023) trustworthy characteristics: valid & reliable; safe; secure & resilient; accountable & transparent; explainable & interpretable; privacy-enhanced; fair with harmful bias managed.
- EU High-Level Expert Group "Ethics Guidelines for Trustworthy AI" (2019): 7 requirements incl. human agency & oversight, technical robustness, transparency, diversity/non-discrimination/fairness, accountability.
- Existing toolkits this project is inspired by: IBM AIF360, Microsoft Fairlearn, SHAP, LIME, Google What-If Tool, DiCE (counterfactuals). This project implements the core methods from scratch (only scikit-learn, pandas, numpy, plotly, streamlit) so every number can be explained.
- US "four-fifths rule" (EEOC Uniform Guidelines, 1978): selection rate of any group below 80% of the highest group's rate is evidence of adverse impact.
- Impossibility result: when groups have different base rates, calibration/predictive parity and equal error rates cannot all hold at once (Kleinberg, Mullainathan & Raghavan 2016; Chouldechova 2017).

## Demo scenario (synthetic data, 4,000 loan applications, generated in code)
A bank trains an AI on past loan decisions made by biased humans. Historical decision makers penalised women, applicants under 25, and part-time workers. Part-time work is much more common among women in the data (45% vs 10%), so `part_time` is a proxy for gender. Historical approval rate: Male 68.8%, Female 39.9%. The bias strength can be changed with a slider (0 = fair history, 2 = very biased).
Features: age, gender, age_group, education, part_time, income, employment_years, credit_score, loan_amount, debt_to_income, late_payments. Target: approved (0/1). Models offered: Logistic Regression, Random Forest, Gradient Boosting. 70/30 train/test split; the audit runs on the 1,200 test decisions.

## The five audit pillars and how each is measured
Overall trust score = weighted average: Fairness 35%, Transparency 20%, Robustness 15%, Performance 15%, Accountability 15%.
Verdict: >=80 and no critical issue -> TRUSTWORTHY; 60-80 -> CONDITIONAL (fix before deployment); <60 -> NOT TRUSTWORTHY. A critical issue (80% rule violated, >5% decisions flip when only the protected attribute changes, or tampered audit log) blocks a TRUSTWORTHY verdict regardless of score.

1. Fairness (per protected attribute: gender, age_group). Five group metrics with thresholds:
   - Demographic parity difference (gap in approval rate) <= 0.10
   - Disparate impact ratio (lowest/highest approval rate) >= 0.80 (four-fifths rule)
   - Equal opportunity difference (gap in true positive rate) <= 0.10
   - Equalized odds difference (max of TPR gap and FPR gap) <= 0.10
   - Predictive parity difference (gap in precision) <= 0.10
   Plus: counterfactual flip test (change only gender for every applicant, count changed decisions; Kusner et al. 2017 counterfactual fairness idea); proxy detection (Cramer's V for categorical features, correlation ratio eta for numeric; >=0.2 flagged); intersectional view (gender x age group).
2. Transparency / explainability:
   - Global: permutation feature importance (drop in ROC-AUC when a feature is shuffled; Breiman 2001).
   - Local: model-agnostic Monte-Carlo Shapley values (Strumbelj & Kononenko 2014; same idea as SHAP, Lundberg & Lee 2017), shown as a waterfall chart from "average applicant" to "this applicant", plus a plain-language reason sentence like an adverse-action notice.
   - Fidelity check: contributions must add up to (prediction - average prediction).
   - Score penalised if a protected attribute or proxy is among top-3 most important features.
   - Recourse / counterfactual explanation (Wachter et al. 2017): greedy search for the smallest realistic change to actionable features only (income, credit_score, loan_amount, debt_to_income, late_payments, employment_years) that flips the decision. Gender, age, education never suggested.
   - Per-person "same person, different gender?" check.
3. Robustness: add Gaussian noise of 1%, 2%, 5%, 10% of each numeric feature's standard deviation; measure share of decisions that flip. Pass limit: <=3% flips at 2% noise. Borderline decisions (within 5% of the 0.5 threshold) are flagged for human review (human oversight, Art. 14).
4. Performance: accuracy, F1, ROC-AUC, approval rate.
5. Accountability: tamper-evident audit trail. Every logged decision (inputs, decision, probability, explanation, human-review flag, model fingerprint, timestamp) stores the SHA-256 hash of the previous record (hash chain, like a blockchain without the network). Verify button recomputes the chain; "Simulate tampering" button secretly edits a past decision and the tool reports exactly which record was modified. Each model has a 12-character fingerprint (model ID).

## Bias mitigation (tested, then re-audited)
- Pre-processing: Reweighing (Kamiran & Calders 2012) - sample weights make group and label statistically independent; gender removed from inputs.
- In-processing/feature: remove gender AND its proxy part_time.
- Post-processing: group-specific decision thresholds (Hardt, Price & Srebro 2016 style) to equalise approval rates.
The page shows before/after metrics, accuracy and trust score, and a button to deploy the mitigated model.

## Verified results (Gradient Boosting, default bias, from actual runs)
Baseline (gender used as input):
- Accuracy 87.0%, ROC-AUC 0.948.
- Gender: demographic parity diff 0.304 FAIL, disparate impact 0.568 FAIL, equal opportunity 0.132 FAIL, equalized odds 0.132 FAIL, predictive parity 0.116 FAIL (5 of 5 fail). Least favoured group: Female.
- Age group: disparate impact 0.557 FAIL; 4 of 5 fail (equal opportunity passes); least favoured: Under 25.
- Flip test: changing only gender changes 21.8% of decisions.
- Proxy detected: part_time (Cramer's V 0.374) for gender; age and employment_years for age group.
- Pillar scores: Fairness 15.5, Transparency ~56-62, Robustness ~61, Performance 100, Accountability 100. Overall ~57/100 -> NOT TRUSTWORTHY - do not deploy.
After Reweighing:
- Gender disparate impact 0.825 PASS, equal opportunity 0.043 PASS, demographic parity 0.11 (just fails), equalized odds 0.16 FAIL, predictive parity 0.252 FAIL. Accuracy 84.9% (down from 87.0%).
- Fairness 48.1, Transparency 90.5, overall 74/100 -> CONDITIONAL.
Other mitigations: removing gender + part_time: disparate impact 0.896, accuracy 83.3%, overall 71. Group thresholds: demographic parity 0.031 and DI 0.948 but equal opportunity worsens to 0.192 and accuracy 81.4% - shows the fairness trade-off / impossibility result.
Example explanation (rejected female applicant, income 28,600, credit score 620): approval probability 4.3%; main reasons credit_score (-21%), income (-18.5%), gender = Female (-11%). Same person as Male: 16.7%. Recourse: credit score 620 -> 725 and late payments 1 -> 0 flips decision to APPROVED (55.7%).
External hiring-tool sample audit: 9 of 10 fairness checks fail; gender fairness score 27/100, ethnicity 52/100.
Tamper demo: after tampering, tool reported "TAMPERING DETECTED at record #3: record content was modified".
Automated tests: 10 pytest tests pass (known metric values, bias detection, no false alarm on fair data, Shapley additivity, counterfactual flips decision, flip test, reweighing independence, mitigation improves fairness, tamper detection).

## Project structure
- app.py - Streamlit dashboard with 9 pages: Overview, 1 Model under audit, 2 Trust scorecard, 3 Fairness, 4 Explain a decision, 5 Robustness, 6 Bias mitigation, 7 Audit trail, 8 Audit an external AI.
- run_audit.py - command-line audit (before and after reweighing), prints scorecard and writes reports/audit_before.html and reports/audit_after.html; also writes sample_data/loan_applications.csv and sample_data/hiring_decisions.csv.
- auditor/data.py (synthetic data), model.py (training, model fingerprint), fairness.py, explain.py, counterfactual.py, robustness.py, ledger.py (hash-chained audit log), mitigation.py (+ proxy detection), audit.py (runs all checks, scorecard, recommendations), report.py (HTML report export).
- tests/test_auditor.py - 10 tests. requirements.txt.
- The HTML report can be downloaded from the scorecard page and printed to PDF from the browser.

## How to run (Windows PowerShell, from the project folder C:\My Work\TAI)
First time setup:
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Run the dashboard:
  .\.venv\Scripts\python.exe -m streamlit run app.py
  (opens http://localhost:8501)
Command-line audit:
  .\.venv\Scripts\python.exe run_audit.py
Tests:
  .\.venv\Scripts\python.exe -m pytest

## Limitations (be honest)
- Synthetic data, not real bank data (chosen so the bias is known and the auditor can be validated).
- Shapley values are approximated by sampling; counterfactual search is greedy, not guaranteed minimal.
- Thresholds (0.10, 0.80) are common conventions, not legal certification; scores/weights are a design choice.
- Audit log is a local file; hash chain detects tampering but does not prevent deleting the whole file (production would anchor hashes externally / use WORM storage).
- Group-specific thresholds may be legally restricted in some jurisdictions.
Future work: real datasets (UCI Adult, German Credit), more protected attributes, LLM decision auditing, continuous monitoring/drift detection, privacy checks (membership inference), PDF export.
