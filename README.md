# AI Decision Auditor

**Trustworthy Artificial Intelligence — course project**

AI Decision Auditor is a Python tool that audits the decisions of an AI system against the pillars of Trustworthy AI and produces a scored audit report with recommendations and tested fixes. It works in two ways. In white-box/grey-box mode, it audits a model trained inside the tool itself — a loan-approval AI trained on synthetic data with a known, controllable amount of historical bias. In black-box mode, you can upload a CSV of any AI system's decisions (a decision column, a true outcome column, and protected attributes) and get a fairness audit; a sample is provided for an AI CV-screening/hiring tool with 1,500 candidates and attributes for gender and ethnicity. The tool consists of a Streamlit web dashboard and a command-line script, and it implements its core methods from scratch (only scikit-learn, pandas, numpy, plotly, streamlit) so that every number can be explained.

## Why this matters

AI now decides loans, hiring shortlists, insurance, and bail, and it has failed publicly. Amazon's experimental recruiting tool (reported 2018) downgraded CVs containing the word "women's". ProPublica's 2016 analysis of the COMPAS recidivism tool found higher false-positive rates for Black defendants. Apple Card (2019) faced allegations of giving women lower credit limits.

Regulation is catching up. The EU AI Act (Regulation (EU) 2024/1689) classifies credit scoring and recruitment as high-risk systems, with relevant requirements in Art. 10 (data governance and bias examination), Art. 12 (record-keeping/logging), Art. 13 (transparency), Art. 14 (human oversight), and Art. 15 (accuracy and robustness). The NIST AI Risk Management Framework (AI RMF 1.0, 2023) lists trustworthy characteristics such as valid and reliable, accountable and transparent, explainable and interpretable, and fair with harmful bias managed. The EU High-Level Expert Group's "Ethics Guidelines for Trustworthy AI" (2019) similarly require human oversight, technical robustness, transparency, fairness, and accountability. This project is inspired by existing toolkits — IBM AIF360, Microsoft Fairlearn, SHAP, LIME, Google What-If Tool, and DiCE — but implements the core methods from scratch so every result is explainable.

## Features

The audit is organised around five pillars:

- **Fairness** — Five group fairness metrics per protected attribute (gender, age_group), a counterfactual flip test (change only gender for every applicant and count changed decisions, following the counterfactual fairness idea of Kusner et al. 2017), proxy detection (Cramer's V for categorical features, correlation ratio eta for numeric ones; >= 0.2 is flagged), and an intersectional view (gender x age group).
- **Transparency** — Global permutation feature importance (drop in ROC-AUC when a feature is shuffled; Breiman 2001), local model-agnostic Monte-Carlo Shapley values (Strumbelj & Kononenko 2014; the same idea as SHAP, Lundberg & Lee 2017) shown as a waterfall chart from the "average applicant" to "this applicant" with a plain-language reason sentence like an adverse-action notice, a fidelity check (contributions must add up to prediction minus average prediction), and recourse/counterfactual explanations (Wachter et al. 2017): a greedy search for the smallest realistic change to actionable features only that flips the decision. Gender, age, and education are never suggested. The score is penalised if a protected attribute or proxy is among the top-3 most important features.
- **Robustness** — Gaussian noise of 1%, 2%, 5%, and 10% of each numeric feature's standard deviation is added, and the share of flipped decisions is measured (pass limit: <= 3% flips at 2% noise). Borderline decisions within 5% of the 0.5 threshold are flagged for human review (Art. 14).
- **Performance** — Accuracy, F1, ROC-AUC, and approval rate.
- **Accountability** — A tamper-evident audit trail: every logged decision (inputs, decision, probability, explanation, human-review flag, model fingerprint, timestamp) stores the SHA-256 hash of the previous record — a hash chain, like a blockchain without the network. A verify button recomputes the chain; a "simulate tampering" button secretly edits a past decision and the tool reports exactly which record was modified. Each model has a 12-character fingerprint.

### Fairness metrics and thresholds

| Metric | Definition | Threshold |
|---|---|---|
| Demographic parity difference | Gap in approval rate between groups | <= 0.10 |
| Disparate impact ratio | Lowest/highest approval rate (four-fifths rule, EEOC 1978) | >= 0.80 |
| Equal opportunity difference | Gap in true positive rate | <= 0.10 |
| Equalized odds difference | Max of TPR gap and FPR gap | <= 0.10 |
| Predictive parity difference | Gap in precision | <= 0.10 |

The tool also applies **bias mitigation** and re-audits the result: pre-processing with Reweighing (Kamiran & Calders 2012) plus removing gender from inputs; removing gender AND its proxy `part_time`; and post-processing with group-specific thresholds (Hardt, Price & Srebro 2016 style) to equalise approval rates. Each mitigation shows before/after metrics, accuracy, and trust score, with a button to deploy the mitigated model. **Black-box CSV auditing** lets you audit any external AI's decisions.

## Trust score and verdict

The overall trust score is a weighted average of the pillars:

| Pillar | Weight |
|---|---|
| Fairness | 35% |
| Transparency | 20% |
| Robustness | 15% |
| Performance | 15% |
| Accountability | 15% |

Verdict rules:
- **TRUSTWORTHY** — score >= 80 and no critical issue.
- **CONDITIONAL** — score 60-80 (fix before deployment).
- **NOT TRUSTWORTHY** — score < 60.

A critical issue (four-fifths rule violated, more than 5% of decisions flipping when only the protected attribute changes, or a tampered audit log) blocks a TRUSTWORTHY verdict regardless of score.

## Demo scenario

The demo uses synthetic data: 4,000 loan applications generated in code. A bank trains an AI on past loan decisions made by biased humans. The historical decision makers penalised women, applicants under 25, and part-time workers. Part-time work is much more common among women in the data (45% vs 10%), so `part_time` acts as a proxy for gender. Historical approval rates: Male 68.8%, Female 39.9%. The bias strength can be changed with a slider (0 = fair history, 2 = very biased). Features include age, gender, education, part_time, income, employment_years, credit_score, loan_amount, debt_to_income, and late_payments; the target is `approved`. Models offered: Logistic Regression, Random Forest, Gradient Boosting, with a 70/30 train/test split — the audit runs on the 1,200 test decisions.

## Results

Verified numbers from actual runs (Gradient Boosting, default bias):

| Metric | Baseline | After Reweighing |
|---|---|---|
| Accuracy | 87.0% | 84.9% |
| ROC-AUC | 0.948 | — |
| Demographic parity diff (gender) | 0.304 FAIL | 0.11 (just fails) |
| Disparate impact (gender) | 0.568 FAIL | 0.825 PASS |
| Equal opportunity diff (gender) | 0.132 FAIL | 0.043 PASS |
| Equalized odds diff (gender) | 0.132 FAIL | 0.16 FAIL |
| Predictive parity diff (gender) | 0.116 FAIL | 0.252 FAIL |
| Flip test (gender change) | 21.8% | — |
| Fairness score | 15.5 | 48.1 |
| Transparency score | ~56-62 | 90.5 |
| Overall verdict | ~57/100 NOT TRUSTWORTHY | 74/100 CONDITIONAL |

The baseline fails all five gender fairness metrics, with proxy detection correctly flagging `part_time` (Cramer's V 0.374). Reweighing fixes most approval-rate gaps but costs about 2 points of accuracy and leaves equalized odds and predictive parity failing — a live demonstration of the impossibility result that when groups have different base rates, calibration/predictive parity and equal error rates cannot all hold at once (Kleinberg, Mullainathan & Raghavan 2016; Chouldechova 2017). The alternative mitigations show the same trade-off from different angles: removing gender + part_time gives disparate impact 0.896 at accuracy 83.3% (overall 71), while group thresholds achieve near-parity (demographic parity 0.031, DI 0.948) but equal opportunity worsens to 0.192 and accuracy drops to 81.4%.

For illustration, a rejected female applicant (income 28,600, credit score 620) gets an approval probability of 4.3%, with main reasons credit_score (-21%), income (-18.5%), and gender = Female (-11%); the same person as male would get 16.7%. The recourse suggestion — credit score 620 -> 725 and late payments 1 -> 0 — flips the decision to APPROVED (55.7%). On the external hiring-tool sample, 9 of 10 fairness checks fail (gender fairness 27/100, ethnicity 52/100), and the tamper demo correctly reported "TAMPERING DETECTED at record #3: record content was modified". All 10 pytest tests pass.

## Quick start

Windows PowerShell, from the project folder. First time setup:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the dashboard (opens http://localhost:8501):

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Command-line audit (prints the scorecard and writes `reports/audit_before.html` and `reports/audit_after.html`, plus the sample CSVs):

```powershell
.\.venv\Scripts\python.exe run_audit.py
```

Run the tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Dashboard pages

- **Overview** — introduction and navigation.
- **1 Model under audit** — the synthetic loan data, model choice, and bias slider.
- **2 Trust scorecard** — pillar scores, overall trust score, verdict, and downloadable HTML report.
- **3 Fairness** — the five metrics, flip test, proxy detection, and intersectional view.
- **4 Explain a decision** — Shapley waterfall, plain-language reason, and recourse suggestions for one applicant.
- **5 Robustness** — noise tests, flip rates, and borderline decisions flagged for human review.
- **6 Bias mitigation** — reweighing, feature removal, and group thresholds, with before/after comparison and deploy button.
- **7 Audit trail** — the hash-chained decision log, verify and simulate-tampering buttons.
- **8 Audit an external AI** — black-box CSV upload and fairness audit.

## Project structure

- `app.py` — Streamlit dashboard with the 9 pages above.
- `run_audit.py` — command-line audit (before and after reweighing); prints the scorecard and writes the HTML reports and sample CSVs.
- `auditor/data.py` — synthetic data generation.
- `auditor/model.py` — model training and model fingerprint.
- `auditor/fairness.py` — fairness metrics and flip test.
- `auditor/explain.py` — feature importance and Shapley values.
- `auditor/counterfactual.py` — recourse/counterfactual search.
- `auditor/robustness.py` — noise tests.
- `auditor/ledger.py` — hash-chained audit log.
- `auditor/mitigation.py` — bias mitigation and proxy detection.
- `auditor/audit.py` — runs all checks, builds scorecard and recommendations.
- `auditor/report.py` — HTML report export.
- `tests/test_auditor.py` — 10 automated tests.
- `requirements.txt` — dependencies.

The HTML report can be downloaded from the scorecard page and printed to PDF from the browser.

## Methods and references

- Permutation feature importance — Breiman (2001)
- Monte-Carlo Shapley values — Strumbelj & Kononenko (2014); same idea as SHAP, Lundberg & Lee (2017)
- Counterfactual fairness — Kusner et al. (2017)
- Counterfactual explanations / recourse — Wachter et al. (2017)
- Reweighing — Kamiran & Calders (2012)
- Group-specific thresholds — Hardt, Price & Srebro (2016)
- Impossibility of simultaneous fairness criteria — Kleinberg, Mullainathan & Raghavan (2016); Chouldechova (2017)
- Four-fifths rule — EEOC Uniform Guidelines (1978)
- EU AI Act — Regulation (EU) 2024/1689, Arts. 10, 12-15
- NIST AI Risk Management Framework — AI RMF 1.0 (2023)
- EU HLEG Ethics Guidelines for Trustworthy AI (2019)

## Limitations and future work

Be honest about what this tool does not do:

- The data is synthetic, not real bank data — chosen deliberately so the bias is known and the auditor can be validated against ground truth.
- Shapley values are approximated by sampling; the counterfactual search is greedy and not guaranteed to be minimal.
- The thresholds (0.10, 0.80) are common conventions, not legal certification; the scores and weights are a design choice.
- The audit log is a local file: the hash chain detects tampering but does not prevent deleting the whole file. A production system would anchor hashes externally or use WORM storage.
- Group-specific thresholds may be legally restricted in some jurisdictions.

Future work: real datasets (UCI Adult, German Credit), more protected attributes, LLM decision auditing, continuous monitoring and drift detection, privacy checks (membership inference), and PDF export.
