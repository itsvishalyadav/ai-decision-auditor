# Presentation Guide: "AI Decision Auditor"

**Course:** Trustworthy Artificial Intelligence
**Audience:** Professor examining your college project
**Time:** ~15 minutes presentation + 5–7 minute live demo

---

## Part 1 – Slide Deck Outline

### Slide 1 – Title

- **AI Decision Auditor**
- Auditing AI decisions against the pillars of Trustworthy AI
- Course: Trustworthy Artificial Intelligence
- Your names, date

**Say:** Good morning. Our project is called the AI Decision Auditor. The big idea is simple: AI systems now make decisions about people — who gets a loan, who gets shortlisted for a job — but how do we know those decisions are fair and safe? Our tool audits an AI system's decisions, scores them against the pillars of Trustworthy AI, and suggests tested fixes. Today we'll show the problem, our solution, live results, and an honest list of limitations. Everything you'll see was built from scratch in Python, so we can explain every number on screen.

---

### Slide 2 – The Problem: AI Decides, But Who Checks the AI?

- AI now decides loans, hiring shortlists, insurance, bail
- Real failures: Amazon recruiting tool penalised CVs with "women's" (2018)
- COMPAS: higher false-positive rates for Black defendants (ProPublica, 2016)
- Apple Card (2019): allegations of lower credit limits for women

**Say:** AI is already making high-stakes decisions. And this isn't theoretical — it has gone wrong in public. Amazon built an experimental recruiting tool that reportedly downgraded CVs containing the word "women's". ProPublica's 2016 analysis found the COMPAS recidivism tool produced higher false-positive rates for Black defendants. And Apple Card faced allegations of giving women lower credit limits. The pattern in all three cases: the AI learned bias from historical data, and nobody was auditing the decisions. That's the gap our project addresses.

---

### Slide 3 – What Is "Trustworthy AI"? The Frameworks

- EU HLEG Ethics Guidelines (2019): 7 requirements — oversight, robustness, transparency, fairness, accountability...
- EU AI Act (2024/1689): credit scoring & recruitment = high-risk; Arts. 10, 12–15
- NIST AI RMF (2023): valid & reliable, fair, explainable, accountable, private
- Our tool turns these principles into measurable checks and a score

**Say:** "Trustworthy AI" isn't just a buzzword — there are official frameworks. The EU's High-Level Expert Group published Ethics Guidelines in 2019 with seven requirements, including human oversight, robustness, transparency, fairness and accountability. The EU AI Act of 2024 classifies credit scoring and recruitment as high-risk systems, with specific articles on data governance, logging, transparency, human oversight, and accuracy. In the US, NIST's AI Risk Management Framework lists similar characteristics. The problem: these are principles on paper. Our contribution is turning them into concrete, measurable checks with a score.

---

### Slide 4 – Our Solution & Architecture

- Data (4,000 synthetic loan applications) → model training
- Model makes 1,200 test decisions → Auditor checks them
- 5 pillars → weighted Trust Score (0–100) → verdict
- Mitigation (reweighing, etc.) → re-audit → deploy button
- Also: black-box mode — upload any AI's decisions as CSV

**Say:** Here's the pipeline. We generate 4,000 synthetic loan applications where the historical decisions were made by biased humans — they penalised women, applicants under 25, and part-time workers. We train a loan-approval model on that history, and it makes 1,200 test decisions. The auditor then checks those decisions on five pillars, combines them into a weighted trust score, and gives a verdict: Trustworthy, Conditional, or Not Trustworthy. If it fails, we apply a mitigation and re-audit to prove the fix works. There's also a black-box mode: upload a CSV of any external AI's decisions and get a fairness audit.

---

### Slide 5 – Fairness Metrics, in Plain English

- Demographic parity difference: gap in approval rates between groups
- Disparate impact ratio: lowest rate ÷ highest rate; must be ≥ 0.80 (four-fifths rule)
- Equal opportunity: gap in true positive rates — qualified people treated equally
- Equalized odds: TPR and FPR gaps both small
- Predictive parity: gap in precision
- Plus: flip test, proxy detection, intersectional view

**Say:** Fairness has to be measured, so we use five standard group metrics. Demographic parity asks: is the approval rate similar across groups? Disparate impact ratio divides the lowest group's rate by the highest — if it's below 0.80, that's the "four-fifths rule" from the 1978 EEOC guidelines, evidence of adverse impact. Equal opportunity asks: among truly qualified applicants, are both groups approved equally? Equalized odds adds the false-positive side, and predictive parity compares precision. On top of these, we run a counterfactual flip test — change only gender, see how many decisions change — and we detect proxy features like part-time work that correlate with gender.

---

### Slide 6 – Explainability: Why Was I Rejected?

- Permutation importance: which features matter globally
- Shapley waterfall: from "average applicant" to "this applicant"
- Plain-language reason, like an adverse-action notice
- Recourse: smallest realistic change that flips the decision
- Penalised if gender or a proxy is a top-3 feature

**Say:** A fair AI must also explain itself. Globally, we use permutation feature importance: shuffle one feature, see how much performance drops. Locally, we compute model-agnostic Shapley values by sampling, and show them as a waterfall chart — starting from the average applicant and adding up each factor until we reach this person's prediction. We also generate a plain-language reason sentence, like the adverse-action notices banks must send. And for rejected applicants, we compute recourse: the smallest realistic change — say, raising the credit score — that would flip the decision to approved. Importantly, we never suggest changing gender or age; only actionable features.

---

### Slide 7 – Robustness & Human Oversight

- Gaussian noise at 1%, 2%, 5%, 10% of each feature's spread
- Pass: ≤ 3% of decisions flip at 2% noise
- Borderline decisions (within 5% of the threshold) flagged for human review
- Maps to EU AI Act Art. 14 (human oversight)

**Say:** A trustworthy model shouldn't change its mind because of tiny input noise. We perturb the numeric features with Gaussian noise at increasing levels and count how many decisions flip. Our pass limit is at most 3% flips at 2% noise. Separately, we flag borderline decisions — those within 5% of the decision threshold — for human review. This directly implements the human oversight requirement, Article 14 of the EU AI Act: the AI shouldn't be the final word on cases it's unsure about.

---

### Slide 8 – Accountability: A Tamper-Evident Audit Trail

- Every decision logged: inputs, decision, probability, explanation, timestamp
- Each record stores the SHA-256 hash of the previous one → hash chain
- "Verify" button recomputes the whole chain
- "Simulate tampering" — we'll show it detects the exact modified record
- Each model has a 12-character fingerprint

**Say:** Accountability means you can prove later what the AI decided and why. Every decision is logged with its inputs, output, probability, explanation, and timestamp. Each record also stores the SHA-256 hash of the previous record — a hash chain, like a blockchain without the network. If anyone edits a past record, the hashes stop matching. There's a Verify button that recomputes the chain, and — because we're honest about needing to test this — a Simulate Tampering button that secretly edits an old decision. The tool then reports exactly which record was changed. We'll demo this live.

---

### Slide 9 – Results: Before vs. After Mitigation

- Baseline: accuracy 87%, but 5/5 gender fairness checks FAIL
- Disparate impact 0.568; 21.8% of decisions flip on gender change
- Overall trust score ~57/100 → NOT TRUSTWORTHY
- After Reweighing: DI 0.825 PASS, transparency 90.5, overall 74 → CONDITIONAL
- Accuracy drops 87.0% → 84.9%

**Say:** Now the numbers. Baseline, with gender used as an input, accuracy is 87% and ROC-AUC is 0.948 — it looks like a good model. But all five gender fairness checks fail. Disparate impact is 0.568, far below the 0.80 rule. Changing only gender flips 21.8% of decisions. The tool detects part-time work as a gender proxy, with Cramer's V of 0.374. Overall score: about 57 out of 100 — Not Trustworthy, do not deploy. After applying Reweighing, disparate impact rises to 0.825 and passes, transparency jumps to 90.5, and the overall score becomes 74 — Conditional. Accuracy drops slightly from 87 to 84.9% — fairness costs a little accuracy.

---

### Slide 10 – The Fairness Trade-Off (Impossibility)

- Different base rates → can't satisfy all metrics at once
- Kleinberg et al. 2016; Chouldechova 2017
- Group thresholds: parity diff 0.031 and DI 0.948 — but equal opportunity worsens to 0.192
- So the auditor reports trade-offs instead of pretending

**Say:** Here's a deep result we hit in practice. When groups have different base rates, calibration, equal error rates, and other fairness criteria mathematically cannot all hold at the same time — proven by Kleinberg and colleagues in 2016 and Chouldechova in 2017. We saw it ourselves: using group-specific thresholds, we achieved near-perfect demographic parity — a difference of 0.031 and a DI ratio of 0.948 — but equal opportunity got worse, at 0.192, and accuracy fell to 81.4%. The lesson: there's no magic "fair" button. That's why our tool shows all metrics and the trade-offs, rather than pretending one number solves everything.

---

### Slide 11 – Black-Box Audit of an External Hiring AI

- Upload CSV: decisions, true outcomes, protected attributes
- Sample: AI CV-screening tool, 1,500 candidates, gender & ethnicity
- Result: 9 of 10 fairness checks fail
- Gender fairness 27/100, ethnicity 52/100

**Say:** The tool isn't limited to models we train ourselves. In black-box mode, anyone can upload a CSV of an external AI's decisions — a decision column, a true-outcome column, and protected attributes. As a sample, we included the decisions of an AI CV-screening tool for 1,500 candidates, with gender and ethnicity recorded. The audit found 9 of 10 fairness checks failing. The gender fairness score was 27 out of 100 and ethnicity 52 out of 100. This is the scenario regulators care about: a company deploying a third-party AI can audit its behaviour without access to the model internals.

---

### Slide 12 – Limitations & Future Work

- Synthetic data (chosen so the true bias is known); Shapley approximated by sampling
- Greedy counterfactuals — not guaranteed minimal
- Thresholds are conventions, not legal certification
- Audit log: detects tampering, doesn't prevent file deletion
- Future: real datasets (Adult, German Credit), drift detection, privacy checks

**Say:** We want to be honest about limits. The data is synthetic — deliberately, because we know exactly what bias was planted, so we can verify the auditor catches it. Shapley values are approximated by sampling, and the counterfactual search is greedy, so it finds a good change but not provably the smallest one. Our thresholds like 0.10 and 0.80 are common conventions, not legal certification. And the hash chain detects tampering but can't stop someone deleting the whole file — production systems would anchor hashes externally. Future work includes real datasets like UCI Adult and German Credit, continuous monitoring, drift detection, and privacy checks.

---

### Slide 13 – Conclusion

- Turned Trustworthy AI principles into measurable, scored checks
- Detected, explained, and mitigated bias — with verified numbers
- Tamper-evident accountability and black-box auditing
- Built from scratch: every number explainable
- 10 automated tests pass

**Say:** To sum up: we took the abstract principles of Trustworthy AI — from the EU HLEG, the EU AI Act, and NIST — and turned them into measurable checks with a single trust score. We showed the tool detect real bias patterns, explain individual decisions in plain language, mitigate the bias, and prove the fix with a re-audit. We added tamper-evident accountability and the ability to audit external black-box systems. And we built everything from scratch — only scikit-learn, pandas, numpy, plotly, and Streamlit — backed by 10 automated tests. Thank you. Now we'd like to show it live.

---

## Part 2 – Live Demo Script (5–7 minutes)

**Start the app first, before the presentation:**

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

It opens at http://localhost:8501.

*Note to presenters: the numbers below come from verified runs with Gradient Boosting and default settings. If you changed the model, bias slider, or random settings, your numbers will differ slightly — that's fine, explain that the tool reflects the actual data and settings.*

**1. Overview (30 sec).** "This is the AI Decision Auditor dashboard. A Gradient Boosting model with gender as an input is trained automatically when the app opens, so we're ready to go straight away. The sidebar is for navigation between the pages, and it always shows the name and 12-character fingerprint of the current model under audit — so at any point we know exactly which model made the decisions we're looking at. We'll walk through the pages in order."

**2. Model under audit (40 sec).** "This page sets up the data and model. The bias slider controls the bias strength of the historical data — we leave it at default. The data is 4,000 synthetic loan applications, and the historical human decisions were biased against women, under-25s, and part-time workers. You can see the approval rates by gender in the historical data: **68.8% for men versus 39.9% for women**, shown in the chart of historical decisions. Notice that part-time work is much more common among women — 45% versus 10% — so it acts as a proxy for gender. There's a 'Model type' dropdown — Gradient Boosting — and a checkbox 'Give the model the gender column as input', which is ticked. The default Gradient Boosting model is already loaded, so clicking **Train model** is optional for the demo — if we do click it, we keep Gradient Boosting and the checkbox ticked, which reproduces the numbers you'll see. The model makes 1,200 test decisions that we'll now audit."

**3. Trust scorecard (40 sec).** "Here's the headline result. The overall trust score is about **57 out of 100**, and the verdict is **NOT TRUSTWORTHY — do not deploy**. Look at the pillar breakdown: Performance is 100 and Accountability is 100, but Fairness is only about 15. This is the key insight — a model can look excellent on accuracy, 87%, and still be unfair. There's also a button here to download the full HTML report."

**4. Fairness (60 sec).** "Let's see why Fairness failed. Every one of the five gender metrics fails — for example, disparate impact ratio is **0.568**, well below the 0.80 four-fifths rule. The least favoured group is Female. For age, the least favoured group is Under 25. Scroll down: the counterfactual flip test shows changing only gender flips **21.8%** of decisions — direct evidence the model depends on gender. And the proxy detector flagged `part_time` with Cramer's V of 0.374. So even if we hid gender, the model could still discriminate through the proxy."

**5. Explain a decision (60 sec).** "The page opens with 'Pick from test set' and 'Show only rejected applicants' ticked, and the default **Applicant # 1** is exactly the rejected female applicant we want — age 30, income 28,600, credit score 620 — so we leave her selected. Point at the waterfall chart. This is a Monte-Carlo Shapley waterfall. Her approval probability is just **4.3%**. The biggest negative bars: credit score contributed about **−21%**, income **−18.5%**, and gender = Female **−11%**. Below is a plain-language reason sentence, like an adverse-action notice. The 'same person, different gender' check: identical applicant as male gets **16.7%**. Scroll to recourse: if she raises her credit score from 620 to 725 and clears her one late payment, the decision flips to APPROVED at **55.7%**. That's actionable feedback, and we never suggest changing gender or age."

**6. Robustness (30 sec).** "Robustness: the tool adds noise at 1, 2, 5 and 10% of each feature's spread and counts flipped decisions. It also lists borderline decisions — within 5% of the threshold — flagged for human review, implementing Article 14 of the EU AI Act's human oversight requirement."

**7. Bias mitigation (60 sec).** "Here's the fix. In the 'Mitigation technique' dropdown, Reweighing is already selected — click **Apply and re-audit**. The before/after table shows disparate impact going from 0.568 to **0.825 — now passing**. Equal opportunity drops to 0.043 — passing. Accuracy dips from 87.0% to 84.9%, and the overall trust score rises from about 57 to **74 — CONDITIONAL**, because a couple of metrics like predictive parity still fail. This also demonstrates the impossibility result: not every fairness metric can pass at once. The other two techniques — removing gender plus part-time, and group thresholds — are chosen from the same dropdown if the professor asks about them. There's also a button, **Deploy the mitigated model (make it the model under audit)** — deploying is optional, and note that if we do, the numbers on the other pages change to reflect the mitigated model."

**8. Audit trail (40 sec).** "Accountability. Every decision is logged and hash-chained — each record stores the SHA-256 hash of the previous one. The log may be empty at this point, so first click **Log 5 random test decisions** — five decisions from the test set are appended to the chain. The chain status is shown automatically: point at the green **Chain valid** message. Now the fun part: click **Simulate tampering**. The tool secretly edits an old record, and a red **TAMPERING DETECTED at record #N** message appears — in our run it was record #3, though the number depends on how many records are in the log. Look at the tampered record: its decision was flipped (APPROVED became REJECTED or the other way round), so it no longer matches its probability and explanation — that's exactly the kind of manipulation the chain catches. After the demo, click **Reset log** to clean up."

**9. Audit an external AI (30 sec).** "Finally, black-box mode. If no file is uploaded, the sample hiring dataset loads automatically — 1,500 candidates screened by an external AI CV-screening tool. Click **Run audit**, and it finds **9 of 10 fairness checks failing**, gender fairness at **27/100**, ethnicity at **52/100** — without any access to that model's internals. That closes the demo. Thank you — happy to take questions."

---

## Part 3 – Likely Professor Questions & Model Answers

**Q1: Why use synthetic data instead of a real dataset?**
Because we plant the bias ourselves, we know the ground truth. If the historical decisions penalise women, under-25s and part-time workers, and the auditor finds exactly that, the auditor is validated. Real datasets like UCI Adult are on our future-work list, but their true bias is unknown, so you can't verify detection.

**Q2: What is disparate impact, exactly?**
It's the ratio of the lowest group's selection rate to the highest group's. If women are approved at 40% and men at 69%, the ratio is about 0.58. The US EEOC's four-fifths rule from 1978 says a ratio below 0.80 is evidence of adverse impact. In our baseline it was 0.568 — a clear fail.

**Q3: Why not just remove gender from the inputs?**
Because of proxies. In our data, part-time work is 45% common among women versus 10% among men, so the model can reconstruct gender from `part_time`. We detect this with Cramer's V — flagged at 0.374. In fact, removing gender plus `part_time` is one of our tested mitigations: it raised disparate impact to 0.896, at a small accuracy cost.

**Q4: What are Shapley values?**
A concept from game theory: fairly split a prediction among the features, based on each feature's average marginal contribution across all possible orderings. Exact computation is exponential, so we use the Monte-Carlo approximation of Strumbelj and Kononenko — the same idea as SHAP. We verify fidelity: the contributions must sum to the prediction minus the average prediction.

**Q5: Why can't all fairness metrics pass at the same time?**
It's a proven impossibility. When groups have different base rates, calibration and equal error rates cannot hold simultaneously — Kleinberg et al. 2016 and Chouldechova 2017. We observed it directly: group-specific thresholds gave near-perfect demographic parity (0.031 difference) but worsened equal opportunity to 0.192. Fairness requires choosing which definition matters for the context.

**Q6: How does the hash chain detect tampering?**
Each log record stores the SHA-256 hash of the previous record. Change any past record — even one character — and its hash changes, so every later link no longer matches. Verification recomputes the whole chain and reports the exact broken record. It's like a blockchain without the distributed network.

**Q7: How is this different from IBM AIF360 or Microsoft Fairlearn?**
Those are excellent toolkits we drew inspiration from. The difference: we implemented the core methods — fairness metrics, Shapley, recourse, reweighing, thresholds, the audit ledger — from scratch using only scikit-learn, pandas, numpy, plotly and Streamlit. So every number is fully explainable, and we integrate everything into one scored audit with a verdict, mitigation loop, and black-box mode, rather than separate libraries.

**Q8: Is a trust score of 74 "good enough" to deploy?**
By our rubric, 60–80 means CONDITIONAL — fix remaining issues before deployment. After reweighing, predictive parity and equalized odds still fail. The score and its weights (Fairness 35%, Transparency 20%, and 15% each for the rest) are a transparent design choice, not a legal certification.

**Q9: What is a "critical issue" that overrides the score?**
Three conditions block a TRUSTWORTHY verdict regardless of the score: violating the 80% rule, more than 5% of decisions flipping when only the protected attribute changes, or a tampered audit log. In our baseline, both of the first two trigger: disparate impact is 0.568 and 21.8% of decisions flip on gender.

**Q10: What are the main limitations?**
Synthetic data; Shapley values approximated by sampling; greedy — not provably minimal — counterfactuals; thresholds that are conventions rather than law; a local audit log that detects tampering but not deletion of the whole file; and group-specific thresholds that may be legally restricted in some jurisdictions.

**Q11: What does the counterfactual flip test actually prove?**
It tests counterfactual fairness in the spirit of Kusner et al. 2017: take each applicant, change only gender, re-predict. If 21.8% of decisions change, the decision depends directly on gender, which is strong evidence of direct discrimination — beyond what group statistics show.

**Q12: How did you validate correctness?**
Ten automated pytest tests covering known metric values, bias detection, no false alarms on fair data, Shapley additivity, that the counterfactual actually flips the decision, the flip test, reweighing achieving statistical independence, mitigation improving fairness, and tamper detection. All ten pass, and the command-line script reproduces the scorecard in the HTML reports.

---

*Good luck — you know the tool, the numbers, and the honest limitations. That last part impresses examiners most.*
