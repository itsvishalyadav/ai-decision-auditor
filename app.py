"""AI Decision Auditor - Streamlit dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from auditor.audit import PILLARS, run_full_audit
from auditor.counterfactual import find_counterfactual, protected_flip_test
from auditor.data import ACTIONABLE, TARGET, make_hiring_decisions, make_loan_data
from auditor.explain import explanation_text, shapley_explanation
from auditor.fairness import METRIC_INFO, fairness_findings, fairness_score, group_table
from auditor.ledger import AuditLedger
from auditor.mitigation import mitigate_drop_proxies, mitigate_reweighing, mitigate_thresholds
from auditor.model import MODEL_TYPES, train_model
from auditor.report import build_html_report

ROOT = Path(__file__).parent
PROTECTED = ["gender", "age_group"]
st.set_page_config(page_title="AI Decision Auditor", page_icon="⚖️", layout="wide")


# ---------------------------------------------------------------- helpers
@st.cache_data
def load_data(bias_strength: float) -> pd.DataFrame:
    return make_loan_data(bias_strength=bias_strength)


def ledger() -> AuditLedger:
    return AuditLedger(ROOT / "data" / "audit_log.jsonl")


def get_model():
    if "model" not in st.session_state:
        df = load_data(1.0)
        st.session_state.model = train_model(df, "Gradient Boosting", include_protected=True)
        st.session_state.audits = {}
    return st.session_state.model


def get_audit(model, fast=False):
    key = (model.model_id, len(ledger().records()), ledger().verify()["valid"])
    audits = st.session_state.setdefault("audits", {})
    if key not in audits:
        with st.spinner("Auditing model: fairness, explanations, robustness, accountability..."):
            audits[key] = run_full_audit(model, PROTECTED, ledger=ledger(), fast=fast)
    return audits[key]


def score_color(s):
    return "#1a7f37" if s >= 80 else "#d4a017" if s >= 60 else "#c53030"


def gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={"text": title},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": score_color(value)},
               "steps": [{"range": [0, 60], "color": "#fde2e2"}, {"range": [60, 80], "color": "#fff3cd"},
                         {"range": [80, 100], "color": "#d4edda"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def fmt_val(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:,.3f}".rstrip("0").rstrip(".") if abs(v) < 10 else f"{v:,.0f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return str(v)


def findings_df(findings):
    df = pd.DataFrame(findings)[["attribute", "metric", "value", "threshold", "status", "question"]]
    return df.style.map(lambda s: "color:#1a7f37;font-weight:bold" if s == "PASS"
                        else "color:#c53030;font-weight:bold" if s == "FAIL" else "", subset=["status"])


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚖️ AI Decision Auditor")
st.sidebar.caption("Trustworthy AI course project")
PAGES = ["🏠 Overview", "⚙️ 1. Model under audit", "📋 2. Trust scorecard", "⚖️ 3. Fairness",
         "🔍 4. Explain a decision", "🛡️ 5. Robustness", "🔧 6. Bias mitigation",
         "🔗 7. Audit trail", "📂 8. Audit an external AI"]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
model = get_model()
st.sidebar.divider()
st.sidebar.markdown(f"**Model under audit**  \n{model.name}  \n`{model.model_id}`")
for n in model.notes:
    st.sidebar.caption(f"🔧 {n}")
if "gender" in model.features:
    st.sidebar.caption("⚠️ Uses gender as an input")


# ================================================================ OVERVIEW
if page == PAGES[0]:
    st.title("AI Decision Auditor")
    st.markdown(
        "#### An independent auditor that checks whether an AI system's decisions can be trusted.\n"
        "AI models now decide who gets a **loan, a job interview, insurance or bail**. "
        "These decisions can be **biased, unexplainable, unstable or untraceable**. "
        "This tool audits an AI decision-maker against the principles of Trustworthy AI and produces "
        "a scored audit report with concrete fixes."
    )
    cols = st.columns(5)
    info = {
        "Fairness": ("⚖️", "Are groups (gender, age) treated equally? 5 fairness metrics, flip test, proxy detection."),
        "Transparency": ("🔍", "Why was this decision made? Shapley-value explanations + recourse."),
        "Robustness": ("🛡️", "Do tiny input changes flip the decision?"),
        "Performance": ("🎯", "Is the model accurate enough to be used at all?"),
        "Accountability": ("🔗", "Is every decision logged in a tamper-evident audit trail?"),
    }
    for c, (p, (icon, txt)) in zip(cols, info.items()):
        c.markdown(f"### {icon}\n**{p}**  \n{txt}  \n\n<small>EU AI Act {PILLARS[p][1]}</small>",
                   unsafe_allow_html=True)
    st.divider()
    st.subheader("How the audit works")
    st.graphviz_chart("""
    digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];
      D [label="Historical data\\n(loan applications)"]; M [label="AI model\\n(black box)"];
      P [label="Decisions\\napprove / reject"]; A [label="AUDITOR", fillcolor="#fde68a"];
      F [label="Fairness"]; E [label="Explainability"]; R [label="Robustness"]; L [label="Audit trail"];
      S [label="Trust score\\n+ report", fillcolor="#bbf7d0"]; X [label="Mitigation\\n& re-audit", fillcolor="#fecaca"];
      D->M->P->A; A->F; A->E; A->R; A->L; F->S; E->S; R->S; L->S; S->X; X->M [style=dashed]; }
    """)
    st.subheader("Demo scenario")
    st.markdown(
        "A bank trained an AI on **past loan decisions made by humans**. Those humans were biased against "
        "**women** and **applicants under 25**, and they penalised **part-time work** (which in this data is "
        "mostly done by women). The AI learns these patterns. Our auditor has to discover this bias "
        "*without being told about it*, explain individual decisions, and recommend a fix."
    )
    st.info("Start with **⚙️ 1. Model under audit** in the sidebar, then go through the pages in order.")


# ================================================================ MODEL
elif page == PAGES[1]:
    st.title("⚙️ The AI model under audit")
    bias = st.slider("Bias in the historical training data (0 = fair history, 1 = default, 2 = very biased)",
                     0.0, 2.0, 1.0, 0.25)
    df = load_data(bias)
    c1, c2, c3 = st.columns(3)
    c1.metric("Applications", f"{len(df):,}")
    c2.metric("Approval rate - Male", f"{df[df.gender == 'Male'][TARGET].mean():.1%}")
    c3.metric("Approval rate - Female", f"{df[df.gender == 'Female'][TARGET].mean():.1%}")
    with st.expander("Preview the training data"):
        st.dataframe(df.head(50))
    st.plotly_chart(px.histogram(df, x="credit_score", color=df[TARGET].map({1: "approved", 0: "rejected"}),
                                 facet_col="gender", barmode="overlay", nbins=40,
                                 title="Historical decisions by credit score and gender"))
    st.subheader("Train the AI")
    c1, c2 = st.columns(2)
    mtype = c1.selectbox("Model type", list(MODEL_TYPES), index=2)
    incl = c2.checkbox("Give the model the 'gender' column as input", value=True,
                       help="Untick to see 'fairness through unawareness' - and why it fails because of proxies.")
    if st.button("Train model", type="primary"):
        with st.spinner("Training..."):
            st.session_state.model = train_model(df, mtype, include_protected=incl)
        st.success(f"Trained {mtype}. Now open **📋 2. Trust scorecard**.")
        st.rerun()
    m = st.session_state.model
    st.markdown(f"**Current model:** {m.name} &nbsp; | &nbsp; **inputs:** {', '.join(m.features)}")
    st.json(m.performance())


# ================================================================ SCORECARD
elif page == PAGES[2]:
    st.title("📋 Trust scorecard")
    r = get_audit(model)
    color = score_color(r["overall"] if not r["critical"] else min(r["overall"], 79))
    st.markdown(f"<div style='padding:18px;border-radius:10px;background:{color};color:white;font-size:24px'>"
                f"Verdict: <b>{r['verdict']}</b> &nbsp;·&nbsp; overall {r['overall']:.0f}/100</div>",
                unsafe_allow_html=True)
    if r["critical"]:
        st.caption("A critical issue was found (e.g. 80% rule violated, direct discrimination, or tampered log), "
                   "so the model cannot be rated TRUSTWORTHY regardless of its score.")
    c1, c2 = st.columns([1, 2])
    c1.plotly_chart(gauge(r["overall"], "Overall trust score"))
    sc = pd.DataFrame({"pillar": list(r["scores"]), "score": list(r["scores"].values())})
    fig = px.bar(sc, x="score", y="pillar", orientation="h", range_x=[0, 100], text="score",
                 color="score", color_continuous_scale=["#c53030", "#d4a017", "#1a7f37"], range_color=[0, 100])
    fig.update_layout(height=280, coloraxis_showscale=False, margin=dict(t=20))
    c2.plotly_chart(fig)
    st.subheader("Recommendations")
    for i, rec in enumerate(r["recommendations"], 1):
        st.markdown(f"{i}. {rec}")
    st.subheader("Regulatory mapping")
    st.dataframe(pd.DataFrame([{"Pillar": p, "Score": r["scores"][p], "Weight": f"{w:.0%}", "EU AI Act": eu,
                                "NIST AI RMF": n} for p, (w, eu, n) in PILLARS.items()]), hide_index=True)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download full audit report (HTML)", build_html_report(r),
                       file_name=f"audit_report_{r['model_id']}.html", mime="text/html", type="primary")
    if c2.button("Re-run audit"):
        st.session_state.audits = {}
        st.rerun()


# ================================================================ FAIRNESS
elif page == PAGES[3]:
    st.title("⚖️ Fairness audit")
    r = get_audit(model)
    attr = st.radio("Protected attribute", PROTECTED, horizontal=True)
    t = r["group_tables"][attr]
    c1, c2 = st.columns(2)
    f1 = px.bar(t.reset_index(), x="group", y=["selection_rate", "base_rate"], barmode="group",
                title="AI approval rate vs. true qualification rate", labels={"value": "rate"})
    f1.add_hline(y=t.selection_rate.max() * 0.8, line_dash="dot", annotation_text="80% of best group")
    c1.plotly_chart(f1)
    f2 = px.bar(t.reset_index(), x="group", y=["tpr", "fpr", "precision"], barmode="group",
                title="Error rates per group", labels={"value": "rate"})
    c2.plotly_chart(f2)
    st.subheader("Fairness checks")
    st.dataframe(findings_df([f for f in r["fairness_findings"] if f["attribute"] == attr]), hide_index=True)
    with st.expander("What do these metrics mean?"):
        for k, (title, q) in METRIC_INFO.items():
            st.markdown(f"**{title}** - {q}")
        st.markdown("Note: several of these metrics cannot all be satisfied at once when groups have "
                    "different base rates (Kleinberg et al. 2016; Chouldechova 2017). The auditor reports all of "
                    "them so the humans can decide which notion of fairness matters for the use-case.")
    st.subheader("Counterfactual flip test")
    ft = r["flip_tests"][attr]
    if ft.get("tested"):
        st.metric(f"Decisions that change when ONLY '{attr}' is flipped", f"{ft['flip_rate']:.1%}",
                  help="Direct discrimination: identical applicants, different decision.")
        if not ft["in_model"]:
            st.caption(f"'{attr}' is not a model input, so the flip test is 0 - but bias can still enter "
                       "through proxy features (below).")
    else:
        st.caption(f"Flip test needs a binary attribute; '{attr}' has more than two groups.")
    st.subheader("Proxy detection")
    st.caption("Features strongly associated with the protected attribute can leak it into the model.")
    st.dataframe(r["proxies"][attr], hide_index=True)
    st.subheader("Intersectional view")
    X = model.X_test.copy()
    X["decision"] = model.predict(X)
    inter = X.groupby(["gender", "age_group"])["decision"].agg(["mean", "count"]).reset_index()
    st.plotly_chart(px.bar(inter, x="age_group", y="mean", color="gender", barmode="group",
                           title="Approval rate by gender x age group", labels={"mean": "approval rate"}))


# ================================================================ EXPLAIN
elif page == PAGES[4]:
    st.title("🔍 Explain an individual decision")
    X = model.X_test
    mode = st.radio("Applicant", ["Pick from test set", "Enter manually"], horizontal=True)
    if mode == "Pick from test set":
        preds = model.predict(X)
        only_rej = st.checkbox("Show only rejected applicants", value=True)
        idxs = [i for i in range(len(X)) if (preds[i] == 0 or not only_rej)][:300]
        i = st.selectbox("Applicant #", idxs)
        x = X.iloc[i].copy()
    else:
        base = X.iloc[0]
        c = st.columns(4)
        x = base.copy()
        x["gender"] = c[0].selectbox("gender", ["Male", "Female"])
        x["age"] = c[1].number_input("age", 18, 75, 30)
        x["age_group"] = "Under 25" if x["age"] < 25 else "25-59" if x["age"] < 60 else "60+"
        x["education"] = c[2].selectbox("education", ["High School", "Bachelor", "Master", "PhD"])
        x["part_time"] = c[3].selectbox("part_time", ["No", "Yes"])
        x["income"] = float(c[0].number_input("income", 10000, 400000, 45000, 1000))
        x["credit_score"] = c[1].number_input("credit_score", 300, 850, 650)
        x["loan_amount"] = float(c[2].number_input("loan_amount", 1000, 500000, 30000, 1000))
        x["debt_to_income"] = c[3].number_input("debt_to_income", 0.0, 1.0, 0.3, 0.01)
        x["employment_years"] = c[0].number_input("employment_years", 0, 45, 5)
        x["late_payments"] = c[1].number_input("late_payments", 0, 20, 1)
    st.dataframe(x.to_frame().T, hide_index=True)

    p = float(model.predict_proba(x)[0])
    d = int(model.predict(x)[0])
    c1, c2 = st.columns([1, 2])
    c1.markdown(f"<div style='padding:20px;border-radius:10px;color:white;font-size:26px;text-align:center;"
                f"background:{'#1a7f37' if d else '#c53030'}'>{'APPROVED' if d else 'REJECTED'}<br>"
                f"<span style='font-size:16px'>approval probability {p:.1%}</span></div>", unsafe_allow_html=True)
    review = abs(p - model.threshold) < 0.05
    if review:
        c1.warning("Borderline case: route to a human reviewer.")

    bg = model.X_train.sample(150, random_state=0)
    with st.spinner("Computing Shapley explanation..."):
        e = shapley_explanation(model, x, bg, n_permutations=150)
    base_p = float(model.predict_proba(bg).mean())
    fig = go.Figure(go.Waterfall(
        orientation="h",
        y=["Average applicant"] + [f"{r.feature} = {fmt_val(r.value)}" for r in e.itertuples()] + ["THIS applicant"],
        x=[base_p] + list(e.contribution) + [0], measure=["absolute"] + ["relative"] * len(e) + ["total"],
        decreasing={"marker": {"color": "#c53030"}}, increasing={"marker": {"color": "#1a7f37"}},
        totals={"marker": {"color": "#3b5bdb"}}, connector={"line": {"color": "#999", "dash": "dot"}}))
    fig.add_vline(x=model.threshold, line_dash="dash", annotation_text="approval threshold")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(title="Why? Each factor pushes the approval chance up (green) or down (red)",
                      height=460, margin=dict(t=60))
    c2.plotly_chart(fig)
    reason = explanation_text(e, d)
    st.info(reason)
    if "gender" in list(e.feature.head(3)):
        st.error("⚠️ 'gender' is one of the top 3 reasons for this decision - this is direct discrimination.")

    st.subheader("Recourse: what would change the decision?")
    cf = find_counterfactual(model, x)
    if cf["success"] and cf["changes"]:
        st.success("The decision would flip to **" + ("APPROVED" if cf["target"] else "REJECTED") +
                   f"** (probability {cf['new_probability']:.1%}) with these changes:")
        st.table(pd.DataFrame([{"feature": c["feature"], "current": fmt_val(c["from"]),
                                "needed": fmt_val(c["to"])} for c in cf["changes"]]))
    else:
        st.warning("No realistic change of the actionable features flips this decision.")
    st.caption("Only actionable features are changed: " + ", ".join(ACTIONABLE) +
               ". Gender, age and education are never suggested.")

    st.subheader("Same person, different gender?")
    xf = x.copy()
    xf["gender"] = "Female" if x["gender"] == "Male" else "Male"
    pf, df_ = float(model.predict_proba(xf)[0]), int(model.predict(xf)[0])
    st.write(f"If this applicant were **{xf['gender']}**, approval probability would be **{pf:.1%}** "
             f"({'APPROVED' if df_ else 'REJECTED'}).")
    if df_ != d:
        st.error("The decision changes with gender alone - this individual is treated unfairly.")

    if st.button("📝 Record this decision in the audit trail", type="primary"):
        rec = ledger().append(model.model_id, x.to_dict(), d, p, reason, reviewer_flag=review)
        st.success(f"Logged as record #{rec['index']} with hash `{rec['hash'][:16]}...`")


# ================================================================ ROBUSTNESS
elif page == PAGES[5]:
    st.title("🛡️ Robustness & stability")
    r = get_audit(model)
    rob = r["robustness"].copy()
    rob["noise"] = (rob.noise_level * 100).map("{:.0f}% of std-dev".format)
    fig = px.line(rob, x="noise", y="flip_rate", markers=True, title="Share of decisions that flip under small random noise")
    fig.add_hline(y=0.03, line_dash="dot", annotation_text="limit at 2% noise")
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig)
    st.markdown("A trustworthy model should give the **same answer** if income is 52,000 or 52,300. "
                "Flips under tiny noise mean the decision is close to arbitrary for those people.")
    st.subheader(f"Borderline decisions ({len(r['boundary_cases'])})")
    st.caption("Applicants whose approval probability is within 5% of the threshold. Human oversight "
               "(EU AI Act Art. 14) suggests these should be reviewed by a person.")
    st.dataframe(r["boundary_cases"].round(3), hide_index=True)
    st.subheader("Global feature importance")
    st.plotly_chart(px.bar(r["importance"], x="importance", y="feature", orientation="h", error_x="std",
                           title="Drop in ROC-AUC when a feature is shuffled").update_yaxes(autorange="reversed"))


# ================================================================ MITIGATION
elif page == PAGES[6]:
    st.title("🔧 Bias mitigation")
    st.markdown("The auditor does not just find problems - it tests fixes and **re-audits** them.")
    attr = "gender"
    methods = {
        "Pre-processing: Reweighing (Kamiran & Calders 2012)": lambda m: mitigate_reweighing(m, attr),
        "In-processing: remove gender AND its proxy 'part_time'": lambda m: mitigate_drop_proxies(m, attr, ["part_time"]),
        "Post-processing: group-specific thresholds (Hardt et al. 2016)": lambda m: mitigate_thresholds(m, attr),
    }
    choice = st.selectbox("Mitigation technique", list(methods))
    base = st.session_state.get("baseline_model", model)
    if st.button("Apply and re-audit", type="primary"):
        with st.spinner("Retraining and re-auditing..."):
            st.session_state.baseline_model = base
            st.session_state.mitigated = methods[choice](base)
    if "mitigated" in st.session_state:
        mm = st.session_state.mitigated
        rb, rm = get_audit(base, fast=True), get_audit(mm, fast=True)
        rows = []
        for f0, f1 in zip(rb["fairness_findings"], rm["fairness_findings"]):
            if f0["attribute"] == attr:
                rows.append({"metric": f0["metric"], "before": f0["value"], "after": f1["value"],
                             "status before": f0["status"], "status after": f1["status"]})
        rows.append({"metric": "Accuracy", "before": round(rb["performance"]["accuracy"], 3),
                     "after": round(rm["performance"]["accuracy"], 3), "status before": "", "status after": ""})
        rows.append({"metric": "Overall trust score", "before": rb["overall"], "after": rm["overall"],
                     "status before": rb["verdict"].split(" - ")[0], "status after": rm["verdict"].split(" - ")[0]})
        st.caption(" · ".join(mm.notes))
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        comp = pd.concat([rb["group_tables"][attr].assign(model="before"),
                          rm["group_tables"][attr].assign(model="after")]).reset_index()
        st.plotly_chart(px.bar(comp, x="group", y="selection_rate", color="model", barmode="group",
                               title="Approval rate by gender, before vs after"))
        st.info("Trade-offs are expected: equalising approval rates can lower accuracy or worsen another "
                "fairness metric (impossibility theorem). The auditor makes the trade-off visible.")
        if st.button("✅ Deploy the mitigated model (make it the model under audit)"):
            st.session_state.model = mm
            st.session_state.pop("mitigated")
            st.session_state.pop("baseline_model", None)
            st.rerun()


# ================================================================ AUDIT TRAIL
elif page == PAGES[7]:
    st.title("🔗 Tamper-evident audit trail")
    st.markdown("Every recorded decision stores the **SHA-256 hash of the previous record**. Editing any past "
                "record breaks the chain, so tampering is detected (EU AI Act Art. 12 - record-keeping).")
    L = ledger()
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("➕ Log 5 random test decisions"):
        for _, x in model.X_test.sample(5).iterrows():
            p = float(model.predict_proba(x)[0])
            d = int(model.predict(x)[0])
            e = shapley_explanation(model, x, model.X_train.sample(80, random_state=0), n_permutations=40)
            L.append(model.model_id, x.to_dict(), d, p, explanation_text(e, d), abs(p - model.threshold) < 0.05)
    if c2.button("🔎 Verify integrity"):
        st.session_state.verify_clicked = True
    if c3.button("😈 Simulate tampering"):
        L.tamper(index=max(0, len(L.records()) // 2))
        st.warning("Someone secretly changed a past decision in the log file...")
    if c4.button("🗑️ Reset log"):
        L.clear()
    v = L.verify()
    if v["valid"]:
        st.success(f"✅ Chain valid - {len(L.records())} records, {v['reason']}.")
    else:
        st.error(f"❌ TAMPERING DETECTED at record #{v['broken_at']}: {v['reason']}.")
    recs = L.records()
    if recs:
        df = pd.DataFrame([{**{k: r[k] for k in ["index", "timestamp", "model_id", "decision", "probability",
                                                 "needs_human_review", "explanation"]},
                            "prev_hash": r["prev_hash"][:12] + "...", "hash": r["hash"][:12] + "..."} for r in recs])
        st.dataframe(df, hide_index=True)
    else:
        st.caption("No decisions logged yet. Use the button above or the 'Explain a decision' page.")


# ================================================================ EXTERNAL
elif page == PAGES[8]:
    st.title("📂 Audit an external AI system")
    st.markdown("Audit **any** AI system as a black box - you only need a CSV of its decisions. Example: an "
                "AI CV-screening tool that shortlisted job candidates.")
    up = st.file_uploader("Upload decisions CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
    else:
        df = make_hiring_decisions()
        st.caption("No file uploaded - using the sample AI hiring-tool decisions (sample_data/hiring_decisions.csv).")
    st.dataframe(df.head(20))
    cols = list(df.columns)
    c1, c2, c3 = st.columns(3)
    pred_col = c1.selectbox("AI decision column (0/1)", cols, index=cols.index("ai_shortlisted") if "ai_shortlisted" in cols else 0)
    true_col = c2.selectbox("True outcome column (0/1)", cols, index=cols.index("actually_qualified") if "actually_qualified" in cols else 0)
    prot = c3.multiselect("Protected attributes", cols, default=[c for c in ["gender", "ethnicity"] if c in cols])
    if st.button("Run audit", type="primary") and prot:
        allf = []
        for a in prot:
            f = fairness_findings(df[true_col], df[pred_col], df[a], a)
            allf += f
            st.subheader(f"{a}  -  fairness score {fairness_score(f):.0f}/100")
            t = group_table(df[true_col], df[pred_col], df[a])
            st.plotly_chart(px.bar(t.reset_index(), x="group", y=["selection_rate", "base_rate", "tpr"],
                                   barmode="group", title=f"Selection rate vs. qualification by {a}"))
        st.dataframe(findings_df(allf), hide_index=True)
        n_fail = sum(f["status"] == "FAIL" for f in allf)
        (st.error if n_fail else st.success)(f"{n_fail} of {len(allf)} fairness checks failed.")
