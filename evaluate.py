"""Full evaluation behind the project report.

Audits three model types, a gender-blind model, the three mitigations, a
fair-history control and the external hiring-tool sample, and prints every
number used in the report.  Usage:  python evaluate.py
"""
import time, warnings; warnings.filterwarnings("ignore")
from auditor.data import make_loan_data, make_hiring_decisions
from auditor.model import train_model
from auditor.audit import run_full_audit
from auditor.mitigation import mitigate_reweighing, mitigate_drop_proxies, mitigate_thresholds
from auditor.fairness import fairness_findings, fairness_score
from auditor.ledger import AuditLedger
df=make_loan_data(); print("rows",len(df), "approval", df.groupby("gender").approved.mean().round(3).to_dict(), df.groupby("age_group").approved.mean().round(3).to_dict())
print("part_time share", df.groupby("gender").part_time.apply(lambda s:(s=="Yes").mean()).round(3).to_dict())
def show(name,r):
    print("\n==",name, r["model_id"], r["notes"]); print("perf",{k:round(v,3) for k,v in r["performance"].items()})
    print("scores",r["scores"],"overall",r["overall"],r["verdict"])
    for f in r["fairness_findings"]: print("  ",f["attribute"],f["key"],f["value"],f["status"])
    print("flip",{k:(round(v["flip_rate"],3),v.get("in_model")) for k,v in r["flip_tests"].items() if v.get("tested")})
    print("proxies g",r["proxies"]["gender"].query("possible_proxy")[["feature","association"]].values.tolist(),"age",r["proxies"]["age_group"].query("possible_proxy")[["feature","association"]].values.tolist())
    print("rob",r["robustness"].round(4).values.tolist(),"boundary",len(r["boundary_cases"]),"fid",round(r["fidelity_error"],4))
    print("top3",list(r["importance"].feature.head(3)))
for mt in ["Logistic Regression","Random Forest","Gradient Boosting"]:
    t=time.time(); m=train_model(df,mt); r=run_full_audit(m,["gender","age_group"]); show(mt+f" ({time.time()-t:.1f}s)",r)
show("GB no gender", run_full_audit(train_model(df,"Gradient Boosting",include_protected=False),["gender","age_group"]))
for nm,fn in [("reweigh",lambda:mitigate_reweighing(m,"gender")),("drop",lambda:mitigate_drop_proxies(m,"gender",["part_time"])),("thr",lambda:mitigate_thresholds(m,"gender"))]:
    show(nm, run_full_audit(fn(),["gender","age_group"]))
show("fair history", run_full_audit(train_model(make_loan_data(bias_strength=0.0),"Gradient Boosting",include_protected=False),["gender","age_group"]))
h=make_hiring_decisions()
for a in ["gender","ethnicity"]:
    f=fairness_findings(h.actually_qualified,h.ai_shortlisted,h[a],a); print(a,fairness_score(f),[(x["key"],x["value"],x["status"]) for x in f])
