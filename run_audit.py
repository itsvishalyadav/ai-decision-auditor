"""Command-line audit: trains the demo model, audits it, applies a mitigation,
re-audits, and writes HTML reports to ./reports.

Usage:  python run_audit.py [--model "Gradient Boosting"] [--no-gender]
"""
import argparse
import warnings
from pathlib import Path

from auditor.audit import run_full_audit
from auditor.data import make_hiring_decisions, make_loan_data
from auditor.ledger import AuditLedger
from auditor.mitigation import mitigate_reweighing
from auditor.model import MODEL_TYPES, train_model
from auditor.report import build_html_report

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent


def summary(title, r):
    print(f"\n=== {title} ===")
    print(f"Model {r['model_name']} [{r['model_id']}]  {' '.join(r['notes'])}")
    print(f"Overall trust score: {r['overall']:.0f}/100  ->  {r['verdict']}")
    for p, s in r["scores"].items():
        print(f"  {p:<15}{s:>6.1f}  " + "#" * int(s / 5))
    print("Fairness checks:")
    for f in r["fairness_findings"]:
        print(f"  [{f['status']}] {f['attribute']:<10} {f['metric']:<30} {f['value']}")
    print("Recommendations:")
    for x in r["recommendations"]:
        print("  -", x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Gradient Boosting", choices=list(MODEL_TYPES))
    ap.add_argument("--no-gender", action="store_true", help="do not give gender to the model")
    a = ap.parse_args()

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "sample_data").mkdir(exist_ok=True)
    df = make_loan_data()
    df.to_csv(ROOT / "sample_data" / "loan_applications.csv", index=False)
    make_hiring_decisions().to_csv(ROOT / "sample_data" / "hiring_decisions.csv", index=False)

    ledger = AuditLedger(ROOT / "data" / "audit_log.jsonl")
    model = train_model(df, a.model, include_protected=not a.no_gender)
    before = run_full_audit(model, ["gender", "age_group"], ledger=ledger)
    summary("BEFORE mitigation", before)
    (ROOT / "reports" / "audit_before.html").write_text(build_html_report(before), encoding="utf-8")

    fixed = mitigate_reweighing(model, "gender")
    after = run_full_audit(fixed, ["gender", "age_group"], ledger=ledger)
    summary("AFTER reweighing", after)
    (ROOT / "reports" / "audit_after.html").write_text(build_html_report(after), encoding="utf-8")
    print("\nReports written to", ROOT / "reports")


if __name__ == "__main__":
    main()
