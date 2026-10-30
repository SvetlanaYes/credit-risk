"""V8 — Monitoring specification and validation conclusion. Not a live system."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import PROJECT_ROOT

HOLD = PROJECT_ROOT / "reports" / "validation" / "holdout" / "holdout_summary.json"
CUT = PROJECT_ROOT / "reports" / "validation" / "cutoffs" / "cutoff_summary.json"
OUT = PROJECT_ROOT / "reports" / "validation" / "08_monitoring_and_conclusion.md"


def main() -> None:
    hold = json.loads(HOLD.read_text())
    cut = json.loads(CUT.read_text()) if CUT.exists() else {"rows": []}
    c = hold["candidate"]
    r = hold["constant_development_rate_reference"]
    top10 = next((row for row in cut.get("rows", []) if row["rule"].startswith("top_10")), None)

    top10_line = ""
    if top10:
        top10_line = (
            f"Example (not a policy): flagging development-OOF top 10% on test "
            f"(threshold {top10['threshold']:.3f}) flags {top10['flag_rate']:.1%} of rows, "
            f"captures {top10['capture_rate']:.1%} of labelled defaults, and leaves "
            f"event rate {top10['event_rate_below_cutoff']:.3f} below the cutoff."
        )

    text = f"""# V8 — Monitoring spec and conclusion

**Production decision:** **not approved.**  
This pack is a **specification** of what would be watched if the bank later used a score like this. It is not a live monitoring service.

All High findings VAL-01–VAL-05 remain **open**. VAL-10: this is not organizationally independent validation.

---

## What was shown

Frozen logistic `C=0.1`, 14 inputs including `interest_rate`, recovery excluded.

| | Frozen model | Constant development rate |
| --- | ---: | ---: |
| Holdout ROC AUC | {c['roc_auc']:.4f} | {r['roc_auc']:.4f} |
| Holdout AP | {c['average_precision']:.4f} | {r['average_precision']:.4f} |
| Holdout log loss | {c['log_loss']:.4f} | {r['log_loss']:.4f} |
| Holdout Brier | {c['brier_score']:.4f} | {r['brier_score']:.4f} |

Development OOF AUC was 0.717. Holdout AUC is 0.718 on 4,000 rows / 629 events. Ranking of the **supplied label** on this mixture is modest and stable versus a constant rate. Mean $\\hat{{p}}$ 0.160 vs observed 0.157. Score bands tracked observed rates.

That does **not** make $\\hat{{p}}$ a verified origination PD:

$$
\\hat{{p}}(x)=\\widehat{{P}}(Y_{{\\mathrm{{supplied}}}}=1\\mid X=x)
$$

not

$$
P\\big(Y_{{(t_0,\\,t_0+H]}}=1\\mid X(t_0),\\;\\text{{eligible population}}\\big).
$$

Segments: block 2 ranked higher than block 1; `past_due_90>past_due_60` was weak; age 99 ranking withheld. Coefficients: `past_default` and `dti` dominate the scaled log-odds; `interest_rate` is small and still covered by VAL-03.

{top10_line}

---

## Monitoring we would specify (if used later)

Baselines are **development**. Do not retune alerts on the holdout.

| Watch | Why | Starting baseline |
| --- | --- | --- |
| Volume and mix of scored applications | Population shift (VAL-05) | Development vs any later book |
| Missingness of the 14 inputs, especially income | VAL-07 | Development missing rates |
| Age = 99 rate | VAL-08 | Development share |
| Factor-ten DTI / loan / income breaks | VAL-04 | 73 rows in the full file |
| `past_due_90 > past_due_60` rate | VAL-09 | Development share |
| `interest_rate` distribution and missingness | VAL-03 | Development 0.03–0.24 range (units unverified) |
| Mean $\\hat{{p}}$ vs matured observed rate | Calibration | Development rate ~0.157 |
| Log loss / Brier / AUC once labels exist | Skill of the supplied event | Holdout log loss {c['log_loss']:.4f}, AUC {c['roc_auc']:.4f} |
| Share flagged at the **predeclared** OOF top-decile rule | Workload drift | Test flag rate in the V7 table |
| Recovery-like fields entering the pipeline | VAL-06 | Must stay excluded |

Traffic-light idea (not bank policy): investigate if missingness or age-99 rate doubles, if mean $\\hat{{p}}$ and observed rate diverge by a large persistent gap, or if a recovery-style column appears. Exact thresholds belong to the bank.

---

## What would have to close before production

1. Dictionary for the event, horizon $H$, $t_0$, and incomplete outcomes (VAL-01).  
2. As-of lineage for every input, especially delinquency history (VAL-02).  
3. Confirmation that `interest_rate` is known at the intended decision point and is not circular with pricing (VAL-03).  
4. DTI formula and the 73 factor-ten rows (VAL-04).  
5. Population, row unit, IDs, dates (VAL-05).  
6. A reviewer who did not train the model (VAL-10).

Until then the honest line is: **useful ranking of the file label on a random-row holdout; not a production PD.**

---

## Interview close

We selected on development log loss, locked the test, checked the joblib, scored once, and did not retune. Discrimination is real on this mixture and not a 12-month PD. SHAP was not required. Cutoffs are illustrations. Monitoring here is a spec, not a platform.
"""
    OUT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
