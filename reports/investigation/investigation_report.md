# Data investigation

This is a description of the **raw** 20,000-row file. No split, no imputation, no training.
Recorded values were **not** changed. Age 99 and other flags are left as-is.

- Rows: 20000
- Target = 1: 3146 (15.73%)
- Unique pandas index: False (repeated labels are not treated as customer IDs)
- Features that will be modelled later: client_history_in_bank, age, income, employment_years, loan_amount, interest_rate, num_loans, credit_utilization, past_due_30, past_due_60, past_due_90, past_default, dti, region_risk
- Excluded from models: `recovery_flag_after_90d` (timing). `interest_rate` is included; validation will challenge its timing.

## Quality flags (not automatic deletions)

- Income missing: 700; utilization missing: 17
- Income equal to 30,000: 12505
- Age equal to 99: 120
- Loan/(income×DTI) near 1 / near 10: 19227 / 73
- 60+ count > 30+ count: 2558; 90+ > 60+: 797
- Interest rate range: 0.03 to 0.242794 (units unverified)
- Recovery flag = 1: 3148 (field still excluded from models)
