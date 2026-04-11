# Three-Tier Human-in-the-Loop (HiTL) Governance

## Overview

AgHealth+ implements a three-tier governance model operating at different
timescales. Each tier provides a feedback signal that flows back into the system.

```
Tier 1 — Patient Feedback         (seconds)
Tier 2 — Clinician Override        (minutes)
Tier 3 — Governance Committee      (weekly)
```

---

## Tier 1 — Patient Feedback Loop

**Actor:** Patient via mobile app  
**Actions:** Accept | Modify | Reject recommendation  
**Timescale:** Real-time (seconds)  
**File:** `governance/hitl/patient_feedback.py`

### Signal flow
```
Patient taps feedback → PatientFeedbackHandler.record()
    → R_t^adherence ∈ {+1, 0, -1}
    → Buffered → RL offline update
```

### Reward mapping
| Feedback | R_t^adherence |
|----------|---------------|
| Accept   | +1.0          |
| Modify   | 0.0           |
| Reject   | -1.0          |

### Usage
```python
from governance.hitl.patient_feedback import PatientFeedbackHandler
handler = PatientFeedbackHandler(audit_log)
reward  = handler.record("patient_42", "rec_001", "accept")
# reward = 1.0
```

---

## Tier 2 — Clinician Override

**Actor:** Clinician via dashboard  
**Actions:** Accept | Override | Escalate further  
**Timescale:** Minutes (triggered by escalation alert)  
**File:** `governance/hitl/clinician_override.py`

### Override tuple (5-tuple from paper)
```
⟨s_t, a_proposed, a_override, clinician_id, rationale⟩
```

### Signal flow
```
Clinician override → ClinicianOverrideHandler.record_override()
    → 5-tuple stored in encrypted audit log
    → flush_overrides() called during weekly offline policy update
    → RL policy updated: rejected actions penalised in similar states
```

### Usage
```python
from governance.hitl.clinician_override import ClinicianOverrideHandler
handler = ClinicianOverrideHandler(audit_log)
handler.record_override(
    state          = current_state_dict,
    proposed_action = "dietary_modification",
    override_action = "escalate",
    clinician_id   = "dr_smith",
    rationale      = "Patient has active infection, escalation warranted",
)
```

---

## Tier 3 — Governance Committee Review

**Actor:** Clinical committee (weekly)  
**Actions:** Update Policy Registry, adjust thresholds, flag fairness concerns  
**Timescale:** Weekly  
**File:** `governance/hitl/governance_review.py`

### Report contents
- Override frequency by agent (identifies systematic issues)
- False-positive escalation rate (target: < 5%)
- Action type distribution
- Subgroup disparity flags

### Policy update process
1. Committee reviews `GovernanceReview.generate_weekly_report()`
2. If FP rate > 15%: tighten escalation thresholds in `configs/escalation_thresholds.yaml`
3. If override rate high: update `data/policy_registry/prescriber_rules.json`
4. Deploy via blue-green strategy (never in-place update of running policy)

### Usage
```python
from governance.hitl.governance_review import GovernanceReview
review = GovernanceReview(override_handler, audit_log)
report = review.generate_weekly_report(period_days=7)
print(report["recommendation"])
```
