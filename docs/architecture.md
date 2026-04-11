# System Architecture

## Four-Layer Overview

### Layer 1 — Sensing & Data Acquisition
- Wearables (CGM, smartwatch, BP cuff): BP, HR, SpO2, glucose every 5 minutes
- EHR & Lab Records: historical diagnostics, eGFR, AST/ALT
- Patient Inputs: diet logs, symptom reports, feedback

### Layer 2 — Preprocessing & Knowledge
- **Ingestion & Stream Processing**: MQTT/HTTP ingest, median filter denoise,
  unit normalisation, z-score (Algorithm 1)
- **Feature Store**: rolling mean (5/15/60 min), slope, volatility
- **Clinical Knowledge Graph**: Drug-Food, Condition-Contraindication,
  Nutrient-Deficiency triples (ADA/AHA/DrugBank)
- **Policy & Constraints Registry**: prescriber rules, dose ceilings,
  allergy exclusions, escalation thresholds

### Layer 3 — Agentic AI Core
- **Orchestrator**: CMDP state construction, goal formulation, task planning
- **RL Constrained Policy Loop**: MaskablePPO with Lagrangian safety constraints
- **Constraint Filter**: hard action masking (infeasible actions never sampled)
- **Conflict Resolver**: cross-domain consistency (e.g. sodium budget)
- **BDI Agents**: Medicine, Food & Nutrition, Sleep & Lifestyle, Emergency

### Layer 4 — Governance & Outputs
- **Encrypted Data Lake**: AES-256, HIPAA/GDPR compliant
- **Immutable Audit Log**: SHA-256 hash-chained append-only log
- **Three-Tier HiTL**: Patient feedback (Tier 1), Clinician override (Tier 2),
  Governance review (Tier 3)
- **Patient App**: explanations, adherence tracking
- **Clinician Dashboard**: alerts, override, notes
- **Clinician Alert**: secure escalation packet

## CMDP Formulation
State:  s_t = (v_t, h_t, e_t, p_t) — 25-dimensional
Action: a_t ∈ {∅, med, food, lifestyle, escalate}
Reward: R_t = R_clinical + λ_adh·R_adherence + λ_safe·R_safety
Constraint: ∀t: a_t ⊨ C  (enforced via action masking)
