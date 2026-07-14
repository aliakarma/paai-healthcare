# Limitations & Future Work

## Known Limitations

### 1. Synthetic Data Limitations
- **Limited real-world distribution**: While the synthetic cohort (12 months, 500 patients) matches MIMIC-IV demographics, 
  it does not capture all real-world heterogeneity in disease progression
- **Assumption of Markovian dynamics**: Patient state assumes conditional independence given current observations; 
  long-term dependencies may be underrepresented
- **Controlled event injection**: Rare events (hypoglycemia, hypertensive crises) are artificially injected; 
  natural frequency may differ in real populations

### 2. Knowledge Graph Completeness
- **Incomplete medical ontology**: Drug-food interactions sourced from DrugBank may miss newly identified interactions
- **Hard-coded thresholds**: Escalation criteria (Table 1 in paper) are fixed across patient subgroups; 
  personalization not yet implemented
- **No temporal contraindications**: Knowledge graph does not model time-dependent contraindications 
  (e.g., photosensitivity onset for certain antibiotics)

### 3. MIMIC-IV Evaluation Limitations
- **Retrospective data**: MIMIC-IV is an ICU dataset; results may not generalize to outpatient/home settings
- **Survivor bias**: Only completed ICU episodes are included; acute-stage or fatal outcomes less represented
- **Feature availability**: Not all EHR variables may be available in external datasets (time-to-implementation risk)
- **No real-world feedback loop**: MIMIC-IV validation uses logged data, not prospective intervention

### 4. Real Wearable Dataset (OhioT1DM, WESAD, PPG-DaLiA) Limitations
- **Retrospective offline evaluation**: The validation on real datasets uses windowed static data features rather than running the active closed-loop Gymnasium `PatientEnv`. Direct prospective clinical testing is required.
- **Sensor quality and dropouts**: Wearable telemetry is subject to significant noise, artifacts, and packet loss, which require extensive preprocessing (Algorithm 1) in real deployment.
- **Labeling proxy**: Ground-truth labels for risks (such as diabetic risk in OhioT1DM) are derived using physiological thresholds and future windowed alerts rather than direct physician clinical notes.

### 5. Reinforcement Learning Constraints
- **Sample efficiency**: RL policy requires ~2M environment steps (500 patients × 12 months × timesteps); 
  not immediately deployable without significant clinical validation period
- **Generalization**: Policy trained on MIMIC-IV demographics may have performance drift on different populations 
  (e.g., pediatric, geriatric, different ethnic backgrounds)
- **Partial observability**: Real-world patient state may have unmeasured confounders not captured in feature vector

### 6. Human-in-the-Loop Limitations
- **Feedback loop lag**: Tier 2 clinician overrides only integrated during weekly offline policy update; 
  no real-time closed-loop adaptation
- **Scalability**: Three-tier governance requires clinical staffing; may not scale to millions of patients
- **Clinician compliance**: System effectiveness depends on clinician engagement in feedback loops; 
  implementation burden not quantified

### 7. Privacy & Security Scope
- **Implementation-level privacy**: Code uses standard cryptography (AES-256, SHA-256); 
  no differential privacy or formal privacy guarantees provided
- **Audit log integrity**: Hash-chaining prevents tampering but does not prevent unauthorized access; 
  requires OS-level access controls
- **Deployment assumptions**: Model assumes trustworthy infrastructure; vulnerable to compromised edge nodes

## Future Work

### Near-term (6–12 months)
1. **Prospective Clinical Trial**: Deployment in controlled pilot (50–100 patients, 3–6 months) to validate HiTL feedback loops
2. **Fairness Analysis**: Subgroup performance analysis across age, gender, ethnicity, socioeconomic status
3. **Explainability Enhancement**: Integration of SHAP or attention mechanisms for clinician interpretability
4. **Feature Store Optimization**: Migration from in-memory Redis to low-latency OLAP store for production scalability

### Medium-term (1–2 years)
1. **Temporal Knowledge Graph**: Incorporate time-dependent interactions and contraindications
2. **Multi-Agent Negotiation**: Extend conflict resolution to explicit negotiation protocols (e.g., Contract Nets)
3. **Transfer Learning**: Evaluate domain adaptation from MIMIC-IV to external health system data
4. **Pediatric & Geriatric Extensions**: Adapt knowledge graph thresholds for age-specific populations

### Long-term (2+ years)
1. **Federated Learning**: Multi-institutional collaborative training while preserving local data privacy
2. **Causal Inference**: Integrate causal discovery (e.g., causal forests) to identify optimal personalized interventions
3. **Uncertainty Quantification**: Bayesian RL to provide confidence intervals on recommendations
4. **Continuous Deployment**: Safe online learning with automatic audit log-based policy rollback

## Experimental Design Considerations

### For Practitioners Using This Code

When deploying or extending this work, consider:

1. **Population Characteristics**: Validate thresholds (escalation, sodium caps) for your specific patient population
2. **Data Requirements**: Ensure feature availability matches `envs/spaces.py` STATE_DIM definitions
3. **Clinical Workflow Integration**: Assess HiTL governance fit with your institution's clinical protocols
4. **Regulatory Compliance**: Audit against local regulations (e.g., FDA software validation for Class II devices in USA)
5. **Bias & Fairness**: Run subgroup analysis before deployment using `evaluation/statistical_tests.py`

## Reproducibility Notes

- **Random seeds**: All final results are reported as the `mean \pm SD` across 5 random seeds (seeds 42, 1337, 2024, 7, 91) to capture seed-level training variance. Individual runs and local cohort generation default to `--seed 42` for consistency.
- **Hardware variability**: GPU vendor and CUDA version can affect RL convergence timing (not final performance)
- **Dependency versions**: Exact versions in `requirements.txt` are pinned; deviations may cause numerical differences
- **Temporal drift**: MIMIC-IV features are time-dependent; evaluation performed on specific date range (see `data/mimic/extract_cohort.py`)

## Citation for Limitations

If citing this work, please acknowledge both the contributions **and** the limitations documented here. This ensures fair representation in downstream research.

---

Last updated: March 2026
