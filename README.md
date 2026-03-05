# AgHealth+ · PAAI — Privacy-Aware Agentic AI for IoT Healthcare

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)


Official implementation of the **PAAI** framework — a four-layer,
privacy-aware agentic AI architecture for chronic disease management
in IoT healthcare (AgHealth+ system).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 · Sensing & Data Acquisition                           │
│  Wearables (BP/HR/SpO₂/Glucose) · EHR Records · Patient Inputs │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Algorithm 1 (signal_pipeline)
┌───────────────────────────▼─────────────────────────────────────┐
│  Layer 2 · Preprocessing & Knowledge                            │
│  Feature Store · Clinical Knowledge Graph · Policy Registry     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ CMDP state s_t
┌───────────────────────────▼─────────────────────────────────────┐
│  Layer 3 · Agentic AI Core                                      │
│  Orchestrator (Listing 1) → RL π(a|s) → Constraint Filter      │
│  → Conflict Resolver → Task Router                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────────┐  │
│  │Medicine  │ │Nutrition │ │Lifestyle │ │Emergency Escalat. │  │
│  │  Agent   │ │  Agent   │ │  Agent   │ │      Agent        │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ CIP: encrypt + hash-chain
┌───────────────────────────▼─────────────────────────────────────┐
│  Layer 4 · Governance & Outputs                                 │
│  Encrypted Data Lake · Immutable Audit Log · 3-Tier HiTL        │
│  Patient App · Clinician Dashboard · Clinician Alert            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Results (Table 2)

| Method           | Accuracy      | ROC AUC | Med. Latency | Med. Precision |
|------------------|---------------|---------|--------------|----------------|
| Rules-only (B1)  | 0.81 ± 0.01   | 0.87    | 4.9 s        | 0.71           |
| Predictive (B2)  | 0.88 ± 0.01   | 0.92    | 3.1 s        | 0.79           |
| Human sched (B3) | 0.76 ± 0.02   | 0.83    | 9.8 s        | 0.75           |
| **AgHealth+**    | **0.92 ± 0.01** | **0.96** | **1.8 s**  | **0.87**       |

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/toqeersyed/paai-healthcare
cd paai-healthcare
pip install -r requirements.txt
pip install -e .

# 2. Generate synthetic cohort (500 patients, 12 months)
python data/synthetic/generate_patients.py

# 3. Train RL policy (2 M steps, ~3–6 h on CPU)
python rl/train.py

# 4. Reproduce all paper results
python evaluation/run_evaluation.py --mode all
```

---

paai-healthcare/
│
├── [P1] setup.py                    ← package install, console scripts
├── [P1] requirements.txt            ← pinned deps (numpy, torch, sb3, gymnasium…)
├── [P1] LICENSE                     ← Apache-2.0
├── [P1] CITATION.cff                ← CFF citation metadata
├── [P1] README.md
├── [P1] MERGE_GUIDE.md              ← step-by-step unzip & verify instructions
├── [P1] verify_merge.py             ← 5-step merge verifier
│
├── [P1] configs/
│   ├── patient_sim.yaml             ← 500 patients, 12-month sim parameters
│   ├── rl_training.yaml             ← PPO/Lagrangian hypers + reward weights
│   ├── escalation_thresholds.yaml   ← SBP/glucose/SpO₂ alert thresholds
│   ├── preprocessing.yaml           ← denoise, rolling-window, anomaly gates
│   ├── mimic_extraction.yaml        ← MIMIC-IV cohort SQL + feature spec
│   └── knowledge_graph.yaml         ← RDF endpoint, guideline sources
│
├── [P1] data/
│   ├── synthetic/
│   │   ├── generate_patients.py     ← 500-patient longitudinal generator (Algorithm 1 input)
│   │   ├── adherence_model.py       ← stochastic dose-miss model
│   │   ├── hazard_model.py          ← rare event injection (hypo/hypertensive)
│   │   └── patient_schema.json
│   ├── knowledge_graph/
│   │   ├── drug_food_triples.json   ← DrugBank-derived drug–food pairs
│   │   ├── condition_contraindications.json  ← ADA/AHA contraindications
│   │   ├── nutrient_deficiency.json ← WHO/USDA nutrient links
│   │   └── kg_schema.ttl
│   ├── policy_registry/
│   │   ├── prescriber_rules.json    ← dose ceilings, timing windows, sodium caps
│   │   ├── allergy_exclusions.json  ← allergen → exclude tags
│   │   ├── escalation_criteria.json ← watch/escalate vital thresholds
│   │   └── validate_registry.py     ← JSON schema validator
│   └── [P4] mimic/
│       ├── extract_cohort.py        ← MIMIC-IV SQL cohort builder
│       └── README.md
│
├── [P1] preprocessing/
│   ├── signal_pipeline.py           ← Algorithm 1: denoise→normalise→featurise→gate
│   ├── denoise.py                   ← median filter, dropout bridging
│   ├── normalise.py                 ← unit→SI + per-channel z-score
│   └── feature_extraction.py        ← rolling mean/slope/z-score/volatility
│
├── [P2] knowledge/
│   ├── knowledge_graph.py           ← RDF/property-graph KG, SPARQL/Cypher
│   ├── feature_store.py             ← Redis hot-path + Parquet long-term
│   ├── policy_registry.py           ← concrete PolicyRegistry (implements Protocol)
│   └── drug_checker.py              ← renal/hepatic constraint checker
│
├── [P2] agents/
│   ├── base_agent.py                ← BaseAgent/BDIAgent — perceive/deliberate/act
│   │                                   + execute()/update_beliefs() for orchestrator
│   ├── medicine_agent.py            ← Listing 2: dose/timing/interaction checks
│   ├── nutrition_agent.py           ← Listing 3: macro/micro meal planner
│   ├── lifestyle_agent.py           ← Listing 4: sleep/walk/caffeine nudges
│   └── emergency_agent.py           ← Listing 5 / Algorithm 3: watch-and-repeat
│
├── [P2] governance/
│   ├── audit_log.py                 ← SHA-256 hash-chain, AES-encrypted audit log
│   ├── consent_manager.py           ← GDPR/HIPAA consent scopes + tracking
│   ├── encryption.py                ← Fernet key generation + data-at-rest
│   └── hitl/
│       ├── patient_feedback.py      ← one-tap feedback → RL reward loop
│       ├── clinician_override.py    ← Tier 2 override with audit entry
│       └── governance_review.py     ← quarterly policy drift review
│
├── [P3] orchestrator/
│   ├── orchestrator.py              ← Algorithm 2 / Listing 1 — BDI orchestration
│   ├── constraint_filter.py         ← Figure 2 — RL action constrained by registry
│   ├── conflict_resolver.py         ← drug–food/lifestyle conflict resolution
│   └── task_router.py               ← task-type → agent dispatch table
│
├── [P3] envs/
│   ├── spaces.py                    ← STATE_DIM=25, N_ACTIONS=5 CMDP constants
│   ├── patient_env.py               ← Gymnasium PatientEnv with step/reset/render
│   ├── reward_function.py           ← Eq.1 reward: stability + adherence − alarm
│   └── constraint_set.py            ← Lagrangian safety constraint set C
│
├── [P3] rl/
│   ├── train.py                     ← PPO + Lagrangian multiplier training
│   ├── callbacks.py                 ← SB3 callbacks: constraint monitor, policy saver
│   ├── lagrangian.py                ← dual-variable update λ for CMDP
│   ├── evaluate_policy.py           ← off-policy rollout + metric collection
│   ├── checkpoints/                 ← saved policy weights (.gitkeep)
│   └── tensorboard/                 ← training logs (.gitkeep)
│
├── [P3] baselines/
│   ├── rules_only.py                ← B1: threshold rules, no learning (AUC 0.87)
│   ├── predictive_only.py           ← B2: sklearn anomaly model (AUC 0.92)
│   └── human_schedule.py            ← B3: static human-authored schedule
│
├── [P4] evaluation/
│   ├── run_evaluation.py            ← master runner: 500-patient synthetic eval
│   ├── metrics.py                   ← ROC-AUC, precision/recall/F1, latency CDF
│   ├── statistical_tests.py         ← DeLong, Wilcoxon, paired-t, Bonferroni
│   ├── ablation.py                  ← no-KG / no-policy / no-orchestrator runs
│   ├── mimic_evaluation.py          ← real-EHR evaluation harness (MIMIC-IV)
│   └── plots/
│       ├── plot_roc.py              ← Figure 3 — ROC curves
│       ├── plot_med_quality.py      ← Figure 4 — precision/recall/F1 bars
│       ├── plot_latency_cdf.py      ← Figure 5 — alert latency CDF
│       ├── plot_adherence.py        ← Figure 6 — 8-week adherence trend
│       └── plot_learning_curves.py  ← RL reward + constraint curves
│
├── [P4] tests/
│   ├── test_signal_pipeline.py      ← denoise/normalise/featurise
│   ├── test_agents.py               ← BDI cycle, execute(), safety gate
│   ├── test_constraint_filter.py    ← filter blocks unsafe RL actions
│   ├── test_audit_log.py            ← hash-chain tamper detection
│   ├── test_reward_function.py      ← Eq.1 reward sanity checks
│   └── test_orchestrator.py         ← end-to-end orchestration cycle
│
├── [P4] docs/
│   ├── architecture.md              ← system diagram narrative + layer descriptions
│   ├── rl_training_guide.md         ← how to train, resume, evaluate policies
│   ├── mimic_setup.md               ← MIMIC-IV access, SQL extraction, FHIR notes
│   └── hitl_guide.md                ← clinician dashboard, override, consent flows
│
└── [P4] .github/
    ├── workflows/
    │   ├── test.yml                 ← CI: pytest + flake8 on push/PR
    │   └── lint.yml                 ← CI: black + mypy type checks
    └── ISSUE_TEMPLATE/bug_report.md

---

---

## 📦 Data Availability & Reproducibility

- **Synthetic Data**: Fully reproducible via `python data/synthetic/generate_patients.py --seed 42`
- **MIMIC-IV Integration**: Scripts provided in `data/mimic/`; access requires [PhysioNet credentialing](https://physionet.org/content/mimiciv/)
- **Trained Models**: Checkpoints saved to `models/` during training; example weights included in release v1.0
- **Persistent Archive**: This repository is archived at Zenodo: [10.5281/zenodo.XXXXXXX](https://zenodo.org/records/XXXXXXX) (CC BY 4.0)

> 🔄 **Exact Reproducibility**: All experiments use `--seed 42`. Results in Table 2 can be regenerated via:
> ```bash
> python evaluation/run_evaluation.py --mode all --seed 42 --reproduce-table2
> ```

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10 | 3.11 |
| **CPU** | 8 cores | 16+ cores |
| **RAM** | 16 GB | 32+ GB |
| **Storage** | 10 GB SSD | 50+ GB NVMe |
| **GPU** | Optional (CPU-only mode) | NVIDIA RTX 3060+ (for faster RL training) |

*Training time*: ~3–6 hours on CPU; ~45 minutes on GPU  
*Evaluation time*: ~15 minutes (all baselines + AgHealth+)

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: paai_healthcare` | Run `pip install -e .` after cloning |
| MIMIC-IV access denied | Apply for credentials at [physionet.org](https://physionet.org) |
| CUDA out of memory | Add `--cpu` flag to training/evaluation scripts |
| RL training unstable | Verify `configs/rl_config.yaml` matches paper hyperparameters |
| Statistical test failures | Ensure `--seed 42` is used for exact replication |

---

## 📚 Citation

If you use this code or framework in your research, please cite:

```bibtex
@article{syed2026paai,
  title   = {PAAI — From Sensing to Action: A Privacy-Aware Agentic AI 
             Architecture for IoT Healthcare},
  author  = {Syed, Toqeer Ali and Akarma, Ali and Ali, Ahmad and 
             Lee, It Ee and Jan, Salman and Khan, Sohail and Nauman, Muhammad},
  journal = {},
  year    = {2026},
  volume  = {},
  issue   = {},
  pages   = {},
  doi     = {}
}

## License

Apache 2.0 — see [LICENSE](LICENSE).
