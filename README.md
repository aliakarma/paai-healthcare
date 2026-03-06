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
git clone git clone https://github.com/aliakarma/paai-healthcare
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

## 📂 Repository Structure

```bash
paai-healthcare/
│
├── ⚙️ setup.py # package install, console scripts
├── 📦 requirements.txt # pinned dependencies (numpy, torch, sb3, gymnasium…)
├── 📜 LICENSE # Apache-2.0 license
├── 📖 CITATION.cff # citation metadata for academic use
├── 📘 README.md
├── ✅ test_bugfixes.py # bugfix tests
│
├── ⚙️ configs/
│ ├── patient_sim.yaml # 500 patients, 12-month simulation parameters
│ ├── rl_training.yaml # PPO/Lagrangian hyperparameters + reward weights
│ ├── escalation_thresholds.yaml # SBP/glucose/SpO₂ alert thresholds
│ ├── preprocessing.yaml # denoise, rolling-window, anomaly gates
│ ├── mimic_extraction.yaml # MIMIC-IV cohort SQL + feature specification
│ └── knowledge_graph.yaml # RDF endpoint and guideline sources
│
├── 🗄️ data/
│ ├── synthetic/
│ │ ├── generate_patients.py # 500-patient longitudinal generator
│ │ ├── adherence_model.py # stochastic medication adherence model
│ │ ├── hazard_model.py # rare event injection (hypoglycemia / hypertensive)
│ │ └── patient_schema.json
│ │
│ ├── knowledge_graph/
│ │ ├── drug_food_triples.json # DrugBank-derived drug–food pairs
│ │ ├── condition_contraindications.json # ADA/AHA contraindications
│ │ ├── nutrient_deficiency.json # WHO/USDA nutrient links
│ │ └── kg_schema.ttl
│ │
│ ├── policy_registry/
│ │ ├── prescriber_rules.json # dose ceilings, timing windows, sodium caps
│ │ ├── allergy_exclusions.json # allergen → exclusion tags
│ │ ├── escalation_criteria.json # watch / escalate vital thresholds
│ │ └── validate_registry.py # JSON schema validator
│ │
│ └── mimic/
│ ├── extract_cohort.py # MIMIC-IV SQL cohort builder
│ └── README.md
│
├── 🔧 preprocessing/
│ ├── signal_pipeline.py # Algorithm 1: denoise → normalise → featurise → gate
│ ├── denoise.py # median filtering, dropout bridging
│ ├── normalise.py # unit → SI conversion + per-channel z-score
│ └── feature_extraction.py # rolling mean, slope, z-score, volatility
│
├── 🧠 knowledge/
│ ├── knowledge_graph.py # RDF/property graph interface (SPARQL/Cypher)
│ ├── feature_store.py # Redis hot-path + Parquet long-term store
│ ├── policy_registry.py # concrete PolicyRegistry implementation
│ └── drug_checker.py # renal/hepatic safety constraint checker
│
├── 🤖 agents/
│ ├── base_agent.py # BaseAgent / BDI Agent (perceive → deliberate → act)
│ ├── medicine_agent.py # medication dose/timing/interaction checks
│ ├── nutrition_agent.py # macro/micro meal planning
│ ├── lifestyle_agent.py # sleep/walking/caffeine behavioural nudges
│ └── emergency_agent.py # emergency escalation agent
│
├── 🛡️ governance/
│ ├── audit_log.py # SHA-256 hash-chain audit log
│ ├── consent_manager.py # GDPR/HIPAA consent tracking
│ ├── encryption.py # Fernet encryption utilities
│ └── hitl/
│ ├── patient_feedback.py # patient feedback → RL reward loop
│ ├── clinician_override.py # Tier 2 override logging
│ └── governance_review.py # governance committee policy review
│
├── 🎛️ orchestrator/
│ ├── orchestrator.py # multi-agent BDI orchestrator
│ ├── constraint_filter.py # RL action safety constraint layer
│ ├── conflict_resolver.py # drug–food–lifestyle conflict resolution
│ └── task_router.py # task → agent dispatch routing
│
├── 🌍 envs/
│ ├── spaces.py # STATE_DIM=25, N_ACTIONS=5 constants
│ ├── patient_env.py # Gymnasium PatientEnv implementation
│ ├── reward_function.py # composite reward function
│ └── constraint_set.py # Lagrangian safety constraint set
│
├── 🧮 rl/
│ ├── train.py # PPO + Lagrangian constraint training
│ ├── callbacks.py # training callbacks and monitoring
│ ├── lagrangian.py # dual-variable update λ
│ ├── evaluate_policy.py # policy evaluation runner
│ └── checkpoints/ # saved policy weights
│
├── 📊 baselines/
│ ├── rules_only.py # baseline B1: rule-based thresholds
│ ├── predictive_only.py # baseline B2: anomaly detection model
│ └── human_schedule.py # baseline B3: static clinician schedule
│
├── 📈 evaluation/
│ ├── run_evaluation.py # full synthetic evaluation pipeline
│ ├── metrics.py # ROC-AUC, precision/recall/F1, latency
│ ├── statistical_tests.py # DeLong, Wilcoxon, Bonferroni tests
│ ├── ablation.py # ablation experiments
│ ├── mimic_evaluation.py # real-world EHR evaluation
│ └── plots/
│ ├── plot_roc.py
│ ├── plot_med_quality.py
│ ├── plot_latency_cdf.py
│ ├── plot_adherence.py
│ └── plot_learning_curves.py
│
├── 🧪 tests/
│ ├── test_signal_pipeline.py
│ ├── test_agents.py
│ ├── test_constraint_filter.py
│ ├── test_audit_log.py
│ ├── test_reward_function.py
│ └── test_orchestrator.py
│
├── 📚 docs/
│ ├── architecture.md
│ ├── rl_training_guide.md
│ ├── mimic_setup.md
│ └── hitl_guide.md
│
└── ⚡ .github/
├── workflows/
│ ├── test.yml
│ └── lint.yml
└── ISSUE_TEMPLATE/
└── bug_report.md

```
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

Apache 2.0 — see (LICENSE).
