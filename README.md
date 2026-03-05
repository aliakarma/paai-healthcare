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

## Repository Structure

```
paai-healthcare/
├── configs/              # All YAML configuration files
├── data/
│   ├── synthetic/        # Patient generator + adherence/hazard models
│   ├── knowledge_graph/  # Drug-Food, Condition-Contraindication triples
│   ├── policy_registry/  # Prescriber rules, allergies, escalation criteria
│   └── mimic/            # MIMIC-IV extraction scripts (requires PhysioNet access)
├── preprocessing/        # Algorithm 1 — Signal denoising, features, anomaly gating
├── knowledge/            # KG loader, Feature Store, Policy Registry, Drug Checker
├── agents/               # Four BDI agents (Listings 2–5)
├── governance/           # Audit log, consent, encryption, 3-tier HiTL
├── orchestrator/         # Algorithm 2 / Listing 1 — Central reasoning hub
├── envs/                 # Custom Gymnasium environment (CMDP)
├── rl/                   # MaskablePPO + Lagrangian training
├── baselines/            # Rules-only, Predictive-only, Human-schedule
├── evaluation/           # Metrics, statistical tests, all figures
├── tests/                # Unit tests (pytest)
└── docs/                 # Architecture, HiTL guide, MIMIC setup, RL guide
```

---

## Citation

```bibtex
@article{syed2025paai,
  title   = {PAAI — From Sensing to Action: A Privacy-Aware Agentic AI
             Architecture for IoT Healthcare},
  author  = {Syed, Toqeer Ali and Akarma, Ali and Ali, Ahmad and
             Lee, It Ee and Jan, Salman and Khan, Sohail and Nauman, Muhammad},
  journal = {Journal Not Specified},
  year    = {2025},
  doi     = {10.32604/journal.2025.012345}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
