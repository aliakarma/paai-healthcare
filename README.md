<div align="center">

<img src="https://raw.githubusercontent.com/aliakarma/paai-healthcare/main/docs/assets/banner.png" alt="AgHealth+ PAAI" width="100%" onerror="this.style.display='none'"/>

# PAAI · Privacy-Aware Agentic AI for IoT Healthcare

**A four-layer privacy-preserving agentic AI architecture for chronic disease management**

<br/>

[![License](https://img.shields.io/badge/License-Apache_2.0-22c55e?style=flat-square&logo=apache&logoColor=white)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3b82f6?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18891707-f59e0b?style=flat-square&logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.18891707)
[![Framework](https://img.shields.io/badge/RL-PPO_+_Lagrangian-8b5cf6?style=flat-square&logo=pytorch&logoColor=white)](docs/rl_training_guide.md)
[![HIPAA](https://img.shields.io/badge/Compliance-HIPAA_%2F_GDPR-ef4444?style=flat-square&logo=shield&logoColor=white)](#ethics-statement)
[![CI](https://img.shields.io/badge/Tests-Passing-22c55e?style=flat-square&logo=github-actions&logoColor=white)](.github/workflows/test.yml)

<br/>

[![Architecture Docs](https://img.shields.io/badge/📐_Architecture-docs%2Farchitecture.md-0ea5e9?style=for-the-badge)](docs/architecture.md)
[![RL Training](https://img.shields.io/badge/🧠_RL_Training-docs%2Frl__training__guide.md-7c3aed?style=for-the-badge)](docs/rl_training_guide.md)
[![HiTL Governance](https://img.shields.io/badge/🛡️_HiTL_Governance-docs%2Fhitl__guide.md-0f766e?style=for-the-badge)](docs/hitl_guide.md)
[![MIMIC Setup](https://img.shields.io/badge/🗄️_MIMIC_Setup-docs%2Fmimic__setup.md-b45309?style=for-the-badge)](docs/mimic_setup.md)

<br/>

[![Contributing](https://img.shields.io/badge/Contributing-CONTRIBUTING.md-1d4ed8?style=flat-square)](CONTRIBUTING.md)
[![Data Availability](https://img.shields.io/badge/Data_Availability-DATA__AVAILABILITY.md-16a34a?style=flat-square)](DATA_AVAILABILITY.md)
[![Limitations](https://img.shields.io/badge/Limitations-LIMITATIONS.md-dc2626?style=flat-square)](LIMITATIONS.md)
[![Declarations](https://img.shields.io/badge/Author_Declarations-DECLARATIONS.md-9333ea?style=flat-square)](DECLARATIONS.md)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Results](#key-results)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [System Requirements](#system-requirements)
- [Data Availability & Reproducibility](#data-availability--reproducibility)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Authors & Contributions](#authors--contributions)
- [Ethics Statement](#ethics-statement)
- [License](#license)

---

## Overview

**AgHealth+** implements the **PAAI** (Privacy-Aware Agentic AI) framework — a production-grade, four-layer multi-agent system for personalized chronic disease management at scale. The architecture combines reinforcement learning, Bayesian decision-making, and Human-in-the-Loop governance to deliver real-time, privacy-preserving clinical recommendations across heterogeneous IoT health data streams.

| Feature | Details |
|---|---|
| 🤖 **Agent Architecture** | BDI-based multi-agent system (Medicine, Nutrition, Lifestyle, Emergency) |
| 🔐 **Privacy Layer** | AES-256 encryption + SHA-256 hash-chain audit log (HIPAA/GDPR compliant) |
| 🧠 **Learning Core** | PPO + Lagrangian constrained RL (Constrained MDP formulation) |
| 👥 **Governance** | 3-tier Human-in-the-Loop oversight (Patient · Clinician · Committee) |
| 📡 **Data Sources** | Wearables (BP/HR/SpO₂/Glucose), EHR, synthetic + MIMIC-IV validation |
| ⚡ **Inference** | Median decision latency: **1.8 s** (5.4× faster than human scheduling) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 1  ·  Sensing & Data Acquisition                                  │
│  Wearables (BP / HR / SpO₂ / Glucose)  ·  EHR Records  ·  Patient I/O    │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │  Algorithm 1  (signal_pipeline)
┌──────────────────────────────▼───────────────────────────────────────────┐
│  Layer 2  ·  Preprocessing & Knowledge                                   │
│  Feature Store  ·  Clinical Knowledge Graph  ·  Policy Registry          │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │  CMDP state  sₜ
┌──────────────────────────────▼───────────────────────────────────────────┐
│  Layer 3  ·  Agentic AI Core                                             │
│  Orchestrator (Listing 1) → RL π(a|s) → Constraint Filter                │
│                           → Conflict Resolver → Task Router              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐    │
│  │  Medicine   │ │  Nutrition  │ │  Lifestyle  │ │   Emergency      │    │
│  │    Agent    │ │    Agent    │ │    Agent    │ │  Escalation Agt  │    │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────────────┘    │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │  CIP: encrypt + hash-chain
┌──────────────────────────────▼───────────────────────────────────────────┐
│  Layer 4  ·  Governance & Outputs                                        │
│  Encrypted Data Lake  ·  Immutable Audit Log  ·  3-Tier HiTL             │
│  Patient App  ·  Clinician Dashboard  ·  Clinician Alert                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Key Results

> Results reported on the **500-patient synthetic cohort** (12-month longitudinal simulation, seed 42).  
> All comparisons are statistically significant after Bonferroni correction (*p* < 0.001).

| Method | Accuracy | ROC AUC | Med. Latency | Med. Precision |
|---|:---:|:---:|:---:|:---:|
| Rules-only (B1) | 0.81 ± 0.01 | 0.87 | 4.9 s | 0.71 |
| Predictive-only (B2) | 0.88 ± 0.01 | 0.92 | 3.1 s | 0.79 |
| Human schedule (B3) | 0.76 ± 0.02 | 0.83 | 9.8 s | 0.75 |
| **AgHealth+ (PAAI)** | **0.92 ± 0.01** | **0.96** | **1.8 s** | **0.87** |

> ✅ AgHealth+ achieves **+4 pp accuracy**, **+0.04 AUC**, and **5.4× lower latency** vs. the best baseline.

---

## Quick Start

### Prerequisites

```bash
Python ≥ 3.10   |   RAM ≥ 16 GB   |   Storage ≥ 10 GB
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/aliakarma/paai-healthcare
cd paai-healthcare

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Run in 3 Steps

```bash
# Step 1 — Generate synthetic cohort (500 patients, 12 months)
python data/synthetic/generate_patients.py --seed 42

# Step 2 — Train RL policy (2 M steps · ~3–6 h CPU · ~45 min GPU)
python rl/train.py

# Step 3 — Reproduce all paper results (Table 2)
python evaluation/run_evaluation.py --mode all --seed 42
```

> 💡 **GPU acceleration**: Add `--gpu` to `rl/train.py` for ~8× faster training on NVIDIA hardware.  
> 💡 **CPU-only mode**: Add `--cpu` if CUDA is unavailable.

---

## Repository Structure

```
paai-healthcare/
│
├── ⚙️  setup.py                        # Package install & console scripts
├── 📦  requirements.txt                # Pinned dependencies
├── 📜  LICENSE                         # Apache-2.0
├── 📖  CITATION.cff                    # Academic citation metadata
│
├── ⚙️  configs/
│   ├── patient_sim.yaml                # 500-patient simulation parameters
│   ├── rl_training.yaml                # PPO/Lagrangian hyperparameters
│   ├── escalation_thresholds.yaml      # SBP/glucose/SpO₂ alert thresholds
│   ├── preprocessing.yaml              # Denoise, rolling-window, anomaly gates
│   ├── mimic_extraction.yaml           # MIMIC-IV cohort SQL + features
│   └── knowledge_graph.yaml            # RDF endpoint & guideline sources
│
├── 🗄️  data/
│   ├── synthetic/                      # 500-patient longitudinal generator
│   ├── knowledge_graph/                # DrugBank, ADA/AHA, WHO/USDA triples
│   ├── policy_registry/                # Dose rules, allergen exclusions
│   └── mimic/                          # MIMIC-IV cohort extraction scripts
│
├── 🔧  preprocessing/
│   ├── signal_pipeline.py              # Algorithm 1: denoise→normalise→gate
│   ├── denoise.py                      # Median filtering, dropout bridging
│   ├── normalise.py                    # Unit→SI + per-channel z-score
│   └── feature_extraction.py          # Rolling mean, slope, z-score, volatility
│
├── 🧠  knowledge/
│   ├── knowledge_graph.py              # RDF/property graph (SPARQL/Cypher)
│   ├── feature_store.py                # Redis hot-path + Parquet long-term
│   ├── policy_registry.py              # Concrete PolicyRegistry
│   └── drug_checker.py                 # Renal/hepatic safety constraints
│
├── 🤖  agents/
│   ├── base_agent.py                   # BaseAgent / BDI (perceive→deliberate→act)
│   ├── medicine_agent.py               # Dose/timing/interaction checks
│   ├── nutrition_agent.py              # Macro/micro meal planning
│   ├── lifestyle_agent.py              # Sleep/walking/caffeine nudges
│   └── emergency_agent.py             # Emergency escalation
│
├── 🛡️  governance/
│   ├── audit_log.py                    # SHA-256 hash-chain audit log
│   ├── consent_manager.py              # GDPR/HIPAA consent tracking
│   ├── encryption.py                   # Fernet encryption utilities
│   └── hitl/                           # 3-tier Human-in-the-Loop modules
│
├── 🎛️  orchestrator/
│   ├── orchestrator.py                 # Multi-agent BDI orchestrator
│   ├── constraint_filter.py            # RL action safety constraint layer
│   ├── conflict_resolver.py            # Drug–food–lifestyle conflict resolution
│   └── task_router.py                  # Task→agent dispatch routing
│
├── 🌍  envs/
│   ├── patient_env.py                  # Gymnasium PatientEnv (STATE_DIM=25)
│   ├── reward_function.py              # Composite reward function
│   └── constraint_set.py              # Lagrangian safety constraint set
│
├── 🧮  rl/
│   ├── train.py                        # PPO + Lagrangian constraint training
│   ├── lagrangian.py                   # Dual-variable update λ
│   ├── evaluate_policy.py              # Policy evaluation runner
│   └── checkpoints/                    # Saved policy weights
│
├── 📊  baselines/
│   ├── rules_only.py                   # B1: Rule-based thresholds
│   ├── predictive_only.py              # B2: Anomaly detection model
│   └── human_schedule.py              # B3: Static clinician schedule
│
├── 📈  evaluation/
│   ├── run_evaluation.py               # Full synthetic evaluation pipeline
│   ├── metrics.py                      # ROC-AUC, precision/recall/F1, latency
│   ├── statistical_tests.py           # DeLong, Wilcoxon, Bonferroni tests
│   ├── ablation.py                     # Ablation experiments
│   └── mimic_evaluation.py            # Real-world EHR evaluation
│
└── 🧪  tests/
    ├── test_signal_pipeline.py
    ├── test_agents.py
    ├── test_constraint_filter.py
    ├── test_audit_log.py
    ├── test_reward_function.py
    └── test_orchestrator.py
```

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **Python** | 3.10 | 3.11 |
| **CPU** | 8 cores | 16+ cores |
| **RAM** | 16 GB | 32+ GB |
| **Storage** | 10 GB SSD | 50+ GB NVMe |
| **GPU** | Optional (CPU-only mode supported) | NVIDIA RTX 3060+ |

| Task | CPU Time | GPU Time |
|---|---|---|
| RL Training (2 M steps) | ~3–6 hours | ~45 minutes |
| Full Evaluation (all baselines) | ~15 minutes | ~5 minutes |

---

## Data Availability & Reproducibility

| Dataset | Access | Reproducibility |
|---|---|---|
| **Synthetic Cohort** | Included | `python data/synthetic/generate_patients.py --seed 42` |
| **MIMIC-IV** | [PhysioNet credentialing required](https://physionet.org/content/mimiciv/) | Scripts in `data/mimic/` |
| **Trained Checkpoints** | Included in release v1.0 | Saved to `rl/checkpoints/` during training |
| **Zenodo Archive** | [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18891707-f59e0b?style=flat-square)](https://doi.org/10.5281/zenodo.18891707) | Persistent archive with fixed seed |

> 🔄 **Exact Reproducibility**: All experiments use `--seed 42`. Full Table 2 regeneration:
> ```bash
> python evaluation/run_evaluation.py --mode all --seed 42
> ```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: paai_healthcare` | Run `pip install -e .` after cloning |
| MIMIC-IV access denied | Apply for credentials at [physionet.org](https://physionet.org) |
| CUDA out of memory | Add `--cpu` flag to training/evaluation scripts |
| RL training unstable | Verify `configs/rl_training.yaml` matches paper hyperparameters |
| Statistical test failures | Ensure `--seed 42` is set for exact replication |

---

## Citation

If AgHealth+ or the PAAI framework contributed to your research, please cite:

```bibtex
@article{syed2026paai,
  title   = {PAAI: A Privacy-Aware Agentic AI Architecture for
             Chronic Disease Management in IoT Healthcare},
  author  = {Syed, Toqeer Ali and Akarma, Ali and Ali, Ahmad and
             Lee, It Ee and Jan, Salman and Khan, Sohail and Nauman, Muhammad},
  journal = {Not Specified Yet},
  year    = {2026}
}
```

---

## Authors & Contributions

This research is a collaborative effort across multiple institutions.

| Author | Affiliation | Role |
|---|---|---|
| **Toqeer Ali Syed** | Islamic University of Madinah | Lead — Conceptualization, methodology, architecture, RL framework, evaluation, writing |
| **Ali Akarma** | Islamic University of Madinah | Lead — BDI agents, knowledge graph, orchestrator, governance module, writing |
| **Ahmad Ali** | Islamic University of Madinah | Synthetic data generation, baseline implementations, statistical testing |
| **It Ee Lee** | Multimedia University | Critical review, validation |
| **Salman Jan** | Arab Open University | Critical review, governance framework |
| **Sohail Khan** | Effat University | Critical review, validation |
| **Muhammad Nauman** | Effat University | Critical review, governance framework |

---

## Ethics Statement

| Principle | Implementation |
|---|---|
| 🔐 **Patient Privacy** | AES-256 encryption + SHA-256 hash-chain audit log (HIPAA/GDPR compliant) |
| 👥 **Data Governance** | 3-tier Human-in-the-Loop oversight prevents autonomous risk escalation |
| 📋 **Transparency** | All model decisions logged and auditable via immutable audit log |
| ⚖️ **Bias Mitigation** | Subgroup disparity analysis (`governance/hitl/governance_review.py`) |
| 🏥 **Clinical Validation** | MIMIC-IV evaluation bridges synthetic and real-world performance |

> This research was conducted independently and is not endorsed by or affiliated with any commercial entity.  
> We acknowledge the PhysioNet team for access to the MIMIC-IV dataset.

---

## License

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-22c55e?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)

Distributed under the **Apache 2.0 License**. See [`LICENSE`](LICENSE) for full terms.

<br/>

*PAAI — Advancing privacy-preserving AI for patient-centred healthcare*

</div>
