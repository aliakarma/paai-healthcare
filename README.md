<div align="center">

<img src="docs/assets/banner.png" alt="PAAI / AgHealth+ banner" width="100%" />

# PAAI · Privacy-Aware Agentic AI for IoT Healthcare

A research-grade, split-aware ML systems repository for chronic disease management with constrained RL, safety rules, and Human-in-the-Loop governance.

<p>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick_Start-0f766e?style=for-the-badge&logo=rocket&logoColor=white" alt="Quick Start"></a>
  <a href="#documentation-index"><img src="https://img.shields.io/badge/Documentation_Index-1d4ed8?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation Index"></a>
  <a href="#reproducibility-contract"><img src="https://img.shields.io/badge/Reproducibility_Contract-334155?style=for-the-badge&logo=checkmarx&logoColor=white" alt="Reproducibility"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Citation-c026d3?style=for-the-badge&logo=academia&logoColor=white" alt="Citation"></a>
</p>

</div>

---

## Documentation Index

[![Architecture](https://img.shields.io/badge/Architecture-0ea5e9?style=for-the-badge&logo=gitbook&logoColor=white)](docs/architecture.md)
[![RL Training](https://img.shields.io/badge/RL_Training-7c3aed?style=for-the-badge&logo=pytorch&logoColor=white)](docs/rl_training_guide.md)
[![HiTL Governance](https://img.shields.io/badge/HiTL_Governance-0f766e?style=for-the-badge&logo=shield&logoColor=white)](docs/hitl_guide.md)
[![MIMIC Setup](https://img.shields.io/badge/MIMIC_Setup-b45309?style=for-the-badge&logo=database&logoColor=white)](docs/mimic_setup.md)
[![Contributing](https://img.shields.io/badge/Contributing-1d4ed8?style=for-the-badge&logo=github&logoColor=white)](CONTRIBUTING.md)
[![Data Availability](https://img.shields.io/badge/Data_Availability-16a34a?style=for-the-badge&logo=files&logoColor=white)](DATA_AVAILABILITY.md)
[![Limitations](https://img.shields.io/badge/Limitations-dc2626?style=for-the-badge&logo=warning&logoColor=white)](LIMITATIONS.md)
[![Declarations](https://img.shields.io/badge/Declarations-9333ea?style=for-the-badge&logo=clipboard&logoColor=white)](DECLARATIONS.md)

---

## Overview

This repository implements a full healthcare-AI research pipeline:

1. Synthetic cohort generation and MIMIC extraction.
2. Signal preprocessing and feature construction.
3. Policy-aware orchestration and constrained decision making.
4. RL training with patient-level train/val/test separation.
5. Evaluation with split-aware baselines and statistical reporting.

### What Is In Scope

| Capability | Status in repo | Primary entrypoint |
|---|---|---|
| Synthetic patient simulation | Implemented | data/synthetic/generate_patients.py |
| Patient-level split generation | Implemented | data/synthetic/generate_patients.py |
| RL policy training | Implemented | rl/train.py |
| Split-aware benchmark evaluation | Implemented | evaluation/run_evaluation.py |
| MIMIC anomaly validation | Implemented | evaluation/run_evaluation.py --mode mimic |
| Governance (audit/consent/encryption/HiTL) | Implemented | governance/ |

---

## Reproducibility Contract

- Training and evaluation are split-aware by patient IDs (train/val/test).
- Split files are generated automatically at data creation time under data/synthetic/cohort/splits/.
- Baseline B2 is trained only on train split IDs and evaluated on eval split IDs.
- Synthetic results are reported from a chosen split (default: test).
- A trained RL checkpoint is required for AgHealth+ evaluation.

---

## Installation

### Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 16 GB | 32+ GB |
| CPU | 8 cores | 16+ cores |
| GPU | Optional | NVIDIA GPU for faster training |

### Setup

```bash
git clone https://github.com/aliakarma/paai-healthcare.git
cd paai-healthcare

pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

### 1) Generate Cohort And Splits

```bash
python data/synthetic/generate_patients.py --config configs/patient_sim.yaml
```

Outputs include:

- data/synthetic/cohort/patients_static.csv
- data/synthetic/cohort/vitals_longitudinal.csv
- data/synthetic/cohort/events.csv
- data/synthetic/cohort/splits/train_ids.json
- data/synthetic/cohort/splits/val_ids.json
- data/synthetic/cohort/splits/test_ids.json

### 2) Train RL On Train Split, Validate On Val Split

```bash
python rl/train.py \
  --config configs/rl_training.yaml \
  --patient_config configs/patient_sim.yaml \
  --cohort_dir data/synthetic/cohort \
  --split_dir data/synthetic/cohort/splits \
  --train_split train \
  --eval_split val \
  --device cpu
```

### 3) Evaluate On Held-Out Test Split

```bash
python evaluation/run_evaluation.py \
  --mode synthetic \
  --cohort_dir data/synthetic/cohort \
  --split_dir data/synthetic/cohort/splits \
  --train_split train \
  --eval_split test \
  --model_path rl/checkpoints/aghealth_final.zip
```

### 4) Full Pipeline (Synthetic + MIMIC)

```bash
python evaluation/run_evaluation.py \
  --mode all \
  --cohort_dir data/synthetic/cohort \
  --split_dir data/synthetic/cohort/splits \
  --train_split train \
  --eval_split test \
  --model_path rl/checkpoints/aghealth_final.zip
```

---

## Command Reference

| Command | Purpose |
|---|---|
| python data/synthetic/generate_patients.py | Build synthetic cohort + split files |
| python rl/train.py --device cpu | Train RL policy |
| python evaluation/run_evaluation.py --mode synthetic | Split-aware synthetic evaluation |
| python evaluation/run_evaluation.py --mode mimic | MIMIC anomaly validation |
| python data/mimic/extract_cohort.py --config configs/mimic_extraction.yaml | Extract MIMIC cohort artifacts |
| python data/policy_registry/validate_registry.py | Validate clinical policy assets |
| python -m pytest tests/ -v | Run test suite |

---

## Repository Layout

| Path | Description |
|---|---|
| agents/ | Domain agents (medicine, nutrition, lifestyle, emergency) |
| baselines/ | B1 rules-only, B2 predictive-only, B3 human-schedule |
| configs/ | RL, simulation, preprocessing, escalation, and MIMIC configs |
| data/synthetic/ | Synthetic cohort generator and models |
| data/mimic/ | MIMIC extractor and usage docs |
| envs/ | Gymnasium patient environment and reward/constraints |
| evaluation/ | End-to-end evaluation, stats, plots, ablations |
| governance/ | Audit log, encryption, consent, HiTL modules |
| knowledge/ | Knowledge graph and policy registry integrations |
| orchestrator/ | Routing, conflict resolution, and constraints |
| preprocessing/ | Signal denoise/normalize/feature extraction |
| rl/ | Training, policy evaluation, callbacks, checkpoints |
| tests/ | Automated tests |

---

## Data Governance

- Raw MIMIC data must not be committed.
- Synthetic outputs, checkpoints, tensorboard logs, and evaluation results are generated artifacts.
- See DATA_AVAILABILITY.md and docs/mimic_setup.md for access and compliance details.

---

## Citation

Use CITATION.cff as the source of truth. Minimal BibTeX:

```bibtex
@software{paai_healthcare_2025,
  title   = {AgHealth+: Privacy-Aware Agentic AI for IoT Healthcare (PAAI Framework)},
  author  = {Syed, Toqeer Ali and Akarma, Ali and Ali, Ahmad and Lee, It Ee and Jan, Salman and Khan, Sohail and Nauman, Muhammad},
  year    = {2025},
  version = {1.0.0},
  url     = {https://github.com/aliakarma/paai-healthcare}
}
```

---

## License

Apache 2.0. See LICENSE.
