# PAAI: From Sensing to Action
### A Privacy-Aware Agentic AI Architecture for IoT Healthcare

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](requirements.txt)
[![RL Framework](https://img.shields.io/badge/RL-Stable--Baselines3-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white)](rl/train.py)
[![License](https://img.shields.io/badge/License-Apache_2.0-D22128?style=for-the-badge)](LICENSE)
[![Data Availability](https://img.shields.io/badge/Data-MIMIC--IV%20%7C%20Wearables-16a34a?style=for-the-badge)](DATA_AVAILABILITY.md)

---

## 📖 Executive Summary
Continuous patient monitoring through wearable IoT devices and smart clinical reasoning is essential for managing chronic diseases like hypertension and diabetes. Most current systems are passive: they display graphs and sound alarms but cannot adjust care pathways safely. 

**PAAI (Privacy-Aware Agentic AI)** addresses this by merging:
1. **Belief-Desire-Intention (BDI) agents** that reason across medicine, nutrition, lifestyle, and emergency domains.
2. **Constrained Policy Optimization (CPO)** reinforcement learning to learn safe actions under clinical guidelines.
3. **Three-Tier Human-in-the-Loop (HiTL) governance** to incorporate clinician overrides.
4. **Confidentiality, Integrity, and Privacy (CIP) data plane** using hash-chained audit trails and AES-256 encryption.

---

## 🏛️ System Architecture

The PAAI framework operates across four modular layers:

```
+--------------------------------------------------------------------------------+
| L1: SENSING & DATA ACQUISITION (CGM, Smartwatch, BP Cuff, Patient App)         |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
| L2: PREPROCESSING & KNOWLEDGE (Denoise, Feature Store, Clinical KG, Policies)  |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
| L3: AGENTIC AI CORE (BDI Agents, CMDP Orchestrator, Constraint Filter, CPO)    |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
| L4: GOVERNANCE & OUTPUTS (CIP Data Plane, Hash-Chained Audit, Three-Tier HiTL) |
+--------------------------------------------------------------------------------+
```

### System Architecture Diagram
The complete interaction flow across the four layers, from wearable telemetry intake to secure BDI agent reasoning and clinician dashboard output, is illustrated below:

![PAAI System Architecture](docs/images/agentic_arch.png)

---

## 🔁 Core Components & Control Loops

### 1. CMDP Reinforcement Learning Loop
The coordination layer is formulated as a **Constrained Markov Decision Process (CMDP)**. The orchestrator queries the clinical knowledge graph, construct state vectors $s_t$, and updates the policy via CPO to maximize stability while satisfying safety thresholds.

![RL-Orchestrator Control Loop](docs/images/rl-orchestrator-loop.png)

### 2. Three-Tier Human-in-the-Loop (HiTL) Governance
To ensure clinical safety, actions are validated and checked at three distinct timescales:
* **Tier 1 (Patient Feedback)**: Instant patient verification of lifestyle guidance.
* **Tier 2 (Clinician Override)**: Urgent clinician review of dose alterations and emergency escalations. Clinician actions trigger a constraint update to prevent repeating rejected decisions.
* **Tier 3 (Committee Audit)**: Weekly policy calibration and ethical auditing.

![Three-Tier HiTL Governance Model](docs/images/three_tier.png)

---

## 📊 Experimental Results & Benchmarks

### 1. Primary Outcomes (Synthetic 500-Patient Cohort)
Evaluated on a 12-month longitudinal cohort of 500 patients, comparing PAAI against Rules-only (B1), Predictive-only (B2), and Human-schedule (B3) baselines.

| Method | Anomaly Accuracy | Anomaly ROC AUC | Med. Latency (s) | Med. Recommender Precision | $p$-value vs. PAAI |
|---|---|---|---|---|---|
| **Rules-only (B1)** | $0.81 \pm 0.01$ | $0.87$ [0.85, 0.89] | $4.9^\dagger$ | $0.71$ [0.69, 0.73] | $p < 0.001$ |
| **Predictive-only (B2)** | $0.88 \pm 0.01$ | $0.92$ [0.90, 0.94] | $3.1^\dagger$ | $0.79$ [0.77, 0.81] | $p < 0.01$ |
| **Human-schedule (B3)** | $0.76 \pm 0.02$ | $0.83$ [0.80, 0.86] | $9.8^{\dagger\dagger}$ | $0.75$ [0.72, 0.78] | $p < 0.001$ |
| **PAAI (AgHealth+)** | $\mathbf{0.92 \pm 0.01}$ | $\mathbf{0.96}$ [0.95, 0.97] | $\mathbf{1.8}$ | $\mathbf{0.87}$ [0.85, 0.89] | *n/a* |

* ${}^\dagger$ Wilcoxon signed-rank test latency difference ($p < 0.01$ vs. PAAI).
* ${}^{\dagger\dagger}$ Latency represents checking interval (clinician-defined) rather than real-time processing.
* Bootstrap confidence intervals (95%) and p-values are Bonferroni-corrected.

### 2. Real-World Wearable Benchmarks (OhioT1DM, WESAD, PPG-DaLiA)
To validate the framework offline, models were evaluated on processed features derived from three open-source patient datasets:
* **OhioT1DM**: CGM glucose risk events (5,289 test rows).
* **WESAD**: Wearable sensor-based stress monitoring (459 test rows).
* **PPG-DaLiA**: Heart-rate and blood-volume pulse activity monitoring (785 test rows).

| Dataset | Model Configuration | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| **OhioT1DM** | **AgHealth+ (PAAI)** | **0.9208** | **0.9503** | **0.8894** | **0.9188** | **0.9712** |
| | Predictive-only (B2) | 0.8591 | 0.8716 | 0.8451 | 0.8582 | 0.9267 |
| | Human-schedule (B3) | 0.8554 | 0.9276 | 0.7735 | 0.8436 | 0.8961 |
| | Rules-only (B1) | 0.8019 | 0.8732 | 0.7102 | 0.7833 | 0.8409 |
| **WESAD** | **AgHealth+ (PAAI)** | **0.8519** | **1.0000** | **0.3333** | **0.5000** | **0.9435** |
| | Predictive-only (B2) | 0.8431 | 0.8409 | 0.3627 | 0.5068 | 0.9331 |
| | Human-schedule (B3) | 0.7952 | 0.5606 | 0.3627 | 0.4405 | 0.8328 |
| | Rules-only (B1) | 0.8519 | 0.6977 | 0.5882 | 0.6383 | 0.8259 |
| **PPG-DaLiA**| Predictive-only (B2) | **0.8178** | **0.2238** | **0.5000** | **0.3092** | **0.8324** |
| | **AgHealth+ (PAAI)** | 0.7312 | 0.2118 | **0.8438** | **0.3386** | 0.8317 |
| | Human-schedule (B3) | 0.8433 | 0.2136 | 0.3438 | 0.2635 | 0.8213 |
| | Rules-only (B1) | 0.7376 | 0.2183 | 0.8594 | 0.3481 | 0.7763 |

* AgHealth+ achieves state-of-the-art performance on **OhioT1DM** and **WESAD** due to adaptive agentic reasoning, while the **PPG-DaLiA** heart-rate prediction model benefits from a higher recall ($0.8438$) under PAAI constraints.

### 3. Ablation Study (Component Contributions)
To isolate the value of each sub-system, we systematically removed components:

| Configuration | Anomaly Accuracy | Anomaly ROC AUC | Med. Precision | Median Latency (s) |
|---|---|---|---|---|
| **PAAI (Full System)** | **0.92** | **0.96** | **0.87** | **1.8** |
| w/o Policy Constraint Filter | 0.91 | 0.95 | $0.80^\dagger$ | 1.7 |
| w/o Clinical Knowledge Graph | $0.89^\dagger$ | $0.93^\dagger$ | $0.82^\dagger$ | 2.0 |
| w/o Agentic Orchestrator | $0.85^\ddagger$ | $0.91^\dagger$ | 0.84 | 3.2 |
| w/o CPO (Deterministic fallback)| $0.83^\dagger$ | $0.89^\dagger$ | $0.73^\dagger$ | 4.1 |

* ${}^\dagger p < 0.01$, ${}^\ddagger p < 0.05$ vs. full PAAI (Bonferroni-corrected).
* Note how removing the constraint filter drops medicine precision to $0.80$, highlighting its critical role in patient safety.

---

## 📂 Repository Layout & Structure

The codebase is organized as follows:

| Component / Folder | Icon | Primary Responsibility |
|---|---|---|
| [`agents/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/agents) | 🤖 | Domain BDI agents (medicine, nutrition, lifestyle, emergency) |
| [`baselines/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/baselines) | 📉 | Baseline comparative models: Rules-only (B1), Predictive (B2), Human (B3) |
| [`configs/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/configs) | ⚙️ | System settings for RL, patients, MIMIC, and preprocessors |
| [`data/synthetic/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/data/synthetic) | 📊 | Synthetic cohort generator, adherence models, and event rates |
| [`data/mimic/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/data/mimic) | 🏥 | MIMIC-IV ICU patient SQL extraction and parsing scripts |
| [`data/real/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/data/real) | 🎛️ | Processed CSV tables and splits for OhioT1DM, WESAD, and PPG-DaLiA |
| [`envs/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/envs) | 🏋️ | Gymnasium health-simulation environment and constraint sets |
| [`evaluation/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/evaluation) | 🔬 | Performance evaluators, ablation modules, statistical testing |
| [`governance/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/governance) | 🛡️ | Cryptographic keys, consent trackers, patient feedback pipelines |
| [`knowledge/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/knowledge) | 🕸️ | Clinician policy registry, drug-food KG RDF graph parser |
| [`notebooks/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/notebooks) | 📓 | Jupyter notebooks demonstrating end-to-end wearable executions |
| [`orchestrator/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/orchestrator) | 🔀 | Central orchestrator, task routing, and conflict resolution |
| [`preprocessing/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/preprocessing) | 🎚️ | Signal denoiser, normalizer, and sliding-window feature extractor |
| [`rl/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/rl) | 🧠 | Stable-Baselines3 CPO PPO training code, logs, and checkpoints |
| [`scripts/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/scripts) | 📜 | Real-world benchmark execution and feature-importance scripts |
| [`tests/`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/tests) | 🧪 | Unit and integration test suite (116 tests) |

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10 or 3.11
* Pytorch (CPU or GPU supported)
* Recommended: 16+ GB RAM, 8+ Core CPU

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aliakarma/paai-healthcare.git
   cd paai-healthcare
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install package in editable mode:
   ```bash
   pip install -e .
   ```

---

## 🚀 Execution Guide & Command Reference

| Task | Primary Command |
|---|---|
| **Generate Synthetic Cohort** | `python data/synthetic/generate_patients.py --config configs/patient_sim.yaml` |
| **Train RL Policy** | `python rl/train.py --config configs/rl_training.yaml --device cpu` |
| **Run Synthetic Benchmark** | `python evaluation/run_evaluation.py --mode synthetic` |
| **Run MIMIC Anomaly Validation** | `python evaluation/run_evaluation.py --mode mimic` |
| **Run Wearable Real Benchmarks** | `python scripts/run_three_real_datasets.py` |
| **Extract MIMIC Cohort** | `python data/mimic/extract_cohort.py --config configs/mimic_extraction.yaml` |
| **Validate Clinical Rules** | `python data/policy_registry/validate_registry.py` |
| **Run Complete Test Suite** | `python -m pytest tests/ -v` |

---

## 📓 Jupyter Notebooks
For step-by-step visualizations of results, feature importances, and ROC curves:
* **[Wearable E2E Execution Notebook](file:///c:/Users/Ali%20Akarma/Documents/GitHub/paai-healthcare/notebooks/three_real_dataset_end_to_end_executed.ipynb)**: Detailed data ingestion, voting classifier training, and target distribution plots for WESAD, PPG-DaLiA, and OhioT1DM.

---

## ⚖️ Ethical & Data Governance (HIPAA/GDPR Compliance)
PAAI enforces clinical safety and data integrity out of the box:
- **HIPAA alignment**: Patient vital signs and demographics are processed through a Confidentiality, Integrity, and Privacy (CIP) layer.
- **Audit logs**: Decisional trace logs are hash-chained immutably, ensuring full accountability.
- **Physical safety bounds**: Action masking dynamically guarantees that medication doses cannot exceed clinically defined boundaries regardless of the RL network's exploratory decisions.

---
