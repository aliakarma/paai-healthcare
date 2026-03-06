# Data Availability Statement

## Synthetic Evaluation Cohort

**Availability**: Fully open-source and reproducible.

### Access Method
Generate the synthetic cohort locally using:
```bash
python data/synthetic/generate_patients.py --seed 42 --num_patients 500 --months 12
```

### About the Dataset
| Property | Details |
|----------|---------|
| **Cohort Size** | 500 patients, 12-month longitudinal |
| **Generation Code** | `data/synthetic/generate_patients.py` + `data/synthetic/adherence_model.py` + `data/synthetic/hazard_model.py` |
| **GitHub Repository** | [github.com/aliakarma/paai-healthcare](https://github.com/aliakarma/paai-healthcare) |
| **License** | Apache 2.0 (same as codebase) |
| **Reproducibility** | Deterministic given `--seed 42` |
| **Zenodo Archive** | Will be published upon journal acceptance |

### Output Format
```
data/synthetic/cohort_500_12mo_seed42/
├── patients_static.csv           # Demographics, conditions, medications
├── vitals_longitudinal.csv       # Time-series: BP, HR, SpO2, glucose, weight
├── medications_adherence.csv     # Adherence patterns for each patient
└── cohort_metadata.json          # Dataset generation parameters
```

### Data Dictionary
All synthetic patient features are defined in `data/synthetic/patient_schema.json`.

---

## Real-World Evaluation (MIMIC-IV)

**Availability**: Restricted; requires credentialing.

### Access Method
1. Apply for PhysioNet credentials: [physionet.org](https://physionet.org)
2. Sign MIMIC-IV Data Use Agreement
3. Download using scripts in `data/mimic/extract_cohort.py`

### Dataset Details
| Property | Details |
|----------|---------|
| **Source** | MIMIC-IV v2.2 (Johnson et al., 2023) |
| **Subset** | ICU patients with ≥2 chronic conditions, 48+ hour stay |
| **GitHub Handling** | **Raw data NOT committed** — only extraction scripts and aggregates |
| **Ethical Requirements** | DUA compliance enforced via `.gitignore` (blocks `data/mimic/raw/` and individual `.csv` files) |
| **Reproducibility** | Extraction fully specified in `data/mimic/extract_cohort.py` with SQL queries |

### Available Aggregates (Non-Identifiable)
The following aggregate statistics may be shared without DUA violation:
- `data/mimic/cohort_summary.json` — basic cohort statistics (counts, age distributions)
- `data/mimic/feature_distributions.json` — aggregate feature means/stds

---

## RL Policy Checkpoints

**Availability**: Distributed with repository; model weights in public release.

### Checkpoint Details
| File | Details |
|------|---------|
| `rl/checkpoints/aghealth_ppo_500k_steps.zip` | Policy trained for 500k environment steps |
| `rl/checkpoints/aghealth_ppo_2m_steps.zip` | **Recommended for paper results** — final policy (2M steps) |
| `rl/checkpoints/baseline_predictive_*.pkl` | Pretrained anomaly detector (sklearn) for baseline B2 |

### How to Use
```python
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Load trained policy
model = PPO.load("rl/checkpoints/aghealth_ppo_2m_steps")

# Evaluate or continue training
obs, info = env.reset()
action, _states = model.predict(obs)
next_obs, reward, done, info = env.step(action)
```

### Reproducibility
All checkpoints are trained with:
- Fixed seed: `--seed 42`
- Hyperparameters: `configs/rl_training.yaml`
- Exact environment: `envs/patient_env.py`

Results should be bitwise identical on same hardware/CUDA version.

---

## Knowledge Graphs & Policy Registries

**Availability**: Bundled with repository.

### Locations
```
data/knowledge_graph/
├── drug_food_triples.json            # 2,347 drug-food pairs from DrugBank
├── condition_contraindications.json  # ADA/AHA clinical contraindications
├── nutrient_deficiency.json          # WHO/USDA nutrient links
└── kg_schema.ttl                     # RDF schema definition

data/policy_registry/
├── prescriber_rules.json             # Dosing rules, safety windows
├── allergy_exclusions.json           # Allergen tags
├── escalation_criteria.json          # Vital thresholds for alert escalation
└── validate_registry.py              # JSON schema validation
```

### Source Attribution
- **DrugBank**: [drugbank.ca](https://drugbank.ca) (CC BY-NC 4.0)
- **ADA/AHA**: American Diabetes Association & American Heart Association clinical guidelines (public guidelines, parameterized)
- **WHO/USDA**: Public nutrition databases

### License
DrugBank content is included under CC BY-NC 4.0; derivative clinical guidelines are public domain parameterizations.

---

## Code & Documentation

**Availability**: Fully open-source.

- **Repository**: [github.com/aliakarma/paai-healthcare](https://github.com/aliakarma/paai-healthcare)
- **License**: Apache 2.0
- **Persistent Archive**: Will be assigned Zenodo DOI upon publication
- **GitHub Release**: Tagged release `v1.0` includes all code, configs, and benchmarks

---

## Supplementary Materials

**Availability**: Included in `/docs` directory and GitHub repository.

| Document | Location | Content |
|----------|----------|---------|
| Architecture Reference | `docs/architecture.md` | System design and mathematical formulation |
| RL Training Guide | `docs/rl_training_guide.md` | Hyperparameter tuning and convergence debugging |
| HiTL Governance | `docs/hitl_guide.md` | Three-tier feedback loop implementation |
| MIMIC Setup | `docs/mimic_setup.md` | Credentialing and data extraction |

---

## Contact for Data Access Issues

- **Synthetic cohort issues**: [GitHub Issues](https://github.com/aliakarma/paai-healthcare/issues)
- **MIMIC-IV access**: [PhysioNet Help](https://physionet.org/help/)
- **General inquiries**: 443059463@stu.iu.edu.sa

---

## Fair Use & Citation

If you use code or data from this repository:

1. **Synthetic cohort**: Cite this paper + cite `patient_schema.json` and generation parameters
2. **MIMIC-IV derivative**: Cite MIMIC-IV original paper (Johnson et al., 2023) + this paper
3. **Knowledge graphs**: Cite original sources (DrugBank, ADA/AHA, WHO) as indicated in `data/`
4. **RL policy**: Cite this paper + `rl/train.py` methodology

See CITATION.cff for comprehensive citation information.

---

**Last updated**: March 2026
