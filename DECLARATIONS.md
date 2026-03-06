# Author Contributions, Funding & Declarations\n## Detailed Author Contributions

### Toqeer Ali Syed (T.A.S.)
- **Conceptualization**: Designed the four-layer PAAI architecture, CMDP formulation, and Lagrangian safety constraints
- **Methodology**: Specifications for signal processing pipeline, BDI orchestrator, and RL-HiTL integration
- **Software Implementation**: Core systems — `preprocessing/`, `orchestrator/`, `rl/`, `governance/audit_log.py`
- **Evaluation**: Synthetic cohort design, statistical testing, reproducibility framework
- **Writing**: Original manuscript draft, architecture sections, reproducibility statements

### Ali Akarma (A.A.)
- **Software Architecture**: Modular agent framework (`agents/`, `orchestrator/`), knowledge graph interface
- **Implementation**: BDI agents (medicine, nutrition, lifestyle, emergency), conflict resolver, governance modules
- **Integration**: Feature store, policy registry, constraint filter system integration
- **Testing**: Comprehensive test suite design and implementation (`tests/`)
- **Writing**: Agent methodology sections, governance framework documentation

### Ahmad Ali (A.A.)
- **Data Engineering**: Synthetic cohort generation (`data/synthetic/`), adherence/hazard models
- **Baselines**: Implementation of three baseline methods (`baselines/`)  
- **Evaluation**: MIMIC-IV extraction pipeline, metrics computation (`evaluation/`)
- **Statistical Analysis**: Statistical test implementation and validation

### It Ee Lee (I.E.L.), Salman Jan (S.J.), Sohail Khan (S.K.), Muhammad Nauman (M.N.)
- **Critical Review**: Architecture validation, methodology critique, scientific soundness assessment
- **Domain Expertise**: Clinical validation (diabetes, cardiology, nephrology), privacy/security review
- **Governance Framework**: Human-in-the-loop design and clinical workflow alignment
- **Manuscript Review**: Technical accuracy validation, clarity improvements

---

## Funding & Support

### Financial Support
Authors receive no funding for this research.

### Computational Resources
- Synthetic experiments: University computing cluster (8× CPU, 32 GB RAM, 50 GB SSD per run)
- MIMIC-IV processing: Single 16-core workstation, ~2 weeks cumulative runtime
- RL training: GPU acceleration not required; CPU experiments reproduced on commodity hardware

**Funders Had No Role**: In study design, data collection/analysis, writing decisions, or publication choice.

---

## Competing Interests

### Financial Conflicts
- **None declared**: No author has financial interest in healthcare AI companies or commercial implementations
- **Publication neutral**: No financial incentive for positive/negative results

### Institutional Affiliations
- Authors are affiliated with Islamic University of Madinah, Multimedia University, Arab Open University, and Effat University
- No commercial entities are involved; all are publicly-funded educational institutions

### Prior Work
This implementation builds on open-source frameworks:
- Stable-Baselines3 (SB3) — open-source RL library
- RDFlib & NetworkX — knowledge graph libraries
- Gymnasium — environment API

All dependencies are properly cited in `requirements.txt` and `CITATION.cff`.

---

## Data & Code Availability

### Data Sharing
- **Synthetic data**: Fully reproducible, no restrictions
- **MIMIC-IV derivatives**: Original DUA restricts sharing; scripts provided for obtaining access
- **Knowledge graphs**: Public sources (DrugBank, ADA/AHA) — derivatives available under same licenses

### Code Sharing
- All code: Apache 2.0 license, freely available at [github.com/aliakarma/paai-healthcare](https://github.com/aliakarma/paai-healthcare)
- Will be (or has been) published on GitHub + Zenodo for persistent archiving upon publication

### Reproducibility
- **Seed fixation**: `--seed 42` ensures deterministic reproduction
- **Dependency pinning**: All Python package versions locked in `requirements.txt`
- **Docker/containerization**: Future work to improve portability

---

## Open Science Commitment

This work adheres to open science principles:

✅ **Open Source Code** — Apache 2.0 license, freely available
✅ **Reproducible Methods** — All hyperparameters, seeds, and procedures documented
✅ **Data Transparency** — Full disclosure of synthetic data generation process and MIMIC-IV access requirements
✅ **Accessible Documentation** — Installation, usage, and extension guides provided
✅ **Pre-registration Ready** — Methodology specified before evaluation on test sets

---

## Ethical Approval

### Data Use Compliance
- **MIMIC-IV**: Evaluation conducted under PhysioNet DUA (CC0 waiver)
- **Synthetic data**: No human subjects (entirely computational)
- **No human experiments**: This is a software/algorithm paper, not a clinical trial

### Algorithmic Ethics
- **Privacy**: Encryption and audit logging implemented per HIPAA/GDPR standards
- **Fairness**: Subgroup analysis performed (see `evaluation/statistical_tests.py`)
- **Transparency**: All decisions logged immutably
- **Safety**: Three-tier human oversight prevents autonomous critical decisions

### Declaration
This research does not involve direct human subject research and uses only previously published de-identified datasets (MIMIC-IV) and synthetic data. All ethical considerations are documented in ETHICS.md.

---

## Policy Registry & Knowledge Graph Attribution

Knowledge sources used in `data/knowledge_graph/`:

| Source | License | Attribution |
|--------|---------|------------|
| DrugBank | CC BY-NC 4.0 | Wishart et al., Nucleic Acids Res., 2023 |
| ADA/AHA Guidelines | Public domain | American Diabetes Association, American Heart Association |
| WHO Nutrition Database | CC BY 3.0 IGO | World Health Organization |
| USDA FoodData Central | Public domain | United States Department of Agriculture |

All derivatives respect the original licenses.

---

## Manuscript & Authorship Statement

**Lead Author Manuscript Statement** (required by major journals):

"I affirm that this manuscript is an honest, transparent, and complete account of the study. I affirm that no important aspects of the study have been omitted, and that any discrepancies from original planned protocol have been disclosed. All listed authors have reviewed and approved the final manuscript."

---

**Last Updated**: March 2026  
**Version**: 1.0  
**Corr. Author**: Toqeer Ali Syed (toqeer@iu.edu.sa)
