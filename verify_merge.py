#!/usr/bin/env python3
"""
verify_merge.py — Run after extracting all 4 parts to confirm merge is complete.

Usage:
    cd paai-healthcare
    python verify_merge.py
"""
import sys, os, json, hashlib, importlib
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

B = "\033[1m"; G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"; X = "\033[0m"

def ok(m):   print(f"  {G}✓{X}  {m}")
def fail(m): print(f"  {R}✗{X}  {m}")
def warn(m): print(f"  {Y}!{X}  {m}")
def hdr(m):  print(f"\n{B}{C}{m}{X}")

# ── 1. Manifest files ─────────────────────────────────────────────────────────
def check_manifests():
    hdr("── Step 1: Manifest presence ─────────────────────────────────────")
    missing = []
    for i in range(1, 5):
        p = ROOT / f".paai_part{i}_manifest.json"
        if p.exists(): ok(f"Part {i} manifest present")
        else:          fail(f"Part {i} manifest missing  ← extract paai_part{i}.zip"); missing.append(i)
    return missing

# ── 2. Required files ─────────────────────────────────────────────────────────
REQUIRED = {
    1: [
        "requirements.txt","setup.py","LICENSE","CITATION.cff",".gitignore",
        "README.md","MERGE_GUIDE.md","verify_merge.py",
        "configs/patient_sim.yaml","configs/rl_training.yaml",
        "configs/escalation_thresholds.yaml","configs/preprocessing.yaml",
        "configs/mimic_extraction.yaml","configs/knowledge_graph.yaml",
        "data/synthetic/generate_patients.py","data/synthetic/adherence_model.py",
        "data/synthetic/hazard_model.py",
        "data/knowledge_graph/drug_food_triples.json",
        "data/knowledge_graph/condition_contraindications.json",
        "data/knowledge_graph/nutrient_deficiency.json",
        "data/policy_registry/prescriber_rules.json",
        "data/policy_registry/allergy_exclusions.json",
        "data/policy_registry/escalation_criteria.json",
        "data/policy_registry/validate_registry.py",
        "preprocessing/signal_pipeline.py","preprocessing/denoise.py",
        "preprocessing/normalise.py","preprocessing/feature_extraction.py",
    ],
    2: [
        "knowledge/knowledge_graph.py","knowledge/feature_store.py",
        "knowledge/policy_registry.py","knowledge/drug_checker.py",
        "agents/base_agent.py","agents/medicine_agent.py",
        "agents/nutrition_agent.py","agents/lifestyle_agent.py",
        "agents/emergency_agent.py",
        "governance/audit_log.py","governance/consent_manager.py",
        "governance/encryption.py","governance/hitl/patient_feedback.py",
        "governance/hitl/clinician_override.py","governance/hitl/governance_review.py",
    ],
    3: [
        "orchestrator/orchestrator.py","orchestrator/constraint_filter.py",
        "orchestrator/conflict_resolver.py","orchestrator/task_router.py",
        "envs/spaces.py","envs/patient_env.py",
        "envs/reward_function.py","envs/constraint_set.py",
        "rl/train.py","rl/callbacks.py","rl/lagrangian.py","rl/evaluate_policy.py",
        "baselines/rules_only.py","baselines/predictive_only.py",
        "baselines/human_schedule.py",
    ],
    4: [
        "evaluation/run_evaluation.py","evaluation/metrics.py",
        "evaluation/statistical_tests.py","evaluation/ablation.py",
        "evaluation/mimic_evaluation.py",
        "evaluation/plots/plot_roc.py","evaluation/plots/plot_med_quality.py",
        "evaluation/plots/plot_latency_cdf.py","evaluation/plots/plot_adherence.py",
        "evaluation/plots/plot_learning_curves.py",
        "data/mimic/extract_cohort.py",
        "tests/test_signal_pipeline.py","tests/test_agents.py",
        "tests/test_constraint_filter.py","tests/test_audit_log.py",
        "tests/test_reward_function.py","tests/test_orchestrator.py",
        "docs/architecture.md","docs/rl_training_guide.md",
        "docs/mimic_setup.md","docs/hitl_guide.md",
        ".github/workflows/test.yml",
    ],
}

def check_files():
    hdr("── Step 2: File completeness ──────────────────────────────────────")
    all_ok = True
    for part, files in REQUIRED.items():
        missing = [f for f in files if not (ROOT / f).exists()]
        if not missing:
            ok(f"Part {part}: all {len(files)} required files present")
        else:
            for f in missing: fail(f"Part {part} missing: {f}")
            all_ok = False
    return all_ok

# ── 3. Checksums ──────────────────────────────────────────────────────────────
def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def check_checksums():
    hdr("── Step 3: Checksum integrity ──────────────────────────────────────")
    all_ok = True
    for i in range(1, 5):
        mp = ROOT / f".paai_part{i}_manifest.json"
        if not mp.exists(): continue
        data   = json.loads(mp.read_text())
        # support both key names
        files  = data.get("files_in_part") or data.get("files", {})
        bad    = [f for f, h in files.items() if (ROOT/f).exists() and sha256(ROOT/f) != h]
        if not bad:
            ok(f"Part {i}: all {len(files)} checksums match")
        else:
            for f in bad: fail(f"Part {i} checksum mismatch: {f}")
            all_ok = False
    return all_ok

# ── 4. Import checks ──────────────────────────────────────────────────────────
# Partition imports by whether they need pip packages not in stdlib
STDLIB_IMPORTS = [
    ("preprocessing.signal_pipeline", "SignalPipeline",  "Algorithm 1 — signal pipeline"),
    ("knowledge.knowledge_graph",      "KnowledgeGraph", "Clinical knowledge graph"),
    ("knowledge.policy_registry",      "PolicyRegistry", "Policy registry"),
    ("knowledge.feature_store",        "FeatureStore",   "Feature store"),
    ("knowledge.drug_checker",         "DrugChecker",    "Drug checker"),
    ("agents.base_agent",              "BDIAgent",       "BDI agent base class"),
    ("agents.medicine_agent",          "MedicineAgent",  "Medicine agent (Listing 2)"),
    ("agents.nutrition_agent",         "NutritionAgent", "Nutrition agent (Listing 3)"),
    ("agents.lifestyle_agent",         "LifestyleAgent", "Lifestyle agent (Listing 4)"),
    ("agents.emergency_agent",         "EmergencyAgent", "Emergency agent (Listing 5)"),
    ("governance.audit_log",           "AuditLog",       "Immutable audit log"),
    ("governance.consent_manager",     "ConsentManager", "Consent manager"),
    ("governance.encryption",          "generate_key",   "AES encryption helpers"),
    ("orchestrator.orchestrator",      "Orchestrator",   "Main orchestrator (Algorithm 2)"),
    ("orchestrator.constraint_filter", "ConstraintFilter","Constraint filter"),
    ("orchestrator.conflict_resolver", "ConflictResolver","Conflict resolver"),
    ("orchestrator.task_router",       "TaskRouter",     "Task router"),
    ("evaluation.metrics",             "compute_roc_auc","Evaluation metrics"),
    ("evaluation.statistical_tests",   "delong_test",    "Statistical tests"),
    ("evaluation.run_evaluation",      "run_synthetic",  "Master evaluation runner"),
]
# These need gymnasium / stable-baselines3 — only warn if missing
OPTIONAL_IMPORTS = [
    ("envs.spaces",       "STATE_DIM",   "CMDP state/action constants"),
    ("envs.patient_env",  "PatientEnv",  "Patient Gym environment"),
    ("envs.reward_function","compute_reward","Reward function Eq.1"),
    ("baselines.rules_only",     "evaluate","Rules-only baseline (B1)"),
    ("baselines.predictive_only","evaluate","Predictive-only (B2)"),
    ("baselines.human_schedule", "evaluate","Human-schedule (B3)"),
]

def check_imports():
    hdr("── Step 4: Import resolution ───────────────────────────────────────")
    all_ok = True
    for mod, attr, desc in STDLIB_IMPORTS:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, attr): ok(f"{mod}.{attr}  [{desc}]")
            else:                fail(f"{mod}.{attr} — attribute missing"); all_ok = False
        except ImportError as e:
            fail(f"{mod} — ImportError: {e}"); all_ok = False
        except Exception as e:
            warn(f"{mod} — {type(e).__name__}: {e}")

    print()
    for mod, attr, desc in OPTIONAL_IMPORTS:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, attr): ok(f"{mod}.{attr}  [{desc}]")
            else:                warn(f"{mod}.{attr} — attribute missing (run: pip install -r requirements.txt)")
        except ImportError as e:
            pkg = str(e).split("'")[1] if "'" in str(e) else str(e)
            warn(f"{mod} — needs `pip install {pkg}`  (not a merge error)")
        except Exception as e:
            warn(f"{mod} — {type(e).__name__}: {e}")
    return all_ok

# ── 5. Consistency ────────────────────────────────────────────────────────────
def check_consistency():
    hdr("── Step 5: Cross-part consistency ──────────────────────────────────")
    all_ok = True
    try:
        import yaml
        sim = yaml.safe_load((ROOT/"configs/patient_sim.yaml").read_text())
        n, seed = sim.get("n_patients",0), sim.get("seed",-1)
        (ok if n==500 else fail)(f"n_patients = {n}  {'✓' if n==500 else '(expected 500)'}"); all_ok &= n==500
        (ok if seed==42 else warn)(f"seed = {seed}  {'✓' if seed==42 else '(expected 42)'}")
    except Exception as e: warn(f"patient_sim.yaml: {e}")

    try:
        from preprocessing.signal_pipeline import CHANNELS
        exp = ["sbp","dbp","glucose_mgdl","heart_rate","spo2"]
        (ok if CHANNELS==exp else fail)(f"CHANNELS = {CHANNELS}"); all_ok &= CHANNELS==exp
    except Exception as e: warn(f"CHANNELS: {e}")

    try:
        from envs.spaces import STATE_DIM, N_ACTIONS
        (ok if STATE_DIM==25 else fail)(f"STATE_DIM = {STATE_DIM}  {'✓' if STATE_DIM==25 else '(expected 25)'}"); all_ok &= STATE_DIM==25
        (ok if N_ACTIONS==5  else fail)(f"N_ACTIONS = {N_ACTIONS}  {'✓' if N_ACTIONS==5 else '(expected 5)'}");  all_ok &= N_ACTIONS==5
    except ImportError:
        warn("envs.spaces — skipped (install gymnasium first)")
    except Exception as e: warn(f"envs.spaces: {e}")
    return all_ok

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{B}{'='*64}{X}")
    print(f"{B}  AgHealth+ / PAAI — Repository Merge Verification{X}")
    print(f"{B}{'='*64}{X}")

    missing = check_manifests()
    if len(missing) == 4:
        print(f"\n{R}No manifests found. Extract at least paai_part1.zip first.{X}")
        sys.exit(1)

    f_ok = check_files()
    c_ok = check_checksums()
    i_ok = check_imports()
    s_ok = check_consistency()

    hdr("── Summary ──────────────────────────────────────────────────────────")
    results = [
        ("Manifests (all 4 parts extracted)", len(missing)==0),
        ("Files (all required files present)", f_ok),
        ("Checksums (no corruption)",          c_ok),
        ("Core imports (no missing modules)",  i_ok),
        ("Cross-part consistency",             s_ok),
    ]
    for label, passed in results:
        print(f"  {G+'PASS'+X if passed else R+'FAIL'+X}  {label}")

    print()
    if all(r[1] for r in results):
        print(f"{B}{G}✓ Merge complete — all 4 parts verified successfully!{X}")
        print(f"\n  Next steps:")
        print(f"    pip install -r requirements.txt")
        print(f"    python data/synthetic/generate_patients.py --sample 10")
        print(f"    python rl/train.py --sample 10   # smoke test")
        print(f"    pytest tests/ -v")
    else:
        print(f"{B}{R}✗ Merge has issues — review failures above.{X}")
        print(f"\n  Common fixes:")
        print(f"    Missing files  → re-extract the corresponding part zip")
        print(f"    Import errors  → pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
