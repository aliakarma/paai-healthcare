# MIMIC-IV Data — Access Instructions

> **IMPORTANT**: Raw MIMIC-IV data files are **never** committed to this repository.
> This directory contains only processing scripts and a non-identifiable summary.
> Committing raw MIMIC data violates the PhysioNet Data Use Agreement (DUA).

---

## Step 1 — Obtain PhysioNet Access (≈ 1–2 weeks)

1. Create a free account at https://physionet.org/register/
2. Complete the **CITI Data or Specimens Only Research** training
   - URL: https://about.citiprogram.org/
   - Takes approximately 2–3 hours
   - Download your completion certificate
3. Upload the certificate to your PhysioNet profile
4. Submit a **credentialing request** at PhysioNet
5. Once credentialed, request access to **MIMIC-IV** at:
   https://physionet.org/content/mimiciv/

Approval typically takes 24–72 hours after credentialing is complete.

---

## Step 2 — Download MIMIC-IV Files

After approval, download the following tables from MIMIC-IV (v2.2 or later):

**From `hosp/` module:**
- `diagnoses_icd.csv.gz`
- `admissions.csv.gz`
- `prescriptions.csv.gz`
- `labevents.csv.gz`

**From `icu/` module:**
- `icustays.csv.gz`
- `chartevents.csv.gz`   ← Large file (~30 GB compressed)

Place all files under `data/mimic/raw/` (this folder is git-ignored).

---

## Step 3 — Extract the Cohort

```bash
python data/mimic/extract_cohort.py \
  --config configs/mimic_extraction.yaml \
  --mimic_dir data/mimic/raw \
  --output_dir data/mimic/extracted
```

This produces `data/mimic/extracted/vitals_mimic.csv` and
`data/mimic/extracted/events_mimic.csv` — also git-ignored.

---

## Step 4 — Run MIMIC Validation

```bash
python evaluation/run_evaluation.py --mode mimic
```

Results are saved to `evaluation/results/mimic_validation_results.json`.

---

## What IS committed to this repo

- `data/mimic/extract_cohort.py` — extraction script
- `data/mimic/preprocess_mimic.py` — preprocessing script
- `data/mimic/README.md` — this file
- `data/mimic/extracted/cohort_summary.json` — non-identifiable aggregate stats only
