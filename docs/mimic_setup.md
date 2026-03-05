# MIMIC-IV Setup Guide

## Step 1 — Complete CITI Training (2 hours, free)
1. Go to https://citiprogram.org
2. Create account → Add course → "Data or Specimens Only Research"
3. Complete "Biomedical Research" course
4. Download certificate

## Step 2 — Apply for PhysioNet Access
1. Go to https://physionet.org/register/
2. Create account and upload CITI certificate
3. Wait for approval (typically 1–3 days)

## Step 3 — Sign MIMIC-IV Data Use Agreement
1. Go to https://physionet.org/content/mimiciv/
2. Click "Request Access" and sign DUA
3. Wait for approval (same day to 48 hours)

## Step 4 — Download Required Files
```bash
mkdir -p data/mimic/raw/hosp data/mimic/raw/icu
wget -r -N -c -np \
  --user YOUR_PHYSIONET_USERNAME \
  --ask-password \
  https://physionet.org/files/mimiciv/2.2/hosp/diagnoses_icd.csv.gz \
  -P data/mimic/raw/hosp/
# Repeat for: icustays, chartevents, labevents, prescriptions, admissions
```

## Step 5 — Run Extraction
```bash
python data/mimic/extract_cohort.py --config configs/mimic_extraction.yaml
```

## CRITICAL REMINDERS
- **NEVER commit raw MIMIC data to git** (DUA violation + legal risk)
- `.gitignore` already blocks `data/mimic/raw/` and `data/mimic/extracted/*.csv`
- Only `cohort_summary.json` (non-identifiable aggregate stats) may be committed
