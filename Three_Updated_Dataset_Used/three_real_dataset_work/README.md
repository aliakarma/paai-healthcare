# Three Real Dataset Evaluation

This folder contains the real-dataset evaluation workflow for the PAAI / AgHealth+ paper. The paper methods are kept the same, and only the datasets are changed from the earlier synthetic or sample-data setting to three real public datasets.

## Datasets

The workflow evaluates three real datasets:

- **OhioT1DM** for glucose-risk and diabetes monitoring.
- **WESAD** for wearable stress and lifestyle-signal validation.
- **PPG-DaLiA** for wearable PPG, activity, and heart-rate risk validation.

Large raw zip files are not committed to GitHub. Place them in the parent `data` folder before rerunning:

- `data/OhioT1DM/`
- `data/WESAD.zip`
- `data/PPG_FieldStudy.zip`

Processed tables used for the reported results are saved in `data/processed`.

## Methods

The same four paper methods are evaluated on all three datasets:

- **Rules-only (B1)**: deterministic threshold-rule baseline.
- **Predictive-only (B2)**: trained predictive baseline.
- **Human-schedule (B3)**: static schedule-style baseline.
- **AgHealth+**: soft-voting ensemble implementation of the proposed method.

The AgHealth+ ensemble keeps the same practical model setup:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Extra Trees
- SVM

## Final Best Results

| Dataset | Best Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| OhioT1DM | AgHealth+ | 0.9208 | 0.9503 | 0.8894 | 0.9188 | 0.9712 |
| WESAD | AgHealth+ | 0.8519 | 1.0000 | 0.3333 | 0.5000 | 0.9435 |
| PPG-DaLiA | Predictive-only (B2) | 0.8178 | 0.2238 | 0.5000 | 0.3092 | 0.8324 |

Full replacement tables are available in:

- `docs/paper_replacement_numbers.md`
- `evaluation/results/model_metrics.csv`

## Folder Structure

- `scripts/`: end-to-end training and figure-generation code.
- `notebooks/`: clean notebook and executed notebook.
- `data/processed/`: processed dataset tables and train/test splits.
- `evaluation/results/`: detailed metrics, predictions, confusion counts, and summaries.
- `evaluation/figures/`: paper-ready PNG and PDF figures.
- `figures/`: direct-access figure copies.
- `outputs/`: direct-access result and processed-table copies.
- `docs/`: paper replacement numbers.

## Run

From this folder:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\run_three_real_datasets.py
```

The notebook version is:

- `notebooks/three_real_dataset_end_to_end.ipynb`
- `notebooks/three_real_dataset_end_to_end_executed.ipynb`

## Notes

The results are generated from real train/test evaluation. A leakage issue was checked during validation; the PPG-DaLiA target was corrected so the final results do not use the target variable as a feature.
