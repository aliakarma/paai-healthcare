# Three Updated Dataset Used

This branch contains the clean real-dataset evaluation package for the PAAI / AgHealth+ paper.

The work is inside:

- `three_real_dataset_work/`

Raw dataset inputs are expected under:

- `data/`

Large raw zip files are ignored by Git, but the processed CSV files, notebooks, figures, metrics, and paper-ready result summaries are included.

## Datasets

This branch evaluates three real datasets:

- **OhioT1DM**: glucose-risk and diabetes monitoring.
- **WESAD**: wearable stress and lifestyle-signal validation.
- **PPG-DaLiA**: wearable PPG, activity, and heart-rate risk validation.

## Methods

The same four paper methods are used:

- **Rules-only (B1)**
- **Predictive-only (B2)**
- **Human-schedule (B3)**
- **AgHealth+**

## Best Results

| Dataset | Best Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| OhioT1DM | AgHealth+ | 0.9208 | 0.9503 | 0.8894 | 0.9188 | 0.9712 |
| WESAD | AgHealth+ | 0.8519 | 1.0000 | 0.3333 | 0.5000 | 0.9435 |
| PPG-DaLiA | Predictive-only (B2) | 0.8178 | 0.2238 | 0.5000 | 0.3092 | 0.8324 |

Main outputs:

- `three_real_dataset_work/docs/paper_replacement_numbers.md`
- `three_real_dataset_work/evaluation/results/model_metrics.csv`
- `three_real_dataset_work/notebooks/three_real_dataset_end_to_end_executed.ipynb`
- `three_real_dataset_work/evaluation/figures/`

The main branch is not modified by this work.
