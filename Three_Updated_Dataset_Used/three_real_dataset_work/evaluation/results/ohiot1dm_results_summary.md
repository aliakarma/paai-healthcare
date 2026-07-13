# OhioT1DM Results

Dataset: OhioT1DM
Rows: 28341
Train rows: 23052
Test rows: 5289
Features: 10
Target: future 30 minute hypo/hyperglycemia event
Cutoff: 0.0000
Split: official training/testing XML split
Best model: AgHealth+

| model | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| AgHealth+ | 0.9208 | 0.9503 | 0.8894 | 0.9188 | 0.9712 |
| Predictive-only (B2) | 0.8591 | 0.8716 | 0.8451 | 0.8582 | 0.9267 |
| Human-schedule (B3) | 0.8554 | 0.9276 | 0.7735 | 0.8436 | 0.8961 |
| Rules-only (B1) | 0.8019 | 0.8732 | 0.7102 | 0.7833 | 0.8409 |