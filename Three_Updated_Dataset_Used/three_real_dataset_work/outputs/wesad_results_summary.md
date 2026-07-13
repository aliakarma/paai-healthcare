# WESAD Results

Dataset: WESAD
Rows: 2326
Train rows: 1867
Test rows: 459
Features: 10
Target: stress state label equals 2
Cutoff: 0.5000
Split: subject-level 80/20 split
Best model: AgHealth+

| model | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| AgHealth+ | 0.8519 | 1.0000 | 0.3333 | 0.5000 | 0.9435 |
| Predictive-only (B2) | 0.8431 | 0.8409 | 0.3627 | 0.5068 | 0.9331 |
| Human-schedule (B3) | 0.7952 | 0.5606 | 0.3627 | 0.4405 | 0.8328 |
| Rules-only (B1) | 0.8519 | 0.6977 | 0.5882 | 0.6383 | 0.8259 |