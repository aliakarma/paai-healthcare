# PPG-DaLiA Results

Dataset: PPG-DaLiA
Rows: 4020
Train rows: 3235
Test rows: 785
Features: 10
Target: top 25 percent training heart-rate windows
Cutoff: 104.4868
Split: subject-level 80/20 split
Best model: Predictive-only (B2)

| model | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| Predictive-only (B2) | 0.8178 | 0.2238 | 0.5000 | 0.3092 | 0.8324 |
| AgHealth+ | 0.7312 | 0.2118 | 0.8438 | 0.3386 | 0.8317 |
| Human-schedule (B3) | 0.8433 | 0.2136 | 0.3438 | 0.2635 | 0.8213 |
| Rules-only (B1) | 0.7376 | 0.2183 | 0.8594 | 0.3481 | 0.7763 |