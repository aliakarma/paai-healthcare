# Paper Replacement Numbers

The same four methods were evaluated on three real datasets.

## Best Models

| dataset | model | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| OhioT1DM | AgHealth+ | 0.9208 | 0.9503 | 0.8894 | 0.9188 | 0.9712 |
| PPG-DaLiA | Predictive-only (B2) | 0.8178 | 0.2238 | 0.5000 | 0.3092 | 0.8324 |
| WESAD | AgHealth+ | 0.8519 | 1.0000 | 0.3333 | 0.5000 | 0.9435 |

## All Results

| dataset | model | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| OhioT1DM | AgHealth+ | 0.9208 | 0.9503 | 0.8894 | 0.9188 | 0.9712 |
| OhioT1DM | Predictive-only (B2) | 0.8591 | 0.8716 | 0.8451 | 0.8582 | 0.9267 |
| OhioT1DM | Human-schedule (B3) | 0.8554 | 0.9276 | 0.7735 | 0.8436 | 0.8961 |
| OhioT1DM | Rules-only (B1) | 0.8019 | 0.8732 | 0.7102 | 0.7833 | 0.8409 |
| WESAD | AgHealth+ | 0.8519 | 1.0000 | 0.3333 | 0.5000 | 0.9435 |
| WESAD | Predictive-only (B2) | 0.8431 | 0.8409 | 0.3627 | 0.5068 | 0.9331 |
| WESAD | Human-schedule (B3) | 0.7952 | 0.5606 | 0.3627 | 0.4405 | 0.8328 |
| WESAD | Rules-only (B1) | 0.8519 | 0.6977 | 0.5882 | 0.6383 | 0.8259 |
| PPG-DaLiA | Predictive-only (B2) | 0.8178 | 0.2238 | 0.5000 | 0.3092 | 0.8324 |
| PPG-DaLiA | AgHealth+ | 0.7312 | 0.2118 | 0.8438 | 0.3386 | 0.8317 |
| PPG-DaLiA | Human-schedule (B3) | 0.8433 | 0.2136 | 0.3438 | 0.2635 | 0.8213 |
| PPG-DaLiA | Rules-only (B1) | 0.7376 | 0.2183 | 0.8594 | 0.3481 | 0.7763 |

## Dataset Notes

### OhioT1DM

Rows: 28341
Train rows: 23052
Test rows: 5289
Target rule: future 30 minute hypo/hyperglycemia event
Split: official training/testing XML split

### WESAD

Rows: 2326
Train rows: 1867
Test rows: 459
Target rule: stress state label equals 2
Split: subject-level 80/20 split

### PPG-DaLiA

Rows: 4020
Train rows: 3235
Test rows: 785
Target rule: top 25 percent training heart-rate windows
Split: subject-level 80/20 split
