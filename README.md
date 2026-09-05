# Fraud Detection with Temporal Feature Engineering

This project investigates whether temporal transaction features improve machine-learning fraud detection under severe class imbalance.

## Project question

**Do engineered temporal behavioral features provide meaningful additional signal for fraud detection compared with a baseline without those engineered features?**

The analysis compares **Logistic Regression**, **Random Forest**, and **XGBoost** using transaction-level data.

## Methods

The project includes:

- data preprocessing and categorical encoding;
- temporal feature engineering;
- class-imbalance handling with SMOTE;
- Logistic Regression, Random Forest, and XGBoost;
- evaluation with accuracy, precision, recall, F1-score, and ROC-AUC;
- XGBoost feature-importance analysis.

### Engineered temporal features

Examples include:

- 1-day transaction count;
- 1-day and 7-day rolling transaction amount averages;
- balance change rate;
- time since the previous transaction;
- customer-level transaction amount z-score;
- hour of day and nighttime indicator;
- weekday;
- daily transaction count.

## Results

The original study reported:

| Model | Temporal Features | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | Yes | 0.9500 | 0.0400 | 0.0000 | 0.0000 | 0.4936 |
| Random Forest | Yes | 0.7900 | 0.0500 | 0.1700 | 0.0800 | 0.4956 |
| Logistic Regression | Yes | 0.7600 | 0.0500 | 0.2100 | 0.0800 | 0.5005 |
| XGBoost | No | 0.7896 | 0.0508 | 0.1794 | 0.0792 | 0.5010 |
| Random Forest | No | 0.7404 | 0.0507 | 0.2340 | 0.0833 | 0.5015 |
| Logistic Regression | No | 0.6371 | 0.0521 | 0.3605 | 0.0911 | 0.5022 |

### Main finding

Temporal features did **not** substantially improve fraud discrimination in this experiment. Although some configurations increased overall accuracy, minority-class recall/F1 and ROC-AUC did not improve meaningfully. Static/account-level features remained more influential.

## Repository structure

```text
.
├── fraud_detection.ipynb
├── requirements.txt
└── data/
    └── README.md
```

The raw dataset is **not included** in this repository until its redistribution license is confirmed.

## Methodological notes

This repository documents the original exploratory experiment while also making its limitations explicit.

- SMOTE was applied only after the train-test split, so the held-out test set was not oversampled.
- However, SMOTE was applied after integer encoding categorical variables; nearest-neighbor interpolation between category codes is not an ideal treatment of nominal features.
- Temporal features were engineered before a random train-test split. A stricter temporal study should use chronological splitting and leakage-aware feature construction.
- The original "without temporal features" baseline retained raw date/time-related columns after encoding, so it is better understood as a comparison **without engineered temporal features**.

## Future improvements

A follow-up version could use:

- chronological train/validation/test splits;
- train-only preprocessing and feature engineering;
- one-hot, target, or native categorical handling where appropriate;
- class weighting or categorical-aware oversampling;
- threshold tuning and precision-recall analysis;
- anomaly-detection or hybrid approaches for rare-event detection.

## Tech stack

Python, Pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, Matplotlib, Seaborn
