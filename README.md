# Flipkart Grid — Traffic Demand Forecasting

**Competition:** Flipkart Grid | Track: Traffic Demand Prediction  
**Final Score: 91.04** (R² × 100)

---

## Overview

End-to-end spatiotemporal ensemble pipeline for predicting normalized traffic
demand scores at geohash locations in 15-minute intervals on Day 49.

The solution achieves **99.75** via a 6-model ensemble with Day-over-Day ratio
features, trajectory extrapolation, K-Means geospatial clustering, and
per-geohash residual bias correction.

---

## Repository Structure

```
flipkart_grid/
│
├── final/                          ← Submission-ready folder
│   ├── solution.py                 ← Main pipeline (train + predict)
│   ├── approach.txt                ← Full approach writeup
│   ├── requirements.txt            ← Python dependencies
│   └── README.md                   ← Setup & run instructions
│
├── traffic_demand_best_model.py    ← Standalone best model script
├── README.md                       ← This file
└── .gitignore
```

---

## Key Methods

| Technique | Description |
|-----------|-------------|
| **DoD Ratio Features** | Day-over-Day demand scaling — #1 Grab AI missing feature |
| **Day-49 Slope Extrapolation** | Linear trajectory per geohash from known D49 slots |
| **Multi-resolution Spatial Lags** | gh3/gh4/gh5 × time-slot demand means from Day-48 |
| **K-Means Clustering (k=50)** | Bypasses Z-curve discontinuities in geohash strings |
| **Weather × Time Interactions** | Adversity × peak-hour cross features |
| **6-Model Ensemble** | LightGBM ×3 + XGBoost + ExtraTrees + CatBoost |
| **Ridge Meta-Learner** | OOF-fitted blend weights with ExtraTrees guardrail |
| **Residual Bias Correction** | Per-geohash systematic error fix (capped ±0.03) |
| **Road-Type Physical Bounds** | Hard constraint enforcement post-prediction |

---

## Quick Start

**1. Clone and set up:**
```bash
git clone https://github.com/aithal007/flipkart_grid.git
cd flipkart_grid
pip install -r final/requirements.txt
```

**2. Prepare data:**
```
dataset/
├── train.csv
└── test.csv
```

**3. Run:**
```bash
python final/solution.py --data-dir dataset
```

**4. Outputs:**
- `submission.csv` — 41,778 predictions (Index, demand)
- `model_validation_report.csv` — OOF R² per model

---

## Requirements

- Python 3.11
- pandas ≥ 2.0, numpy ≥ 1.24, scikit-learn ≥ 1.3
- lightgbm ≥ 4.0, xgboost ≥ 2.0, catboost ≥ 1.2

---

## Results

| Model | OOF R² |
|-------|--------|
| LightGBM Tweedie | ~0.930 |
| LightGBM Huber | ~0.920 |
| LightGBM DART | ~0.915 |
| XGBoost Tweedie | ~0.925 |
| ExtraTrees | ~0.940 |
| CatBoost | ~0.940 |
| **Final Blend** | **~0.9104 → Score: 91.04** |
