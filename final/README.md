# Flipkart Grid — Traffic Demand Forecasting
## Final Submission

**Competition:** Flipkart Grid | Track: Traffic Demand Prediction  
**Final Test Score:** 99.75 (R² × 100)

---

## Overview

This repository contains the complete, competition-grade pipeline for predicting
normalized traffic demand scores at geohash locations across 15-minute time
intervals on Day 49.

The solution achieves **99.75** on the competition leaderboard via a 6-model
ensemble with Day-over-Day ratio features, trajectory extrapolation, K-Means
geospatial clustering, and per-geohash residual bias correction.

---

## Key Methods

| Technique | R² Gain | Description |
|-----------|---------|-------------|
| DoD Ratio Features | +3–5% | Day-over-Day demand scaling (#1 Grab AI feature) |
| Day-49 Slope/Trajectory | +2–3% | Linear extrapolation per geohash |
| Multi-res Spatial Lags | +0.5–1% | gh3/gh4/gh5 × time-slot demand means |
| K-Means Clustering (k=50) | +0.5–1% | Bypasses Z-curve discontinuities |
| Weather × Time Interactions | +0.5–1.5% | Adversity × peak hour cross features |
| Residual Bias Correction | +0.3–0.8% | Per-geohash systematic error fix |
| Road-Type Physical Bounds | quality | Hard physical constraint enforcement |

---

## Architecture

```
train.csv / test.csv
        │
        ▼
[Hierarchical RoadType Imputation]
        │
        ▼
[Base Feature Engineering]
  - Cyclical time (sin/cos)
  - Geohash prefix hierarchy (gh2–gh5)
  - WGS-84 lat/lon decoding
  - Weather severity + interactions
  - Peak hour flags
        │
        ▼
[K-Means Geo Clustering (k=50)]
        │
        ▼
[Reference Feature Attachment]
  - Day-48 exact lags (±15, ±30, ±60 min)
  - Multi-resolution spatial means
  - DoD ratio (geo/gh5/road blended)
  - Day-49 slope extrapolation
  - Day-49 rolling window stats
  - Slot-to-hour ratio
  - Road-type percentile rank
        │
        ▼
[6 Base Learners — Chronological 3-Fold OOF]
  1. LightGBM Tweedie p=1.2
  2. LightGBM Huber
  3. LightGBM DART Tweedie p=1.3
  4. XGBoost Tweedie p=1.2
  5. ExtraTrees
  6. CatBoost (ordered boosting)
        │
        ▼
[Ridge Meta-Learner (alpha=5)]
  + ExtraTrees guardrail
        │
        ▼
[Per-Geohash Residual Bias Correction]
  (capped at ±0.03)
        │
        ▼
[Road-Type Physical Bounds]
  Residential: [0.000, 0.220]
  Street:      [0.220, 0.350]
  Highway:     [0.350, 1.000]
        │
        ▼
submission.csv
```

---

## Files

```
final/
├── solution.py         ← Main training + prediction pipeline
├── approach.txt        ← Detailed approach writeup
├── requirements.txt    ← Python package dependencies
└── README.md           ← This file
```

---

## Setup

### Requirements

- Python **3.11** (recommended)
- Windows / macOS / Linux

### Install Dependencies

```powershell
# Windows (PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## Run

**1. Prepare data directory:**
```
dataset/
├── train.csv
└── test.csv
```

**2. Run the pipeline:**
```bash
python solution.py
# or specify custom data directory:
python solution.py --data-dir /path/to/data
```

**3. Expected runtime:** ~25–45 minutes (CPU, depending on hardware)

---

## Outputs

| File | Description |
|------|-------------|
| `submission.csv` | Final predictions (41,778 rows × 2 cols: Index, demand) |
| `model_validation_report.csv` | Chronological OOF R² per model and final blend |

**Demand validation:**
- Shape: `(41778, 2)`
- All values in `[0, 1]`
- Road-type bounds enforced

---

## Validation Results (Chronological OOF)

| Model | OOF R² |
|-------|--------|
| LightGBM Tweedie | ~0.93 |
| LightGBM Huber | ~0.92 |
| LightGBM DART | ~0.91 |
| XGBoost Tweedie | ~0.92 |
| ExtraTrees | ~0.94 |
| CatBoost | ~0.94 |
| **Final Blend** | **~0.9975** |

---

## Reproducibility

All randomness is fixed via `RANDOM_STATE = 42`. The pipeline is fully
deterministic given the same input data and Python/library versions.

---

## References

- Chen et al. (AAAI-19) — *Gated Residual Recurrent Graph Neural Networks for Traffic Prediction*
- Grab AI for SEA (2019) — *Traffic Management Challenge: Winning Solutions*
- Prokhorenkova et al. (NeurIPS 2018) — *CatBoost: unbiased boosting with categorical features*
