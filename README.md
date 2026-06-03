# flipkart_grid

Traffic demand forecasting pipeline for the Flipkart Grid dataset. This repo contains a single best model script that:

- Re-imputes missing RoadType variables hierarchically.
- Extracts Day-over-Day (DoD) ratios, temporal slope extrapolation, multi-resolution spatial lags, and K-Means geohash clusters.
- Performs chronological Out-of-Fold (OOF) validation across 3 folds on Day-49.
- Trains an ensemble of LightGBM, XGBoost, and ExtraTrees, with an ExtraTrees guardrail.
- Corrects per-geohash residual bias based on known early Day-49 slots.
- Applies strict road-type physical bounds and writes a competition-ready `submission.csv`.

## Repo Contents

- `traffic_demand_best_model.py` — main training/validation + submission writer
- `dataset/`
	- `train.csv`
	- `test.csv`
	- `sample_submission.csv`
- `submission.csv` — generated predictions (`Index`, `demand`)
- `model_validation_report.csv` — validation scores for models and blend

## Requirements

- Windows / macOS / Linux
- Python **3.11** (recommended) or 3.12

Python packages:

- `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `xgboost`

## Setup

From the repo root:

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install numpy pandas scikit-learn lightgbm xgboost
```

### macOS / Linux (bash/zsh)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install numpy pandas scikit-learn lightgbm xgboost
```

## Run

### Model training & prediction

```powershell
python traffic_demand_best_model.py
```

### Options

- `--data-dir dataset` — change input data directory

## Outputs

After a successful run, you should see:

- `submission.csv`
	- Shape: `(41778, 2)`
	- Columns: `Index`, `demand` (clipped to road-type bounds)
- `model_validation_report.csv`
	- Chronological OOF R2 metrics for individual models and the final ensemble.

## Notes

- Validation is performed on day 49 using chronological cutoff slots (minutes 90, 105, 150).
- The pipeline applies a Ridge regression meta-learner with a strict guardrail: if ExtraTrees alone performs better than the ensemble, it falls back to ExtraTrees only to prevent overfitting.
- Randomness is fixed via `RANDOM_STATE = 42`.
