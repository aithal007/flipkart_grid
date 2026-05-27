# flipkart_grid

Traffic demand forecasting pipeline for the Flipkart Grid dataset. This repo contains a single “best model” script that:

- Builds time + geohash + contextual features
- Runs a small validation routine on day-49 continuation slices
- Trains the final model(s) and writes a competition-ready `submission.csv`

## Repo Contents

- `traffic_demand_best_model.py` — main training/validation + submission writer
- `traffic_demand_best_model.ipynb` — optional notebook version
- `dataset/`
	- `train.csv`
	- `test.csv`
	- `sample_submission.csv`
- `submission.csv` — generated predictions (`Index`, `demand`)
- `model_validation_report.csv` — validation scores + chosen strategy/weights

## Requirements

- Windows / macOS / Linux
- Python **3.11** (recommended) or 3.12
	- Note: Python 3.13 can fail to install ML dependencies on some machines due to missing wheels.

Python packages:

- `numpy`, `pandas`
- `scikit-learn`
- `lightgbm`
- `xgboost`

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

### Full run (recommended)

Recomputes validation + retrains final model(s) + overwrites outputs.

```powershell
python traffic_demand_best_model.py --force-validation --force-final
```

### Fast run

Skips XGBoost and ExtraTrees training. Use this when you want a quicker sanity run.

```powershell
python traffic_demand_best_model.py --fast --force-validation --force-final
```

### Other useful flags

- `--data-dir dataset` — change input directory
- `--output submission.csv` — change output filename
- `--force-validation` — recompute validation even if `model_validation_report.csv` exists
- `--force-final` — retrain final models even if the output CSV already exists

## Outputs

After a successful run, you should see:

- `submission.csv`
	- Shape: `(41778, 2)`
	- Columns: `Index`, `demand`
	- `demand` is clipped to `[0, 1]`
- `model_validation_report.csv`
	- Per-fold model metrics + a “chosen strategy” line
- `feature_importance_lightgbm.csv` (optional)
	- Written only when the final run trains `lgbm_l2_deep` (if the chosen strategy is ExtraTrees-only, this file may not be produced).

## Notes

- The script includes strict shape/consistency assertions against the provided dataset.
- Validation is performed on day 49 by holding out later minutes (see `VALIDATION_CUTOFFS` in the script).
- Final predictions can be a weighted blend; depending on validation, the script may choose an `extra_trees_fallback` strategy.

## Reproducibility

Most randomness is fixed via a constant seed (`RANDOM_STATE = 42`), but exact results can still vary slightly across OS/library versions.
