"""Advanced Spatiotemporal Modeling for Traffic Demand Forecasting.

This script implements the complete, highly optimized pipeline:
- Hierarchical RoadType Imputation
- Spatial Neighbor Maps (Cold-Start handling)
- Momentum features & Cyclical time
- OOF Stacking with LightGBM, XGBoost, and ExtraTrees
- Strict RoadType Bound Post-Processing
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
RANDOM_STATE = 42
VALIDATION_CUTOFFS = [90, 105]

ROAD_BOUNDS = {
    "Residential": (0.0, 0.219997),
    "Street": (0.220016, 0.349908),
    "Highway": (0.350009, 1.0),
}

def parse_timestamp(value: str) -> int:
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)

def decode_geohash(geohash: str) -> tuple[float, float]:
    lat = [-90.0, 90.0]
    lon = [-180.0, 180.0]
    even_bit = True
    for char in geohash:
        bits = BASE32.index(char)
        for mask in (16, 8, 4, 2, 1):
            if even_bit:
                mid = (lon[0] + lon[1]) / 2
                if bits & mask:
                    lon[0] = mid
                else:
                    lon[1] = mid
            else:
                mid = (lat[0] + lat[1]) / 2
                if bits & mask:
                    lat[0] = mid
                else:
                    lat[1] = mid
            even_bit = not even_bit
    return (lat[0] + lat[1]) / 2, (lon[0] + lon[1]) / 2

def mode_or_none(values: pd.Series) -> str | None:
    modes = values.dropna().mode()
    return None if modes.empty else str(modes.iloc[0])

def impute_road_type(train: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    known = train.dropna(subset=["RoadType"]).copy()
    key_sets = [
        ["geohash", "NumberofLanes", "LargeVehicles", "Landmarks"],
        ["geohash", "NumberofLanes", "LargeVehicles"],
        ["geohash", "NumberofLanes"],
        ["NumberofLanes", "LargeVehicles", "Landmarks"],
        ["NumberofLanes", "LargeVehicles"],
        ["geohash"],
    ]
    for keys in key_sets:
        mapping = known.groupby(keys)["RoadType"].agg(mode_or_none).dropna().to_dict()
        mask = out["RoadType"].isna()
        if not mask.any():
            break
        vals = out.loc[mask, keys].apply(lambda row: mapping.get(tuple(row)), axis=1)
        out.loc[mask, "RoadType"] = vals.values

    # Final structural rules for edge cases
    mask = out["RoadType"].isna()
    out.loc[mask & (out["NumberofLanes"] >= 4), "RoadType"] = "Highway"
    
    mask = out["RoadType"].isna()
    out.loc[
        mask & (out["NumberofLanes"] == 1) & (out["LargeVehicles"] == "Not Allowed") & (out["Landmarks"] == "Yes"),
        "RoadType"
    ] = "Street"
    
    out["RoadType"] = out["RoadType"].fillna("Residential")
    return out

def apply_bounds(pred: np.ndarray, road_types: pd.Series) -> np.ndarray:
    out = pred.astype(float).copy()
    roads = road_types.astype(str).to_numpy()
    for road, (lo, hi) in ROAD_BOUNDS.items():
        mask = roads == road
        out[mask] = np.clip(out[mask], lo, hi)
    return np.clip(out, 0, 1)

def build_spatial_neighbors(train_df: pd.DataFrame, test_df: pd.DataFrame, k: int = 8) -> dict[str, list[str]]:
    train_geos = train_df[["geohash"]].drop_duplicates()
    test_geos = test_df[["geohash"]].drop_duplicates()
    all_geos = pd.concat([train_geos, test_geos], ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    decoded = all_geos["geohash"].map(decode_geohash)
    all_geos["lat"] = [pair[0] for pair in decoded]
    all_geos["lon"] = [pair[1] for pair in decoded]
    
    coords = all_geos[["lat", "lon"]].to_numpy()
    train_coords = train_geos["geohash"].map(decode_geohash)
    train_coords = np.array([[pair[0], pair[1]] for pair in train_coords])
    train_names = train_geos["geohash"].to_numpy()
    
    mapping = {}
    for idx, row in all_geos.iterrows():
        dist = np.sum((train_coords - coords[idx]) ** 2, axis=1)
        order = np.argsort(dist)
        if row["geohash"] in train_names:
            neighbors = train_names[order[1 : k + 1]].tolist()
        else:
            neighbors = train_names[order[:k]].tolist()
        mapping[row["geohash"]] = neighbors
    return mapping

def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["minute"] = out["timestamp"].map(parse_timestamp)
    out["slot"] = out["minute"] // 15
    out["hour"] = out["minute"] // 60
    
    # Cyclical time encodings
    out["time_sin"] = np.sin(2 * np.pi * out["slot"] / 96)
    out["time_cos"] = np.cos(2 * np.pi * out["slot"] / 96)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    # Geohash prefixes
    out["gh4"] = out["geohash"].str[:4]
    out["gh5"] = out["geohash"].str[:5]
    
    decoded = out["geohash"].map(decode_geohash)
    out["lat"] = [pair[0] for pair in decoded]
    out["lon"] = [pair[1] for pair in decoded]

    out["Weather"] = out["Weather"].fillna("__MISSING__")
    out["Temperature_filled"] = out["Temperature"].fillna(out["Temperature"].median())
    
    # Treat NumberofLanes as categorical effectively by string casting later, but numeric for interaction
    out["is_large_allowed"] = (out["LargeVehicles"] == "Allowed").astype(np.int8)
    out["has_landmark"] = (out["Landmarks"] == "Yes").astype(np.int8)
    out["lanes_x_large"] = out["NumberofLanes"] * out["is_large_allowed"]
    return out

def get_neighbor_stats(day48: pd.DataFrame, target: pd.DataFrame, neighbor_map: dict[str, list[str]], prefix: str) -> pd.DataFrame:
    lookup = day48.set_index(["geohash", "minute"])["demand"].to_dict()
    means, maxes = [], []
    for geohash, minute in zip(target["geohash"], target["minute"]):
        values = [lookup.get((neighbor, minute), np.nan) for neighbor in neighbor_map.get(geohash, [])]
        arr = np.array(values, dtype=float)
        if np.isfinite(arr).any():
            means.append(float(np.nanmean(arr)))
            maxes.append(float(np.nanmax(arr)))
        else:
            means.append(np.nan)
            maxes.append(np.nan)
    out = target.copy()
    out[f"{prefix}_neighbor_mean"] = means
    out[f"{prefix}_neighbor_max"] = maxes
    return out

def calibration_tables(history: pd.DataFrame, day48: pd.DataFrame) -> dict[str, pd.DataFrame | float]:
    known49 = history[(history["day"] == 49) & history["demand"].notna()].copy()
    if known49.empty:
        return {"global_ratio": 1.0}

    day48_exact = day48[["geohash", "minute", "demand"]].rename(columns={"demand": "d48"})
    joined = known49.merge(day48_exact, on=["geohash", "minute"], how="inner")
    joined = joined[(joined["d48"].notna()) & (joined["d48"] > 0)]
    if joined.empty:
        return {"global_ratio": 1.0}

    global_ratio = float(joined["demand"].sum() / joined["d48"].sum())
    tables: dict[str, pd.DataFrame | float] = {"global_ratio": global_ratio}

    for keys, name, strength in [
        (["geohash"], "ratio_geo", 0.25),
        (["gh5"], "ratio_gh5", 0.75),
        (["RoadType"], "ratio_road", 2.0),
    ]:
        agg = joined.groupby(keys, dropna=False).agg(y=("demand", "sum"), x=("d48", "sum")).reset_index()
        agg[name] = (agg["y"] + strength * global_ratio) / (agg["x"] + strength)
        tables[name] = agg[keys + [name]]
    return tables

def attach_reference_features(df: pd.DataFrame, reference: pd.DataFrame, history: pd.DataFrame, neighbor_map: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    day48 = reference[reference["day"] == 48].copy()
    global_mean = float(reference["demand"].mean())

    # Target encodings
    geo_mean = day48.groupby("geohash")["demand"].mean().rename("geo_mean")
    road_slot = day48.groupby(["RoadType", "minute"])["demand"].mean().rename("road_slot_mean")
    out = out.merge(geo_mean, on="geohash", how="left")
    out = out.merge(road_slot, on=["RoadType", "minute"], how="left")

    # Exact lag
    exact = day48[["geohash", "minute", "demand"]].rename(columns={"demand": "lag_day48_exact"})
    out = out.merge(exact, on=["geohash", "minute"], how="left")

    # Shifted lags (15m, 30m, 60m)
    for shift, label in [(-15, "prev15"), (15, "next15"), (-60, "prev60")]:
        shifted = exact.copy()
        shifted["minute"] = shifted["minute"] - shift
        shifted = shifted.rename(columns={"lag_day48_exact": f"day48_{label}"})
        out = out.merge(shifted, on=["geohash", "minute"], how="left")

    # Neighbor features
    out = get_neighbor_stats(day48, out, neighbor_map, "day48")

    # Momentum ratios
    ratios = calibration_tables(history, day48)
    out["ratio_global"] = float(ratios.get("global_ratio", 1.0))
    for name, keys in [
        ("ratio_geo", ["geohash"]),
        ("ratio_gh5", ["gh5"]),
        ("ratio_road", ["RoadType"]),
    ]:
        table = ratios.get(name)
        if isinstance(table, pd.DataFrame):
            out = out.merge(table, on=keys, how="left")
        else:
            out[name] = np.nan

    for col in ["ratio_geo", "ratio_gh5", "ratio_road"]:
        out[col] = out[col].fillna(out["ratio_global"])

    out["ratio_blend"] = 0.5 * out["ratio_geo"] + 0.3 * out["ratio_gh5"] + 0.1 * out["ratio_road"] + 0.1 * out["ratio_global"]
    out["prior_calibrated"] = out["lag_day48_exact"].fillna(out["geo_mean"]).fillna(global_mean) * out["ratio_blend"]
    
    # Cold-start fill for prior_calibrated utilizing neighbors
    mask = out["prior_calibrated"].isna()
    if mask.any():
        out.loc[mask, "prior_calibrated"] = out.loc[mask, "day48_neighbor_mean"] * out.loc[mask, "ratio_blend"]

    # Short term lags from Day 49
    known49 = history[(history["day"] == 49) & history["demand"].notna()].copy()
    if not known49.empty:
        lag_source = known49[["geohash", "minute", "demand"]].copy()
        for lag in [15, 30]:
            shifted = lag_source.copy()
            shifted["minute"] = shifted["minute"] + lag
            shifted = shifted.rename(columns={"demand": f"lag49_t{lag}"})
            out = out.merge(shifted, on=["geohash", "minute"], how="left")
    else:
        out["lag49_t15"] = np.nan
        out["lag49_t30"] = np.nan

    return out

def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    exclude = {"demand", "Index", "timestamp"}
    categorical = [
        "geohash", "RoadType", "LargeVehicles", "Landmarks", 
        "Weather", "gh4", "gh5", "NumberofLanes"
    ]
    categorical = [c for c in categorical if c in df.columns]
    numeric = [c for c in df.columns if c not in exclude and c not in categorical]
    return numeric + categorical, numeric, categorical

def prepare_categories(train_df: pd.DataFrame, predict_df: pd.DataFrame, categorical: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.copy()
    pred_out = predict_df.copy()
    for col in categorical:
        train_out[col] = train_out[col].astype(str).fillna("__MISSING__")
        pred_out[col] = pred_out[col].astype(str).fillna("__MISSING__")
        categories = sorted(set(train_out[col]) | set(pred_out[col]))
        train_out[col] = pd.Categorical(train_out[col], categories=categories)
        pred_out[col] = pd.Categorical(pred_out[col], categories=categories)
    return train_out, pred_out

def make_models(numeric: list[str], categorical: list[str]) -> dict[str, object]:
    models = {
        "lgbm_l2": LGBMRegressor(
            objective="regression_l2", n_estimators=260, learning_rate=0.06, num_leaves=95,
            subsample=0.88, colsample_bytree=0.82, random_state=RANDOM_STATE, verbose=-1
        ),
        "lgbm_huber": LGBMRegressor(
            objective="huber", alpha=0.9, n_estimators=220, learning_rate=0.065, num_leaves=63,
            subsample=0.9, colsample_bytree=0.86, random_state=RANDOM_STATE+1, verbose=-1
        ),
        "lgbm_tweedie_1p2": LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.2, n_estimators=260, learning_rate=0.06,
            num_leaves=95, subsample=0.9, colsample_bytree=0.85, random_state=RANDOM_STATE+2, verbose=-1
        ),
        "lgbm_tweedie_1p5": LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.5, n_estimators=240, learning_rate=0.06,
            num_leaves=75, subsample=0.9, colsample_bytree=0.88, random_state=RANDOM_STATE+3, verbose=-1
        ),
        "xgb_tweedie": XGBRegressor(
            objective="reg:tweedie", tweedie_variance_power=1.2, n_estimators=180, learning_rate=0.065,
            max_depth=7, subsample=0.88, colsample_bytree=0.86, tree_method="hist",
            enable_categorical=True, random_state=RANDOM_STATE+4, n_jobs=-1
        )
    }
    
    transformer = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="__MISSING__"), OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), categorical)
    ])
    et = ExtraTreesRegressor(n_estimators=110, min_samples_leaf=2, max_features=0.82, n_jobs=-1, random_state=RANDOM_STATE+5)
    models["extra_trees"] = make_pipeline(transformer, et)
    
    return models

def train_predict_models(
    train_df: pd.DataFrame, predict_df: pd.DataFrame, features: list[str], 
    numeric: list[str], categorical: list[str]
) -> tuple[pd.DataFrame, dict[str, object]]:
    train_ready, pred_ready = prepare_categories(train_df, predict_df, categorical)
    y = train_ready["demand"].to_numpy()
    predictions = {}
    fitted_models = {}
    
    models = make_models(numeric, categorical)
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        if "lgbm" in name:
            model.fit(train_ready[features], y, categorical_feature=categorical)
        elif name == "xgb_tweedie":
            model.fit(train_ready[features], y)
        else:
            model.fit(train_ready[features], y)
            
        predictions[name] = model.predict(pred_ready[features])
        fitted_models[name] = model

    pred_df = pd.DataFrame(predictions, index=predict_df.index)
    return pred_df, fitted_models

def get_blend_weights(preds: pd.DataFrame, y: np.ndarray, road_types: pd.Series) -> dict[str, float]:
    columns = list(preds.columns)
    best_score = -1e9
    best_weights = {col: 1.0 / len(columns) for col in columns}
    
    candidates = [np.full(len(columns), 1.0 / len(columns))]
    for i in range(len(columns)):
        one = np.zeros(len(columns))
        one[i] = 1.0
        candidates.append(one)
        
    rng = np.random.default_rng(RANDOM_STATE)
    for _ in range(3000):
        raw = rng.random(len(columns)) ** 1.6
        candidates.append(raw / raw.sum())
        
    values = preds[columns].to_numpy()
    for weights in candidates:
        blended = np.clip(values @ weights, 0, 1)
        blended = apply_bounds(blended, road_types) # Evaluate with bounds!
        score = r2_score(y, blended)
        if score > best_score:
            best_score = score
            best_weights = dict(zip(columns, weights))
    return best_weights, best_score

def run_pipeline(data_dir: Path) -> None:
    print("Loading data...", flush=True)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    
    # 1. Hierarchical Imputation
    print("Imputing RoadType...", flush=True)
    train = impute_road_type(train, train)
    test = impute_road_type(train, test)
    
    # 2. Build Neighbor Map (over entire space)
    print("Building spatial neighborhood map...", flush=True)
    neighbor_map = build_spatial_neighbors(train, test, k=8)
    
    # 3. Base feature extraction
    train_base = add_base_features(train)
    test_base = add_base_features(test)
    day48 = train_base[train_base["day"] == 48].copy()
    day49 = train_base[train_base["day"] == 49].copy()
    
    print("\n--- Running Chronological OOF Validation ---")
    oof_pred_parts = []
    oof_y_parts = []
    oof_roads = []
    
    for val_start in VALIDATION_CUTOFFS:
        print(f"Validation fold: hold out day 49 from minute {val_start}...", flush=True)
        fit_raw = train_base[(train_base["day"] == 48) | ((train_base["day"] == 49) & (train_base["minute"] < val_start))].copy()
        val_raw = day49[day49["minute"] >= val_start].copy()
        
        fit_feat = attach_reference_features(fit_raw, day48, fit_raw, neighbor_map)
        val_feat = attach_reference_features(val_raw, day48, fit_raw, neighbor_map)
        features, numeric, categorical = feature_columns(fit_feat)
        
        model_preds, _ = train_predict_models(fit_feat, val_feat, features, numeric, categorical)
        
        preds = pd.DataFrame(index=val_feat.index)
        preds["prior_calibrated"] = val_feat["prior_calibrated"].to_numpy()
        preds = pd.concat([preds, model_preds], axis=1).clip(0, 1)
        
        oof_pred_parts.append(preds.reset_index(drop=True))
        oof_y_parts.append(val_feat["demand"].to_numpy())
        oof_roads.append(val_feat["RoadType"])

    oof_preds = pd.concat(oof_pred_parts, ignore_index=True).fillna(0)
    oof_y = np.concatenate(oof_y_parts)
    oof_road_types = pd.concat(oof_roads, ignore_index=True)

    weights, blend_score = get_blend_weights(oof_preds, oof_y, oof_road_types)
    print(f"\nFinal Stacked Validation R2 (with RoadType bounds): {blend_score:.5f}")
    print("Optimal Weights:", json.dumps(weights, indent=2))
    
    print("\n--- Training Final Models on All Data ---")
    train_feat = attach_reference_features(train_base, day48, train_base, neighbor_map)
    test_feat = attach_reference_features(test_base, day48, train_base, neighbor_map)
    features, numeric, categorical = feature_columns(train_feat)
    
    model_preds, _ = train_predict_models(train_feat, test_feat, features, numeric, categorical)
    
    preds = pd.DataFrame(index=test_feat.index)
    preds["prior_calibrated"] = test_feat["prior_calibrated"].to_numpy()
    preds = pd.concat([preds, model_preds], axis=1).clip(0, 1)
    
    values = preds[list(weights)].to_numpy()
    final_pred = np.clip(values @ np.array(list(weights.values())), 0, 1)
    
    print("\nApplying RoadType bounds to final predictions...", flush=True)
    final_pred = apply_bounds(final_pred, test_feat["RoadType"])
    
    submission = pd.DataFrame({"Index": test["Index"].to_numpy(), "demand": final_pred})
    
    out_path = Path("submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"\nWrote final predictions to {out_path.resolve()}")
    print(submission["demand"].describe())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset", type=Path)
    args = parser.parse_args()
    run_pipeline(args.data_dir)
