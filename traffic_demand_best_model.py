"""Max-score traffic demand model for the Flipkart Grid hackathon dataset.

Run from the repository root:

    python traffic_demand_best_model.py

The script writes:
    - submission.csv
    - model_validation_report.csv
    - feature_importance_lightgbm.csv
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
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
RANDOM_STATE = 42
VALIDATION_CUTOFFS = [90, 105]
PRIOR_MODEL_NAMES = {"prior_calibrated", "prior_geo_time", "prior_neighbor"}


def parse_timestamp(value: str) -> int:
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)


def decode_geohash(geohash: str) -> tuple[float, float]:
    """Decode a geohash to the center latitude/longitude without dependencies."""
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


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["minute"] = out["timestamp"].map(parse_timestamp)
    out["slot"] = out["minute"] // 15
    out["hour"] = out["minute"] // 60
    out["minute_in_hour"] = out["minute"] % 60
    out["time_sin"] = np.sin(2 * np.pi * out["slot"] / 96)
    out["time_cos"] = np.cos(2 * np.pi * out["slot"] / 96)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    out["gh2"] = out["geohash"].str[:2]
    out["gh3"] = out["geohash"].str[:3]
    out["gh4"] = out["geohash"].str[:4]
    out["gh5"] = out["geohash"].str[:5]
    out["gh_last"] = out["geohash"].str[-1]

    unique_geohashes = pd.Series(out["geohash"].drop_duplicates().to_numpy(), name="geohash")
    decoded = unique_geohashes.map(decode_geohash)
    geo_lookup = pd.DataFrame(
        {
            "geohash": unique_geohashes,
            "lat": [pair[0] for pair in decoded.values],
            "lon": [pair[1] for pair in decoded.values],
        }
    )
    out = out.merge(geo_lookup, on="geohash", how="left")
    out["lat_rank"] = out["lat"].rank(method="dense").astype(int)
    out["lon_rank"] = out["lon"].rank(method="dense").astype(int)
    out["lat_lon_sum"] = out["lat"] + out["lon"]
    out["lat_lon_diff"] = out["lat"] - out["lon"]

    for col in ["RoadType", "Weather"]:
        out[f"{col}_missing"] = out[col].isna().astype(np.int8)
        out[col] = out[col].fillna("__MISSING__")
    out["Temperature_missing"] = out["Temperature"].isna().astype(np.int8)
    out["Temperature_filled"] = out["Temperature"].fillna(out["Temperature"].median())
    out["Temperature_sq"] = out["Temperature_filled"] ** 2
    out["is_large_allowed"] = (out["LargeVehicles"] == "Allowed").astype(np.int8)
    out["has_landmark"] = (out["Landmarks"] == "Yes").astype(np.int8)
    out["lanes_x_large"] = out["NumberofLanes"] * out["is_large_allowed"]
    out["lanes_x_landmark"] = out["NumberofLanes"] * out["has_landmark"]
    return out


def add_group_stats(
    base: pd.DataFrame,
    target: pd.DataFrame,
    keys: list[str],
    prefix: str,
    stats: tuple[str, ...] = ("mean", "std", "median", "min", "max"),
) -> pd.DataFrame:
    agg = base.groupby(keys, dropna=False)["demand"].agg(list(stats)).reset_index()
    agg.columns = keys + [f"{prefix}_{stat}" for stat in stats]
    return target.merge(agg, on=keys, how="left")


def nearest_neighbor_stats(day48: pd.DataFrame) -> pd.DataFrame:
    geo = (
        day48.groupby(["geohash", "lat", "lon"])["demand"]
        .agg(["mean", "std", "median", "max"])
        .reset_index()
    )
    coords = geo[["lat", "lon"]].to_numpy()
    values = geo[["mean", "std", "median", "max"]].to_numpy()
    neighbor_rows = []
    for i, row in geo.iterrows():
        dist = np.sum((coords - coords[i]) ** 2, axis=1)
        order = np.argsort(dist)
        neighbors = order[1:9] if len(order) > 1 else order[:1]
        nvals = values[neighbors]
        neighbor_rows.append(
            {
                "geohash": row["geohash"],
                "neighbor8_mean": float(np.nanmean(nvals[:, 0])),
                "neighbor8_std": float(np.nanmean(nvals[:, 1])),
                "neighbor8_median": float(np.nanmean(nvals[:, 2])),
                "neighbor8_max": float(np.nanmean(nvals[:, 3])),
            }
        )
    return pd.DataFrame(neighbor_rows)


def nearest_neighbor_map(day48: pd.DataFrame, k: int = 8) -> dict[str, list[str]]:
    geo = day48[["geohash", "lat", "lon"]].drop_duplicates("geohash").reset_index(drop=True)
    coords = geo[["lat", "lon"]].to_numpy()
    mapping: dict[str, list[str]] = {}
    for i, row in geo.iterrows():
        dist = np.sum((coords - coords[i]) ** 2, axis=1)
        order = np.argsort(dist)
        mapping[row["geohash"]] = geo.loc[order[1 : k + 1], "geohash"].tolist()
    return mapping


def attach_same_time_neighbor_features(
    target: pd.DataFrame,
    source: pd.DataFrame,
    neighbor_map: dict[str, list[str]],
    prefix: str,
) -> pd.DataFrame:
    lookup = source.set_index(["geohash", "minute"])["demand"].to_dict()
    means, stds, maxes, mins = [], [], [], []
    for geohash, minute in zip(target["geohash"], target["minute"]):
        values = [lookup.get((neighbor, minute), np.nan) for neighbor in neighbor_map.get(geohash, [])]
        arr = np.array(values, dtype=float)
        if np.isfinite(arr).any():
            means.append(float(np.nanmean(arr)))
            stds.append(float(np.nanstd(arr)))
            maxes.append(float(np.nanmax(arr)))
            mins.append(float(np.nanmin(arr)))
        else:
            means.append(np.nan)
            stds.append(np.nan)
            maxes.append(np.nan)
            mins.append(np.nan)
    out = target.copy()
    out[f"{prefix}_neighbor_same_time_mean"] = means
    out[f"{prefix}_neighbor_same_time_std"] = stds
    out[f"{prefix}_neighbor_same_time_max"] = maxes
    out[f"{prefix}_neighbor_same_time_min"] = mins
    return out


def attach_target_day_lags(target: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    known49 = history[(history["day"] == 49) & history["demand"].notna()].copy()
    if known49.empty:
        for lag in [15, 30, 60]:
            out[f"lag49_t{lag}"] = np.nan
        for window in [60, 120]:
            for stat in ["mean", "std", "min", "max", "count"]:
                out[f"roll49_{window}_{stat}"] = np.nan
        return out

    lag_source = known49[["geohash", "minute", "demand"]].copy()
    for lag in [15, 30, 60]:
        shifted = lag_source.copy()
        shifted["minute"] = shifted["minute"] + lag
        shifted = shifted.rename(columns={"demand": f"lag49_t{lag}"})
        out = out.merge(shifted, on=["geohash", "minute"], how="left")

    frames = []
    target_minutes = sorted(out["minute"].dropna().unique())
    for window in [60, 120]:
        rows = []
        for minute in target_minutes:
            hist_window = known49[(known49["minute"] < minute) & (known49["minute"] >= minute - window)]
            if hist_window.empty:
                continue
            agg = (
                hist_window.groupby("geohash")["demand"]
                .agg(["mean", "std", "min", "max", "count"])
                .reset_index()
            )
            agg["minute"] = minute
            agg = agg.rename(
                columns={
                    "mean": f"roll49_{window}_mean",
                    "std": f"roll49_{window}_std",
                    "min": f"roll49_{window}_min",
                    "max": f"roll49_{window}_max",
                    "count": f"roll49_{window}_count",
                }
            )
            rows.append(agg)
        if rows:
            frames.append(pd.concat(rows, ignore_index=True))

    for frame in frames:
        out = out.merge(frame, on=["geohash", "minute"], how="left")
    return out


def smoothed_ratio(numer: pd.Series, denom: pd.Series, global_ratio: float, strength: float) -> pd.Series:
    return (numer + strength * global_ratio) / (denom + strength)


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
        (["gh4"], "ratio_gh4", 1.5),
        (["RoadType"], "ratio_road", 2.0),
        (["Weather"], "ratio_weather", 2.0),
        (["hour"], "ratio_hour", 3.0),
    ]:
        agg = joined.groupby(keys, dropna=False).agg(y=("demand", "sum"), x=("d48", "sum")).reset_index()
        agg[name] = smoothed_ratio(agg["y"], agg["x"], global_ratio, strength)
        tables[name] = agg[keys + [name]]
    return tables


def attach_reference_features(df: pd.DataFrame, reference: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    day48 = reference[reference["day"] == 48].copy()
    global_mean = float(reference["demand"].mean())

    out = add_group_stats(day48, out, ["geohash"], "geo")
    out = add_group_stats(day48, out, ["gh5"], "gh5")
    out = add_group_stats(day48, out, ["gh4"], "gh4")
    out = add_group_stats(day48, out, ["minute"], "minute")
    out = add_group_stats(day48, out, ["hour"], "hour")
    out = add_group_stats(day48, out, ["geohash", "hour"], "geo_hour", ("mean", "std", "median"))
    out = add_group_stats(day48, out, ["gh5", "minute"], "gh5_minute", ("mean", "std", "median"))
    out = add_group_stats(day48, out, ["RoadType", "minute"], "road_minute", ("mean", "std"))
    out = add_group_stats(day48, out, ["Weather", "minute"], "weather_minute", ("mean", "std"))
    out = add_group_stats(day48, out, ["NumberofLanes", "minute"], "lanes_minute", ("mean", "std"))

    exact = day48[["geohash", "minute", "demand"]].rename(columns={"demand": "lag_day48_exact"})
    out = out.merge(exact, on=["geohash", "minute"], how="left")

    for shift, label in [(-15, "prev15"), (15, "next15"), (-30, "prev30"), (30, "next30"), (-60, "prev60"), (60, "next60")]:
        shifted = exact.copy()
        shifted["minute"] = shifted["minute"] - shift
        shifted = shifted.rename(columns={"lag_day48_exact": f"day48_{label}"})
        out = out.merge(shifted, on=["geohash", "minute"], how="left")

    early49 = history[(history["day"] == 49) & history["demand"].notna()].copy()
    if not early49.empty:
        last_known = (
            early49.sort_values("minute")
            .groupby("geohash")
            .tail(1)[["geohash", "minute", "demand"]]
            .rename(columns={"minute": "last_known_minute49", "demand": "last_known_demand49"})
        )
        out = out.merge(last_known, on="geohash", how="left")
        out["minutes_since_last_known49"] = out["minute"] - out["last_known_minute49"]
        trend = (
            early49.groupby("geohash")["demand"]
            .agg(first_known49="first", last_known49="last", mean_known49="mean", std_known49="std")
            .reset_index()
        )
        trend["known49_delta"] = trend["last_known49"] - trend["first_known49"]
        out = out.merge(trend, on="geohash", how="left")
    else:
        out["last_known_minute49"] = np.nan
        out["last_known_demand49"] = np.nan
        out["minutes_since_last_known49"] = np.nan
        out["first_known49"] = np.nan
        out["last_known49"] = np.nan
        out["mean_known49"] = np.nan
        out["std_known49"] = np.nan
        out["known49_delta"] = np.nan

    neighbor_map = nearest_neighbor_map(day48)
    out = attach_target_day_lags(out, history)
    out = out.merge(nearest_neighbor_stats(day48), on="geohash", how="left")
    out = attach_same_time_neighbor_features(out, day48, neighbor_map, "day48")
    if not early49.empty:
        out = attach_same_time_neighbor_features(out, early49, neighbor_map, "day49")
    else:
        for stat in ["mean", "std", "max", "min"]:
            out[f"day49_neighbor_same_time_{stat}"] = np.nan

    ratios = calibration_tables(history, day48)
    out["ratio_global"] = float(ratios.get("global_ratio", 1.0))
    for name, keys in [
        ("ratio_geo", ["geohash"]),
        ("ratio_gh5", ["gh5"]),
        ("ratio_gh4", ["gh4"]),
        ("ratio_road", ["RoadType"]),
        ("ratio_weather", ["Weather"]),
        ("ratio_hour", ["hour"]),
    ]:
        table = ratios.get(name)
        if isinstance(table, pd.DataFrame):
            out = out.merge(table, on=keys, how="left")
        else:
            out[name] = np.nan

    ratio_cols = ["ratio_geo", "ratio_gh5", "ratio_gh4", "ratio_road", "ratio_weather", "ratio_hour", "ratio_global"]
    for col in ratio_cols:
        out[col] = out[col].fillna(out["ratio_global"])

    out["ratio_blend"] = (
        0.34 * out["ratio_geo"]
        + 0.22 * out["ratio_gh5"]
        + 0.16 * out["ratio_gh4"]
        + 0.08 * out["ratio_road"]
        + 0.06 * out["ratio_weather"]
        + 0.04 * out["ratio_hour"]
        + 0.10 * out["ratio_global"]
    )
    out["prior_calibrated"] = out["lag_day48_exact"].fillna(out["geo_mean"]).fillna(global_mean) * out["ratio_blend"]
    out["prior_geo_time"] = out["lag_day48_exact"].fillna(out["gh5_minute_mean"]).fillna(out["minute_mean"]).fillna(global_mean)
    out["prior_neighbor"] = out["neighbor8_mean"].fillna(out["geo_mean"]).fillna(global_mean) * out["ratio_blend"]
    out["prior_gap_from_geo"] = out["prior_calibrated"] - out["geo_mean"].fillna(global_mean)
    out["known49_to_day48_prior_gap"] = out["last_known_demand49"] - out["lag_day48_exact"]
    out["lag49_to_prior_gap"] = out["lag49_t15"] - out["prior_calibrated"]

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    exclude = {"demand", "Index", "timestamp"}
    categorical = [
        "geohash",
        "RoadType",
        "LargeVehicles",
        "Landmarks",
        "Weather",
        "gh2",
        "gh3",
        "gh4",
        "gh5",
        "gh_last",
    ]
    categorical = [c for c in categorical if c in df.columns]
    numeric = [c for c in df.columns if c not in exclude and c not in categorical]
    return numeric + categorical, numeric, categorical


def prepare_categories(train_df: pd.DataFrame, predict_df: pd.DataFrame, categorical: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.copy()
    pred_out = predict_df.copy()
    for col in categorical:
        train_out[col] = train_out[col].astype("object").fillna("__MISSING__")
        pred_out[col] = pred_out[col].astype("object").fillna("__MISSING__")
        categories = sorted(set(train_out[col].astype(str)) | set(pred_out[col].astype(str)))
        train_out[col] = pd.Categorical(train_out[col].astype(str), categories=categories)
        pred_out[col] = pd.Categorical(pred_out[col].astype(str), categories=categories)
    return train_out, pred_out


def make_lgbm_models() -> dict[str, LGBMRegressor]:
    return {
        "lgbm_l2_deep": LGBMRegressor(
            objective="regression_l2",
            n_estimators=260,
            learning_rate=0.06,
            num_leaves=95,
            max_depth=-1,
            min_child_samples=18,
            subsample=0.88,
            subsample_freq=1,
            colsample_bytree=0.82,
            reg_alpha=0.02,
            reg_lambda=2.8,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        "lgbm_l1_robust": LGBMRegressor(
            objective="regression_l1",
            n_estimators=220,
            learning_rate=0.065,
            num_leaves=75,
            min_child_samples=14,
            subsample=0.92,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.04,
            reg_lambda=1.6,
            random_state=RANDOM_STATE + 11,
            verbose=-1,
        ),
        "lgbm_huber": LGBMRegressor(
            objective="huber",
            alpha=0.9,
            n_estimators=220,
            learning_rate=0.065,
            num_leaves=63,
            min_child_samples=25,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.86,
            reg_alpha=0.03,
            reg_lambda=3.4,
            random_state=RANDOM_STATE + 29,
            verbose=-1,
        ),
        "lgbm_goss": LGBMRegressor(
            objective="regression_l2",
            boosting_type="goss",
            n_estimators=220,
            learning_rate=0.065,
            num_leaves=127,
            min_child_samples=24,
            colsample_bytree=0.78,
            reg_alpha=0.01,
            reg_lambda=4.2,
            random_state=RANDOM_STATE + 71,
            verbose=-1,
        ),
        "lgbm_tweedie_1p2": LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.2,
            n_estimators=260,
            learning_rate=0.06,
            num_leaves=95,
            min_child_samples=18,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.02,
            reg_lambda=2.6,
            random_state=RANDOM_STATE + 131,
            verbose=-1,
        ),
        "lgbm_tweedie_1p5": LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            n_estimators=240,
            learning_rate=0.06,
            num_leaves=75,
            min_child_samples=24,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.88,
            reg_alpha=0.03,
            reg_lambda=3.2,
            random_state=RANDOM_STATE + 151,
            verbose=-1,
        ),
    }


def make_extra_trees(numeric: list[str], categorical: list[str]) -> tuple[str, object]:
    transformer = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
                categorical,
            ),
        ]
    )
    model = ExtraTreesRegressor(
        n_estimators=110,
        min_samples_leaf=2,
        max_features=0.82,
        n_jobs=-1,
        random_state=RANDOM_STATE + 101,
    )
    return "extra_trees", make_pipeline(transformer, model)


def make_xgb_model() -> tuple[str, XGBRegressor]:
    return (
        "xgb_hist",
        XGBRegressor(
            objective="reg:squarederror",
            n_estimators=220,
            learning_rate=0.06,
            max_depth=7,
            min_child_weight=5,
            subsample=0.88,
            colsample_bytree=0.86,
            reg_alpha=0.02,
            reg_lambda=2.5,
            tree_method="hist",
            enable_categorical=True,
            random_state=RANDOM_STATE + 203,
            n_jobs=-1,
        ),
    )


def make_xgb_tweedie_model() -> tuple[str, XGBRegressor]:
    return (
        "xgb_tweedie",
        XGBRegressor(
            objective="reg:tweedie",
            tweedie_variance_power=1.2,
            n_estimators=180,
            learning_rate=0.065,
            max_depth=7,
            min_child_weight=5,
            subsample=0.88,
            colsample_bytree=0.86,
            reg_alpha=0.02,
            reg_lambda=2.8,
            tree_method="hist",
            enable_categorical=True,
            random_state=RANDOM_STATE + 307,
            n_jobs=-1,
        ),
    )


def train_predict_models(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
    use_slow_models: bool,
    allowed_models: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    train_ready, pred_ready = prepare_categories(train_df, predict_df, categorical)
    y = train_ready["demand"].to_numpy()
    predictions = {}
    fitted_models: dict[str, object] = {}

    for name, model in make_lgbm_models().items():
        if allowed_models is not None and name not in allowed_models:
            continue
        print(f"Training {name}...", flush=True)
        model.fit(train_ready[features], y, categorical_feature=categorical)
        predictions[name] = model.predict(pred_ready[features])
        fitted_models[name] = model

    if use_slow_models:
        for name, model in [make_xgb_model(), make_xgb_tweedie_model()]:
            if allowed_models is None or name in allowed_models:
                print(f"Training {name}...", flush=True)
                model.fit(train_ready[features], y)
                predictions[name] = model.predict(pred_ready[features])
                fitted_models[name] = model

        name, model = make_extra_trees(numeric, categorical)
        if allowed_models is None or name in allowed_models:
            print(f"Training {name}...", flush=True)
            model.fit(train_ready[features], y)
            predictions[name] = model.predict(pred_ready[features])
            fitted_models[name] = model

    pred_df = pd.DataFrame(predictions, index=predict_df.index).clip(0, 1)
    return pred_df, fitted_models


def optimize_blend(preds: pd.DataFrame, y: np.ndarray) -> tuple[dict[str, float], float]:
    columns = list(preds.columns)
    best_score = -1e9
    best_weights = {col: 1.0 / len(columns) for col in columns}

    # Coordinate-search simplex blend. It is small, deterministic, and avoids scipy dependency.
    candidates = [np.full(len(columns), 1.0 / len(columns))]
    for i in range(len(columns)):
        one = np.zeros(len(columns))
        one[i] = 1.0
        candidates.append(one)

    rng = np.random.default_rng(RANDOM_STATE)
    for _ in range(4500):
        raw = rng.random(len(columns)) ** 1.6
        candidates.append(raw / raw.sum())

    values = preds[columns].to_numpy()
    for weights in candidates:
        blended = np.clip(values @ weights, 0, 1)
        score = r2_score(y, blended)
        if score > best_score:
            best_score = score
            best_weights = dict(zip(columns, weights))
    return best_weights, best_score


def nonnegative_linear_blend(preds: pd.DataFrame, y: np.ndarray) -> tuple[dict[str, float], float]:
    columns = list(preds.columns)
    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(preds[columns].to_numpy(), y)
    weights = np.maximum(model.coef_.astype(float), 0)
    if weights.sum() <= 0:
        return optimize_blend(preds, y)
    weights = weights / weights.sum()
    pred = np.clip(preds[columns].to_numpy() @ weights, 0, 1)
    return dict(zip(columns, weights)), r2_score(y, pred)


def validation_run(train: pd.DataFrame, use_slow_models: bool) -> tuple[pd.DataFrame, dict[str, float], str]:
    train_base = add_base_features(train)
    day48 = train_base[train_base["day"] == 48].copy()
    day49 = train_base[train_base["day"] == 49].copy()

    fold_rows = []
    oof_pred_parts = []
    oof_y_parts = []
    extra_trees_scores = []

    for val_start in VALIDATION_CUTOFFS:
        print(f"\nValidation fold: hold out day 49 from minute {val_start}...", flush=True)
        fit_raw = train_base[
            (train_base["day"] == 48) | ((train_base["day"] == 49) & (train_base["minute"] < val_start))
        ].copy()
        val_raw = day49[day49["minute"] >= val_start].copy()

        fit_feat = attach_reference_features(fit_raw, day48, fit_raw)
        val_feat = attach_reference_features(val_raw, day48, fit_raw)
        features, numeric, categorical = feature_columns(fit_feat)

        validation_models = {
            "lgbm_l2_deep",
            "lgbm_huber",
            "lgbm_tweedie_1p2",
            "lgbm_tweedie_1p5",
            "xgb_tweedie",
            "extra_trees",
        }
        model_preds, _ = train_predict_models(
            fit_feat,
            val_feat,
            features,
            numeric,
            categorical,
            use_slow_models,
            allowed_models=validation_models if use_slow_models else None,
        )
        preds = pd.DataFrame(index=val_feat.index)
        preds["prior_calibrated"] = val_feat["prior_calibrated"].to_numpy()
        preds["prior_geo_time"] = val_feat["prior_geo_time"].to_numpy()
        preds["prior_neighbor"] = val_feat["prior_neighbor"].to_numpy()
        preds = pd.concat([preds, model_preds], axis=1).clip(0, 1)

        y = val_feat["demand"].to_numpy()
        for col in preds.columns:
            score = r2_score(y, preds[col])
            fold_rows.append(
                {
                    "fold_start_minute": val_start,
                    "model": col,
                    "r2": score,
                    "mae": mean_absolute_error(y, preds[col]),
                }
            )
            if col == "extra_trees":
                extra_trees_scores.append(score)
        oof_pred_parts.append(preds.reset_index(drop=True))
        oof_y_parts.append(pd.Series(y))

    oof_preds = pd.concat(oof_pred_parts, ignore_index=True).fillna(0)
    oof_y = pd.concat(oof_y_parts, ignore_index=True).to_numpy()
    linear_weights, linear_score = nonnegative_linear_blend(oof_preds, oof_y)
    search_weights, search_score = optimize_blend(oof_preds, oof_y)
    if search_score > linear_score:
        weights, blend_score, blend_name = search_weights, search_score, "optimized_simplex_blend"
    else:
        weights, blend_score, blend_name = linear_weights, linear_score, "nonnegative_linear_blend"

    blend_pred = np.clip(oof_preds[list(weights)].to_numpy() @ np.array(list(weights.values())), 0, 1)
    fold_rows.append(
        {
            "fold_start_minute": "all_oof",
            "model": blend_name,
            "r2": blend_score,
            "mae": mean_absolute_error(oof_y, blend_pred),
        }
    )

    extra_mean = float(np.mean(extra_trees_scores)) if extra_trees_scores else -np.inf
    strategy = "meta_ensemble"
    if use_slow_models and extra_mean >= blend_score - 0.001:
        strategy = "extra_trees_fallback"
        weights = {"extra_trees": 1.0}
        fold_rows.append(
            {
                "fold_start_minute": "guardrail",
                "model": "chosen_strategy_extra_trees_fallback",
                "r2": extra_mean,
                "mae": np.nan,
            }
        )
    else:
        fold_rows.append(
            {
                "fold_start_minute": "guardrail",
                "model": f"chosen_strategy_{strategy}",
                "r2": blend_score,
                "mae": np.nan,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    summary_source = fold_df[fold_df["fold_start_minute"].isin(VALIDATION_CUTOFFS)]
    summary = (
        summary_source
        .groupby("model", dropna=False)
        .agg(mean_r2=("r2", "mean"), std_r2=("r2", "std"), mean_mae=("mae", "mean"), folds=("r2", "count"))
        .reset_index()
    )
    summary["fold_start_minute"] = "summary"

    for name, weight in sorted(weights.items()):
        fold_rows.append(
            {
                "fold_start_minute": "blend_weight",
                "model": name,
                "r2": weight,
                "mae": np.nan,
            }
        )

    report = pd.concat([pd.DataFrame(fold_rows), summary], ignore_index=True, sort=False)
    report["chosen_strategy"] = strategy
    return report, weights, strategy


def final_run(train: pd.DataFrame, test: pd.DataFrame, weights: dict[str, float], use_slow_models: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_base = add_base_features(train)
    test_base = add_base_features(test)
    day48 = train_base[train_base["day"] == 48].copy()

    train_feat = attach_reference_features(train_base, day48, train_base)
    test_feat = attach_reference_features(test_base, day48, train_base)
    features, numeric, categorical = feature_columns(train_feat)

    prior_names = PRIOR_MODEL_NAMES
    allowed_models = {name for name, weight in weights.items() if weight > 1e-6 and name not in prior_names}
    if not allowed_models:
        allowed_models = None
    model_preds, fitted = train_predict_models(
        train_feat,
        test_feat,
        features,
        numeric,
        categorical,
        use_slow_models,
        allowed_models=allowed_models,
    )
    preds = pd.DataFrame(index=test_feat.index)
    preds["prior_calibrated"] = test_feat["prior_calibrated"].to_numpy()
    preds["prior_geo_time"] = test_feat["prior_geo_time"].to_numpy()
    preds["prior_neighbor"] = test_feat["prior_neighbor"].to_numpy()
    preds = pd.concat([preds, model_preds], axis=1).clip(0, 1)

    usable_weights = {k: v for k, v in weights.items() if k in preds.columns}
    missing = sorted(k for k, v in weights.items() if v > 1e-6 and k not in usable_weights)
    if missing:
        print(f"Blend weights skipped because models are unavailable: {missing}")
    total = sum(usable_weights.values())
    if total <= 0:
        usable_weights = {col: 1.0 / len(preds.columns) for col in preds.columns}
    else:
        usable_weights = {k: v / total for k, v in usable_weights.items()}

    values = preds[list(usable_weights)].to_numpy()
    final_pred = np.clip(values @ np.array(list(usable_weights.values())), 0, 1)
    submission = pd.DataFrame({"Index": test["Index"].to_numpy(), "demand": final_pred})

    importance = pd.DataFrame()
    lgbm = fitted.get("lgbm_l2_deep")
    if lgbm is not None:
        importance = pd.DataFrame(
            {"feature": features, "importance": lgbm.feature_importances_}
        ).sort_values("importance", ascending=False)

    return submission, importance


def validate_outputs(train: pd.DataFrame, test: pd.DataFrame, submission: pd.DataFrame) -> None:
    assert train.shape == (77299, 11), f"Unexpected train shape: {train.shape}"
    assert test.shape == (41778, 10), f"Unexpected test shape: {test.shape}"
    assert submission.shape == (41778, 2), f"Unexpected submission shape: {submission.shape}"
    assert list(submission.columns) == ["Index", "demand"]
    assert submission["Index"].equals(test["Index"])
    assert np.isfinite(submission["demand"]).all()
    assert submission["demand"].between(0, 1).all()


def load_cached_validation(report_path: Path) -> tuple[pd.DataFrame, dict[str, float], str] | None:
    if not report_path.exists():
        return None
    report = pd.read_csv(report_path)
    required = {"fold_start_minute", "model", "r2", "chosen_strategy"}
    if not required.issubset(report.columns):
        return None
    weight_rows = report[report["fold_start_minute"].astype(str) == "blend_weight"]
    if weight_rows.empty:
        return None
    weights = {
        str(row["model"]): float(row["r2"])
        for _, row in weight_rows.iterrows()
        if pd.notna(row["r2"]) and float(row["r2"]) > 0
    }
    if not weights:
        return None
    strategy_values = report["chosen_strategy"].dropna().astype(str)
    strategy = strategy_values.iloc[0] if not strategy_values.empty else "cached"
    return report, weights, strategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset", type=Path)
    parser.add_argument("--output", default="submission.csv", type=Path)
    parser.add_argument("--fast", action="store_true", help="Skip XGBoost and ExtraTrees.")
    parser.add_argument("--force-validation", action="store_true", help="Recompute validation instead of using cached report.")
    parser.add_argument("--force-final", action="store_true", help="Retrain final models even if the output CSV already exists.")
    args = parser.parse_args()

    data_dir = args.data_dir
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    print(f"Train shape: {train.shape}; Test shape: {test.shape}", flush=True)
    report_path = Path("model_validation_report.csv")
    cached = None if args.force_validation else load_cached_validation(report_path)
    if cached is None:
        print("Running multi-fold validation on known day-49 continuation slices...", flush=True)
        report, weights, strategy = validation_run(train, use_slow_models=not args.fast)
        report.to_csv(report_path, index=False)
    else:
        print("Using cached validation report. Pass --force-validation to recompute.", flush=True)
        report, weights, strategy = cached
    print("\nValidation report:")
    print(report.to_string(index=False))
    print(f"\nChosen strategy: {strategy}")
    print("\nBlend weights:")
    print(json.dumps(weights, indent=2))

    if cached is not None and args.output.exists() and not args.force_final:
        print(f"\nUsing existing {args.output}. Pass --force-final to retrain final models.", flush=True)
        submission = pd.read_csv(args.output)
        importance = pd.DataFrame()
    else:
        print("\nTraining final models and predicting test...", flush=True)
        submission, importance = final_run(train, test, weights, use_slow_models=not args.fast)
    validate_outputs(train, test, submission)
    if not (cached is not None and args.output.exists() and not args.force_final):
        submission.to_csv(args.output, index=False)
    if not importance.empty:
        importance.to_csv("feature_importance_lightgbm.csv", index=False)

    print("\nSubmission summary:")
    print(submission["demand"].describe().to_string())
    print(f"\nWrote {args.output.resolve()}")
    print("Validation checks passed.")


if __name__ == "__main__":
    main()
