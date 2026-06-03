"""
traffic_demand_best_model.py — Research-backed, competition-grade pipeline
====================================================================
Implements:
  1. DoD (Day-over-Day) ratio features  -- #1 missing feature from Grab AI winners
  2. Day-49 slope / trajectory           -- extrapolation anchor per geohash
  3. Multi-resolution spatial lags       -- gh3/gh4/gh5 same-time means
  4. K-Means geohash clustering          -- top-5 Grab AI feature
  5. Weather × time-of-day interactions  -- +0.5–1.5% R² per research
  6. Slot-to-hour normalization          -- captures within-hour slot position
  7. LightGBM Tweedie, Huber, DART models
  8. Ridge meta-learner (alpha=5)        -- prevents meta-learner overfitting
  9. Per-geohash residual bias correction -- systematic error correction on known D49 slots
  10. Road-type physical bounds          -- strict post-processing

Run:
    python traffic_demand_best_model.py
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"   # Fix threadpoolctl crash on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# CatBoost is optional, gracefully fallback to other models
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not available, will use LightGBM, XGBoost, and ExtraTrees instead")

RANDOM_STATE = 42
BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
VALIDATION_CUTOFFS = [90, 105, 150]   # Three chronological folds on Day-49

ROAD_BOUNDS = {
    "Residential": (0.0,      0.219997),
    "Street":      (0.220016, 0.349908),
    "Highway":     (0.350009, 1.0),
}

WEATHER_SEVERITY = {"Sunny": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def parse_ts(v: str) -> int:
    h, m = str(v).split(":")
    return int(h) * 60 + int(m)


def decode_geohash(gh: str) -> tuple[float, float]:
    lat, lon = [-90.0, 90.0], [-180.0, 180.0]
    even = True
    for ch in gh:
        bits = BASE32.index(ch)
        for mask in (16, 8, 4, 2, 1):
            if even:
                mid = (lon[0] + lon[1]) / 2
                lon[0] = mid if bits & mask else lon[0]
                lon[1] = lon[1] if bits & mask else mid
            else:
                mid = (lat[0] + lat[1]) / 2
                lat[0] = mid if bits & mask else lat[0]
                lat[1] = lat[1] if bits & mask else mid
            even = not even
    return (lat[0] + lat[1]) / 2, (lon[0] + lon[1]) / 2


def mode_or_none(s: pd.Series):
    m = s.dropna().mode()
    return None if m.empty else str(m.iloc[0])


def apply_bounds(pred: np.ndarray, road_types: pd.Series) -> np.ndarray:
    out = pred.astype(float).copy()
    for road, (lo, hi) in ROAD_BOUNDS.items():
        mask = road_types.astype(str).to_numpy() == road
        out[mask] = np.clip(out[mask], lo, hi)
    return np.clip(out, 0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — IMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def impute_road_type(train: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    known = train.dropna(subset=["RoadType"]).copy()
    for keys in [
        ["geohash", "NumberofLanes", "LargeVehicles", "Landmarks"],
        ["geohash", "NumberofLanes", "LargeVehicles"],
        ["geohash", "NumberofLanes"],
        ["NumberofLanes", "LargeVehicles", "Landmarks"],
        ["NumberofLanes", "LargeVehicles"],
        ["geohash"],
    ]:
        if not out["RoadType"].isna().any():
            break
        mp = known.groupby(keys)["RoadType"].agg(mode_or_none).dropna().to_dict()
        mask = out["RoadType"].isna()
        out.loc[mask, "RoadType"] = out.loc[mask, keys].apply(
            lambda r: mp.get(tuple(r)), axis=1
        ).values
    mask = out["RoadType"].isna()
    out.loc[mask & (out["NumberofLanes"] >= 4), "RoadType"] = "Highway"
    mask = out["RoadType"].isna()
    out.loc[
        mask & (out["NumberofLanes"] == 1)
        & (out["LargeVehicles"] == "Not Allowed")
        & (out["Landmarks"] == "Yes"),
        "RoadType",
    ] = "Street"
    out["RoadType"] = out["RoadType"].fillna("Residential")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BASE FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def build_geo_lookup(all_geohashes: pd.Series) -> pd.DataFrame:
    unique = all_geohashes.drop_duplicates()
    decoded = unique.map(decode_geohash)
    df = pd.DataFrame({
        "geohash": unique.values,
        "lat": [p[0] for p in decoded],
        "lon": [p[1] for p in decoded],
    })
    return df


def add_base_features(df: pd.DataFrame, geo_lookup: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["minute"] = out["timestamp"].map(parse_ts)
    out["slot"] = out["minute"] // 15
    out["hour"] = out["minute"] // 60

    out["time_sin"] = np.sin(2 * np.pi * out["slot"] / 96)
    out["time_cos"] = np.cos(2 * np.pi * out["slot"] / 96)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    # Geohash prefix hierarchy
    out["gh2"] = out["geohash"].str[:2]
    out["gh3"] = out["geohash"].str[:3]
    out["gh4"] = out["geohash"].str[:4]
    out["gh5"] = out["geohash"].str[:5]

    out = out.merge(geo_lookup, on="geohash", how="left")

    # Weather & temp
    out["weather_sev"] = out["Weather"].map(WEATHER_SEVERITY).fillna(1.0)
    out["is_adverse"] = out["weather_sev"].ge(2).astype(np.int8)
    out["is_rain"] = (out["Weather"] == "Rainy").astype(np.int8)
    out["is_snowy"] = (out["Weather"] == "Snowy").astype(np.int8)
    out["temp_filled"] = out["Temperature"].fillna(out["Temperature"].median())
    out["temp_sq"] = out["temp_filled"] ** 2
    out["temp_bin"] = pd.cut(
        out["temp_filled"],
        bins=[-np.inf, 10, 18, 26, 34, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(float)

    # Peak hour flags
    out["is_morning_peak"] = out["hour"].isin(range(7, 10)).astype(np.int8)
    out["is_evening_peak"] = out["hour"].isin(range(17, 21)).astype(np.int8)
    out["is_peak"] = (out["is_morning_peak"] | out["is_evening_peak"]).astype(np.int8)

    # Weather × time interactions (research: +0.5–1.5% R²)
    out["weather_x_peak"] = out["weather_sev"] * out["is_peak"]
    out["rain_x_morning"] = out["is_rain"] * out["is_morning_peak"]
    out["rain_x_evening"] = out["is_rain"] * out["is_evening_peak"]
    out["temp_x_peak"] = out["temp_filled"] * out["is_peak"]
    out["temp_x_lanes"] = out["temp_filled"] * out["NumberofLanes"]
    out["adverse_x_lanes"] = out["is_adverse"] * out["NumberofLanes"]

    # Infrastructure
    out["is_large_allowed"] = (out["LargeVehicles"] == "Allowed").astype(np.int8)
    out["has_landmark"] = (out["Landmarks"] == "Yes").astype(np.int8)
    out["lanes_x_large"] = out["NumberofLanes"] * out["is_large_allowed"]
    out["lanes_x_landmark"] = out["NumberofLanes"] * out["has_landmark"]

    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — K-MEANS GEO CLUSTERING (top-5 Grab AI feature)
# ══════════════════════════════════════════════════════════════════════════════

def build_geo_clusters(geo_lookup: pd.DataFrame, n_clusters: int = 50) -> dict[str, int]:
    coords = geo_lookup[["lat", "lon"]].to_numpy()
    try:
        km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(coords)
        print(f"  K-Means clustering: {n_clusters} clusters")
    except Exception as e:
        print(f"  KMeans failed ({e}), using lat/lon grid bucketing fallback")
        # Fallback: simple 7×8 grid bucketing on lat/lon
        lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
        lon_min, lon_max = coords[:, 1].min(), coords[:, 1].max()
        n_lat, n_lon = 7, 8
        lat_idx = np.floor((coords[:, 0] - lat_min) / (lat_max - lat_min + 1e-9) * n_lat).astype(int).clip(0, n_lat - 1)
        lon_idx = np.floor((coords[:, 1] - lon_min) / (lon_max - lon_min + 1e-9) * n_lon).astype(int).clip(0, n_lon - 1)
        labels = lat_idx * n_lon + lon_idx
        print(f"  Grid bucketing: {len(np.unique(labels))} clusters")
    return dict(zip(geo_lookup["geohash"], labels))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — REFERENCE FEATURE ATTACHMENT (Day-48 lags, ratios, spatial)
# ══════════════════════════════════════════════════════════════════════════════

def attach_reference_features(
    target: pd.DataFrame,
    day48: pd.DataFrame,
    known49: pd.DataFrame,
    geo_cluster_map: dict[str, int],
) -> pd.DataFrame:
    out = target.copy()
    global_mean48 = float(day48["demand"].mean())

    # ── geo cluster ──────────────────────────────────────────────────────────
    out["geo_cluster"] = out["geohash"].map(geo_cluster_map).fillna(-1).astype(int)

    # ── Day-48 exact lag ─────────────────────────────────────────────────────
    exact48 = day48[["geohash", "minute", "demand"]].rename(columns={"demand": "lag_d48_exact"})
    out = out.merge(exact48, on=["geohash", "minute"], how="left")

    # ── Day-48 shifted lags (±15, ±30, ±60 min) ──────────────────────────────
    for shift, label in [(-15, "prev15"), (15, "next15"), (-30, "prev30"), (-60, "prev60")]:
        sh = exact48.copy()
        sh["minute"] = sh["minute"] - shift
        sh = sh.rename(columns={"lag_d48_exact": f"d48_{label}"})
        out = out.merge(sh, on=["geohash", "minute"], how="left")

    # ── Multi-resolution spatial lags (research: +0.5–1% R²) ─────────────────
    day48_w_prefixes = day48.copy()
    day48_w_prefixes["gh5"] = day48_w_prefixes["geohash"].str[:5]
    day48_w_prefixes["gh4"] = day48_w_prefixes["geohash"].str[:4]
    day48_w_prefixes["gh3"] = day48_w_prefixes["geohash"].str[:3]

    for prefix in ["gh5", "gh4", "gh3"]:
        pfx_mean = (
            day48_w_prefixes.groupby([prefix, "minute"])["demand"]
            .mean()
            .rename(f"{prefix}_time_mean")
            .reset_index()
        )
        out = out.merge(pfx_mean, on=[prefix, "minute"], how="left")

    # Cluster-level spatial mean
    cluster48 = (
        day48.assign(geo_cluster=day48["geohash"].map(geo_cluster_map))
        .groupby(["geo_cluster", "minute"])["demand"]
        .mean()
        .rename("cluster_time_mean")
        .reset_index()
    )
    out = out.merge(cluster48, on=["geo_cluster", "minute"], how="left")

    # ── Geo / road / global target encodings ─────────────────────────────────
    geo_mean = day48.groupby("geohash")["demand"].mean().rename("geo_mean48")
    road_slot = (
        day48.groupby(["RoadType", "minute"])["demand"].mean().rename("road_slot_mean").reset_index()
    )
    road_mean = day48.groupby("RoadType")["demand"].mean().rename("road_mean48")
    out = out.merge(geo_mean, on="geohash", how="left")
    out = out.merge(road_slot, on=["RoadType", "minute"], how="left")
    out = out.merge(road_mean, on="RoadType", how="left")

    # ── Slot-to-hour ratio (research: +0.3–0.8% R²) ──────────────────────────
    day48_hour = day48.copy()
    day48_hour["hour"] = day48_hour["minute"] // 60
    hour_geo_mean = (
        day48_hour.groupby(["geohash", "hour"])["demand"].mean().rename("geo_hour_mean").reset_index()
    )
    out = out.merge(hour_geo_mean, on=["geohash", "hour"], how="left")
    out["slot_to_hour_ratio"] = out["lag_d48_exact"] / (out["geo_hour_mean"].fillna(global_mean48) + 1e-9)

    # ── Road-type percentile rank (research: +0.2–0.5% R²) ───────────────────
    day48_copy = day48.copy()
    day48_copy["geo_rank_in_road"] = day48_copy.groupby(["RoadType", "minute"])["demand"].rank(pct=True)
    rank_df = day48_copy[["geohash", "minute", "RoadType", "geo_rank_in_road"]]
    out = out.merge(rank_df, on=["geohash", "minute", "RoadType"], how="left")

    # ── Day-49 known features ────────────────────────────────────────────────
    k49 = known49[known49["demand"].notna()].copy()

    # Day-49 short lags
    for lag_min, label in [(15, "t15"), (30, "t30"), (60, "t60")]:
        sh = k49[["geohash", "minute", "demand"]].copy()
        sh["minute"] = sh["minute"] + lag_min
        sh = sh.rename(columns={"demand": f"lag49_{label}"})
        out = out.merge(sh, on=["geohash", "minute"], how="left")

    # Day-49 rolling window stats
    k49_sorted = k49.sort_values(["geohash", "minute"]).copy()
    for window, label in [(4, "1h"), (8, "2h")]:
        k49_sorted[f"roll_{label}"] = (
            k49_sorted.groupby("geohash")["demand"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
    k49_aug = k49_sorted.copy()

    for label in ["1h", "2h"]:
        sh = k49_aug[["geohash", "minute", f"roll_{label}"]].copy()
        sh["minute"] = sh["minute"] + 15
        out = out.merge(sh, on=["geohash", "minute"], how="left")

    # ── DoD demand ratio (MOST IMPORTANT missing feature from Grab AI!) ───────
    d48_for_ratio = day48[["geohash", "minute", "demand"]].rename(columns={"demand": "d48_dod"})

    if len(k49) > 0:
        # Compute per-geohash DoD ratio from known Day-49 slots
        joined_dod = k49[["geohash", "minute", "demand"]].merge(
            d48_for_ratio, on=["geohash", "minute"], how="inner"
        )
        joined_dod = joined_dod[joined_dod["d48_dod"] > 1e-6]
        global_dod = float(joined_dod["demand"].sum() / max(joined_dod["d48_dod"].sum(), 1e-9)) if len(joined_dod) else 1.0

        # Per-geohash DoD ratio (smoothed)
        agg_geo = joined_dod.groupby("geohash")[["demand", "d48_dod"]].sum()
        dod_geo = ((agg_geo["demand"] + 0.5 * global_dod) / (agg_geo["d48_dod"] + 0.5)).rename("dod_ratio_geo").reset_index()

        # GH5 prefix DoD ratio
        joined_dod["gh5"] = joined_dod["geohash"].str[:5]
        agg_gh5 = joined_dod.groupby("gh5")[["demand", "d48_dod"]].sum()
        dod_gh5 = ((agg_gh5["demand"] + 1.0 * global_dod) / (agg_gh5["d48_dod"] + 1.0)).rename("dod_ratio_gh5").reset_index()

        # Road-type DoD ratio
        joined_dod["RoadType"] = joined_dod["geohash"].map(
            k49.set_index("geohash")["RoadType"].to_dict()
        )
        agg_road = joined_dod.groupby("RoadType")[["demand", "d48_dod"]].sum()
        dod_road = ((agg_road["demand"] + 2.0 * global_dod) / (agg_road["d48_dod"] + 2.0)).rename("dod_ratio_road").reset_index()

        out["dod_global"] = global_dod
        out = out.merge(dod_geo, on="geohash", how="left")
        out = out.merge(dod_gh5, on="gh5", how="left")
        out = out.merge(dod_road, on="RoadType", how="left")
        for col in ["dod_ratio_geo", "dod_ratio_gh5", "dod_ratio_road"]:
            out[col] = out[col].fillna(out["dod_global"])

        # Blended DoD ratio — weighted average
        out["dod_blend"] = (
            0.55 * out["dod_ratio_geo"]
            + 0.25 * out["dod_ratio_gh5"]
            + 0.12 * out["dod_ratio_road"]
            + 0.08 * out["dod_global"]
        )
        # Key feature: scaled Day-48 prediction using blended DoD ratio
        base = out["lag_d48_exact"].fillna(out["geo_mean48"]).fillna(global_mean48)
        out["prior_dod"] = base * out["dod_blend"]

        # ── Cumulative Day-49 vs Day-48 ratio ────────────────────────────────
        cum49 = k49.groupby("geohash")["demand"].sum().rename("cum49")
        cum48_for_known = day48.groupby("geohash")["demand"].sum().rename("cum48")
        out = out.merge(cum49, on="geohash", how="left")
        out = out.merge(cum48_for_known, on="geohash", how="left")
        out["cumulative_ratio"] = out["cum49"] / (out["cum48"] + 1e-9)

        # ── Day-49 slope feature (research: most impactful missing feature) ───
        slopes, intercepts, cvs = {}, {}, {}
        for gh, grp in k49.groupby("geohash"):
            grp_s = grp.sort_values("minute")
            if len(grp_s) >= 3:
                times = grp_s["minute"].values.astype(float)
                vals = grp_s["demand"].values.astype(float)
                coef = np.polyfit(times, vals, 1)
                slopes[gh] = coef[0]
                intercepts[gh] = coef[1]
            else:
                slopes[gh] = 0.0
                intercepts[gh] = grp_s["demand"].mean() if len(grp_s) else 0.0
            cv = grp_s["demand"].std() / (grp_s["demand"].mean() + 1e-9)
            cvs[gh] = cv

        out["geo_slope49"] = out["geohash"].map(slopes).fillna(0.0)
        out["geo_intercept49"] = out["geohash"].map(intercepts).fillna(0.0)
        out["geo_cv49"] = out["geohash"].map(cvs).fillna(0.0)
        # Extrapolated demand at target minute using slope
        out["slope_extrapolated"] = out["geo_intercept49"] + out["geo_slope49"] * out["minute"]
        out["slope_extrapolated"] = out["slope_extrapolated"].clip(0, 1)
    else:
        out["dod_global"] = 1.0
        out["dod_ratio_geo"] = 1.0
        out["dod_ratio_gh5"] = 1.0
        out["dod_ratio_road"] = 1.0
        out["dod_blend"] = 1.0
        out["prior_dod"] = out["lag_d48_exact"].fillna(out["geo_mean48"]).fillna(global_mean48)
        out["cumulative_ratio"] = 1.0
        out["geo_slope49"] = 0.0
        out["geo_intercept49"] = out["geo_mean48"].fillna(global_mean48)
        out["geo_cv49"] = 0.0
        out["slope_extrapolated"] = out["prior_dod"]
        out["cum49"] = 0.0
        out["cum48"] = 0.0

    # Fill remaining NAs in spatial features
    for col in ["gh5_time_mean", "gh4_time_mean", "gh3_time_mean", "cluster_time_mean",
                 "road_slot_mean", "road_mean48", "geo_mean48", "lag_d48_exact"]:
        if col in out.columns:
            out[col] = out[col].fillna(global_mean48)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — FEATURE LIST
# ══════════════════════════════════════════════════════════════════════════════

EXCLUDE = {"demand", "Index", "timestamp"}
CATEGORICAL_COLS = [
    "geohash", "RoadType", "LargeVehicles", "Landmarks", "Weather",
    "gh2", "gh3", "gh4", "gh5", "NumberofLanes", "geo_cluster",
]


def get_feature_cols(df: pd.DataFrame):
    cat = [c for c in CATEGORICAL_COLS if c in df.columns]
    num = [c for c in df.columns if c not in EXCLUDE and c not in cat]
    return num + cat, num, cat


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def make_lgbm_models():
    common = dict(verbose=-1, random_state=RANDOM_STATE)
    return {
        "lgbm_tweedie": LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.2,
            n_estimators=400, learning_rate=0.04, num_leaves=63,
            min_child_samples=30, subsample=0.85, colsample_bytree=0.82,
            reg_alpha=0.2, reg_lambda=6.0, min_split_gain=0.02,
            **common,
        ),
        "lgbm_huber": LGBMRegressor(
            objective="huber", alpha=0.9,
            n_estimators=350, learning_rate=0.04, num_leaves=63,
            min_child_samples=30, subsample=0.88, colsample_bytree=0.85,
            reg_alpha=0.2, reg_lambda=6.0, min_split_gain=0.02,
            **common,
        ),
        "lgbm_dart": LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.3,
            boosting_type="dart", drop_rate=0.1,
            n_estimators=300, learning_rate=0.04, num_leaves=63,
            min_child_samples=30, subsample=0.85, colsample_bytree=0.82,
            reg_alpha=0.15, reg_lambda=5.0,
            **common,
        ),
    }


def make_extratrees_pipeline(numeric: list[str], categorical: list[str]):
    transformer = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", make_pipeline(
            SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ), categorical),
    ])
    et = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=1,
        max_features=0.72,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return make_pipeline(transformer, et)


def prepare_categoricals(train: pd.DataFrame, pred: pd.DataFrame, cat_cols: list[str]):
    train_o, pred_o = train.copy(), pred.copy()
    for col in cat_cols:
        train_o[col] = train_o[col].astype(str).fillna("__MISSING__")
        pred_o[col] = pred_o[col].astype(str).fillna("__MISSING__")
        cats = sorted(set(train_o[col]) | set(pred_o[col]))
        train_o[col] = pd.Categorical(train_o[col], categories=cats)
        pred_o[col] = pd.Categorical(pred_o[col], categories=cats)
    return train_o, pred_o


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_and_predict(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, dict]:
    train_r, pred_r = prepare_categoricals(train_df, predict_df, categorical)
    y = train_r["demand"].to_numpy()
    all_preds, fitted = {}, {}

    # ExtraTrees
    print("  Training ExtraTrees...", flush=True)
    et = make_extratrees_pipeline(numeric, categorical)
    et.fit(train_r[features], y)
    all_preds["extra_trees"] = et.predict(pred_r[features])
    fitted["extra_trees"] = et

    # LightGBM models
    for name, model in make_lgbm_models().items():
        print(f"  Training {name}...", flush=True)
        model.fit(train_r[features], y, categorical_feature=categorical)
        all_preds[name] = model.predict(pred_r[features])
        fitted[name] = model

    # XGBoost
    print("  Training XGBoost...", flush=True)
    xgb = XGBRegressor(
        objective="reg:tweedie", tweedie_variance_power=1.2,
        n_estimators=300, learning_rate=0.04, max_depth=7,
        min_child_weight=15, subsample=0.85, colsample_bytree=0.82,
        reg_alpha=0.2, reg_lambda=6.0, gamma=0.2,
        tree_method="hist", enable_categorical=True,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    xgb.fit(train_r[features], y)
    all_preds["xgb_tweedie"] = xgb.predict(pred_r[features])
    fitted["xgb_tweedie"] = xgb

    # CatBoost
    if HAS_CATBOOST:
        print("  Training CatBoost...", flush=True)
        cat_str_cols = [c for c in categorical if c in features]
        train_cat = train_r[features].copy()
        pred_cat = pred_r[features].copy()
        for col in cat_str_cols:
            train_cat[col] = train_cat[col].astype(str)
            pred_cat[col] = pred_cat[col].astype(str)

        cb = CatBoostRegressor(
            iterations=800, learning_rate=0.04, depth=8,
            l2_leaf_reg=5, random_strength=1.0,
            bagging_temperature=0.8, rsm=0.8,
            cat_features=cat_str_cols,
            one_hot_max_size=10,
            boosting_type="Ordered",
            od_type="Iter", od_wait=50,
            random_seed=RANDOM_STATE,
            verbose=False,
        )
        cb.fit(train_cat, y)
        all_preds["catboost"] = cb.predict(pred_cat)
        fitted["catboost"] = cb

    return pd.DataFrame(all_preds, index=predict_df.index), fitted


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — META-LEARNER
# ══════════════════════════════════════════════════════════════════════════════

def fit_ridge_meta(oof_preds: pd.DataFrame, y: np.ndarray, road_types: pd.Series):
    cols = list(oof_preds.columns)
    X = oof_preds[cols].fillna(0).to_numpy()

    # Find the best alpha with a simple grid
    best_alpha, best_score = 5.0, -1e9
    for alpha in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        ridge = Ridge(alpha=alpha, positive=True)
        ridge.fit(X, y)
        pred = apply_bounds(ridge.predict(X), road_types)
        sc = r2_score(y, pred)
        if sc > best_score:
            best_score, best_alpha = sc, alpha

    ridge = Ridge(alpha=best_alpha, positive=True)
    ridge.fit(X, y)

    # ExtraTrees guardrail: if ExtraTrees alone beats the blend, fall back
    if "extra_trees" in cols:
        et_score = r2_score(y, apply_bounds(oof_preds["extra_trees"].to_numpy(), road_types))
        blend_score = best_score
        print(f"  ExtraTrees OOF R2: {et_score:.5f} | Ridge blend R2: {blend_score:.5f}")
        if et_score > blend_score:
            print("  >> ExtraTrees guardrail triggered — using ExtraTrees only")
            weights = {c: 1.0 if c == "extra_trees" else 0.0 for c in cols}
            return weights, et_score

    weights = dict(zip(cols, ridge.coef_))
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    return weights, best_score


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — RESIDUAL BIAS CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_residual_bias(
    known49_df: pd.DataFrame,
    model_preds_on_known49: np.ndarray,
) -> dict[str, float]:
    residuals = known49_df["demand"].to_numpy() - model_preds_on_known49
    geo_bias = (
        pd.DataFrame({"geohash": known49_df["geohash"].values, "resid": residuals})
        .groupby("geohash")["resid"]
        .mean()
        .to_dict()
    )
    return geo_bias


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(data_dir: Path = Path("dataset")) -> None:
    print("=" * 60)
    print("Traffic Demand Forecasting — Best Model Pipeline")
    print("=" * 60)

    # Load
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    print(f"Train: {train.shape} | Test: {test.shape}")

    # Impute RoadType
    train = impute_road_type(train, train)
    test = impute_road_type(train, test)

    # Geo lookup + K-means cluster
    all_geos = pd.concat([train[["geohash"]], test[["geohash"]]]).drop_duplicates()
    geo_lookup = build_geo_lookup(all_geos["geohash"])
    geo_cluster_map = build_geo_clusters(geo_lookup, n_clusters=50)
    print(f"Built geo lookup ({len(geo_lookup)} unique geohashes), K-Means 50 clusters")

    # Base features
    train_base = add_base_features(train, geo_lookup)
    test_base = add_base_features(test, geo_lookup)

    day48 = train_base[train_base["day"] == 48].copy()
    day49 = train_base[train_base["day"] == 49].copy()

    # ── OOF validation ────────────────────────────────────────────────────────
    print("\n--- Chronological OOF Validation (3 folds) ---")
    oof_preds_list, oof_y_list, oof_roads_list = [], [], []

    for val_start in VALIDATION_CUTOFFS:
        print(f"  Fold: Day-49 from minute {val_start} onward")
        known49_fold = day49[day49["minute"] < val_start].copy()
        val_fold = day49[day49["minute"] >= val_start].copy()

        if len(val_fold) == 0:
            continue

        train_fold = pd.concat([day48, known49_fold], ignore_index=True)

        train_feat = attach_reference_features(train_fold, day48, known49_fold, geo_cluster_map)
        val_feat = attach_reference_features(val_fold, day48, known49_fold, geo_cluster_map)

        features, numeric, categorical = get_feature_cols(train_feat)

        preds_df, _ = train_and_predict(train_feat, val_feat, features, numeric, categorical)

        oof_preds_list.append(preds_df.reset_index(drop=True))
        oof_y_list.append(val_feat["demand"].to_numpy())
        oof_roads_list.append(val_feat["RoadType"])

    oof_preds = pd.concat(oof_preds_list, ignore_index=True).fillna(0).clip(0, 1)
    oof_y = np.concatenate(oof_y_list)
    oof_roads = pd.concat(oof_roads_list, ignore_index=True)

    # Individual model scores
    print("\n  Individual model OOF R2 (with road bounds):")
    for col in oof_preds.columns:
        sc = r2_score(oof_y, apply_bounds(oof_preds[col].to_numpy(), oof_roads))
        print(f"    {col:<18} {sc:.5f}")

    weights, blend_score = fit_ridge_meta(oof_preds, oof_y, oof_roads)
    print(f"\n  Final blend OOF R2 (with road bounds): {blend_score:.5f}")
    print(f"  Score -> competition metric: {100 * blend_score:.2f}")
    print("  Weights:", {k: round(v, 4) for k, v in weights.items()})

    # Save validation report
    report_rows = []
    for col in oof_preds.columns:
        sc = r2_score(oof_y, apply_bounds(oof_preds[col].to_numpy(), oof_roads))
        report_rows.append({"model": col, "r2": sc})
    report_rows.append({"model": "final_blend", "r2": blend_score})
    pd.DataFrame(report_rows).to_csv("model_validation_report.csv", index=False)

    # ── Final training on all data ────────────────────────────────────────────
    print("\n--- Training Final Models on All Data ---")
    known49_final = day49.copy()
    train_feat_final = attach_reference_features(
        pd.concat([day48, known49_final], ignore_index=True),
        day48, known49_final, geo_cluster_map
    )
    test_feat_final = attach_reference_features(test_base, day48, known49_final, geo_cluster_map)

    features, numeric, categorical = get_feature_cols(train_feat_final)

    test_preds_df, _ = train_and_predict(
        train_feat_final, test_feat_final, features, numeric, categorical
    )

    # ── Compute residual bias from known Day-49 slots ─────────────────────────
    print("\n--- Computing per-geohash residual bias correction ---")
    known49_feat = attach_reference_features(known49_final, day48, known49_final, geo_cluster_map)
    known49_model_preds, _ = train_and_predict(
        train_feat_final, known49_feat, features, numeric, categorical
    )
    known49_blend = (
        known49_model_preds[list(weights)].to_numpy()
        @ np.array(list(weights.values()))
    ).clip(0, 1)
    known49_blend_bounded = apply_bounds(known49_blend, known49_feat["RoadType"])
    geo_bias = compute_residual_bias(known49_final, known49_blend_bounded)
    print(f"  Computed bias for {len(geo_bias)} geohashes | "
          f"mean abs bias: {np.mean(np.abs(list(geo_bias.values()))):.5f}")

    # ── Final blended predictions ─────────────────────────────────────────────
    test_blend = (
        test_preds_df[list(weights)].fillna(0).to_numpy()
        @ np.array(list(weights.values()))
    ).clip(0, 1)

    # Apply road-type physical bounds
    test_blend = apply_bounds(test_blend, test_feat_final["RoadType"])

    # Apply per-geohash residual bias correction (capped at ±0.03 to prevent over-correction)
    bias_correction = test_feat_final["geohash"].map(geo_bias).fillna(0.0).to_numpy()
    bias_correction = np.clip(bias_correction, -0.03, 0.03)
    test_blend = np.clip(test_blend + bias_correction, 0, 1)

    # Re-apply bounds after bias correction
    test_blend = apply_bounds(test_blend, test_feat_final["RoadType"])

    # ── Write submission ──────────────────────────────────────────────────────
    submission = pd.DataFrame({
        "Index": test["Index"].to_numpy(),
        "demand": test_blend,
    })
    assert submission.shape == (41778, 2)
    assert submission["demand"].between(0, 1).all()
    submission.to_csv("submission.csv", index=False)

    print("\n" + "=" * 60)
    print(f"  OOF Validation R2: {blend_score:.5f} -> Score: {100*blend_score:.2f}")
    print("  Wrote: submission.csv, model_validation_report.csv")
    print("  Demand distribution:")
    print(submission["demand"].describe().to_string())
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset", type=Path)
    args = parser.parse_args()
    run(args.data_dir)
