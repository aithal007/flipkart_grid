"""Generate high-signal competition candidate submissions.

These candidates are designed for leaderboard probing after the public score
showed that local validation was optimistic. They exploit the strongest
dataset-specific facts:

- test is day 49 from 02:15 to 13:45
- day 48 same geohash/time is the main prior
- day 49 00:00-02:00 gives a per-location drift ratio
- demand is road-regime bounded

Outputs:
    submission_comp_bounds_current.csv
    submission_comp_scaled_geo.csv
    submission_comp_additive_geo.csv
    submission_comp_regime_blend.csv
    competition_candidate_report.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


ROAD_BOUNDS = {
    "Residential": (0.0, 0.219997),
    "Street": (0.220016, 0.349908),
    "Highway": (0.350009, 1.0),
}
ROAD_ORDER = ["Residential", "Street", "Highway"]


def parse_timestamp(value: str) -> int:
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)


def add_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["minute"] = out["timestamp"].map(parse_timestamp)
    out["slot"] = out["minute"] // 15
    return out


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

    mask = out["RoadType"].isna()
    out.loc[mask & (out["NumberofLanes"] >= 4), "RoadType"] = "Highway"
    mask = out["RoadType"].isna()
    out.loc[
        mask
        & (out["NumberofLanes"] == 1)
        & (out["LargeVehicles"] == "Not Allowed")
        & (out["Landmarks"] == "Yes"),
        "RoadType",
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


def build_candidate_frame(train: pd.DataFrame, target: pd.DataFrame, known49: pd.DataFrame) -> pd.DataFrame:
    day48 = train[train["day"] == 48].copy()
    global_mean = float(day48["demand"].mean())
    out = target.copy()

    exact = day48[["geohash", "minute", "demand"]].rename(columns={"demand": "d48_exact"})
    out = out.merge(exact, on=["geohash", "minute"], how="left")

    geo_mean = day48.groupby("geohash")["demand"].mean().rename("geo_mean")
    gh5_mean = day48.assign(gh5=day48["geohash"].str[:5]).groupby("gh5")["demand"].mean().rename("gh5_mean")
    road_slot = day48.groupby(["RoadType", "minute"])["demand"].mean().rename("road_slot_mean")
    road_mean = day48.groupby("RoadType")["demand"].mean().rename("road_mean")
    out["gh5"] = out["geohash"].str[:5]
    out = out.merge(geo_mean, on="geohash", how="left")
    out = out.merge(gh5_mean, on="gh5", how="left")
    out = out.merge(road_slot, on=["RoadType", "minute"], how="left")
    out = out.merge(road_mean, on="RoadType", how="left")

    # Per-geohash/prefix/road drift: compare known day49 window with day48 same window.
    known = known49[known49["demand"].notna()].copy()
    d48_known = day48[["geohash", "minute", "demand"]].rename(columns={"demand": "d48_known"})
    joined = known.merge(d48_known, on=["geohash", "minute"], how="inner")
    joined = joined[(joined["d48_known"] > 0) & joined["demand"].notna()].copy()
    global_ratio = float(joined["demand"].sum() / max(joined["d48_known"].sum(), 1e-9)) if len(joined) else 1.0

    ratio_tables = {}
    for keys, name, strength in [
        (["geohash"], "ratio_geo", 0.50),
        (["gh5"], "ratio_gh5", 1.00),
        (["RoadType"], "ratio_road", 2.00),
    ]:
        tmp = joined.copy()
        tmp["gh5"] = tmp["geohash"].str[:5]
        agg = tmp.groupby(keys).agg(y=("demand", "sum"), x=("d48_known", "sum")).reset_index()
        agg[name] = (agg["y"] + strength * global_ratio) / (agg["x"] + strength)
        ratio_tables[name] = (keys, agg[keys + [name]])

    out["ratio_global"] = global_ratio
    for name, (keys, table) in ratio_tables.items():
        out = out.merge(table, on=keys, how="left")
        out[name] = out[name].fillna(out["ratio_global"])

    # Additive drift often works better than multiplicative scaling for low-demand residential cells.
    drift = joined.groupby("geohash").agg(y=("demand", "mean"), x=("d48_known", "mean")).reset_index()
    drift["delta_geo"] = drift["y"] - drift["x"]
    out = out.merge(drift[["geohash", "delta_geo"]], on="geohash", how="left")
    out["delta_geo"] = out["delta_geo"].fillna(0.0)

    out["base_prior"] = (
        out["d48_exact"]
        .fillna(out["geo_mean"])
        .fillna(out["gh5_mean"])
        .fillna(out["road_slot_mean"])
        .fillna(out["road_mean"])
        .fillna(global_mean)
    )
    out["scaled_geo"] = out["base_prior"] * (
        0.68 * out["ratio_geo"] + 0.18 * out["ratio_gh5"] + 0.08 * out["ratio_road"] + 0.06 * out["ratio_global"]
    )
    out["additive_geo"] = out["base_prior"] + out["delta_geo"]

    # Regime-specific blend: Street is stable; Highway needs more slot prior; Residential benefits from drift.
    out["regime_blend"] = out["scaled_geo"]
    street = out["RoadType"] == "Street"
    res = out["RoadType"] == "Residential"
    hwy = out["RoadType"] == "Highway"
    out.loc[street, "regime_blend"] = 0.75 * out.loc[street, "road_slot_mean"].fillna(out.loc[street, "base_prior"]) + 0.25 * out.loc[street, "scaled_geo"]
    out.loc[res, "regime_blend"] = 0.55 * out.loc[res, "scaled_geo"] + 0.35 * out.loc[res, "additive_geo"] + 0.10 * out.loc[res, "road_slot_mean"].fillna(out.loc[res, "base_prior"])
    out.loc[hwy, "regime_blend"] = 0.70 * out.loc[hwy, "scaled_geo"] + 0.20 * out.loc[hwy, "road_slot_mean"].fillna(out.loc[hwy, "base_prior"]) + 0.10 * out.loc[hwy, "base_prior"]

    for col in ["base_prior", "scaled_geo", "additive_geo", "regime_blend"]:
        out[col] = apply_bounds(out[col].to_numpy(), out["RoadType"])
    return out


def score_candidates(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    y = frame["demand"].to_numpy()
    for col in cols:
        rows.append({"candidate": col, "segment": "global", "r2": r2_score(y, frame[col]), "mae": mean_absolute_error(y, frame[col])})
        for road in ROAD_ORDER:
            part = frame[frame["RoadType"] == road]
            if len(part) > 2:
                rows.append({"candidate": col, "segment": road, "r2": r2_score(part["demand"], part[col]), "mae": mean_absolute_error(part["demand"], part[col])})
    return pd.DataFrame(rows)


def write_submission(test: pd.DataFrame, pred: np.ndarray, path: str) -> None:
    sub = pd.DataFrame({"Index": test["Index"].to_numpy(), "demand": pred})
    assert sub.shape == (41778, 2)
    assert list(sub.columns) == ["Index", "demand"]
    assert sub["Index"].equals(test["Index"])
    assert np.isfinite(sub["demand"]).all()
    assert sub["demand"].between(0, 1).all()
    sub.to_csv(path, index=False)


def main() -> None:
    data_dir = Path("dataset")
    train = add_time(pd.read_csv(data_dir / "train.csv"))
    test = add_time(pd.read_csv(data_dir / "test.csv"))
    train = impute_road_type(train, train)
    test = impute_road_type(train, test)

    # Validation: pretend only day49 before 01:15 is known, predict the rest of known day49.
    known_cutoff = 75
    known49_val = train[(train["day"] == 49) & (train["minute"] < known_cutoff)].copy()
    val = train[(train["day"] == 49) & (train["minute"] >= known_cutoff)].copy()
    val_frame = build_candidate_frame(train, val, known49_val)
    report = score_candidates(val_frame, ["base_prior", "scaled_geo", "additive_geo", "regime_blend"])

    # Final candidates use all known day49 rows.
    known49_final = train[train["day"] == 49].copy()
    final_frame = build_candidate_frame(train, test, known49_final)

    write_submission(test, final_frame["scaled_geo"].to_numpy(), "submission_comp_scaled_geo.csv")
    write_submission(test, final_frame["additive_geo"].to_numpy(), "submission_comp_additive_geo.csv")
    write_submission(test, final_frame["regime_blend"].to_numpy(), "submission_comp_regime_blend.csv")

    current_path = Path("submission.csv")
    if current_path.exists():
        current = pd.read_csv(current_path)
        bounded = test[["Index", "RoadType"]].merge(current, on="Index", how="left")
        bounded_pred = apply_bounds(bounded["demand"].to_numpy(), bounded["RoadType"])
        write_submission(test, bounded_pred, "submission_comp_bounds_current.csv")

    report.to_csv("competition_candidate_report.csv", index=False)
    print(report.sort_values(["segment", "r2"], ascending=[True, False]).to_string(index=False))
    print("\nWrote competition candidate submissions.")
    for path in [
        "submission_comp_scaled_geo.csv",
        "submission_comp_additive_geo.csv",
        "submission_comp_regime_blend.csv",
        "submission_comp_bounds_current.csv",
    ]:
        if Path(path).exists():
            sub = pd.read_csv(path)
            print(path, sub["demand"].describe().to_dict())


if __name__ == "__main__":
    main()
