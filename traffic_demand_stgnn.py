"""PyTorch spatiotemporal graph model for the traffic demand challenge.

This script is intentionally separate from the tabular model runner. It builds a
geohash graph, trains a compact graph-recurrent model, and writes:

    submission_stgnn.csv
    submission_stgnn_blend.csv
    stgnn_validation_report.csv

Run:
    python traffic_demand_stgnn.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn


BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
ROAD_BOUNDS = {
    "Residential": (0.0, 0.219997),
    "Street": (0.220016, 0.349908),
    "Highway": (0.350009, 1.0),
}
ROAD_ORDER = ["Residential", "Street", "Highway"]
WEATHER_ORDER = ["Sunny", "Rainy", "Foggy", "Snowy", "__MISSING__"]
TEST_MINUTES = list(range(135, 826, 15))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_timestamp(value: str) -> int:
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)


def decode_geohash(geohash: str) -> tuple[float, float]:
    lat = [-90.0, 90.0]
    lon = [-180.0, 180.0]
    even = True
    for char in geohash:
        bits = BASE32.index(char)
        for mask in (16, 8, 4, 2, 1):
            if even:
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
            even = not even
    return (lat[0] + lat[1]) / 2, (lon[0] + lon[1]) / 2


def add_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["minute"] = out["timestamp"].map(parse_timestamp)
    out["slot"] = out["minute"] // 15
    return out


def mode_or_none(values: pd.Series) -> str | None:
    modes = values.dropna().mode()
    return None if modes.empty else str(modes.iloc[0])


def impute_road_type(train: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Impute RoadType with progressively coarser keys and rule fallbacks."""
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


def apply_road_bounds(pred: np.ndarray, road_types: list[str]) -> np.ndarray:
    bounded = pred.copy()
    for road, (lo, hi) in ROAD_BOUNDS.items():
        mask = np.array(road_types) == road
        bounded[mask] = np.clip(bounded[mask], lo, hi)
    return np.clip(bounded, 0, 1)


def apply_road_bounds_matrix(pred: np.ndarray, road_types: list[str]) -> np.ndarray:
    bounded = pred.copy()
    road_arr = np.array(road_types)
    for road, (lo, hi) in ROAD_BOUNDS.items():
        mask = road_arr == road
        bounded[:, mask] = np.clip(bounded[:, mask], lo, hi)
    return np.clip(bounded, 0, 1)


def make_node_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    all_geos = sorted(set(train["geohash"]) | set(test["geohash"]))
    rows = []
    full = pd.concat([train.drop(columns=["demand"], errors="ignore"), test], ignore_index=True)
    full = impute_road_type(train, full)
    for geo in all_geos:
        part = full[full["geohash"] == geo]
        lat, lon = decode_geohash(geo)
        rows.append(
            {
                "geohash": geo,
                "lat": lat,
                "lon": lon,
                "RoadType": mode_or_none(part["RoadType"]) or "Residential",
                "NumberofLanes": int(part["NumberofLanes"].mode().iloc[0]),
                "LargeVehicles": mode_or_none(part["LargeVehicles"]) or "Not Allowed",
                "Landmarks": mode_or_none(part["Landmarks"]) or "No",
                "Temperature": float(part["Temperature"].median())
                if part["Temperature"].notna().any()
                else float(full["Temperature"].median()),
                "Weather": mode_or_none(part["Weather"]) or "__MISSING__",
            }
        )
    nodes = pd.DataFrame(rows)
    nodes["road_id"] = nodes["RoadType"].map({v: i for i, v in enumerate(ROAD_ORDER)}).fillna(0).astype(int)
    nodes["weather_id"] = nodes["Weather"].fillna("__MISSING__").map(
        {v: i for i, v in enumerate(WEATHER_ORDER)}
    ).fillna(len(WEATHER_ORDER) - 1).astype(int)
    return nodes


def make_static_features(nodes: pd.DataFrame) -> np.ndarray:
    lat = nodes["lat"].to_numpy()
    lon = nodes["lon"].to_numpy()
    temp = nodes["Temperature"].fillna(nodes["Temperature"].median()).to_numpy()
    features = []
    features.append((lat - lat.mean()) / (lat.std() + 1e-6))
    features.append((lon - lon.mean()) / (lon.std() + 1e-6))
    features.append((temp - temp.mean()) / (temp.std() + 1e-6))
    features.append(nodes["NumberofLanes"].to_numpy() / 5.0)
    features.append((nodes["LargeVehicles"] == "Allowed").astype(float).to_numpy())
    features.append((nodes["Landmarks"] == "Yes").astype(float).to_numpy())
    for road in ROAD_ORDER:
        features.append((nodes["RoadType"] == road).astype(float).to_numpy())
    for weather in WEATHER_ORDER:
        features.append((nodes["Weather"].fillna("__MISSING__") == weather).astype(float).to_numpy())
    return np.vstack(features).T.astype(np.float32)


def make_adjacency(nodes: pd.DataFrame, k: int = 8) -> torch.Tensor:
    coords = nodes[["lat", "lon"]].to_numpy()
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=min(k + 1, len(nodes)))
    n = len(nodes)
    adj = np.eye(n, dtype=np.float32)
    scale = np.nanmedian(distances[:, 1:]) + 1e-9
    for i in range(n):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            weight = math.exp(-float(dist) / scale)
            adj[i, j] = max(adj[i, j], weight)
            adj[j, i] = max(adj[j, i], weight)
    deg = adj.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    norm = inv_sqrt[:, None] * adj * inv_sqrt[None, :]
    rows, cols = np.nonzero(norm > 0)
    values = norm[rows, cols].astype(np.float32)
    indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    return torch.sparse_coo_tensor(indices, torch.tensor(values), size=norm.shape).coalesce()


@dataclass
class TensorData:
    nodes: pd.DataFrame
    static: np.ndarray
    adj: torch.Tensor
    demand48: np.ndarray
    mask48: np.ndarray
    demand49_known: np.ndarray
    mask49_known: np.ndarray
    train: pd.DataFrame
    test: pd.DataFrame


def build_tensor_data(data_dir: Path) -> TensorData:
    train = add_time(pd.read_csv(data_dir / "train.csv"))
    test = add_time(pd.read_csv(data_dir / "test.csv"))
    train = impute_road_type(train, train)
    test = impute_road_type(train, test)
    nodes = make_node_table(train, test)
    geo_to_idx = {g: i for i, g in enumerate(nodes["geohash"])}
    n = len(nodes)
    demand48 = np.full((96, n), np.nan, dtype=np.float32)
    demand49 = np.full((96, n), np.nan, dtype=np.float32)
    for row in train.itertuples(index=False):
        idx = geo_to_idx[row.geohash]
        slot = int(row.slot)
        if int(row.day) == 48:
            demand48[slot, idx] = float(row.demand)
        elif int(row.day) == 49:
            demand49[slot, idx] = float(row.demand)
    mask48 = np.isfinite(demand48).astype(np.float32)
    mask49 = np.isfinite(demand49).astype(np.float32)
    global_mean = float(np.nanmean(demand48))
    demand48 = np.nan_to_num(demand48, nan=global_mean)
    demand49 = np.nan_to_num(demand49, nan=0.0)
    return TensorData(
        nodes=nodes,
        static=make_static_features(nodes),
        adj=make_adjacency(nodes),
        demand48=demand48,
        mask48=mask48,
        demand49_known=demand49,
        mask49_known=mask49,
        train=train,
        test=test,
    )


def early_ratio(data: TensorData) -> np.ndarray:
    num = (data.demand49_known[:9] * data.mask49_known[:9]).sum(axis=0)
    den = (data.demand48[:9] * data.mask48[:9]).sum(axis=0)
    global_ratio = float(num.sum() / max(den.sum(), 1e-9))
    count = data.mask49_known[:9].sum(axis=0)
    ratio = (num + 0.5 * global_ratio) / (den + 0.5)
    ratio[count == 0] = global_ratio
    return np.clip(ratio, 0.4, 2.5).astype(np.float32)


def make_step_features(
    static: np.ndarray,
    prev: np.ndarray,
    lag2: np.ndarray,
    lag4: np.ndarray,
    day48_same: np.ndarray,
    day48_prev: np.ndarray,
    ratio: np.ndarray,
    slot: int,
) -> np.ndarray:
    n = static.shape[0]
    slot_sin = np.full((n, 1), math.sin(2 * math.pi * slot / 96), dtype=np.float32)
    slot_cos = np.full((n, 1), math.cos(2 * math.pi * slot / 96), dtype=np.float32)
    scaled = (day48_same * ratio).reshape(-1, 1)
    dyn = np.vstack([prev, lag2, lag4, day48_same, day48_prev, scaled.reshape(-1), ratio]).T.astype(np.float32)
    return np.hstack([static, dyn, slot_sin, slot_cos]).astype(np.float32)


def make_training_examples(data: TensorData, start_slot: int = 8, end_slot: int = 96) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratio48 = np.ones(data.demand48.shape[1], dtype=np.float32)
    xs, ys, masks = [], [], []
    for slot in range(start_slot, end_slot):
        x = make_step_features(
            data.static,
            data.demand48[slot - 1],
            data.demand48[max(slot - 2, 0)],
            data.demand48[max(slot - 4, 0)],
            data.demand48[slot],
            data.demand48[max(slot - 1, 0)],
            ratio48,
            slot,
        )
        xs.append(x)
        ys.append(data.demand48[slot])
        masks.append(data.mask48[slot])
    return (
        torch.tensor(np.stack(xs), dtype=torch.float32),
        torch.tensor(np.stack(ys), dtype=torch.float32),
        torch.tensor(np.stack(masks), dtype=torch.float32),
    )


class GraphGRU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.gcn_in = nn.Linear(in_dim, hidden_dim)
        self.gcn_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def graph_encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: T, N, F
        ax = torch.stack([torch.sparse.mm(adj, x_t) for x_t in x], dim=0)
        h = torch.relu(self.gcn_in(ax))
        ah = torch.stack([torch.sparse.mm(adj, h_t) for h_t in h], dim=0)
        return torch.relu(self.gcn_hidden(ah) + h)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Process each node's temporal sequence with shared GRU after graph mixing.
        h = self.graph_encode(x, adj)
        h_nodes = h.permute(1, 0, 2)
        out, _ = self.gru(h_nodes)
        pred = self.head(out).squeeze(-1).permute(1, 0)
        return pred


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def bounds_penalty(pred: torch.Tensor, nodes: pd.DataFrame) -> torch.Tensor:
    lo = torch.tensor([ROAD_BOUNDS[r][0] for r in nodes["RoadType"]], dtype=pred.dtype, device=pred.device)
    hi = torch.tensor([ROAD_BOUNDS[r][1] for r in nodes["RoadType"]], dtype=pred.dtype, device=pred.device)
    return (torch.relu(lo[None, :] - pred).pow(2) + torch.relu(pred - hi[None, :]).pow(2)).mean()


def train_model(data: TensorData, seed: int, epochs: int, hidden_dim: int) -> tuple[GraphGRU, dict[str, float]]:
    seed_everything(seed)
    x, y, mask = make_training_examples(data)
    in_dim = x.shape[-1]
    model = GraphGRU(in_dim=in_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    patience = 20
    stale = 0
    # Last 20 slots of day48 are a small temporal validation block.
    train_slice = slice(0, max(1, x.shape[0] - 20))
    val_slice = slice(max(1, x.shape[0] - 20), x.shape[0])
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x[train_slice], data.adj)
        loss = masked_mse(pred, y[train_slice], mask[train_slice]) + 0.02 * bounds_penalty(pred, data.nodes)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x[val_slice], data.adj)
            val_loss = masked_mse(val_pred, y[val_slice], mask[val_slice]).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_mse": best_loss, "epochs_ran": epoch + 1}


def recursive_predict(data: TensorData, model: GraphGRU) -> np.ndarray:
    ratio = early_ratio(data)
    history = np.zeros((96, data.demand48.shape[1]), dtype=np.float32)
    history[:9] = data.demand49_known[:9]
    # Fill unobserved early nodes with scaled day48 so recursion has sane inputs.
    for slot in range(9):
        fill = data.demand48[slot] * ratio
        history[slot] = np.where(data.mask49_known[slot] > 0, history[slot], fill)
    preds_by_slot = {}
    model.eval()
    with torch.no_grad():
        for slot in TEST_MINUTES:
            s = slot // 15
            x = make_step_features(
                data.static,
                history[s - 1],
                history[max(s - 2, 0)],
                history[max(s - 4, 0)],
                data.demand48[s],
                data.demand48[max(s - 1, 0)],
                ratio,
                s,
            )
            # Use a short synthetic sequence ending at the current target so the GRU
            # sees recent dynamics without retraining a decoder.
            seq = []
            for prev_s in range(max(1, s - 5), s + 1):
                seq.append(
                    make_step_features(
                        data.static,
                        history[max(prev_s - 1, 0)],
                        history[max(prev_s - 2, 0)],
                        history[max(prev_s - 4, 0)],
                        data.demand48[prev_s],
                        data.demand48[max(prev_s - 1, 0)],
                        ratio,
                        prev_s,
                    )
                )
            xt = torch.tensor(np.stack(seq), dtype=torch.float32)
            pred_seq = model(xt, data.adj).numpy()
            pred = pred_seq[-1]
            # Residual skip from scaled same-slot prior keeps the shallow-history
            # graph model anchored to the strongest deterministic signal.
            scaled_prior = data.demand48[s] * ratio
            pred = 0.55 * pred + 0.45 * scaled_prior
            pred = apply_road_bounds(pred, data.nodes["RoadType"].tolist())
            history[s] = pred.astype(np.float32)
            preds_by_slot[s] = history[s].copy()
    return np.stack([preds_by_slot[m // 15] for m in TEST_MINUTES])


def scaled_lag_baseline(data: TensorData) -> np.ndarray:
    ratio = early_ratio(data)
    rows = []
    for minute in TEST_MINUTES:
        slot = minute // 15
        pred = data.demand48[slot] * ratio
        rows.append(apply_road_bounds(pred, data.nodes["RoadType"].tolist()))
    return np.stack(rows).astype(np.float32)


def road_slot_baseline(data: TensorData) -> np.ndarray:
    train = data.train.copy()
    global_by_road_slot = (
        train.groupby(["RoadType", "slot"])["demand"].mean().to_dict()
    )
    rows = []
    for minute in TEST_MINUTES:
        slot = minute // 15
        vals = []
        for road in data.nodes["RoadType"]:
            vals.append(global_by_road_slot.get((road, slot), global_by_road_slot.get((road, 0), train["demand"].mean())))
        rows.append(apply_road_bounds(np.array(vals, dtype=np.float32), data.nodes["RoadType"].tolist()))
    return np.stack(rows)


def evaluate_day48_proxy(data: TensorData, pred_slots: np.ndarray, name: str) -> dict[str, float | str]:
    rows = []
    y_true, y_pred = [], []
    for i, minute in enumerate(TEST_MINUTES):
        slot = minute // 15
        mask = data.mask48[slot] > 0
        y_true.append(data.demand48[slot][mask])
        y_pred.append(pred_slots[i][mask])
    y = np.concatenate(y_true)
    p = np.concatenate(y_pred)
    rows.append({"model": name, "segment": "global", "r2": r2_score(y, p), "mae": mean_absolute_error(y, p)})
    for road in ROAD_ORDER:
        road_mask_nodes = (data.nodes["RoadType"].to_numpy() == road)
        yt, yp = [], []
        for i, minute in enumerate(TEST_MINUTES):
            slot = minute // 15
            mask = (data.mask48[slot] > 0) & road_mask_nodes
            if mask.any():
                yt.append(data.demand48[slot][mask])
                yp.append(pred_slots[i][mask])
        if yt:
            y = np.concatenate(yt)
            p = np.concatenate(yp)
            rows.append({"model": name, "segment": road, "r2": r2_score(y, p), "mae": mean_absolute_error(y, p)})
    for label, start, end in [
        ("02:15-06:00", 135, 360),
        ("06:15-10:00", 375, 600),
        ("10:15-13:45", 615, 825),
    ]:
        yt, yp = [], []
        for i, minute in enumerate(TEST_MINUTES):
            if start <= minute <= end:
                slot = minute // 15
                mask = data.mask48[slot] > 0
                yt.append(data.demand48[slot][mask])
                yp.append(pred_slots[i][mask])
        if yt:
            y = np.concatenate(yt)
            p = np.concatenate(yp)
            rows.append({"model": name, "segment": label, "r2": r2_score(y, p), "mae": mean_absolute_error(y, p)})
    return rows


def predictions_to_submission(data: TensorData, slot_preds: np.ndarray, output: Path) -> pd.DataFrame:
    geo_to_idx = {g: i for i, g in enumerate(data.nodes["geohash"])}
    minute_to_row = {minute: i for i, minute in enumerate(TEST_MINUTES)}
    preds = []
    for row in data.test.itertuples(index=False):
        node = geo_to_idx[row.geohash]
        mrow = minute_to_row[int(row.minute)]
        preds.append(float(slot_preds[mrow, node]))
    sub = pd.DataFrame({"Index": data.test["Index"].to_numpy(), "demand": preds})
    assert sub.shape == (41778, 2)
    assert list(sub.columns) == ["Index", "demand"]
    assert sub["Index"].equals(data.test["Index"])
    assert np.isfinite(sub["demand"]).all()
    assert sub["demand"].between(0, 1).all()
    sub.to_csv(output, index=False)
    return sub


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2027])
    args = parser.parse_args()

    data = build_tensor_data(args.data_dir)
    print(f"Graph nodes: {len(data.nodes)}; static features: {data.static.shape[1]}")

    scaled = scaled_lag_baseline(data)
    road_slot = road_slot_baseline(data)
    eval_rows = []
    eval_rows.extend(evaluate_day48_proxy(data, scaled, "scaled_lag"))
    eval_rows.extend(evaluate_day48_proxy(data, road_slot, "road_slot"))

    stgnn_preds = []
    train_logs = []
    for seed in args.seeds:
        print(f"Training STGNN seed {seed}...")
        model, log = train_model(data, seed=seed, epochs=args.epochs, hidden_dim=args.hidden_dim)
        log["seed"] = seed
        train_logs.append(log)
        pred = recursive_predict(data, model)
        stgnn_preds.append(pred)
    stgnn = np.mean(stgnn_preds, axis=0)

    # Conservative blend: keep the graph model influential, but anchor to the
    # deterministic same-slot ratio and road-slot pattern.
    blend = apply_road_bounds_matrix(
        0.50 * stgnn + 0.35 * scaled + 0.15 * road_slot,
        data.nodes["RoadType"].tolist(),
    )

    eval_rows.extend(evaluate_day48_proxy(data, stgnn, "stgnn"))
    eval_rows.extend(evaluate_day48_proxy(data, blend, "stgnn_blend"))
    report = pd.DataFrame(eval_rows)
    report["train_logs"] = json.dumps(train_logs)
    report.to_csv("stgnn_validation_report.csv", index=False)
    print(report.to_string(index=False))

    sub_stgnn = predictions_to_submission(data, stgnn, Path("submission_stgnn.csv"))
    sub_blend = predictions_to_submission(data, blend, Path("submission_stgnn_blend.csv"))
    print("\nsubmission_stgnn.csv")
    print(sub_stgnn["demand"].describe().to_string())
    print("\nsubmission_stgnn_blend.csv")
    print(sub_blend["demand"].describe().to_string())


if __name__ == "__main__":
    main()
