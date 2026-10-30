"""
PS-003: Federated Traffic-Flow Prediction with Communication-Cost Evaluation

Dataset support:
  1) PeMSD4 .npz with array key "data" shaped [time, sensors, features]
  2) CSV where each sensor is a numeric column; an optional timestamp column is ignored.

If no dataset is supplied, --demo-data generates a clearly synthetic dataset so the
pipeline can be tested. Do not report synthetic-data metrics as real traffic results.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

WINDOW = 12
N_CLIENTS = 5
SEEDS = [0, 1, 2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_traffic(path, demo_data=False):
    """Return traffic as float32 array [time, sensors]."""
    if demo_data:
        rng = np.random.default_rng(2026)
        t, n = 2400, 15
        x = np.arange(t)
        all_sensors = []
        for s in range(n):
            daily = 18 + 7 * np.sin(2 * np.pi * x / 288 + s * 0.18)
            rush = 8 * np.maximum(0, np.sin(2 * np.pi * (x % 288) / 288 * 2 + s * 0.1))
            noise = rng.normal(0, 1.8 + (s % 3) * 0.5, t)
            all_sensors.append(np.maximum(0, daily + rush + (s - n / 2) * 0.8 + noise))
        return np.stack(all_sensors, axis=1).astype(np.float32)

    if not path:
        raise ValueError("Provide --data PATH, or use --demo-data for a synthetic smoke test.")
    p = Path(path)
    if p.suffix.lower() == ".npz":
        with np.load(p) as z:
            if "data" in z:
                arr = z["data"]
            else:
                arr = z[z.files[0]]
        if arr.ndim == 3:
            # PeMSD4 commonly stores [time, sensors, features]; feature 0 is flow.
            arr = arr[:, :, 0]
        elif arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"Expected NPZ array shaped [time,sensors] or [time,sensors,features], got {arr.shape}")
        return np.asarray(arr, dtype=np.float32)

    df = pd.read_csv(p)
    # Keep numeric columns; timestamp strings are automatically excluded.
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError("CSV needs numeric sensor columns (one column per sensor).")
    return numeric.to_numpy(dtype=np.float32)


def make_windows(data, start, end, mean, std, window=WINDOW):
    """Create windows wholly inside [start, end).

    Returns X=[samples,window,1], y=[samples], sensor_ids=[samples], ordered sensor by sensor.
    """
    segment = data[start:end]
    n_win = len(segment) - window
    if n_win <= 0:
        return (np.empty((0, window, 1), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.int64))
    norm = (segment - mean) / std                       # [time, sensors]
    w = sliding_window_view(norm, window + 1, axis=0)   # [n_win, sensors, window+1]
    x = w[..., :window].transpose(1, 0, 2).reshape(-1, window, 1)
    y = w[..., window].T.reshape(-1)
    ids = np.repeat(np.arange(data.shape[1]), n_win)
    return x.astype(np.float32), y.astype(np.float32), ids


def split_bounds(t):
    a, b = int(t * 0.60), int(t * 0.80)
    if a <= WINDOW or b <= a or t <= b + WINDOW:
        raise ValueError("Dataset is too short for a chronological 60/20/20 split and 12-step windows.")
    return a, b


def prepare_data(raw):
    # Handle missing / infinite values using per-sensor interpolation, then forward/back fill.
    frame = pd.DataFrame(raw).replace([np.inf, -np.inf], np.nan)
    frame = frame.interpolate(limit_direction="both").ffill().bfill()
    data = frame.to_numpy(dtype=np.float32)
    train_end, val_end = split_bounds(len(data))
    # Fit normalization on training timestamps only (no future leakage).
    mean = data[:train_end].mean(axis=0)
    std = data[:train_end].std(axis=0)
    std[std < 1e-6] = 1.0
    train = make_windows(data, 0, train_end, mean, std)
    val = make_windows(data, train_end, val_end, mean, std)
    test = make_windows(data, val_end, len(data), mean, std)
    return train, val, test, mean, std, data.shape[1]


class CompactGRU(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1, :]).squeeze(-1)


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def local_train(global_state, x, y, seed, epochs=1, lr=1e-3, batch_size=128):
    set_seed(seed)
    model = CompactGRU().to(DEVICE)
    model.load_state_dict(global_state)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=min(batch_size, max(1, len(ds))), shuffle=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    return clone_state(model), len(ds)


def fedavg(states, counts):
    total = float(sum(counts))
    result = {}
    for key in states[0]:
        result[key] = sum(state[key] * (count / total) for state, count in zip(states, counts))
    return result


def evaluate(model, x, y, mean, std, sensor_ids, batch_size=512):
    if len(x) == 0:
        return {"mae": float("nan"), "rmse": float("nan")}
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i:i+batch_size]).to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    pred_norm = np.concatenate(preds)
    # Convert normalized values back to the original per-sensor scale.
    sensor_ids = np.asarray(sensor_ids)
    pred = pred_norm * std[sensor_ids] + mean[sensor_ids]
    true = y * std[sensor_ids] + mean[sensor_ids]
    return {"mae": float(np.mean(np.abs(pred - true))),
            "rmse": float(np.sqrt(np.mean((pred - true) ** 2)))}


def make_clients(train, n_sensors, seed, mode, n_clients=N_CLIENTS):
    x, y, sensor_ids = train
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    if mode == "iid":
        rng.shuffle(indices)
        shards = np.array_split(indices, n_clients)
    elif mode == "non_iid":
        # Each client receives a disjoint group of sensors (contiguous sensor index ranges).
        # Sensor IDs represent virtual intersections; for real data this is a reproducible
        # sensor-cluster proxy. It does not claim to infer geographic clusters.
        sensor_groups = np.array_split(np.arange(n_sensors), n_clients)
        shards = [np.where(np.isin(sensor_ids, group))[0] for group in sensor_groups]
    else:
        raise ValueError(mode)
    clients = []
    for idx in shards:
        if len(idx) == 0:
            raise ValueError("A client has no training windows. Use more sensors or fewer clients.")
        clients.append((x[idx], y[idx]))
    return clients


def train_centralized(train, val, test, mean, std, seed, epochs=10):
    set_seed(seed)
    model = CompactGRU().to(DEVICE)
    xtr, ytr, _ = train
    xval, yval, sidval = val
    xtest, ytest, sidtest = test
    ds = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        vm = evaluate(model, xval, yval, mean, std, sidval)
        history.append({"round": epoch + 1, "val_mae": vm["mae"], "communication_bytes": 0})
    return model, history, evaluate(model, xtest, ytest, mean, std, sidtest)


def run_federated(train, val, test, n_sensors, mean, std, seed, mode,
                  rounds=10, local_epochs=1, n_clients=N_CLIENTS, lr=1e-3):
    set_seed(seed)
    global_model = CompactGRU().to(DEVICE)
    global_state = clone_state(global_model)
    clients = make_clients(train, n_sensors, seed, mode, n_clients)
    xval, yval, sidval = val
    xtest, ytest, sidtest = test
    params = sum(p.numel() for p in global_model.parameters())
    bytes_per_round = 2 * n_clients * params * 4  # float32 downlink + uplink, no protocol overhead
    history = []
    for rnd in range(1, rounds + 1):
        states, counts = [], []
        for client_id, (xc, yc) in enumerate(clients):
            state, count = local_train(global_state, xc, yc,
                                       seed=seed * 10000 + rnd * 100 + client_id,
                                       epochs=local_epochs, lr=lr)
            states.append(state)
            counts.append(count)
        global_state = fedavg(states, counts)
        global_model.load_state_dict(global_state)
        val_metrics = evaluate(global_model, xval, yval, mean, std, sidval)
        history.append({
            "round": rnd,
            "val_mae": val_metrics["mae"],
            "communication_bytes": int(bytes_per_round),
            "cumulative_communication_bytes": int(bytes_per_round * rnd),
        })
    global_model.load_state_dict(global_state)
    test_metrics = evaluate(global_model, xtest, ytest, mean, std, sidtest)
    test_metrics["params"] = params
    test_metrics["bytes_per_round"] = int(bytes_per_round)
    test_metrics["total_bytes"] = int(bytes_per_round * rounds)
    return global_model, history, test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="PeMSD4 .npz or CSV")
    parser.add_argument("--demo-data", action="store_true", help="Use synthetic data only to smoke-test the pipeline")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--central-epochs", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    if not args.demo_data and not args.data:
        parser.error("Pass --data PATH for real traffic data, or --demo-data for a synthetic smoke test.")
    raw = load_traffic(args.data, args.demo_data)
    train, val, test, mean, std, n_sensors = prepare_data(raw)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    all_rows, curve_rows = [], []
    print(f"Data: {raw.shape[0]} timestamps x {raw.shape[1]} sensors; device={DEVICE}")
    print(f"Windows: train={len(train[0])}, validation={len(val[0])}, test={len(test[0])}")
    print("Communication counts model weights only (float32 download + upload); excludes protocol overhead.")

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        start = time.time()
        _, central_hist, central_test = train_centralized(
            train, val, test, mean, std, seed, epochs=args.central_epochs)
        all_rows.append({"seed": seed, "method": "centralized", "partition": "pooled",
                         **central_test, "rounds": args.central_epochs})
        for row in central_hist:
            curve_rows.append({"seed": seed, "method": "centralized", "partition": "pooled", **row})

        for mode in ("iid", "non_iid"):
            _, hist, metrics = run_federated(
                train, val, test, n_sensors, mean, std, seed, mode,
                rounds=args.rounds, local_epochs=args.local_epochs, n_clients=args.clients)
            all_rows.append({"seed": seed, "method": "fedavg", "partition": mode,
                             **metrics, "rounds": args.rounds, "local_epochs": args.local_epochs})
            for row in hist:
                curve_rows.append({"seed": seed, "method": "fedavg", "partition": mode, **row})
            print(f"FedAvg {mode}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, "
                  f"bytes/round={metrics['bytes_per_round']:,}")
        print(f"Seed finished in {time.time()-start:.1f}s")

    results = pd.DataFrame(all_rows)
    curves = pd.DataFrame(curve_rows)
    results.to_csv(out / "summary_by_seed.csv", index=False)
    curves.to_csv(out / "convergence_curves.csv", index=False)
    summary = (results.groupby(["method", "partition"], dropna=False)
               .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                    rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"))
               .reset_index())
    summary.to_csv(out / "summary_mean_std.csv", index=False)
    (out / "run_config.json").write_text(json.dumps({
        "seeds": SEEDS, "clients": args.clients, "window": WINDOW,
        "split": "chronological 60/20/20", "normalization": "train timestamps only",
        "rounds": args.rounds, "local_epochs": args.local_epochs,
        "central_epochs": args.central_epochs, "dataset_shape": list(raw.shape),
        "synthetic_demo_data": bool(args.demo_data),
        "communication_accounting": "2 * clients * parameter_count * 4 bytes per round; model weights only"
    }, indent=2))
    print("\nSaved results to:", out.resolve())
    print(summary.to_string(index=False))
    if args.demo_data:
        print("\nWARNING: these are synthetic-data smoke-test results, not benchmark traffic results.")


if __name__ == "__main__":
    main()
