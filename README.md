# PS-003 — Federated Traffic-Flow Prediction

A single-machine simulator that compares centralized training with Federated Averaging (FedAvg)
for traffic forecasting, under IID and non-IID partitions of virtual clients.

**Question it answers:** how much forecasting accuracy is lost, and how many bytes are exchanged,
when a traffic predictor is trained across several clients instead of on pooled data?

## What is included

- Compact 1-layer GRU (hidden size 32) with a linear output head
- Five virtual clients by default
- Task: last 12 readings -> next reading
- Chronological 60/20/20 train/validation/test split
- Normalization fitted on training timestamps only
- Three setups: centralized, FedAvg-IID, FedAvg-non-IID
- Three reproducible seeds: 0, 1, 2
- Metrics: MAE, RMSE, validation convergence, bytes per round, cumulative bytes
- CSV result files and a run configuration file

## 1. Setup (Windows / VS Code)

Requires Python with a matching PyTorch build (PyTorch currently lists Python 3.10-3.14; check
https://pytorch.org/get-started/locally/ if installation fails).

Open this folder in VS Code, then Terminal -> New Terminal:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

For a smaller CPU-only PyTorch install (no NVIDIA GPU):

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy pandas
```

## 2. Quick test with synthetic data

`data/sample_traffic.csv` is **synthetic** (20 sensors, 14 days, 5-minute steps). It only checks
that the pipeline runs. Do not report its metrics as real traffic results.

```bash
python main.py --data data/sample_traffic.csv --rounds 2 --central-epochs 2 --output results_sample
```

`python main.py --demo-data ...` generates a different built-in synthetic dataset.

## 3. Run on the public PeMS04 dataset

The code accepts:

- PeMS04 `.npz` with an array named `data`, shaped `[timestamps, sensors, features]`; feature 0 (flow) is used.
- CSV with one numeric column per sensor and one row per timestamp; non-numeric columns (e.g. timestamps) are ignored.

Get the data (either way), saving it as `data/PEMS04.npz`:

- Run `python get_data.py` (downloads a Hugging Face mirror and prints the array shapes), or
- Download it manually from the ASTGCN project or the mirror page linked in `get_data.py`.

The file is not stored in this repository. Follow the dataset's terms, and cite the original PeMS/ASTGCN source.
Expected structure: key `data`, shape `(16992, 307, 3)`. If yours differs, inspect it and adjust
`load_traffic()` — do not silently treat another feature as traffic flow.

```bash
python main.py --data data/PEMS04.npz --rounds 10 --central-epochs 10 --local-epochs 1 --clients 5 --output results
```

The full dataset has hundreds of sensors and many windows, so a CPU run can be slow. For a quick
trial, temporarily restrict the sensors in `main()` right after loading, e.g. `raw = raw[:, :30]`.

## 4. Results

Written to the folder given by `--output`:

- `summary_by_seed.csv`: one row per method, partition and seed
- `summary_mean_std.csv`: mean and standard deviation across seeds
- `convergence_curves.csv`: validation MAE and communication bytes per round/epoch
- `run_config.json`: parameters and data-provenance flags (including whether data was synthetic)

The program prints a text summary and saves CSV/JSON files; it does not draw charts.

## 5. Communication-cost definition

For each FedAvg round:

`bytes_per_round = 2 * number_of_clients * number_of_parameters * 4`

This assumes float32 weights sent once down and once up per client. It excludes TLS, headers,
retries, and other protocol overhead. Centralized training is assigned zero *federated update*
bytes; that does not mean moving raw data over a network is free.

## 6. Methodological notes

- Windows are built independently inside each chronological split, so no training window crosses into validation/test time.
- Normalization is fitted only on the first 60% of timestamps.
- IID: training windows are shuffled randomly across clients.
- Non-IID: each client gets a disjoint block of sensors (contiguous sensor indices). This is a simple reproducible heterogeneity simulation, not a claim that sensor IDs form geographic clusters.
- Test metrics are in original traffic-count units.
- Not implemented: differential privacy and secure aggregation. Model updates can leak information in principle; do not claim formal privacy guarantees.
- Weights are float32. If float16 transmission is added later, the byte counter must match the real serialization.

## 7. Pseudocode

See `pseudocode.txt`.

## Repository structure

```text
traffic-federated-simulator/
├── main.py
├── get_data.py
├── requirements.txt
├── pseudocode.txt
├── README.md
├── .gitignore
├── data/
│   └── sample_traffic.csv   # synthetic; PEMS04.npz is git-ignored
└── results*/                # generated by runs; git-ignored
```
