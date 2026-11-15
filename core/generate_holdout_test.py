"""generate_holdout_test.py

Generates the held-out test set: 10 additional simulation runs per class
with seeds that do not overlap those of the main dataset (offsets
1000-1029, 2000-2029, 4000-4029, 6000-6029; see generate_multirun_tep.py).
The runs are written to a separate file, tep_holdout_test_data.csv, and
are used only for final held-out evaluation.

Usage:
    python generate_holdout_test.py
"""

import subprocess
import sys
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from coda import ILRTransform

RUN_DIR = Path("tep_sim_runs/holdout")
RUN_DIR.mkdir(parents=True, exist_ok=True)

DURATION_HR = 8.0
FAULT_TIME_HR = 2.0
RECORD_INTERVAL_SEC = 180
N_RUNS_PER_CLASS = 10  # smaller than the 30-runs-per-class training set --
                         # this data is only used once, for final reporting,
                         # not for any search, so it does not need to match
                         # the training set's scale.

FAULT_IDS = {1: 1, 2: 2, 3: 8}

REACTOR_FEED_COLS = slice(22, 28)
PURGE_GAS_COLS = slice(28, 36)
PRODUCT_COLS = slice(36, 41)

# Seed offsets chosen to be far outside every range used by
# generate_multirun_tep.py (1000-1029, 2000-2029, 4000-4029, 6000-6029),
# so there is zero possibility of accidentally regenerating (and thus
# reusing) a simulation run already seen during search/selection.
HOLDOUT_SEED_OFFSETS = {0: 90000, 1: 91000, 2: 92000, 3: 93000}


def _find_tep_sim() -> str:
    exe = shutil.which("tep-sim")
    if exe:
        return exe
    scripts_dir = Path(sys.executable).parent / ("Scripts" if sys.platform == "win32" else "")
    if sys.platform != "win32":
        scripts_dir = Path(sys.executable).parent
    candidate_names = ["tep-sim.exe", "tep-sim"] if sys.platform == "win32" else ["tep-sim"]
    for name in candidate_names:
        candidate = scripts_dir / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "Could not find 'tep-sim'. Run 'cd tep_sim && pip install -e . && cd ..' "
        "in this same terminal (the one showing (.venv))."
    )


_TEP_SIM_EXE = None


def run_sim(seed: int, fault_id, out_path: Path):
    global _TEP_SIM_EXE
    if _TEP_SIM_EXE is None:
        _TEP_SIM_EXE = _find_tep_sim()
    cmd = [
        _TEP_SIM_EXE, "--duration", str(DURATION_HR),
        "--seed", str(seed),
        "--record-interval", str(RECORD_INTERVAL_SEC),
        "--output", str(out_path),
        "--quiet",
    ]
    if fault_id is not None:
        cmd += ["--faults", str(fault_id), "--fault-times", str(FAULT_TIME_HR)]
    subprocess.run(cmd, check=True)


def load_run(path: Path) -> np.ndarray:
    data = np.loadtxt(path, skiprows=1)
    return data[:, 1:]


def main():
    t_start = time.time()
    records = []

    def process(label, fault_id, seed_offset):
        for i in range(N_RUNS_PER_CLASS):
            seed = seed_offset + i
            out_path = RUN_DIR / f"holdout_label{label}_seed{seed}.dat"
            if not out_path.exists():
                run_sim(seed, fault_id, out_path)
            data = load_run(out_path)
            reactor = data[:, REACTOR_FEED_COLS]
            purge = data[:, PURGE_GAS_COLS]
            product = data[:, PRODUCT_COLS]
            fault_intro_sample = int(round(FAULT_TIME_HR * 3600 / RECORD_INTERVAL_SEC)) if fault_id else None
            for t in range(data.shape[0]):
                lab = label
                if fault_id is not None and t < fault_intro_sample:
                    lab = 0
                records.append({
                    "label": lab, "run_id": f"holdout_label{label}_seed{seed}", "t": t,
                    "reactor": reactor[t].copy(), "purge": purge[t].copy(), "product": product[t].copy(),
                })
            print(f"  seed {seed}: {data.shape[0]} samples (elapsed {time.time()-t_start:.0f}s total)")

    print("Generating HELD-OUT normal-operation runs (never used in any search)...")
    process(0, None, seed_offset=HOLDOUT_SEED_OFFSETS[0])
    for label, fault_id in FAULT_IDS.items():
        print(f"Generating HELD-OUT label {label} (IDV({fault_id})) runs...")
        process(label, fault_id, seed_offset=HOLDOUT_SEED_OFFSETS[label])

    print(f"Done generating held-out test set. Total elapsed: {time.time()-t_start:.0f}s")

    reactor_ilr = ILRTransform(6)
    purge_ilr = ILRTransform(8)
    product_ilr = ILRTransform(5)

    def close(x):
        x = np.clip(x, 1e-9, None)
        return x / x.sum()

    rows = []
    for r in records:
        z_reactor = reactor_ilr.forward(close(r["reactor"])[None, :])[0]
        z_purge = purge_ilr.forward(close(r["purge"])[None, :])[0]
        z_product = product_ilr.forward(close(r["product"])[None, :])[0]
        z = np.concatenate([z_reactor, z_purge, z_product])
        rows.append(list(z) + [r["label"], r["run_id"], r["t"]])

    n_z = len(rows[0]) - 3
    cols = [f"z{i+1}" for i in range(n_z)] + ["label", "run_id", "t"]
    df = pd.DataFrame(rows, columns=cols)
    out_csv = "tep_holdout_test_data.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows, {df['run_id'].nunique()} independent HELD-OUT runs, to {out_csv}")
    print(df.groupby("label")["run_id"].nunique())

    # Sanity check: confirm zero overlap with the training dataset's run_ids,
    # if that file exists in the current directory.
    train_path = Path("tep_multirun_composition_data.csv")
    if train_path.exists():
        train_runs = set(pd.read_csv(train_path)["run_id"].unique())
        holdout_runs = set(df["run_id"].unique())
        overlap = train_runs & holdout_runs
        if overlap:
            print(f"WARNING: {len(overlap)} run_ids overlap with the training set! {overlap}")
        else:
            print("Confirmed: zero run_id overlap with tep_multirun_composition_data.csv.")


if __name__ == "__main__":
    main()
