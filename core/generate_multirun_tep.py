"""generate_multirun_tep.py

Generates the multi-run Tennessee Eastman dataset by running the
Downs-Vogel / Braatz simulator (jkitchin/tennessee-eastman-profbraatz,
`tep-sim` CLI) with a different random seed per run, and stores the
composition analyzer blocks (reactor feed, purge, product).

Classes:
  0 : normal operation (no fault)
  1 : IDV(1) A/C feed ratio, B composition constant (step)
  2 : IDV(2) B composition, A/C ratio constant (step)
  3 : IDV(8) A, B, C feed composition (random variation)

In fault runs the disturbance starts at FAULT_TIME_HR; earlier samples
are labelled 0. Separate seeds per run allow run-level train/validation
splits.
"""

import subprocess
import sys
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path("tep_sim_runs/multirun")
RUN_DIR.mkdir(parents=True, exist_ok=True)

DURATION_HR = 8.0
FAULT_TIME_HR = 2.0
RECORD_INTERVAL_SEC = 180  # 3 minutes, matches standard TEP sampling
N_RUNS_PER_CLASS = 30

FAULT_IDS = {1: 1, 2: 2, 3: 8}  # our label -> TEP IDV number

# Column indices (0-indexed) within XMEAS block for the three composition
# sub-blocks, matching the header order printed by tep-sim:
# ... Reactor Feed Component A-F (6), Purge Gas Component A-H (8),
#     Product Component D-H (5) ...
REACTOR_FEED_COLS = slice(22, 28)
PURGE_GAS_COLS = slice(28, 36)
PRODUCT_COLS = slice(36, 41)


def _find_tep_sim() -> str:
    """
    Locate the tep-sim executable reliably across platforms and launch
    methods. On Windows in particular, running a script via an IDE's
    "Run" button does not always add the virtual environment's Scripts
    folder to PATH the way an activated terminal does, so a plain
    subprocess.run(["tep-sim", ...]) can fail to find it even though
    `pip install -e .` succeeded and `tep-sim` works fine when typed
    directly into the terminal.
    """
    exe = shutil.which("tep-sim")
    if exe:
        return exe

    # Fall back to looking next to the current Python interpreter --
    # this is where pip installs console-script launchers for whichever
    # environment is actually running this code.
    scripts_dir = Path(sys.executable).parent / ("Scripts" if sys.platform == "win32" else "")
    if sys.platform != "win32":
        scripts_dir = Path(sys.executable).parent
    candidate_names = ["tep-sim.exe", "tep-sim"] if sys.platform == "win32" else ["tep-sim"]
    for name in candidate_names:
        candidate = scripts_dir / name
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not find the 'tep-sim' command anywhere. This usually means "
        "'pip install -e .' was run inside the tennessee-eastman-profbraatz "
        "folder using a DIFFERENT Python interpreter than the one running "
        "this script. In PyCharm's Terminal tab (which uses your project's "
        "venv), re-run: cd tep_sim && pip install -e . && cd .. -- then try "
        "again. If it still fails, run 'where tep-sim' (Windows) or "
        "'which tep-sim' (Mac/Linux) in the same Terminal tab and check "
        "the path it prints exists."
    )


_TEP_SIM_EXE = None


def run_sim(seed: int, fault_id: int | None, out_path: Path):
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
    # First column is Time(s); skip header row, skip time column.
    data = np.loadtxt(path, skiprows=1)
    return data[:, 1:]  # XMEAS(1..41) + XMV(1..12) roughly, order per header


def main():
    t_start = time.time()
    records = []  # (label, run_id, t, reactor6, purge8, product5)

    def process(label, fault_id, seed_offset):
        for i in range(N_RUNS_PER_CLASS):
            seed = seed_offset + i
            out_path = RUN_DIR / f"label{label}_seed{seed}.dat"
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
                    "label": lab,
                    "run_id": f"label{label}_seed{seed}",
                    "t": t,
                    "reactor": reactor[t].copy(),
                    "purge": purge[t].copy(),
                    "product": product[t].copy(),
                })
            print(f"  seed {seed}: {data.shape[0]} samples "
                  f"(elapsed {time.time()-t_start:.0f}s total)")

    print("Generating normal-operation runs...")
    process(0, None, seed_offset=1000)

    for label, fault_id in FAULT_IDS.items():
        print(f"Generating label {label} (IDV({fault_id})) runs...")
        process(label, fault_id, seed_offset=2000 * label)

    print(f"Done generating. Total elapsed: {time.time()-t_start:.0f}s")
    return records


if __name__ == "__main__":
    records = main()

    import sys
    from coda import ILRTransform

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
    out_csv = "tep_multirun_composition_data.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows, {df['run_id'].nunique()} independent runs, to {out_csv}")
    print(df.groupby("label")["run_id"].nunique())
