# Adversarial robustness of process fault classifiers: analyzer-calibrated budgets and feasibility bounds (TEP)

Code for the manuscript *Adversarial Robustness of Process Fault Classifiers: Analyzer-Calibrated Budgets and Feasibility Bounds on Tennessee Eastman Composition Data* (Imran, Zaidi, Tran).

## Layout
- `core/` – data generation (Braatz TEP simulator, commit 9a6c8e5), ILR transform, classifier, PGD attack, training, earlier hyperparameter search (qNEHVI).
- `analysis/` – scripts behind every table, figure and number in the paper (run from inside `analysis/`).

## Setup
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/jkitchin/tennessee-eastman-profbraatz tep_sim
cd tep_sim && git checkout 9a6c8e5 && pip install -e . && cd ..
```

## Data
Generate the composition datasets with `core/generate_multirun_tep.py` (30 runs per class) and `core/generate_holdout_test.py` (10 held-out runs per class), place the CSVs in `data/raw_dataset/`, or set `TEP_DATA` to the folder that contains `raw_dataset/`.
The 52-variable runs used by `fullvar_published_budgets.py`/`fullvar_layout_ibp.py layout` are produced by the `tep-sim` commands described in `fullvar_published_budgets.py` (written to `analysis/full/`).

## Scripts (run in order; each writes `results_*.csv` in `analysis/`)
| Script | Paper content |
|---|---|
| data_sets.py, calibration_bounds.py, detection_cnn.py, detection_eval*.py | analysis set, calibration (Table 1), feasibility bounds (Table 2, Fig. 1), selection-budget collapse (Tables 3–4), feasible-budget training (Tables 5–6), detection (Table 7), windowed CNN |
| calibration_check_bootstrap.py | calibration cross-check, two-way bootstrap intervals |
| sensitivity_window_threshold.py | transition-window and threshold sensitivity |
| noise_halfdata.py | Gaussian noise, half training data |
| lambda_sweep.py | adversarial-loss weight λ |
| dai_comparison.py | comparison with the hypergraph LP of Dai et al. |
| restricted_attacks.py | restricted (product-block) detection attacks |
| sd_calibration_density_rotation.py | SD-unit calibration, bound vs sample density, rotated ILR bases |
| noise_threat_drift_seeds.py | noise-calibrated threat set, its detection, drift-shaped threat, ten seeds |
| published_budgets_seq_cnn.py | published budgets in SD units, sequence-level CNN attack |
| product_fixed.py, product_fixed_apgd.py | product-fixed threat set (PGD, APGD), feed/purge-only noise |
| fullvar_published_budgets.py | published budgets in the full 52-variable space |
| worstclass_training_timing.py | worst-class (group DRO) training, bound timing |
| fullvar_layout_ibp.py | 52-variable restricted attacks, certified (IBP) training |

Seeds, budgets and settings are fixed in each script; see its docstring.

## License

Released under the MIT License; see `LICENSE`. Anyone may use, modify and redistribute the code, provided the copyright notice is kept. If you use it in research, please cite the paper.
