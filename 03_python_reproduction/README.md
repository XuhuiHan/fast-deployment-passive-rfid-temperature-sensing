# Python From-Scratch Reproduction

This folder contains versioned experimental inputs and the numerical method code. It contains no pretrained model JSON and no precomputed result table.

## Numerical entry points

- `scripts/train_all_models.py`: train all eight candidate formulas.
- `scripts/01_sliding_window_vs_thermotag.py`: evaluate sliding-window fusion.
- `scripts/02_one_point_calibration_regularized.py`: evaluate one-point calibration and construct the ThermoTag-style baseline by averaging the separately fitted `a`, `b`, and `c` parameters of learning tags C250-C282.
- `scripts/03_compare_one_point_calibration_with_thermotag.py`: export the matched comparison with the ThermoTag baseline.
- `scripts/04_calibration_constraint_ablation.py`: retrain and evaluate the four calibration-constraint ablations reported in the paper.

These scripts generate models and numerical outputs only. Manuscript plotting is intentionally outside this reproduction package.

## Run

```powershell
python -m pip install -r requirements.txt
python scripts/check_inputs.py
python run_all.py --jobs 8 --trials 30 --adapt-iters 80
```

`run_all.py` trains a model under `outputs/offline_training/` and writes evaluation tables under `outputs/results/`.

After `run_all.py` completes, run the ablations from this directory:

```powershell
python scripts/04_calibration_constraint_ablation.py --jobs 8 --trials 30 --adapt-iters 80
```

The ablation outputs are generated under `outputs/ablation/`. No model or result archive is versioned in the repository.

The inputs under `data/` are never modified. Verify their SHA-256 values with the repository-level `input_manifest.sha256` before reproduction.
