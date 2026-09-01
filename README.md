# Robust and Fast-Deployment Passive RFID Temperature Sensing

This repository contains the experimental code and versioned input data for the paper **Robust and Fast-Deployment Passive RFID Temperature Sensing**. It supports from-scratch reproduction and contains no pretrained model JSON, expected-result archive, generated numerical result, log, or manuscript image.

## Repository layout

```text
01_data_acquisition/       one minimal Impinj Java collector
02_matlab_processing/      one retained RFID/PT100 processing script
03_python_reproduction/    offline training, fusion, calibration, and evaluation
```

The versioned paper inputs are under `03_python_reproduction/data/`. New Java and MATLAB outputs are created inside their respective stage folders and are ignored by Git; there are no empty top-level data placeholders.

## Full reproduction

The tested Python environment is documented in `03_python_reproduction/requirements.txt`.

```powershell
cd 03_python_reproduction
python -m pip install -r requirements.txt
python run_all.py --jobs 8 --trials 30 --adapt-iters 80
```

The command verifies input hashes, evaluates window lengths 1-10 on the independent C101-C130 selection set, trains all eight candidate formulas, evaluates sliding-window fusion and one-point calibration, and compares the proposed method with the ThermoTag baseline. It generates models and numerical result tables only; manuscript plotting is intentionally outside this reproduction package.

After the full-method outputs have been generated, run the four calibration-constraint ablations with:

```powershell
python scripts/04_calibration_constraint_ablation.py --jobs 8 --trials 30 --adapt-iters 80
```

The ablation command reuses the generated full-method validation results, retrains the four ablated configurations, and writes its models, logs, validation results, and summaries under `03_python_reproduction/outputs/ablation/`.

Runtime artifacts are created under:

```text
01_data_acquisition/output/
02_matlab_processing/output/
03_python_reproduction/outputs/
```

Fresh random search can select slightly different hyperparameters on a different platform. This package reproduces the complete method rather than distributing the archived paper checkpoint.

The Impinj Octane SDK JAR is not redistributed. See `01_data_acquisition/lib/README.md`.
