# Fast-Deployable Passive RFID Temperature Sensing via Robust Persistence-Time Fusion and One-Point Calibration

This repository contains the experimental code and versioned input data for the paper **Fast-Deployable Passive RFID Temperature Sensing via Robust Persistence-Time Fusion and One-Point Calibration**. It supports from-scratch reproduction and contains no pretrained model JSON, expected-result archive, generated numerical result, log, or manuscript image.

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

The command verifies input hashes, trains all eight candidate formulas, evaluates sliding-window fusion and one-point calibration, and compares the proposed method with the ThermoTag baseline. It generates models and numerical result tables only; manuscript plotting is intentionally outside this public reproduction package.

Runtime artifacts are created under:

```text
01_data_acquisition/output/
02_matlab_processing/output/
03_python_reproduction/outputs/
```

Fresh random search can select slightly different hyperparameters on a different platform. The public package reproduces the complete method rather than distributing the archived paper checkpoint.

The Impinj Octane SDK JAR is not redistributed. See `01_data_acquisition/lib/README.md`.
