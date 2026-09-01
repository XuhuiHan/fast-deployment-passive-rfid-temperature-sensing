from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINED_MODEL = PROJECT_ROOT / "outputs" / "offline_training" / "formula_params_studydata_8methods_schemeB.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper pipeline from versioned input data.")
    parser.add_argument("--trials", type=int, default=30, help="Hyperparameter trials per formula (paper: 30).")
    parser.add_argument("--adapt-iters", type=int, default=80, help="Adaptation iterations during tuning (paper: 80).")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="Parallel formula processes.")
    parser.add_argument("--skip-training", action="store_true", help="Reuse a model already generated under outputs/.")
    parser.add_argument("--skip-window-selection", action="store_true", help="Skip the independent 30-tag window-length selection experiment.")
    parser.add_argument("--skip-sliding-window", action="store_true", help="Reuse existing sliding-window outputs.")
    parser.add_argument("--skip-input-check", action="store_true", help="Skip SHA-256 verification of input data.")
    return parser.parse_args()


def run_step(label: str, command: list[str], env: dict[str, str], cwd: Path = PROJECT_ROOT) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")

    run_step("Verify Python environment", [sys.executable, "scripts/check_environment.py"], env)
    if not args.skip_input_check:
        run_step("Verify versioned input data", [sys.executable, "scripts/check_inputs.py"], env)

    if not args.skip_window_selection:
        run_step(
            "Evaluate window lengths 1-10 on the independent 30-tag selection set",
            [sys.executable, "scripts/00_window_length_selection.py"],
            env,
        )

    if not args.skip_training:
        run_step(
            "Train all eight candidate formulas",
            [
                sys.executable,
                "scripts/train_all_models.py",
                "--jobs",
                str(args.jobs),
                "--trials",
                str(args.trials),
                "--adapt-iters",
                str(args.adapt_iters),
            ],
            env,
        )
    elif not TRAINED_MODEL.exists():
        raise FileNotFoundError(
            f"--skip-training was requested, but {TRAINED_MODEL} does not exist."
        )

    if not args.skip_sliding_window:
        run_step(
            "Recompute sliding-window fusion results",
            [sys.executable, "scripts/01_sliding_window_vs_thermotag.py"],
            env,
        )

    run_step(
        "Evaluate regularized one-point calibration",
        [sys.executable, "scripts/02_one_point_calibration_regularized.py"],
        env,
    )
    run_step(
        "Compare one-point calibration with the ThermoTag baseline",
        [sys.executable, "scripts/03_compare_one_point_calibration_with_thermotag.py"],
        env,
    )

    print("\nFrom-scratch pipeline completed successfully.")
    print(f"Models and numerical results: {PROJECT_ROOT / 'outputs'}")


if __name__ == "__main__":
    main()
