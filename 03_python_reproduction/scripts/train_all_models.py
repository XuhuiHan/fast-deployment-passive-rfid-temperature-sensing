from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ["Paper", "M1", "M2", "M2pro", "EXP0", "EXP1", "EXP2", "EXP3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all eight Scheme-B formulas with fixed per-method seeds.")
    parser.add_argument("--trials", type=int, default=30, help="Hyperparameter trials per formula (paper: 30).")
    parser.add_argument("--adapt-iters", type=int, default=80, help="Adaptation iterations during tuning (paper: 80).")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="Parallel formula processes.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "offline_training",
        help="Destination for generated model files and logs.",
    )
    return parser.parse_args()


def run_method(method: str, trials: int, adapt_iters: int, parts_dir: Path, logs_dir: Path) -> str:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_one_formula.py"),
        method,
        "--trials",
        str(trials),
        "--adapt-iters",
        str(adapt_iters),
        "--output-dir",
        str(parts_dir),
    ]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    log_path = logs_dir / f"{method}.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return method


def export_formula_params(merged: dict, destination: Path) -> None:
    methods = {}
    for name, pack in merged["methods"].items():
        methods[name] = {
            "available": bool(pack.get("theta0")),
            "theta0": pack.get("theta0", {}),
            "hp": pack.get("hp", {}),
            "prior_log_s": pack.get("prior_log_s", {}),
            "kept_epcs": pack.get("kept_epcs", []),
        }
    payload = {
        "scheme": "B",
        "dataset_name": "studydata",
        "raw_data_path": "data/training/studydata.txt",
        "TMIN_NORM": merged["TMIN_NORM"],
        "TMAX_NORM": merged["TMAX_NORM"],
        "T_EVAL_MIN": merged["T_EVAL_MIN"],
        "T_EVAL_MAX": merged["T_EVAL_MAX"],
        "methods": methods,
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    parts_dir = output_dir / "parts"
    logs_dir = output_dir / "logs"
    parts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training eight formulas with jobs={args.jobs}, trials={args.trials}.")
    print("The paper configuration is compute intensive; progress is written to outputs/offline_training/logs/.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(run_method, method, args.trials, args.adapt_iters, parts_dir, logs_dir): method
            for method in METHOD_ORDER
        }
        for future in concurrent.futures.as_completed(futures):
            method = future.result()
            print(f"Completed {method}", flush=True)

    merged: dict | None = None
    merged_methods = {}
    for method in METHOD_ORDER:
        part = json.loads((parts_dir / f"results_schemeB_{method}.json").read_text(encoding="utf-8"))
        if merged is None:
            merged = {key: value for key, value in part.items() if key != "methods"}
        merged_methods[method] = part["methods"][method]
    assert merged is not None
    merged["methods"] = merged_methods

    result_path = output_dir / "results_schemeB.json"
    params_path = output_dir / "formula_params_studydata_8methods_schemeB.json"
    result_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    export_formula_params(merged, params_path)
    print(f"Saved {result_path}")
    print(f"Saved {params_path}")
    print("Offline training completed. The generated formula-parameter JSON is ready for one-point evaluation.")


if __name__ == "__main__":
    main()
