from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import offline_tuning_core as core  # noqa: E402


METHOD_ORDER = ["Paper", "M1", "M2", "M2pro", "EXP0", "EXP1", "EXP2", "EXP3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one deterministic Scheme-B candidate formula.")
    parser.add_argument("method", choices=METHOD_ORDER)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--adapt-iters", type=int, default=80, help="Online adaptation iterations during tuning (paper: 80).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "offline_training" / "parts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    core.RAW_DATA_PATH = str(PROJECT_ROOT / "data" / "training" / "studydata.txt")
    core.T_EVAL_MIN = 20.0
    core.T_EVAL_MAX = 80.0
    core.N_TRIALS_PER_METHOD = args.trials
    core.ADAPT_ITERS = args.adapt_iters
    torch.set_num_threads(1)
    torch.manual_seed(core.SEED)
    np.random.seed(core.SEED)

    tags = core.load_tags_from_txt(core.RAW_DATA_PATH)
    methods = {method.name: method for method in core.build_methods()}
    method = methods[args.method]
    pack = core.tune_one_method(method, tags, torch.device("cpu"))
    result = {
        "scheme": "B",
        "raw_data_path": "data/training/studydata.txt",
        "seed": core.SEED,
        "n_trials_per_method": args.trials,
        "device": "cpu",
        "TMIN_NORM": core.TMIN_NORM,
        "TMAX_NORM": core.TMAX_NORM,
        "T_EVAL_MIN": core.T_EVAL_MIN,
        "T_EVAL_MAX": core.T_EVAL_MAX,
        "methods": {args.method: pack},
    }
    destination = args.output_dir / f"results_schemeB_{args.method}.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
