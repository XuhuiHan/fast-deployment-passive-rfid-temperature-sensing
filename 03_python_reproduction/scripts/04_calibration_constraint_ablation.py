"""Train and evaluate four calibration-constraint ablation variants.

The local reproduction pipeline is imported without modification. Existing
full-method results are reused; this script computes only:

1. full method without the training-consistency constraint;
2. full method without the class-model-proximity constraint;
3. full method without the scale-prior constraint; and
4. full method without formula-specific regularization.

For variants that change a tuned objective term, all eight formulas are tuned
again on the 33-tag learning batch. Formulas are trained concurrently in
separate CPU processes, matching the established ``train_all_models.py``
execution strategy. Independent validation tags are never used for tuning.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PIPELINE_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = DEFAULT_PIPELINE_ROOT / "outputs" / "ablation"

FORMULA_ORDER = [f"f{i}" for i in range(1, 9)]
METHOD_ORDER = ["Paper", "M1", "M2", "M2pro", "EXP0", "EXP1", "EXP2", "EXP3"]
METHOD_TO_FORMULA = dict(zip(METHOD_ORDER, FORMULA_ORDER))


@dataclass(frozen=True)
class AblationVariant:
    key: str
    label: str
    stage: str
    zero_hyperparameters: tuple[str, ...] = ()


VARIANTS = (
    AblationVariant(
        key="without_training_consistency",
        label="w/o training consistency",
        stage="online",
        zero_hyperparameters=("alpha_train",),
    ),
    AblationVariant(
        key="without_class_proximity",
        label="w/o class proximity",
        stage="online",
        zero_hyperparameters=("alpha_theta",),
    ),
    AblationVariant(
        key="without_scale_prior",
        label="w/o scale prior",
        stage="offline tuning + online objective",
        zero_hyperparameters=("alpha_sprior",),
    ),
    AblationVariant(
        key="without_formula_regularization",
        label="w/o formula regularization",
        stage="offline class learning + online objective",
        zero_hyperparameters=("alpha_p", "alpha_smooth"),
    ),
)
VARIANT_BY_KEY = {variant.key: variant for variant in VARIANTS}
RESULT_LABELS = {
    "full": "Full method",
    **{variant.key: variant.label for variant in VARIANTS},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain and evaluate four calibration-constraint ablation variants."
    )
    parser.add_argument(
        "--stage",
        choices=("all", "train", "evaluate", "summarize"),
        default="all",
        help="Pipeline stage to run (default: all).",
    )
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=DEFAULT_PIPELINE_ROOT,
        help="Reproduction project root. Defaults to the parent of this scripts directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Ablation output directory.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_BY_KEY),
        default=list(VARIANT_BY_KEY),
        help="Subset of ablation variants to process.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Offline hyperparameter trials per formula (paper setting: 30).",
    )
    parser.add_argument(
        "--adapt-iters",
        type=int,
        default=80,
        help="Online shape-adaptation iterations (paper setting: 80).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of formulas trained concurrently (default: up to 8).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute outputs that already exist.",
    )

    # Internal worker mode. Users do not need these options.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-variant", choices=tuple(VARIANT_BY_KEY), help=argparse.SUPPRESS)
    parser.add_argument("--worker-method", choices=METHOD_ORDER, help=argparse.SUPPRESS)
    parser.add_argument("--worker-parts-dir", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def ensure_pipeline(pipeline_root: Path) -> None:
    required = [
        pipeline_root / "src" / "offline_tuning_core.py",
        pipeline_root / "src" / "fastreg_core.py",
        pipeline_root / "scripts" / "02_one_point_calibration_regularized.py",
        pipeline_root / "scripts" / "01_sliding_window_vs_thermotag.py",
        pipeline_root / "data" / "training" / "studydata.txt",
        pipeline_root
        / "outputs"
        / "results"
        / "fast_registration_precise_alignment"
        / "C201_C301_C350_combined"
        / "all_tags_errors_points.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing paper-pipeline files:\n" + "\n".join(map(str, missing)))


def load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def variant_model_dir(output_dir: Path, variant: AblationVariant) -> Path:
    return output_dir / "models" / variant.key


def variant_params_path(output_dir: Path, variant: AblationVariant) -> Path:
    return variant_model_dir(output_dir, variant) / "formula_params_studydata_8methods_schemeB.json"


def variant_validation_dir(output_dir: Path, variant: AblationVariant) -> Path:
    return output_dir / "validation" / variant.key


def remove_owned_directory(path: Path, output_root: Path) -> None:
    root = output_root.resolve()
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"Refusing to delete outside the ablation output root: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def patch_hyperparameter_sampler(core, variant: AblationVariant) -> None:
    original_sampler = core.sample_hp

    def sample_hp_for_variant(rs, method_name: str) -> dict[str, Any]:
        hp = original_sampler(rs, method_name)
        # Sampling first and then zeroing preserves the original random-number
        # stream and formula-specific deterministic seeds.
        for key in variant.zero_hyperparameters:
            hp[key] = 0.0
        return hp

    core.sample_hp = sample_hp_for_variant


def run_training_worker(args: argparse.Namespace) -> None:
    if not args.worker_variant or not args.worker_method or not args.worker_parts_dir:
        raise ValueError("Worker mode requires variant, method, and parts directory.")

    variant = VARIANT_BY_KEY[args.worker_variant]

    pipeline_root = args.pipeline_root.resolve()
    import torch  # Imported only inside each training process.

    core = load_module(
        pipeline_root / "src" / "offline_tuning_core.py",
        f"offline_tuning_ablation_{variant.key}_{args.worker_method}",
    )
    core.RAW_DATA_PATH = str(pipeline_root / "data" / "training" / "studydata.txt")
    core.T_EVAL_MIN = 20.0
    core.T_EVAL_MAX = 80.0
    core.N_TRIALS_PER_METHOD = int(args.trials)
    core.ADAPT_ITERS = int(args.adapt_iters)
    patch_hyperparameter_sampler(core, variant)

    torch.set_num_threads(1)
    torch.manual_seed(core.SEED)
    np.random.seed(core.SEED)

    tags = core.load_tags_from_txt(core.RAW_DATA_PATH)
    methods = {method.name: method for method in core.build_methods()}
    method = methods[args.worker_method]
    pack = core.tune_one_method(method, tags, torch.device("cpu"))
    pack["ablation"] = {
        "variant": variant.key,
        "label": variant.label,
        "zero_hyperparameters": list(variant.zero_hyperparameters),
    }

    payload = {
        "scheme": "B",
        "raw_data_path": "data/training/studydata.txt",
        "seed": core.SEED,
        "n_trials_per_method": int(args.trials),
        "adapt_iters": int(args.adapt_iters),
        "device": "cpu",
        "TMIN_NORM": core.TMIN_NORM,
        "TMAX_NORM": core.TMAX_NORM,
        "T_EVAL_MIN": core.T_EVAL_MIN,
        "T_EVAL_MAX": core.T_EVAL_MAX,
        "ablation_variant": asdict(variant),
        "methods": {args.worker_method: pack},
    }
    args.worker_parts_dir.mkdir(parents=True, exist_ok=True)
    destination = args.worker_parts_dir / f"results_schemeB_{args.worker_method}.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {destination}")


def launch_worker(
    args: argparse.Namespace,
    variant: AblationVariant,
    method: str,
    parts_dir: Path,
    logs_dir: Path,
) -> str:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-variant",
        variant.key,
        "--worker-method",
        method,
        "--worker-parts-dir",
        str(parts_dir),
        "--pipeline-root",
        str(args.pipeline_root.resolve()),
        "--trials",
        str(args.trials),
        "--adapt-iters",
        str(args.adapt_iters),
    ]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    log_path = logs_dir / f"{method}.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=args.pipeline_root.resolve(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"Training failed for {variant.key}/{method}. See {log_path}\n"
            + "\n".join(tail)
        )
    return method


def export_formula_params(merged: dict[str, Any], variant: AblationVariant) -> dict[str, Any]:
    methods = {}
    for name, pack in merged["methods"].items():
        methods[name] = {
            "available": bool(pack.get("theta0")),
            "theta0": pack.get("theta0", {}),
            "hp": pack.get("hp", {}),
            "prior_log_s": pack.get("prior_log_s", {}),
            "kept_epcs": pack.get("kept_epcs", []),
        }
    return {
        "scheme": "B",
        "dataset_name": "studydata",
        "raw_data_path": "data/training/studydata.txt",
        "TMIN_NORM": merged["TMIN_NORM"],
        "TMAX_NORM": merged["TMAX_NORM"],
        "T_EVAL_MIN": merged["T_EVAL_MIN"],
        "T_EVAL_MAX": merged["T_EVAL_MAX"],
        "ablation_variant": asdict(variant),
        "methods": methods,
    }


def merge_training_parts(
    variant: AblationVariant,
    parts_dir: Path,
    model_dir: Path,
) -> None:
    merged: dict[str, Any] | None = None
    merged_methods: dict[str, Any] = {}
    for method in METHOD_ORDER:
        part_path = parts_dir / f"results_schemeB_{method}.json"
        if not part_path.exists():
            raise FileNotFoundError(part_path)
        part = json.loads(part_path.read_text(encoding="utf-8"))
        if merged is None:
            merged = {key: value for key, value in part.items() if key != "methods"}
        merged_methods[method] = part["methods"][method]
    assert merged is not None
    merged["methods"] = merged_methods
    merged["ablation_variant"] = asdict(variant)

    results_path = model_dir / "results_schemeB.json"
    params_path = model_dir / "formula_params_studydata_8methods_schemeB.json"
    results_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    params_path.write_text(
        json.dumps(export_formula_params(merged, variant), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train_variant(args: argparse.Namespace, variant: AblationVariant) -> None:
    pipeline_root = args.pipeline_root.resolve()
    output_dir = args.output_dir.resolve()
    model_dir = variant_model_dir(output_dir, variant)
    params_path = variant_params_path(output_dir, variant)
    if params_path.exists() and not args.force:
        print(f"[train] Reusing completed model: {params_path}")
        return
    if args.force:
        remove_owned_directory(model_dir, output_dir)

    parts_dir = model_dir / "parts"
    logs_dir = output_dir / "logs" / variant.key
    parts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        method
        for method in METHOD_ORDER
        if args.force or not (parts_dir / f"results_schemeB_{method}.json").exists()
    ]
    if pending:
        print(
            f"[train] {variant.label}: {len(pending)} formulas, "
            f"jobs={max(1, args.jobs)}, trials={args.trials}"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
            futures = {
                executor.submit(
                    launch_worker, args, variant, method, parts_dir, logs_dir
                ): method
                for method in pending
            }
            for future in concurrent.futures.as_completed(futures):
                method = future.result()
                print(f"[train] {variant.key}: completed {method}", flush=True)
    merge_training_parts(variant, parts_dir, model_dir)
    print(f"[train] Saved model archive: {params_path}")


def configure_calibration_module(cal, pipeline_root: Path) -> None:
    cal.PROJECT_ROOT = pipeline_root
    cal.OLD_FASTREG_SCRIPT = pipeline_root / "src" / "fastreg_core.py"
    cal.TRAIN_DATA_PATH = pipeline_root / "data" / "training" / "studydata.txt"
    archived_model = (
        pipeline_root / "data" / "models" / "formula_params_studydata_8methods_schemeB.json"
    )
    generated_model = (
        pipeline_root
        / "outputs"
        / "offline_training"
        / "formula_params_studydata_8methods_schemeB.json"
    )
    cal.FORMULA_PARAMS_PATH = archived_model if archived_model.exists() else generated_model
    cal.DISCHARGE_DIR = pipeline_root / "data" / "validation" / "tags"
    cal.TEMP_CSV = pipeline_root / "data" / "temperature" / "123time_time_corrected.csv"
    cal.HELPER_SCRIPT_CANDIDATES = [
        pipeline_root / "scripts" / "01_sliding_window_vs_thermotag.py"
    ]
    cal.METHOD_ORDER = METHOD_ORDER.copy()
    cal.FORMULA_ORDER = FORMULA_ORDER.copy()


def prepare_validation_inputs(cal, helper):
    tags = cal.all_validation_tags()
    cal.TAG_CACHE = cal.build_sliding_window_cache(helper, tags)
    temp_sec_raw, temp_val_raw, temp_sec_eval, temp_val_eval, temp_shift = (
        cal.read_temperature_pair(helper)
    )
    reg_tags = cal.build_registration_points(helper, temp_sec_raw, temp_val_raw)
    raw_loader = cal.make_raw_eval_loader(helper, temp_sec_eval, temp_val_eval)
    return tags, reg_tags, raw_loader, float(temp_shift)


def evaluate_variants(args: argparse.Namespace, variants: list[AblationVariant]) -> float:
    pipeline_root = args.pipeline_root.resolve()
    output_dir = args.output_dir.resolve()
    calibration_script = pipeline_root / "scripts" / "02_one_point_calibration_regularized.py"
    cal = load_module(calibration_script, "one_point_calibration_ablation_eval")
    configure_calibration_module(cal, pipeline_root)
    cal.OUTPUT_ROOT = output_dir
    cal.OUTPUT_RESULT_ROOT = output_dir / "_shared_input"
    cal.OUTPUT_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    helper = cal.load_helper_module()
    fastreg = load_module(cal.OLD_FASTREG_SCRIPT, "fastreg_ablation_eval")
    cal.patch_fastreg_module(fastreg)
    tags, reg_tags, raw_loader, temp_shift = prepare_validation_inputs(cal, helper)
    train_tags = fastreg.load_tags_from_txt(str(cal.TRAIN_DATA_PATH))
    epc_to_tag = {str(tag["EPC"]).upper(): tag for tag in train_tags}
    epc_to_tag.update({str(tag["EPC"]): tag for tag in train_tags})
    original_adapt_iters = int(fastreg.ADAPT_ITERS)

    for variant in variants:
        params_path = variant_params_path(output_dir, variant)
        if not params_path.exists():
            raise FileNotFoundError(
                f"Missing model for {variant.key}: {params_path}. Run --stage train first."
            )
        validation_dir = variant_validation_dir(output_dir, variant)
        result_file = (
            validation_dir / "C201_C301_C350_combined" / "all_tags_errors_points.csv"
        )
        if result_file.exists() and not args.force:
            print(f"[evaluate] Reusing {result_file}")
            continue
        if args.force:
            remove_owned_directory(validation_dir, output_dir)
        validation_dir.mkdir(parents=True, exist_ok=True)

        params = json.loads(params_path.read_text(encoding="utf-8"))
        method_contexts = fastreg.prepare_method_contexts(params, epc_to_tag)
        missing = [method for method in METHOD_ORDER if method not in method_contexts]
        if missing:
            raise KeyError(f"{params_path} is missing methods: {missing}")

        cal.OUTPUT_ROOT = output_dir
        cal.OUTPUT_RESULT_ROOT = validation_dir
        cal.FORMULA_PARAMS_PATH = params_path
        fastreg.ADAPT_ITERS = int(args.adapt_iters)
        print(f"[evaluate] {variant.label}")
        cal.run_fast_registration_batched(
            fastreg,
            reg_tags,
            {method: method_contexts[method] for method in METHOD_ORDER},
            raw_loader,
        )
        if not result_file.exists():
            raise FileNotFoundError(result_file)
        (validation_dir / "ablation_run_info.json").write_text(
            json.dumps(
                {
                    "variant": asdict(variant),
                    "model_archive": params_path.relative_to(pipeline_root).as_posix(),
                    "validation_tags": len(tags),
                    "temperature_time_shift_seconds": temp_shift,
                    "adapt_iters": int(args.adapt_iters),
                    "input": "sliding-window fused persistence-time observations",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    fastreg.ADAPT_ITERS = original_adapt_iters
    return temp_shift


def normalize_trials(frame: pd.DataFrame, variant_key: str) -> pd.DataFrame:
    out = frame.copy()
    out["EPC"] = out["EPC"].astype(str).str.upper()
    out["reg_idx"] = pd.to_numeric(out["reg_idx"], errors="raise").astype(int)
    out["trial_key"] = out["EPC"] + "::" + out["reg_idx"].astype(str)
    if "formula" not in out.columns:
        out["formula"] = out["method"].map(METHOD_TO_FORMULA)
    out["variant"] = variant_key
    for column in ["MAE", "RMSE", "nPredValid"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def accepted(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["status"].astype(str).str.upper().eq("ACCEPT")
        & np.isfinite(frame["MAE"].to_numpy(dtype=float))
    )


def sample_std(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def load_all_calibration_trials(
    pipeline_root: Path,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    full_path = (
        pipeline_root
        / "outputs"
        / "results"
        / "fast_registration_precise_alignment"
        / "C201_C301_C350_combined"
        / "all_tags_errors_points.csv"
    )
    frames = {"full": normalize_trials(pd.read_csv(full_path), "full")}
    for variant in VARIANTS:
        path = (
            variant_validation_dir(output_dir, variant)
            / "C201_C301_C350_combined"
            / "all_tags_errors_points.csv"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        frames[variant.key] = normalize_trials(pd.read_csv(path), variant.key)
    return frames


def build_formula_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant_key, frame in frames.items():
        for formula in FORMULA_ORDER:
            part = frame[frame["formula"] == formula]
            accepted_part = part[accepted(part)]
            rows.append(
                {
                    "variant": variant_key,
                    "variant_label": RESULT_LABELS[variant_key],
                    "formula": formula,
                    "n_trials_total": int(len(part)),
                    "n_trials_accepted": int(len(accepted_part)),
                    "n_trials_rejected": int(len(part) - len(accepted_part)),
                    "acceptance_rate": float(len(accepted_part) / len(part)),
                    "mean_MAE_accepted_C": float(accepted_part["MAE"].mean()),
                    "std_MAE_accepted_C": sample_std(accepted_part["MAE"]),
                }
            )
    summary = pd.DataFrame(rows)
    full_mae = (
        summary[summary["variant"] == "full"]
        .set_index("formula")["mean_MAE_accepted_C"]
    )
    summary["delta_MAE_vs_full_C"] = summary.apply(
        lambda row: float(row["mean_MAE_accepted_C"] - full_mae.loc[row["formula"]]),
        axis=1,
    )
    return summary


def build_configuration_summary(formula_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant_key in ["full"] + [variant.key for variant in VARIANTS]:
        part = formula_summary[formula_summary["variant"] == variant_key]
        rows.append(
            {
                "variant": variant_key,
                "variant_label": RESULT_LABELS[variant_key],
                "eight_formula_mean_MAE_C": float(part["mean_MAE_accepted_C"].mean()),
                "accepted_tasks": int(part["n_trials_accepted"].sum()),
                "rejected_tasks": int(part["n_trials_rejected"].sum()),
                "total_tasks": int(part["n_trials_total"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_formula_gap_summary(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gap_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for variant_key, frame in frames.items():
        accepted_frame = frame.loc[accepted(frame)].copy()
        per_tag_formula = (
            accepted_frame.groupby(["EPC", "formula"], as_index=False)["MAE"].mean()
        )
        matrix = per_tag_formula.pivot(index="EPC", columns="formula", values="MAE")
        matrix = matrix.reindex(columns=FORMULA_ORDER)
        n_formulas = matrix.notna().sum(axis=1)
        for epc, row in matrix.iterrows():
            values = row.dropna()
            if len(values) < 2:
                continue
            gap_rows.append(
                {
                    "variant": variant_key,
                    "variant_label": RESULT_LABELS[variant_key],
                    "EPC": epc,
                    "n_formulas_used": int(len(values)),
                    "formula_gap_C": float(values.max() - values.min()),
                }
            )
        coverage_rows.append(
            {
                "variant": variant_key,
                "variant_label": RESULT_LABELS[variant_key],
                "tags_in_gap_cdf": int((n_formulas >= 2).sum()),
                "tags_with_all_8_formulas": int((n_formulas == 8).sum()),
            }
        )
    return pd.DataFrame(gap_rows), pd.DataFrame(coverage_rows)


def summarize_results(args: argparse.Namespace, temp_shift: float | None = None) -> None:
    pipeline_root = args.pipeline_root.resolve()
    output_dir = args.output_dir.resolve()
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if temp_shift is None:
        run_info = (
            variant_validation_dir(output_dir, VARIANTS[0])
            / "ablation_run_info.json"
        )
        if run_info.exists():
            saved_info = json.loads(run_info.read_text(encoding="utf-8"))
            saved_shift = saved_info.get("temperature_time_shift_seconds")
            temp_shift = float(saved_shift) if saved_shift is not None else None

    frames = load_all_calibration_trials(pipeline_root, output_dir)
    formula_summary = build_formula_summary(frames)
    configuration_summary = build_configuration_summary(formula_summary)
    formula_gaps, gap_coverage = build_formula_gap_summary(frames)

    formula_summary.to_csv(
        summary_dir / "ablation_formula_summary.csv", index=False, encoding="utf-8-sig"
    )
    configuration_summary.to_csv(
        summary_dir / "ablation_configuration_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    formula_gaps.to_csv(
        summary_dir / "ablation_per_tag_formula_gaps.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gap_coverage.to_csv(
        summary_dir / "ablation_formula_gap_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "created_by": Path(__file__).name,
        "project_root": ".",
        "existing_results_reused": ["Full method"],
        "recomputed_ablation_variants": [asdict(variant) for variant in VARIANTS],
        "offline_training": {
            "learning_tags": 33,
            "trials_per_formula": int(args.trials),
            "formula_processes": int(max(1, args.jobs)),
            "parallelism_note": "CPU process parallelism changes runtime only, not the method.",
        },
        "validation": {
            "tags": 90,
            "formulas": FORMULA_ORDER,
            "adapt_iters": int(args.adapt_iters),
            "temperature_time_shift_seconds": temp_shift,
            "input": "sliding-window fused persistence-time observations",
        },
        "comparison": (
            "Each formula and configuration is summarized over its own accepted "
            "calibration tasks. Rejected tasks and complete eight-formula tag "
            "coverage are reported separately."
        ),
    }
    (summary_dir / "ablation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nAblation summary:")
    print(configuration_summary.to_string(index=False))
    print(f"\nSummary written to: {summary_dir}")


def main() -> None:
    args = parse_args()
    if args.worker:
        run_training_worker(args)
        return

    args.pipeline_root = args.pipeline_root.resolve()
    args.output_dir = args.output_dir.resolve()
    ensure_pipeline(args.pipeline_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = [VARIANT_BY_KEY[key] for key in args.variants]

    if args.stage in ("all", "train"):
        for variant in selected:
            train_variant(args, variant)

    temp_shift: float | None = None
    if args.stage in ("all", "evaluate"):
        temp_shift = evaluate_variants(args, selected)

    if args.stage in ("all", "summarize"):
        if set(args.variants) != set(VARIANT_BY_KEY):
            raise ValueError("Summarization requires all four ablation variants.")
        summarize_results(args, temp_shift)


if __name__ == "__main__":
    main()
