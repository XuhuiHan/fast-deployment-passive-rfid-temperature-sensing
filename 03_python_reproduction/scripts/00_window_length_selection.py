"""Select the sliding-window length using the independent C101-C130 tags.

The analysis reuses the exact data loading, temperature alignment, filtering,
ThermoTag mapping, curve fitting, and error definitions from the manuscript's
sliding-window reproduction script. Only the fusion-window length changes.

For every window length and every tag, the ThermoTag mapping parameters are
refitted from that method-specific persistence-time sequence before inversion.
Window length 1 is therefore the unfused ThermoTag interval baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_SCRIPT = SCRIPT_DIR / "01_sliding_window_vs_thermotag.py"
INPUT_TAG_DIR = PROJECT_ROOT / "data" / "validation" / "tags"
INPUT_TEMP_CSV = PROJECT_ROOT / "data" / "temperature" / "123time_time_corrected.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results" / "window_length_selection"

WINDOW_LENGTHS = tuple(range(1, 11))
CDF_PROBABILITIES = np.linspace(0.0, 1.0, 2001)
EXPECTED_TAGS = tuple(f"C{i}" for i in range(101, 131))


def load_base_module() -> Any:
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(f"Base reproduction script not found: {BASE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("sliding_window_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import base reproduction script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()

_TEMP_SEC_RAW: np.ndarray | None = None
_TEMP_VAL_RAW: np.ndarray | None = None
_TEMP_SEC_EVAL: np.ndarray | None = None
_TEMP_VAL_EVAL: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate window lengths 1 through 10 on C101-C130."
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of tag-level worker processes (default: up to 8).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Destination directory for CSV and JSON results.",
    )
    return parser.parse_args()


def initialize_worker(
    temp_sec_raw: np.ndarray,
    temp_val_raw: np.ndarray,
    temp_sec_eval: np.ndarray,
    temp_val_eval: np.ndarray,
) -> None:
    global _TEMP_SEC_RAW, _TEMP_VAL_RAW, _TEMP_SEC_EVAL, _TEMP_VAL_EVAL
    _TEMP_SEC_RAW = temp_sec_raw
    _TEMP_VAL_RAW = temp_val_raw
    _TEMP_SEC_EVAL = temp_sec_eval
    _TEMP_VAL_EVAL = temp_val_eval


def load_raw_observations(path: Path) -> pd.DataFrame:
    """Load the same raw persistence-time sequence used by the paper script."""
    epc = path.stem
    frame = pd.read_csv(path)
    start_sec, _mid_sec, end_sec = BASE.infer_row_times(frame)
    rows: list[dict[str, Any]] = []

    for row_id, row in frame.iterrows():
        values = BASE.parse_burst_details(row.get("Burst_Details"))
        if len(values) < BASE.WINDOW_SIZE:
            continue
        values = np.asarray(values[: BASE.WINDOW_SIZE], dtype=float)
        if not BASE.valid_burst(values):
            continue

        sub_times = BASE.infer_sub_event_times(start_sec[row_id], end_sec[row_id], values)
        burst_range = float(np.max(values) - np.min(values))
        for sub_index, (ptime, event_sec) in enumerate(zip(values, sub_times), start=1):
            if np.isfinite(ptime) and np.isfinite(event_sec):
                rows.append(
                    {
                        "EPC": epc,
                        "row_id": int(row_id),
                        "sub_index": int(sub_index),
                        "event_sec": float(event_sec),
                        "persistence_time_s": float(ptime),
                        "burst_range_s": burst_range,
                    }
                )

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["event_sec", "row_id", "sub_index"])
        .reset_index(drop=True)
    )


def fuse_observations(raw: pd.DataFrame, window_length: int) -> pd.DataFrame:
    """Apply the manuscript's median-referenced weighting for one length."""
    if window_length < 1:
        raise ValueError("window_length must be at least 1")
    if raw.empty or len(raw) < window_length:
        return pd.DataFrame()

    if window_length == 1:
        out = raw.copy()
        out["median_s"] = out["persistence_time_s"]
        out["window_length"] = 1
        return out

    values = raw["persistence_time_s"].to_numpy(dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(values, window_length)
    medians = np.median(windows, axis=1)
    sigma = np.maximum(BASE.SIGMA_MIN_S, BASE.SIGMA_RATIO * medians)
    weights = np.exp(-np.abs(windows - medians[:, None]) / sigma[:, None])
    weight_sums = np.sum(weights, axis=1)
    fused = np.sum(weights * windows, axis=1) / weight_sums

    out = raw.iloc[window_length - 1 :].copy().reset_index(drop=True)
    out["persistence_time_s"] = fused
    out["median_s"] = medians
    out["burst_range_s"] = np.ptp(windows, axis=1)
    out["window_length"] = window_length
    return out


def process_tag(path_text: str) -> dict[str, Any]:
    if any(x is None for x in (_TEMP_SEC_RAW, _TEMP_VAL_RAW, _TEMP_SEC_EVAL, _TEMP_VAL_EVAL)):
        raise RuntimeError("Worker temperature arrays were not initialized.")

    path = Path(path_text)
    epc = path.stem
    raw = load_raw_observations(path)
    if raw.empty:
        return {"epc": epc, "rows": [], "arrays": {}, "failures": [{"EPC": epc, "error": "no raw observations"}]}

    rows: list[dict[str, Any]] = []
    arrays: dict[int, dict[str, np.ndarray]] = {}
    failures: list[dict[str, Any]] = []

    for length in WINDOW_LENGTHS:
        try:
            observations = fuse_observations(raw, length)
            curve_points, fit = BASE.build_curve_for_method(
                observations,
                _TEMP_SEC_RAW,
                _TEMP_VAL_RAW,
            )
            detail = BASE.evaluate_method(
                epc,
                f"L{length}",
                f"Window length {length}",
                observations,
                fit,
                _TEMP_SEC_EVAL,
                _TEMP_VAL_EVAL,
            )
            if detail.empty:
                raise RuntimeError("no valid evaluation points after alignment")

            errors = detail["absolute_error_C"].to_numpy(dtype=float)
            temperatures = detail["true_temperature_C"].to_numpy(dtype=float)
            metrics = BASE.compute_metrics(errors)
            rows.append(
                {
                    "window_length": length,
                    "EPC": epc,
                    "n_raw_observations": int(len(raw)),
                    "n_fused_observations": int(len(observations)),
                    "n_initial_outputs_lost": int(length - 1),
                    **metrics,
                    "abc_a": fit["a"],
                    "abc_b": fit["b"],
                    "abc_c": fit["c"],
                    "curve_rmse_s": fit["curve_rmse_s"],
                    "curve_r2": fit["curve_r2"],
                    "n_curve_points": fit["n_curve_points"],
                }
            )
            arrays[length] = {
                "absolute_error_C": errors,
                "true_temperature_C": temperatures,
            }
        except Exception as exc:
            failures.append(
                {
                    "EPC": epc,
                    "window_length": length,
                    "error": str(exc),
                }
            )

    return {"epc": epc, "rows": rows, "arrays": arrays, "failures": failures}


def aggregate_results(
    per_tag: pd.DataFrame,
    errors_by_length: dict[int, list[np.ndarray]],
    temperatures_by_length: dict[int, list[np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, Any]] = []
    tempbin_rows: list[dict[str, Any]] = []
    cdf_rows: list[pd.DataFrame] = []

    l1_points = None
    for length in WINDOW_LENGTHS:
        errors = np.concatenate(errors_by_length[length])
        temperatures = np.concatenate(temperatures_by_length[length])
        metrics = BASE.compute_metrics(errors)
        tag_part = per_tag[per_tag["window_length"] == length]
        if length == 1:
            l1_points = int(metrics["n_points"])

        overall_rows.append(
            {
                "window_length": length,
                "n_tags": int(tag_part["EPC"].nunique()),
                **metrics,
                "mean_tag_MAE_C": float(tag_part["mean_absolute_error_C"].mean()),
                "std_tag_MAE_C": float(tag_part["mean_absolute_error_C"].std(ddof=1)),
                "mean_tag_RMSE_C": float(tag_part["root_mean_square_error_C"].mean()),
                "mean_tag_P95_C": float(tag_part["p95_absolute_error_C"].mean()),
                "mean_tag_max_absolute_error_C": float(tag_part["max_absolute_error_C"].mean()),
                "median_tag_max_absolute_error_C": float(tag_part["max_absolute_error_C"].median()),
                "mean_curve_r2": float(tag_part["curve_r2"].mean()),
                "mean_curve_points": float(tag_part["n_curve_points"].mean()),
                "retained_output_fraction_vs_L1": (
                    float(metrics["n_points"]) / float(l1_points) if l1_points else float("nan")
                ),
            }
        )

        quantiles = np.quantile(errors, CDF_PROBABILITIES, method="linear")
        cdf_rows.append(
            pd.DataFrame(
                {
                    "window_length": length,
                    "absolute_error_C": quantiles,
                    "cdf": CDF_PROBABILITIES,
                }
            )
        )

        for bin_index, label in enumerate(BASE.TEMP_BIN_LABELS):
            lo = BASE.TEMP_BIN_EDGES[bin_index]
            hi = BASE.TEMP_BIN_EDGES[bin_index + 1]
            if bin_index == len(BASE.TEMP_BIN_LABELS) - 1:
                mask = (temperatures >= lo) & (temperatures <= hi)
            else:
                mask = (temperatures >= lo) & (temperatures < hi)
            bin_metrics = BASE.compute_metrics(errors[mask])
            tempbin_rows.append(
                {
                    "window_length": length,
                    "temperature_bin_C": label,
                    "temperature_bin_left_C": float(lo),
                    "temperature_bin_right_C": float(hi),
                    **bin_metrics,
                }
            )

    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(tempbin_rows),
        pd.concat(cdf_rows, ignore_index=True),
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tag_files = [INPUT_TAG_DIR / f"{tag}.csv" for tag in EXPECTED_TAGS]
    missing = [path.name for path in tag_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing window-selection tag files: {missing}")

    # Determine the same timestamp correction used by the manuscript script.
    query_parts: list[np.ndarray] = []
    for path in tag_files:
        raw = load_raw_observations(path)
        if not raw.empty:
            query_parts.append(raw["event_sec"].to_numpy(dtype=float))
    query_sec = np.concatenate(query_parts)

    temp_sec_raw, temp_val_raw = BASE.read_temperature_csv(INPUT_TEMP_CSV, smooth_window=1)
    temp_sec_eval, temp_val_eval = BASE.read_temperature_csv(
        INPUT_TEMP_CSV,
        smooth_window=BASE.EVAL_TEMP_SMOOTH_WINDOW,
    )
    temp_shift = BASE.choose_temperature_time_shift(temp_sec_raw, query_sec)
    temp_sec_raw = temp_sec_raw + temp_shift
    temp_sec_eval = temp_sec_eval + temp_shift

    per_tag_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    errors_by_length: dict[int, list[np.ndarray]] = {length: [] for length in WINDOW_LENGTHS}
    temperatures_by_length: dict[int, list[np.ndarray]] = {length: [] for length in WINDOW_LENGTHS}

    jobs = max(1, min(int(args.jobs), len(tag_files)))
    print(f"Evaluating C101-C130, window lengths 1-10, jobs={jobs}", flush=True)
    with ProcessPoolExecutor(
        max_workers=jobs,
        initializer=initialize_worker,
        initargs=(temp_sec_raw, temp_val_raw, temp_sec_eval, temp_val_eval),
    ) as executor:
        futures = {executor.submit(process_tag, str(path)): path.stem for path in tag_files}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            per_tag_rows.extend(result["rows"])
            failures.extend(result["failures"])
            for length, arrays in result["arrays"].items():
                errors_by_length[int(length)].append(arrays["absolute_error_C"])
                temperatures_by_length[int(length)].append(arrays["true_temperature_C"])
            completed += 1
            print(f"[{completed:02d}/{len(tag_files)}] {result['epc']} complete", flush=True)

    per_tag = pd.DataFrame(per_tag_rows).sort_values(["window_length", "EPC"])
    failure_frame = pd.DataFrame(failures)
    for length in WINDOW_LENGTHS:
        if len(errors_by_length[length]) != len(EXPECTED_TAGS):
            raise RuntimeError(
                f"Window length {length} has {len(errors_by_length[length])} valid tags; "
                f"expected {len(EXPECTED_TAGS)}."
            )

    overall, tempbins, cdf = aggregate_results(
        per_tag,
        errors_by_length,
        temperatures_by_length,
    )
    overall.to_csv(output_dir / "01_window_length_overall_summary.csv", index=False, encoding="utf-8-sig")
    per_tag.to_csv(output_dir / "02_window_length_per_tag_summary.csv", index=False, encoding="utf-8-sig")
    tempbins.to_csv(output_dir / "03_window_length_tempbin_summary.csv", index=False, encoding="utf-8-sig")
    cdf.to_csv(output_dir / "04_window_length_point_error_cdf.csv", index=False, encoding="utf-8-sig")
    failure_frame.to_csv(output_dir / "05_failures.csv", index=False, encoding="utf-8-sig")

    best_row = overall.loc[overall["mean_absolute_error_C"].idxmin()]
    run_info = {
        "dataset_role": "Independent window-length selection set not used in subsequent experiments.",
        "tag_ids": list(EXPECTED_TAGS),
        "window_lengths": list(WINDOW_LENGTHS),
        "n_tags": len(tag_files),
        "worker_processes": jobs,
        "temperature_time_shift_days": float(temp_shift / 86400.0),
        "fusion_weight": "exp(-abs(P_i - median(window)) / max(0.005, 0.015*median(window)))",
        "mapping_and_fit": "The basic ThermoTag mapping is refitted separately for every tag and window length.",
        "evaluation_temperature_range_C": [BASE.EVAL_TEMP_MIN_C, BASE.EVAL_TEMP_MAX_C],
        "best_window_length_by_pooled_MAE": int(best_row["window_length"]),
        "best_pooled_MAE_C": float(best_row["mean_absolute_error_C"]),
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2),
        encoding="utf-8",
    )

    print("\nWindow-length sweep completed.", flush=True)
    print(overall.to_string(index=False), flush=True)
    print(f"\nResults: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
