from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TEMP_LABELS = ["10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80"]
BASELINE_LABEL = "ThermoTag-style baseline"
PROPOSED_LABEL = "One-point calibration"
REPRESENTATIVE_FORMULA = "f5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one-point calibration with the ThermoTag baseline and write matched MAE tables."
    )
    parser.add_argument(
        "--output-root",
        "--article-root",
        dest="output_root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
        help="Output root containing results/ (default: PROJECT_ROOT/outputs).",
    )
    return parser.parse_args()


def normalize_temp_bin(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace("_", "-", regex=False).str.strip()


def sample_std(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0


def empirical_cdf(values: pd.Series) -> pd.DataFrame:
    numeric = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float))
    if len(numeric) == 0:
        return pd.DataFrame(columns=["task_mae_C", "cdf"])
    return pd.DataFrame(
        {
            "task_mae_C": numeric,
            "cdf": np.arange(1, len(numeric) + 1, dtype=float) / len(numeric),
        }
    )


def load_task_level_data(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined_dir = result_dir / "C201_C301_C350_combined"
    proposed_overall = pd.read_csv(combined_dir / "all_tags_errors_points.csv")
    proposed_bins = pd.read_csv(combined_dir / "all_tags_errors_by_tempbin.csv")
    baseline_points = pd.read_csv(result_dir / "thermotag_mean_parameter_point_errors.csv")

    proposed_overall = proposed_overall[
        (proposed_overall["formula"] == REPRESENTATIVE_FORMULA)
        & (proposed_overall["status"].astype(str).str.upper() == "ACCEPT")
    ].copy()
    proposed_overall["MAE"] = pd.to_numeric(proposed_overall["MAE"], errors="coerce")
    proposed_overall = proposed_overall.dropna(subset=["MAE"])

    if proposed_overall.duplicated(["EPC", "reg_idx"]).any():
        raise RuntimeError("Accepted proposed task keys are not unique.")
    task_keys = proposed_overall[["EPC", "reg_idx"]].copy()

    baseline_points["abs_error_C"] = pd.to_numeric(baseline_points["abs_error_C"], errors="coerce")
    baseline_points["temp_bin"] = normalize_temp_bin(baseline_points["temp_bin"])
    baseline_tag_overall = (
        baseline_points.groupby("tag", as_index=False)["abs_error_C"]
        .mean()
        .rename(columns={"tag": "EPC", "abs_error_C": "task_mae_C"})
    )
    baseline_tasks = task_keys.merge(baseline_tag_overall, on="EPC", how="left", validate="many_to_one")
    if baseline_tasks["task_mae_C"].isna().any():
        missing = sorted(baseline_tasks.loc[baseline_tasks["task_mae_C"].isna(), "EPC"].unique())
        raise RuntimeError(f"Missing baseline overall MAE for tags: {missing}")

    proposed_task_mae = proposed_overall[["EPC", "reg_idx", "MAE"]].rename(
        columns={"MAE": "task_mae_C"}
    )
    overall_tasks = pd.concat(
        [
            baseline_tasks.assign(method=BASELINE_LABEL),
            proposed_task_mae.assign(method=PROPOSED_LABEL),
        ],
        ignore_index=True,
    )

    proposed_bins = proposed_bins[
        (proposed_bins["formula"] == REPRESENTATIVE_FORMULA)
        & (proposed_bins["status"].astype(str).str.upper() == "ACCEPT")
    ].copy()
    proposed_bins["temp_bin"] = normalize_temp_bin(proposed_bins["temp_bin"])
    proposed_bins["MAE"] = pd.to_numeric(proposed_bins["MAE"], errors="coerce")
    proposed_bins = proposed_bins.dropna(subset=["MAE"])
    proposed_bins = proposed_bins[proposed_bins["temp_bin"].isin(TEMP_LABELS)]

    baseline_tag_bins = (
        baseline_points.groupby(["tag", "temp_bin"], as_index=False)["abs_error_C"]
        .mean()
        .rename(columns={"tag": "EPC", "abs_error_C": "task_bin_mae_C"})
    )

    if proposed_bins.duplicated(["EPC", "reg_idx", "temp_bin"]).any():
        raise RuntimeError("Accepted proposed task-temperature-bin keys are not unique.")
    matched_keys = proposed_bins[["EPC", "reg_idx", "temp_bin"]].copy()
    baseline_bin_tasks = matched_keys.merge(
        baseline_tag_bins,
        on=["EPC", "temp_bin"],
        how="left",
        validate="many_to_one",
    )
    if baseline_bin_tasks["task_bin_mae_C"].isna().any():
        missing = baseline_bin_tasks.loc[
            baseline_bin_tasks["task_bin_mae_C"].isna(), ["EPC", "temp_bin"]
        ].drop_duplicates()
        raise RuntimeError(f"Missing baseline temperature-bin MAE:\n{missing.to_string(index=False)}")

    proposed_bin_tasks = proposed_bins[["EPC", "reg_idx", "temp_bin", "MAE"]].rename(
        columns={"MAE": "task_bin_mae_C"}
    )
    bin_tasks = pd.concat(
        [
            baseline_bin_tasks.assign(method=BASELINE_LABEL),
            proposed_bin_tasks.assign(method=PROPOSED_LABEL),
        ],
        ignore_index=True,
    )
    return overall_tasks, bin_tasks


def summarize_temperature_bins(bin_tasks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in [BASELINE_LABEL, PROPOSED_LABEL]:
        method_rows = bin_tasks[bin_tasks["method"] == method]
        for temp_bin in TEMP_LABELS:
            group = method_rows[method_rows["temp_bin"] == temp_bin]
            rows.append(
                {
                    "method": method,
                    "temp_bin": temp_bin,
                    "n_tasks": int(len(group)),
                    "n_tags": int(group["EPC"].nunique()),
                    "mean_task_mae_C": float(group["task_bin_mae_C"].mean()),
                    "std_task_mae_C": sample_std(group["task_bin_mae_C"]),
                }
            )
    return pd.DataFrame(rows)


def build_cdf_table(overall_tasks: pd.DataFrame) -> pd.DataFrame:
    cdf_parts: list[pd.DataFrame] = []
    for method in [BASELINE_LABEL, PROPOSED_LABEL]:
        cdf = empirical_cdf(
            overall_tasks.loc[overall_tasks["method"] == method, "task_mae_C"]
        )
        cdf.insert(0, "method", method)
        cdf_parts.append(cdf)
    return pd.concat(cdf_parts, ignore_index=True)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    result_dir = output_root / "results" / "fast_registration_precise_alignment"

    overall_tasks, bin_tasks = load_task_level_data(result_dir)
    bin_summary = summarize_temperature_bins(bin_tasks)

    expected_tasks = overall_tasks.groupby("method").size()
    if expected_tasks.nunique() != 1:
        raise RuntimeError(f"Methods do not have the same number of tasks:\n{expected_tasks}")

    bin_summary.to_csv(
        result_dir / "03b_tempbin_task_level_thermotag_vs_one_point.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bin_tasks.to_csv(
        result_dir / "03c_tempbin_task_level_values_thermotag_vs_one_point.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall_tasks.to_csv(
        result_dir / "07b_task_level_mae_values_thermotag_vs_one_point.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cdf = build_cdf_table(overall_tasks)
    cdf.to_csv(
        result_dir / "07c_task_level_mae_cdf_thermotag_vs_one_point.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_summary = (
        overall_tasks.groupby("method")["task_mae_C"]
        .agg(n_tasks="count", mean_task_mae_C="mean", std_task_mae_C="std")
    )
    print(overall_summary.to_string())
    print("\nTemperature-bin task counts:")
    print(bin_summary.pivot(index="temp_bin", columns="method", values="n_tasks").to_string())
    print(f"\nComparison tables written to: {result_dir}")


if __name__ == "__main__":
    main()
