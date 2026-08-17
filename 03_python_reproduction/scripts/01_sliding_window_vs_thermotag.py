# -*- coding: utf-8 -*-
"""
Generate the sliding-window persistence-time fusion vs. ThermoTag comparison.

All generated files are written under the project-local outputs/ directory.
The script reads data/ without modifying the versioned experimental inputs.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_TAG_DIR = PROJECT_ROOT / "data" / "validation" / "tags"
INPUT_TEMP_CSV = PROJECT_ROOT / "data" / "temperature" / "123time_time_corrected.csv"

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
OUTPUT_RESULT_DIR = OUTPUT_ROOT / "results" / "sliding_window_vs_thermotag"


# ---------------------------------------------------------------------------
# Method and evaluation settings
# ---------------------------------------------------------------------------

WINDOW_SIZE = 5
SIGMA_RATIO = 0.015
SIGMA_MIN_S = 0.005

BIN_WIDTH_C = 3.0
MIN_COUNT_PER_CURVE_BIN = 3
CURVE_TIME_MIN_S = 0.2
CURVE_TIME_MAX_S = 1.0
CURVE_LOCAL_WINDOW = 5
CURVE_LOCAL_Z_TH = 3.5

EVAL_TEMP_MIN_C = 10.0
EVAL_TEMP_MAX_C = 80.0
EVAL_TIME_MAX_S = 1.0
EVAL_BURST_RANGE_TH_S = 0.1
EVAL_TEMP_SMOOTH_WINDOW = 5
MAX_TEMP_INTERP_GAP_S = 3.0

METHODS = [
    ("ThermoTag", "ThermoTag", "#1f77b4"),
    ("SlidingWindow", "Sliding-window fusion", "#ff7f0e"),
]

# The paper's sliding-window comparison uses the 63 C2 tags only. The shared
# dataset also contains the independent validation batches for other analyses.
SLIDING_TAG_RANGES = ((201, 230), (250, 282))

TEMP_BIN_EDGES = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=float)
TEMP_BIN_LABELS = [f"{int(TEMP_BIN_EDGES[i])}-{int(TEMP_BIN_EDGES[i + 1])}" for i in range(len(TEMP_BIN_EDGES) - 1)]


@dataclass
class TagObservations:
    epc: str
    thermotag: pd.DataFrame
    sliding_window: pd.DataFrame
    curve_points: dict[str, pd.DataFrame]
    fits: dict[str, dict[str, float]]


def natural_key(text: str) -> tuple:
    parts = re.split(r"(\d+)", str(text))
    return tuple(int(part) if part.isdigit() else part for part in parts)


def is_sliding_window_tag(path: Path) -> bool:
    match = re.fullmatch(r"C(\d+)", path.stem, flags=re.IGNORECASE)
    if match is None:
        return False
    number = int(match.group(1))
    return any(start <= number <= end for start, end in SLIDING_TAG_RANGES)


def ensure_output_dirs() -> None:
    OUTPUT_RESULT_DIR.mkdir(parents=True, exist_ok=True)


def parse_burst_details(text: object) -> np.ndarray:
    """Extract the five persistence-time values from one Burst_Details cell."""
    if not isinstance(text, str) or not text.strip():
        return np.array([], dtype=float)

    # Example cell item: 0.7840(RSSI=-51.0dBm;dphi=0.70;w=0.22;src=reader)
    matches = re.findall(r"([0-9]*\.?[0-9]+)\s*\(", text)
    values = []
    for item in matches:
        try:
            values.append(float(item))
        except ValueError:
            pass
    return np.asarray(values, dtype=float)


def parse_datetime_series(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, errors="coerce")
    seconds = np.full(len(series), np.nan, dtype=float)
    ok = dt.notna().to_numpy()
    if np.any(ok):
        dt_ns = dt[ok].to_numpy(dtype="datetime64[ns]")
        seconds[ok] = dt_ns.astype("int64").astype(float) / 1e9
    return seconds


def parse_epoch_ms_series(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values / 1000.0


def get_time_column_seconds(df: pd.DataFrame, date_name: str, epoch_name: str) -> np.ndarray:
    if date_name in df.columns:
        sec = parse_datetime_series(df[date_name])
        if np.isfinite(sec).any():
            return sec
    if epoch_name in df.columns:
        return parse_epoch_ms_series(df[epoch_name])
    return np.full(len(df), np.nan, dtype=float)


def infer_row_times(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return row start, middle, and latest-event seconds.

    The current online output is aligned to the latest discharge event in the
    five-sample row/window. If EndDateTime exists, it is the latest event time.
    MidDateTime is used only to infer the missing boundary time.
    """
    mid_sec = get_time_column_seconds(df, "MidDateTime", "MidEpochMs")
    end_sec = get_time_column_seconds(df, "EndDateTime", "EndEpochMs")

    if "StartDateTime" in df.columns or "StartEpochMs" in df.columns:
        start_sec = get_time_column_seconds(df, "StartDateTime", "StartEpochMs")
    else:
        start_sec = np.full(len(df), np.nan, dtype=float)

    have_start = np.isfinite(start_sec)
    have_mid = np.isfinite(mid_sec)
    have_end = np.isfinite(end_sec)

    start_from_mid_end = have_mid & have_end & ~have_start
    start_sec[start_from_mid_end] = 2.0 * mid_sec[start_from_mid_end] - end_sec[start_from_mid_end]

    end_from_start_mid = have_start & have_mid & ~have_end
    end_sec[end_from_start_mid] = 2.0 * mid_sec[end_from_start_mid] - start_sec[end_from_start_mid]

    return start_sec, mid_sec, end_sec


def valid_burst(values: np.ndarray) -> bool:
    if len(values) < WINDOW_SIZE:
        return False
    values = values[:WINDOW_SIZE]
    if not np.all(np.isfinite(values)):
        return False
    if not np.all((values > 0) & (values <= EVAL_TIME_MAX_S)):
        return False
    return float(np.max(values) - np.min(values)) <= EVAL_BURST_RANGE_TH_S


def sliding_window_fusion(values: np.ndarray) -> tuple[float, np.ndarray]:
    values = np.asarray(values[:WINDOW_SIZE], dtype=float)
    med = float(np.median(values))
    sigma = max(SIGMA_MIN_S, SIGMA_RATIO * med)
    weights = np.exp(-np.abs(values - med) / sigma)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0 or not np.isfinite(weight_sum):
        return float("nan"), np.full_like(values, np.nan, dtype=float)
    normalized = weights / weight_sum
    return float(np.sum(normalized * values)), normalized


def infer_sub_event_times(start_sec: float, end_sec: float, values: np.ndarray) -> np.ndarray:
    """
    Estimate occurrence time for each raw persistence-time sample.

    For P_j = tau_j - tau_{j-1}, the sample is aligned to tau_j, i.e., the
    latest response timestamp of that discharge interval. This is also why the
    fused sliding-window value is aligned to the fifth tau_j in the window.
    """
    values = np.asarray(values[:WINDOW_SIZE], dtype=float)
    if not np.isfinite(start_sec):
        if np.isfinite(end_sec):
            start_sec = float(end_sec) - float(np.sum(values))
        else:
            return np.full(len(values), np.nan, dtype=float)
    sub_end = float(start_sec) + np.cumsum(values)
    if np.isfinite(end_sec) and len(sub_end):
        # Preserve the measured end timestamp while keeping the intra-window spacing.
        sub_end += float(end_sec) - float(sub_end[-1])
    return sub_end


def load_tag_csv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    epc = path.stem
    df = pd.read_csv(path)
    start_sec, _mid_sec, end_sec = infer_row_times(df)

    thermotag_rows = []

    for row_id, row in df.iterrows():
        values = parse_burst_details(row.get("Burst_Details"))
        if len(values) < WINDOW_SIZE:
            continue
        values = values[:WINDOW_SIZE]
        keep = valid_burst(values)
        if not keep:
            continue

        sub_times = infer_sub_event_times(start_sec[row_id], end_sec[row_id], values)
        burst_range = float(np.max(values) - np.min(values))

        for sub_index, (ptime, sub_sec) in enumerate(zip(values, sub_times), start=1):
            if np.isfinite(ptime) and np.isfinite(sub_sec):
                thermotag_rows.append(
                    {
                        "EPC": epc,
                        "row_id": int(row_id),
                        "sub_index": int(sub_index),
                        "event_sec": float(sub_sec),
                        "persistence_time_s": float(ptime),
                        "burst_range_s": burst_range,
                            }
                        )

    thermotag = pd.DataFrame(thermotag_rows)
    sliding_rows = []
    if not thermotag.empty:
        raw = (
            thermotag.sort_values(["event_sec", "row_id", "sub_index"])
            .reset_index(drop=True)
            .copy()
        )
        pvals = raw["persistence_time_s"].to_numpy(dtype=float)
        for end_pos in range(WINDOW_SIZE - 1, len(raw)):
            win = raw.iloc[end_pos - WINDOW_SIZE + 1 : end_pos + 1]
            values = pvals[end_pos - WINDOW_SIZE + 1 : end_pos + 1]
            if len(values) < WINDOW_SIZE:
                continue
            if not np.all(np.isfinite(values)):
                continue
            if not np.all((values > 0) & (values <= EVAL_TIME_MAX_S)):
                continue
            burst_range = float(np.max(values) - np.min(values))

            fused_time, weights = sliding_window_fusion(values)
            latest = win.iloc[-1]
            first = win.iloc[0]
            if not np.isfinite(fused_time) or not np.isfinite(float(latest["event_sec"])):
                continue
            sliding_rows.append(
                {
                    "EPC": epc,
                    "row_id": int(latest["row_id"]),
                    "sub_index": int(latest["sub_index"]),
                    "event_sec": float(latest["event_sec"]),
                    "persistence_time_s": float(fused_time),
                    "burst_range_s": burst_range,
                    "median_s": float(np.median(values)),
                    "w1": float(weights[0]),
                    "w2": float(weights[1]),
                    "w3": float(weights[2]),
                    "w4": float(weights[3]),
                    "w5": float(weights[4]),
                    "window_start_row_id": int(first["row_id"]),
                    "window_start_sub_index": int(first["sub_index"]),
                    "window_end_row_id": int(latest["row_id"]),
                    "window_end_sub_index": int(latest["sub_index"]),
                }
            )

    return thermotag, pd.DataFrame(sliding_rows)


def parse_temperature_time(value: str) -> float | None:
    value = str(value).strip()
    for fmt in ("%y-%m-%d %H:%M:%S.%f", "%y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(value, fmt)
            return pd.Timestamp(dt).value / 1e9
        except ValueError:
            continue
    return None


def read_temperature_csv(path: Path, smooth_window: int = 1) -> tuple[np.ndarray, np.ndarray]:
    sec_list = []
    temp_list = []

    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        header = handle.readline().strip().split(",")
        columns = {name.strip(): idx for idx, name in enumerate(header)}
        if "Time" not in columns or "CH1" not in columns or "CH2" not in columns:
            raise ValueError(f"Temperature file has unexpected columns: {header}")

        for line in handle:
            parts = line.strip().split(",")
            if len(parts) <= max(columns["Time"], columns["CH1"], columns["CH2"]):
                continue
            sec = parse_temperature_time(parts[columns["Time"]])
            if sec is None:
                continue
            try:
                ch1 = float(parts[columns["CH1"]])
                ch2 = float(parts[columns["CH2"]])
            except ValueError:
                continue
            if ch1 <= 0 or ch2 <= 0:
                continue
            sec_list.append(sec)
            temp_list.append((ch1 + ch2) / 2.0)

    if not sec_list:
        raise ValueError(f"No valid temperature rows in {path}")

    temp_df = pd.DataFrame({"sec": sec_list, "temperature_C": temp_list})
    temp_df = temp_df.groupby("sec", as_index=False)["temperature_C"].mean().sort_values("sec")

    if smooth_window > 1:
        temp_df["temperature_C"] = (
            temp_df["temperature_C"]
            .rolling(window=smooth_window, center=True, min_periods=1)
            .mean()
        )

    return temp_df["sec"].to_numpy(dtype=float), temp_df["temperature_C"].to_numpy(dtype=float)


def choose_temperature_time_shift(temp_sec: np.ndarray, query_sec: np.ndarray) -> float:
    query_sec = query_sec[np.isfinite(query_sec)]
    if len(query_sec) == 0:
        return 0.0

    offsets_days = [0, 365, 366, -365, -366, 730, 731, 732, -730, -731, -732]
    best_offset = 0.0
    best_score = (-1, -1.0, -float("inf"))
    q_min, q_max = float(np.min(query_sec)), float(np.max(query_sec))

    for days in offsets_days:
        offset = days * 86400.0
        shifted_min = float(np.min(temp_sec) + offset)
        shifted_max = float(np.max(temp_sec) + offset)
        overlap = max(0.0, min(q_max, shifted_max) - max(q_min, shifted_min))
        inside = int(np.sum((query_sec >= shifted_min) & (query_sec <= shifted_max)))
        center_gap = abs((shifted_min + shifted_max) / 2.0 - (q_min + q_max) / 2.0)
        score = (inside, overlap, -center_gap)
        if score > best_score:
            best_score = score
            best_offset = offset

    return best_offset


def interpolate_temperature(
    query_sec: np.ndarray,
    temp_sec: np.ndarray,
    temp_val: np.ndarray,
    max_gap_s: float = MAX_TEMP_INTERP_GAP_S,
) -> np.ndarray:
    query_sec = np.asarray(query_sec, dtype=float)
    out = np.full(len(query_sec), np.nan, dtype=float)

    inside = (
        np.isfinite(query_sec)
        & (query_sec >= float(temp_sec[0]))
        & (query_sec <= float(temp_sec[-1]))
    )
    if not np.any(inside):
        return out

    q = query_sec[inside]
    interp = np.interp(q, temp_sec, temp_val)

    right_idx = np.searchsorted(temp_sec, q, side="left")
    left_idx = np.clip(right_idx - 1, 0, len(temp_sec) - 1)
    right_idx = np.clip(right_idx, 0, len(temp_sec) - 1)
    nearest_gap = np.minimum(np.abs(q - temp_sec[left_idx]), np.abs(q - temp_sec[right_idx]))
    interp[nearest_gap > max_gap_s] = np.nan

    out[inside] = interp
    return out


def local_outlier_filter(obs: pd.DataFrame) -> pd.DataFrame:
    if len(obs) < CURVE_LOCAL_WINDOW:
        return obs.copy()

    obs = obs.sort_values("event_sec").reset_index(drop=True).copy()
    values = obs["persistence_time_s"].to_numpy(dtype=float)
    keep = np.ones(len(obs), dtype=bool)

    for i in range(len(obs)):
        lo = max(0, i - CURVE_LOCAL_WINDOW // 2)
        hi = min(len(obs), i + CURVE_LOCAL_WINDOW // 2 + 1)
        window = values[lo:hi]
        med = float(np.median(window))
        mad = float(np.median(np.abs(window - med)))
        if mad <= 1e-12:
            continue
        robust_z = abs(values[i] - med) / (1.4826 * mad)
        if robust_z > CURVE_LOCAL_Z_TH:
            keep[i] = False

    return obs[keep].copy()


def group_curve_points(temp: np.ndarray, ptime: np.ndarray) -> pd.DataFrame:
    valid = np.isfinite(temp) & np.isfinite(ptime)
    temp = temp[valid]
    ptime = ptime[valid]

    if len(temp) == 0:
        return pd.DataFrame(columns=["temperature_C", "persistence_time_s", "n_points"])

    t_min = math.floor(float(np.min(temp)) / BIN_WIDTH_C) * BIN_WIDTH_C
    t_max = math.ceil(float(np.max(temp)) / BIN_WIDTH_C) * BIN_WIDTH_C
    edges = np.arange(t_min, t_max + BIN_WIDTH_C, BIN_WIDTH_C)

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (temp >= lo) & (temp < hi)
        if int(np.sum(mask)) < MIN_COUNT_PER_CURVE_BIN:
            continue
        bin_temp = temp[mask]
        bin_time = ptime[mask]

        med = float(np.median(bin_time))
        mad = float(np.median(np.abs(bin_time - med)))
        if mad > 1e-12:
            robust_z = np.abs(bin_time - med) / (1.4826 * mad)
            inlier = robust_z <= CURVE_LOCAL_Z_TH
            if int(np.sum(inlier)) >= MIN_COUNT_PER_CURVE_BIN:
                bin_temp = bin_temp[inlier]
                bin_time = bin_time[inlier]

        rows.append(
            {
                "temperature_C": float(np.mean(bin_temp)),
                "persistence_time_s": float(np.mean(bin_time)),
                "n_points": int(len(bin_time)),
            }
        )

    return pd.DataFrame(rows)


def thermotag_model(temp: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a / (np.power(2.0, temp / c) + b)


def fit_abc(temp: np.ndarray, ptime: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(temp) & np.isfinite(ptime) & (ptime > 0)
    temp = np.asarray(temp[valid], dtype=float)
    ptime = np.asarray(ptime[valid], dtype=float)
    if len(temp) < 5:
        raise RuntimeError("not enough curve points for fitting")

    def objective(params: np.ndarray) -> float:
        a, b, c = params
        if a <= 0 or b <= 0 or c <= 1:
            return 1e12
        pred = thermotag_model(temp, a, b, c)
        if not np.all(np.isfinite(pred)):
            return 1e12
        return float(np.mean((pred - ptime) ** 2))

    starts = [
        np.array([25.0, 26.0, 14.0], dtype=float),
        np.array([10.0, 10.0, 20.0], dtype=float),
        np.array([max(float(np.max(ptime)) * 10.0, 1.0), 5.0, 25.0], dtype=float),
    ]

    best = None
    for start in starts:
        result = minimize(objective, start, method="Nelder-Mead", options={"maxiter": 20000})
        if best is None or result.fun < best.fun:
            best = result

    if best is None or not best.success:
        raise RuntimeError("curve fitting failed")

    a, b, c = [float(x) for x in best.x]
    fitted = thermotag_model(temp, a, b, c)
    ss_res = float(np.sum((ptime - fitted) ** 2))
    ss_tot = float(np.sum((ptime - float(np.mean(ptime))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    return {
        "a": a,
        "b": b,
        "c": c,
        "curve_rmse_s": math.sqrt(float(best.fun)),
        "curve_r2": r2,
        "n_curve_points": int(len(temp)),
    }


def inverse_temperature(ptime: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    ptime = np.asarray(ptime, dtype=float)
    inner = a / ptime - b
    out = np.full(len(ptime), np.nan, dtype=float)
    valid = np.isfinite(inner) & (inner > 0)
    out[valid] = c * np.log2(inner[valid])
    return out


def build_curve_for_method(
    obs: pd.DataFrame,
    temp_sec_raw: np.ndarray,
    temp_val_raw: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    curve_obs = obs[
        (obs["persistence_time_s"] > CURVE_TIME_MIN_S)
        & (obs["persistence_time_s"] < CURVE_TIME_MAX_S)
        & np.isfinite(obs["event_sec"])
    ].copy()
    curve_obs = local_outlier_filter(curve_obs)

    curve_obs["temperature_C"] = interpolate_temperature(
        curve_obs["event_sec"].to_numpy(dtype=float),
        temp_sec_raw,
        temp_val_raw,
    )
    curve_obs = curve_obs[np.isfinite(curve_obs["temperature_C"])].copy()

    grouped = group_curve_points(
        curve_obs["temperature_C"].to_numpy(dtype=float),
        curve_obs["persistence_time_s"].to_numpy(dtype=float),
    )
    fit = fit_abc(
        grouped["temperature_C"].to_numpy(dtype=float),
        grouped["persistence_time_s"].to_numpy(dtype=float),
    )
    return grouped, fit


def evaluate_method(
    epc: str,
    method_code: str,
    method_label: str,
    obs: pd.DataFrame,
    fit: dict[str, float],
    temp_sec_eval: np.ndarray,
    temp_val_eval: np.ndarray,
) -> pd.DataFrame:
    eval_obs = obs[
        (obs["persistence_time_s"] > 0)
        & (obs["persistence_time_s"] <= EVAL_TIME_MAX_S)
        & np.isfinite(obs["event_sec"])
    ].copy()
    eval_obs["true_temperature_C"] = interpolate_temperature(
        eval_obs["event_sec"].to_numpy(dtype=float),
        temp_sec_eval,
        temp_val_eval,
    )

    eval_obs = eval_obs[
        np.isfinite(eval_obs["true_temperature_C"])
        & (eval_obs["true_temperature_C"] >= EVAL_TEMP_MIN_C)
        & (eval_obs["true_temperature_C"] <= EVAL_TEMP_MAX_C)
    ].copy()

    pred = inverse_temperature(
        eval_obs["persistence_time_s"].to_numpy(dtype=float),
        fit["a"],
        fit["b"],
        fit["c"],
    )
    eval_obs["estimated_temperature_C"] = pred
    eval_obs = eval_obs[np.isfinite(eval_obs["estimated_temperature_C"])].copy()
    eval_obs["signed_error_C"] = eval_obs["estimated_temperature_C"] - eval_obs["true_temperature_C"]
    eval_obs["absolute_error_C"] = np.abs(eval_obs["signed_error_C"])
    eval_obs["method_code"] = method_code
    eval_obs["method"] = method_label
    eval_obs["abc_a"] = fit["a"]
    eval_obs["abc_b"] = fit["b"]
    eval_obs["abc_c"] = fit["c"]

    ordered_cols = [
        "method_code",
        "method",
        "EPC",
        "row_id",
        "event_sec",
        "persistence_time_s",
        "true_temperature_C",
        "estimated_temperature_C",
        "signed_error_C",
        "absolute_error_C",
        "abc_a",
        "abc_b",
        "abc_c",
    ]
    optional = [c for c in ["sub_index", "burst_range_s", "median_s", "w1", "w2", "w3", "w4", "w5"] if c in eval_obs.columns]
    return eval_obs[ordered_cols + optional].copy()


def compute_metrics(errors: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(errors), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "n_points": 0,
            "mean_absolute_error_C": float("nan"),
            "root_mean_square_error_C": float("nan"),
            "median_absolute_error_C": float("nan"),
            "p95_absolute_error_C": float("nan"),
            "max_absolute_error_C": float("nan"),
            "std_absolute_error_C": float("nan"),
        }

    return {
        "n_points": int(len(values)),
        "mean_absolute_error_C": float(np.mean(values)),
        "root_mean_square_error_C": float(np.sqrt(np.mean(values ** 2))),
        "median_absolute_error_C": float(np.median(values)),
        "p95_absolute_error_C": float(np.percentile(values, 95)),
        "max_absolute_error_C": float(np.max(values)),
        "std_absolute_error_C": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def summarize_point_errors(point_errors: pd.DataFrame, fits_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    per_tag_rows = []

    for method_code, method_label, _color in METHODS:
        method_points = point_errors[point_errors["method_code"] == method_code]
        metrics = compute_metrics(method_points["absolute_error_C"])
        fit_part = fits_df[fits_df["method_code"] == method_code]
        overall_rows.append(
            {
                "method_code": method_code,
                "method": method_label,
                "n_tags": int(method_points["EPC"].nunique()),
                **metrics,
                "mean_curve_r2": float(fit_part["curve_r2"].mean()) if len(fit_part) else float("nan"),
                "mean_n_curve_points": float(fit_part["n_curve_points"].mean()) if len(fit_part) else float("nan"),
            }
        )

        for epc, tag_points in method_points.groupby("EPC"):
            tag_metrics = compute_metrics(tag_points["absolute_error_C"])
            fit_row = fit_part[fit_part["EPC"] == epc]
            row = {
                "method_code": method_code,
                "method": method_label,
                "EPC": epc,
                **tag_metrics,
            }
            if not fit_row.empty:
                for col in ["abc_a", "abc_b", "abc_c", "curve_rmse_s", "curve_r2", "n_curve_points"]:
                    row[col] = fit_row.iloc[0][col]
            per_tag_rows.append(row)

    temp_bin_rows = []
    for method_code, method_label, _color in METHODS:
        method_points = point_errors[point_errors["method_code"] == method_code]
        for bin_index, label in enumerate(TEMP_BIN_LABELS):
            lo = TEMP_BIN_EDGES[bin_index]
            hi = TEMP_BIN_EDGES[bin_index + 1]
            if bin_index == len(TEMP_BIN_LABELS) - 1:
                mask = (method_points["true_temperature_C"] >= lo) & (method_points["true_temperature_C"] <= hi)
            else:
                mask = (method_points["true_temperature_C"] >= lo) & (method_points["true_temperature_C"] < hi)
            part = method_points[mask]
            metrics = compute_metrics(part["absolute_error_C"])
            temp_bin_rows.append(
                {
                    "method_code": method_code,
                    "method": method_label,
                    "temperature_bin_C": label,
                    "temperature_bin_left_C": float(lo),
                    "temperature_bin_right_C": float(hi),
                    "n_tags": int(part["EPC"].nunique()) if len(part) else 0,
                    **metrics,
                }
            )

    return pd.DataFrame(overall_rows), pd.DataFrame(per_tag_rows), pd.DataFrame(temp_bin_rows)


def make_cdf(values: np.ndarray) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values.sort()
    if len(values) == 0:
        return pd.DataFrame({"absolute_error_C": [], "cdf": []})
    cdf = np.arange(1, len(values) + 1, dtype=float) / len(values)
    return pd.DataFrame({"absolute_error_C": values, "cdf": cdf})


def save_cdf_tables(point_errors: pd.DataFrame, per_tag_summary: pd.DataFrame) -> None:
    point_cdf_parts = []
    tag_max_cdf_parts = []
    for method_code, method_label, _color in METHODS:
        part = point_errors[point_errors["method_code"] == method_code]
        cdf = make_cdf(part["absolute_error_C"].to_numpy(dtype=float))
        cdf.insert(0, "method", method_label)
        cdf.insert(0, "method_code", method_code)
        point_cdf_parts.append(cdf)

        tag_part = per_tag_summary[per_tag_summary["method_code"] == method_code]
        tag_cdf = make_cdf(tag_part["max_absolute_error_C"].to_numpy(dtype=float))
        tag_cdf = tag_cdf.rename(columns={"absolute_error_C": "tag_max_absolute_error_C"})
        tag_cdf.insert(0, "method", method_label)
        tag_cdf.insert(0, "method_code", method_code)
        tag_max_cdf_parts.append(tag_cdf)

    pd.concat(point_cdf_parts, ignore_index=True).to_csv(
        OUTPUT_RESULT_DIR / "05_point_error_cdf.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(tag_max_cdf_parts, ignore_index=True).to_csv(
        OUTPUT_RESULT_DIR / "06_tag_max_error_cdf.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    ensure_output_dirs()

    if not INPUT_TAG_DIR.exists():
        raise FileNotFoundError(f"Tag data folder not found: {INPUT_TAG_DIR}")
    if not INPUT_TEMP_CSV.exists():
        raise FileNotFoundError(f"Temperature file not found: {INPUT_TEMP_CSV}")

    tag_files = sorted(
        (path for path in INPUT_TAG_DIR.glob("*.csv") if is_sliding_window_tag(path)),
        key=lambda p: natural_key(p.stem),
    )
    if not tag_files:
        raise FileNotFoundError(f"No tag CSV files found under: {INPUT_TAG_DIR}")

    tag_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    all_event_sec = []
    failures = []

    for path in tag_files:
        try:
            thermotag_obs, sliding_obs = load_tag_csv(path)
            tag_data[path.stem] = (thermotag_obs, sliding_obs)
            if len(thermotag_obs):
                all_event_sec.append(thermotag_obs["event_sec"].to_numpy(dtype=float))
            if len(sliding_obs):
                all_event_sec.append(sliding_obs["event_sec"].to_numpy(dtype=float))
        except Exception as exc:
            failures.append({"EPC": path.stem, "stage": "load_tag_csv", "error": str(exc)})

    if not all_event_sec:
        raise RuntimeError("No valid RFID observations were loaded.")
    query_sec = np.concatenate(all_event_sec)

    temp_sec_raw, temp_val_raw = read_temperature_csv(INPUT_TEMP_CSV, smooth_window=1)
    temp_sec_eval, temp_val_eval = read_temperature_csv(INPUT_TEMP_CSV, smooth_window=EVAL_TEMP_SMOOTH_WINDOW)
    temp_shift = choose_temperature_time_shift(temp_sec_raw, query_sec)
    temp_sec_raw = temp_sec_raw + temp_shift
    temp_sec_eval = temp_sec_eval + temp_shift

    point_error_parts = []
    curve_point_parts = []
    fit_rows = []

    for epc in sorted(tag_data, key=natural_key):
        method_obs = {
            "ThermoTag": tag_data[epc][0],
            "SlidingWindow": tag_data[epc][1],
        }
        for method_code, method_label, _color in METHODS:
            obs = method_obs[method_code]
            if len(obs) == 0:
                failures.append({"EPC": epc, "method_code": method_code, "stage": "observation", "error": "no valid observations"})
                continue

            try:
                curve_points, fit = build_curve_for_method(obs, temp_sec_raw, temp_val_raw)
                curve_points = curve_points.copy()
                curve_points.insert(0, "method", method_label)
                curve_points.insert(0, "method_code", method_code)
                curve_points.insert(0, "EPC", epc)
                curve_point_parts.append(curve_points)

                detail = evaluate_method(epc, method_code, method_label, obs, fit, temp_sec_eval, temp_val_eval)
                if len(detail) == 0:
                    raise RuntimeError("no valid evaluation points after temperature alignment")
                point_error_parts.append(detail)

                fit_rows.append(
                    {
                        "EPC": epc,
                        "method_code": method_code,
                        "method": method_label,
                        "abc_a": fit["a"],
                        "abc_b": fit["b"],
                        "abc_c": fit["c"],
                        "curve_rmse_s": fit["curve_rmse_s"],
                        "curve_r2": fit["curve_r2"],
                        "n_curve_points": fit["n_curve_points"],
                        "n_eval_points": int(len(detail)),
                    }
                )
            except Exception as exc:
                failures.append({"EPC": epc, "method_code": method_code, "stage": "fit_or_eval", "error": str(exc)})

    if not point_error_parts:
        raise RuntimeError("No valid evaluation results were generated.")

    point_errors = pd.concat(point_error_parts, ignore_index=True)
    curve_points_df = pd.concat(curve_point_parts, ignore_index=True) if curve_point_parts else pd.DataFrame()
    fits_df = pd.DataFrame(fit_rows)
    failures_df = pd.DataFrame(failures)

    overall_summary, per_tag_summary, temp_bin_summary = summarize_point_errors(point_errors, fits_df)

    overall_summary.to_csv(OUTPUT_RESULT_DIR / "01_overall_summary.csv", index=False, encoding="utf-8-sig")
    per_tag_summary.to_csv(OUTPUT_RESULT_DIR / "02_per_tag_summary.csv", index=False, encoding="utf-8-sig")
    point_errors.to_csv(OUTPUT_RESULT_DIR / "03_point_errors.csv", index=False, encoding="utf-8-sig")
    temp_bin_summary.to_csv(OUTPUT_RESULT_DIR / "04_tempbin_summary.csv", index=False, encoding="utf-8-sig")
    curve_points_df.to_csv(OUTPUT_RESULT_DIR / "07_curve_points.csv", index=False, encoding="utf-8-sig")
    fits_df.to_csv(OUTPUT_RESULT_DIR / "08_fitted_abc_parameters.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(OUTPUT_RESULT_DIR / "09_failures.csv", index=False, encoding="utf-8-sig")
    save_cdf_tables(point_errors, per_tag_summary)

    run_info = {
        "input_tag_dir": str(INPUT_TAG_DIR),
        "input_temperature_csv": str(INPUT_TEMP_CSV),
        "output_root": str(OUTPUT_ROOT),
        "n_tag_files": len(tag_files),
        "n_tags_with_results": int(point_errors["EPC"].nunique()),
        "window_size": WINDOW_SIZE,
        "sliding_window_weight": "exp(-abs(P_i - median(P_window)) / max(0.005, 0.015*median(P_window)))",
        "temperature_alignment": "ThermoTag raw samples are aligned to each sample's inferred response time; sliding-window fusion is aligned to the latest response time in the five-sample window.",
        "temperature_time_shift_seconds": float(temp_shift),
        "temperature_time_shift_days": float(temp_shift / 86400.0),
        "curve_fit": "For each tag and each method, abc is re-fitted from the method-specific persistence-time series before temperature inversion.",
        "methods": [label for _code, label, _color in METHODS],
        "outputs": {
            "results": str(OUTPUT_RESULT_DIR),
        },
    }
    with (OUTPUT_RESULT_DIR / "run_info.json").open("w", encoding="utf-8") as handle:
        json.dump(run_info, handle, ensure_ascii=False, indent=2)

    print("Finished sliding-window comparison.")
    print(f"Tags read: {len(tag_files)}")
    print(f"Tags with valid results: {point_errors['EPC'].nunique()}")
    print(f"Temperature time shift: {temp_shift / 86400.0:.0f} days")
    print(f"Results: {OUTPUT_RESULT_DIR}")
    print()
    print(overall_summary.to_string(index=False))


if __name__ == "__main__":
    main()
