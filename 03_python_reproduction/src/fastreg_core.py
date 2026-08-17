# -*- coding: utf-8 -*-
"""
batch_schemeB_eval_all_tags_raw_sequence_rewrite.py

完整重写版：一次性批量验证所有标签的 SchemeB 单点快速注册误差。

输入：
1) RAW_DATA_PATH：学习样本 rraw.txt。
2) FORMULA_PARAMS_PATH：八方法 SchemeB 参数 JSON。
3) MAPPING_DATA_FILE：所有标签注册点/映射数据文件。
   格式：
       C201 或 0001
       Temp_data = [...];
       Time_data = [...];
   这里 Temp_data / Time_data 就是 Temp_AVG_data / Time_AVG_data。
4) DISCHARGE_DIR：所有标签放电 CSV 文件夹，文件名例如 C201.csv、C202.csv。
5) TEMP_CSV_FILE：总温度记录仪 CSV，包含 Time、CH1、CH2。

评价方式：
- 每个标签的每个 Temp_data/Time_data 点轮流作为单点注册点；
- 八个方法分别完成 SchemeB 单点注册；
- 误差不再只和稀疏注册点比较；
- 而是使用该标签原始放电 CSV 中的 Fused_T(s) 反算温度；
- 再与按 MidDateTime 同步得到的温度记录仪 CH1/CH2 平均温度逐点比较。

新增过滤：
- Max_RSSI(dBm) > 55 的放电点不参与误差计算；
- Fused_T(s) 不在 TIME_MIN~TIME_MAX 的点剔除；
- Fused_T(s) 和周围点偏差大的点剔除。

输出：
- all_tags_errors_points.csv：所有标签×方法×注册点误差明细；
- all_tags_errors_summary.csv：每标签×方法汇总；
- all_methods_overall_summary.csv：真正总表，不再只有 best；
- all_methods_overall_summary_cn.csv：中文列名总表；
- all_tags_params_points.csv：每个注册点的适配参数；
- all_tags_errors_by_tempbin.csv：按温区误差；
- raw_point_filter_report.csv：原始放电点评估前过滤统计。
"""

import os

# 必须放在 numpy / torch 前面，解决 Anaconda 下 OpenMP 重复加载问题
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import csv
import json
import math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch


# =========================
# ===== PATH CONFIG =======
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "training", "studydata.txt")
FORMULA_PARAMS_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "offline_training",
    "formula_params_studydata_8methods_schemeB.json",
)

MAPPING_DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "validation",
    "legacy_registration_points_not_used.csv",
)

# 放电时间 CSV 所在目录。
# 如果三批标签的 CSV 都放在同一个目录，只改 DISCHARGE_DIR 即可。
# 如果三批标签放在多个目录，把 DISCHARGE_DIRS 改成多个目录列表。
DISCHARGE_DIR = os.path.join(PROJECT_ROOT, "data", "validation", "tags")
DISCHARGE_DIRS = [DISCHARGE_DIR]
# Additional project-local tag-data directories can be appended if needed.

TEMP_CSV_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "temperature",
    "123time_time_corrected.csv",
)

# 总输出根目录。程序会在这个目录下自动生成 5 个子文件夹：
# C1、C201、C250、C2、C。
OUT_BASE_DIR = os.path.join(PROJECT_ROOT, "outputs", "results", "legacy_fastreg")

# 五组批量验证任务。
# C1   ：C101-C130
# C201 ：C201-C230
# C250 ：C250-C282
# C2   ：C201-C230 + C250-C282
# C    ：C101-C130 + C201-C230 + C250-C282
EVAL_GROUPS = [
    {
        "name": "C201_C301_C350_combined",
        "ranges": [
            ("C201", "C230"),
            ("C301", "C330"),
            ("C350", "C379"),
        ],
    },
]

# 兼容旧变量名
REG_POINTS_FILE = MAPPING_DATA_FILE


# =========================
# ===== RAW EVAL CONFIG ===
# =========================
TEMP_TIME_COL = "Time"
TEMP_CH1_COL = "CH1"
TEMP_CH2_COL = "CH2"

RFID_TIME_COL = "MidDateTime"
RFID_PTIME_COL = "Fused_T(s)"

# EPC 严格同名匹配：MAPPING_DATA_FILE 里是什么 EPC，就只找同名放电文件。
# 不做 0001 -> C201，不做自动加偏移。

# ============================================================
# ===== 放电数据强筛选：只影响误差验证，不改原始 CSV =====
# ============================================================

# 1) Fused_T(s) 合理范围
ENABLE_FUSED_RANGE_FILTER = True
FUSED_TIME_MIN = 0.2
FUSED_TIME_MAX = 1.0

# 2) Fused_T(s) 局部异常：和前后邻居偏差过大
ENABLE_FUSED_LOCAL_OUTLIER_FILTER = True
LOCAL_WINDOW = 5
LOCAL_Z_TH = 3.5
LOCAL_ABS_DIFF_TH = 0.035      # 秒。与局部中位数差超过该值也剔除；不想用就改成 np.inf

# 3) Burst_Details 子融合数据筛选
ENABLE_BURST_DETAILS_FILTER = True
BURST_DETAILS_COL = "Burst_Details"

# 每条记录理论上有 5 个子放电时间。若不足多少个则剔除。
MIN_SUB_TIMES_REQUIRED = 5

# 子放电时间合理范围。
SUB_TIME_MIN = 0.2
SUB_TIME_MAX = 1.0

# 5 个子放电时间最大值 - 最小值 超过这个阈值则剔除
SUB_RANGE_TH = 0.02

# 某一个子放电时间相对其他四个偏差超过这个阈值则剔除
SUB_ONE_VS_OTHERS_ABS_TH = 0.05

# 某一个子放电时间相对其他四个的 robust z 超过该阈值则剔除
SUB_ONE_VS_OTHERS_Z_TH = 4.0

# Fused_T(s) 与子放电时间中位数差距过大则剔除
FUSED_VS_SUB_MEDIAN_TH = 0.02

# 4) RSSI/SNR 列筛选。Max_RSSI(dBm) 一般是负数，默认关闭。
ENABLE_RSSI_FILTER = False
RSSI_COL_CANDIDATES = [
    "Max_RSSI(dBm)", "Max_RSSI", "Max RSSI(dBm)", "Max RSSI",
    "SNR", "SNR(dB)", "snr", "snr_db", "Fusion_SNR", "Fused_SNR",
    "SignalNoiseRatio", "Signal_Noise_Ratio", "信噪比"
]
RSSI_THRESHOLD = 56.0
RSSI_REJECT_MODE = "reject_gt"   # reject_gt: >阈值剔除；reject_lt: <阈值剔除

# 兼容旧变量名，后面别的函数如果引用也不会报错
ENABLE_SNR_FILTER = ENABLE_RSSI_FILTER
SNR_COL_CANDIDATES = RSSI_COL_CANDIDATES
SNR_THRESHOLD = RSSI_THRESHOLD
SNR_REJECT_MODE = RSSI_REJECT_MODE
USE_DISCHARGE_OUTLIER_REJECT = ENABLE_FUSED_LOCAL_OUTLIER_FILTER
TIME_MIN = FUSED_TIME_MIN
TIME_MAX = FUSED_TIME_MAX

# 温度插值：RFID MidDateTime 距离最近温度记录超过该值就不参与评价
MAX_TEMP_INTERP_GAP_SEC = 3.0

# 误差统计温区：仅使用这里手动设置的范围。
# 不读取、也不允许 JSON 中的 T_EVAL_MIN/T_EVAL_MAX 覆盖。
DEFAULT_T_EVAL_MIN = 10.0
DEFAULT_T_EVAL_MAX = 80.0

# 是否排除注册温度附近的原始评价点。默认 0 表示不排除。
EXCLUDE_REG_TEMP_WINDOW_C = 0.0

# 分温区误差统计宽度
TEMP_BIN_WIDTH = 10.0

# 是否输出逐原始点误差明细。数据会很大，默认 False。
WRITE_RAW_DETAIL = False

# 是否为每个标签单独输出一个子目录
WRITE_PER_TAG_FILES = True

# SchemeB 单点适配迭代设置
ADAPT_ITERS = 80
ADAPT_LR = 0.06


# ===== MODEL CONSTANTS ===
EXP_CLIP = 60.0
EXP_PMIN = 0.8
EXP_PMAX = 4.0
M2PRO_PMIN = 1.2
M2PRO_PMAX = 3.0

TMIN_NORM = -50.0
TMAX_NORM = 100.0


# =========================
# ===== UTILS ============
# =========================
def ensure_dir(p: str) -> None:
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def safe_exp(z: float) -> float:
    if z > EXP_CLIP:
        z = EXP_CLIP
    elif z < -EXP_CLIP:
        z = -EXP_CLIP
    return math.exp(z)

def safe_pow2(z: float) -> float:
    if z > EXP_CLIP:
        z = EXP_CLIP
    elif z < -EXP_CLIP:
        z = -EXP_CLIP
    return 2.0 ** z

def sigmoid_np(z: float) -> float:
    if z >= 60:
        return 1.0
    if z <= -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))

def sigmoid_torch(z: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-z))

def clamp_pow_arg(z: torch.Tensor) -> torch.Tensor:
    return torch.clamp(z, -EXP_CLIP, EXP_CLIP)

def x_norm_np(T: float) -> float:
    x = (T - TMIN_NORM) / (TMAX_NORM - TMIN_NORM)
    if x < 0.0:
        x = 0.0
    if x > 1.0:
        x = 1.0
    return float(x)

def robust_mad(v: np.ndarray, eps: float = 1e-12) -> float:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if (not np.isfinite(mad)) or mad < eps:
        mad = eps
    return float(mad)

def z_robust_scalar(x: float, med: float, mad: float) -> float:
    mad = mad if (mad is not None and np.isfinite(mad) and mad > 1e-12) else 1e-12
    return 0.6745 * (x - med) / mad

def write_csv(path: str, rows: list, fieldnames: list) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# =========================
# ===== PARSER ============
# =========================
_TAG_PATTERN = re.compile(
    r"(?P<epc>[A-Za-z]?\d+)\s*[\r\n]+"
    r"Temp_data\s*=\s*\[(?P<temp>[^\]]+)\];\s*[\r\n]+"
    r"Time_data\s*=\s*\[(?P<time>[^\]]+)\];",
    re.MULTILINE | re.IGNORECASE
)

def _parse_num_list(s: str) -> np.ndarray:
    s = s.replace(",", " ")
    return np.array([float(x) for x in s.split()], dtype=np.float64)

def load_tags_from_txt(path: str) -> List[Dict[str, Any]]:
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    m = _TAG_PATTERN.findall(txt)
    if not m:
        raise RuntimeError("No tags parsed. Check file format.")
    tags = []
    for epc, temp, tim in m:
        T = _parse_num_list(temp)
        y = _parse_num_list(tim)
        tags.append({"EPC": epc, "T": T, "y": y})
    return tags


# =========================
# ===== METHOD OPS ========
# =========================
def basis_paper(theta, T):
    b = float(theta["b"]); c = float(theta["c"])
    return 1.0 / (safe_pow2(T / c) + b)

def invert_paper(theta, s, yobs):
    b = float(theta["b"]); c = float(theta["c"])
    y = np.asarray(yobs, dtype=np.float64)
    Tout = np.full_like(y, np.nan, dtype=np.float64)
    if s <= 0 or b <= 0 or c <= 0:
        return Tout
    v = s / y - b
    ok = np.isfinite(v) & (v >= 1.0) & np.isfinite(y) & (y > 0)
    Tout[ok] = c * (np.log(v[ok]) / np.log(2.0))
    return Tout

def basis_m1(theta, T):
    k = float(theta["k"]); b = float(theta["b"])
    x = x_norm_np(T)
    return 1.0 / (safe_pow2(k * x) + b)

def invert_m1(theta, s, yobs):
    k = float(theta["k"]); b = float(theta["b"])
    y = np.asarray(yobs, dtype=np.float64)
    Tout = np.full_like(y, np.nan, dtype=np.float64)
    if s <= 0 or k <= 0 or b <= 0:
        return Tout
    v = s / y - b
    ok = np.isfinite(v) & (v >= 1.0) & np.isfinite(y) & (y > 0)
    x = np.full_like(y, np.nan, dtype=np.float64)
    x[ok] = (np.log(v[ok]) / np.log(2.0)) / k
    ok2 = ok & np.isfinite(x) & (x >= 0.0) & (x <= 1.0)
    Tout[ok2] = TMIN_NORM + x[ok2] * (TMAX_NORM - TMIN_NORM)
    return Tout

def basis_m2(theta, T):
    lam = float(theta["lambda"]); B = float(theta["B"])
    x = x_norm_np(T)
    return 1.0 / (safe_exp(lam * (x * x)) + B)

def invert_m2(theta, s, yobs):
    lam = float(theta["lambda"]); B = float(theta["B"])
    y = np.asarray(yobs, dtype=np.float64)
    Tout = np.full_like(y, np.nan, dtype=np.float64)
    if s <= 0 or lam <= 0 or B <= 0:
        return Tout
    v = s / y - B
    ok = np.isfinite(v) & (v >= 1.0) & np.isfinite(y) & (y > 0)
    u = np.full_like(y, np.nan, dtype=np.float64)
    u[ok] = np.log(v[ok]) / lam
    ok2 = ok & np.isfinite(u) & (u >= 0.0)
    x = np.full_like(y, np.nan, dtype=np.float64)
    x[ok2] = np.sqrt(u[ok2])
    ok3 = ok2 & np.isfinite(x) & (x >= 0.0) & (x <= 1.0)
    Tout[ok3] = TMIN_NORM + x[ok3] * (TMAX_NORM - TMIN_NORM)
    return Tout

def basis_m2pro(theta, T):
    lam = float(theta["lambda"]); B = float(theta["B"]); p = float(theta["p"])
    x = x_norm_np(T)
    return 1.0 / (safe_exp(lam * (x ** p)) + B)

def invert_m2pro(theta, s, yobs):
    lam = float(theta["lambda"]); B = float(theta["B"]); p = float(theta["p"])
    y = np.asarray(yobs, dtype=np.float64)
    Tout = np.full_like(y, np.nan, dtype=np.float64)
    if s <= 0 or lam <= 0 or B <= 0 or p <= 0:
        return Tout
    v = s / y - B
    ok = np.isfinite(v) & (v >= 1.0) & np.isfinite(y) & (y > 0)
    u = np.full_like(y, np.nan, dtype=np.float64)
    u[ok] = np.log(v[ok]) / lam
    ok2 = ok & np.isfinite(u) & (u >= 0.0)
    x = np.full_like(y, np.nan, dtype=np.float64)
    x[ok2] = u[ok2] ** (1.0 / p)
    ok3 = ok2 & np.isfinite(x) & (x >= 0.0) & (x <= 1.0)
    Tout[ok3] = TMIN_NORM + x[ok3] * (TMAX_NORM - TMIN_NORM)
    return Tout

def exp_pbase(theta, x):
    a0 = float(theta["a0"]); a1 = float(theta["a1"]); a2 = float(theta["a2"]); a3 = float(theta["a3"])
    g = a0 + a1 * x + a2 * x * x + a3 * x * x * x
    return float(EXP_PMIN + (EXP_PMAX - EXP_PMIN) * sigmoid_np(g))

def exp_ptag(theta, mode, x):
    pb = exp_pbase(theta, x)
    delta = float(theta.get("delta", 0.0))
    gamma = float(theta.get("gamma", 1.0))
    tau = float(theta.get("tau", 0.0))
    if mode == "EXP0":
        pt = pb
    elif mode == "EXP1":
        pt = pb + delta
    elif mode == "EXP2":
        pt = gamma * pb + delta
    elif mode == "EXP3":
        pt = pb + delta + tau * (x - 0.5)
    else:
        pt = pb
    pt = min(max(pt, EXP_PMIN), EXP_PMAX)
    return float(pt)

def basis_exp(theta, mode, T):
    lam = float(theta["lambda"]); B = float(theta["B"])
    x = x_norm_np(T)
    pt = exp_ptag(theta, mode, x)
    return 1.0 / (safe_exp(lam * (x ** pt)) + B)

def invert_exp(theta, mode, s, yobs):
    y = np.asarray(yobs, dtype=np.float64)
    Tout = np.full_like(y, np.nan, dtype=np.float64)
    if s <= 0:
        return Tout
    lam = float(theta["lambda"]); B = float(theta["B"])
    if lam <= 0 or B <= 0:
        return Tout

    def y_model(xx: float) -> float:
        xx = min(max(xx, 0.0), 1.0)
        pt = exp_ptag(theta, mode, xx)
        den = safe_exp(lam * (xx ** pt)) + B
        return s / den

    y_low = y_model(0.0)
    y_high = y_model(1.0)

    for i in range(y.size):
        yi = float(y[i])
        if (not np.isfinite(yi)) or yi <= 0:
            continue
        if yi > y_low or yi < y_high:
            continue
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            ym = y_model(mid)
            if ym > yi:
                lo = mid
            else:
                hi = mid
        xhat = 0.5 * (lo + hi)
        Tout[i] = TMIN_NORM + xhat * (TMAX_NORM - TMIN_NORM)

    return Tout


def method_basis_np(method, theta, T):
    if method == "Paper":
        return basis_paper(theta, T)
    if method == "M1":
        return basis_m1(theta, T)
    if method == "M2":
        return basis_m2(theta, T)
    if method == "M2pro":
        return basis_m2pro(theta, T)
    return basis_exp(theta, method, T)

def method_invert_np(method, theta, s, yobs):
    if method == "Paper":
        return invert_paper(theta, s, yobs)
    if method == "M1":
        return invert_m1(theta, s, yobs)
    if method == "M2":
        return invert_m2(theta, s, yobs)
    if method == "M2pro":
        return invert_m2pro(theta, s, yobs)
    return invert_exp(theta, method, s, yobs)


# =========================
# ===== SchemeB adapt (torch) =====
# =========================
def q0_from_theta0(method: str, theta0: Dict[str, float], device: torch.device) -> torch.Tensor:
    if method == "Paper":
        return torch.tensor([math.log(theta0["b"]), math.log(theta0["c"])], dtype=torch.float32, device=device)
    if method == "M1":
        return torch.tensor([math.log(theta0["k"]), math.log(theta0["b"])], dtype=torch.float32, device=device)
    if method == "M2":
        return torch.tensor([math.log(theta0["lambda"]), math.log(theta0["B"])], dtype=torch.float32, device=device)
    if method == "M2pro":
        p = float(theta0["p"])
        p = min(max(p, M2PRO_PMIN + 1e-6), M2PRO_PMAX - 1e-6)
        sp = (p - M2PRO_PMIN) / (M2PRO_PMAX - M2PRO_PMIN)
        qp = math.log(sp / (1.0 - sp))
        return torch.tensor([math.log(theta0["lambda"]), math.log(theta0["B"]), qp], dtype=torch.float32, device=device)

    base = [math.log(theta0["lambda"]), math.log(theta0["B"]),
            float(theta0["a0"]), float(theta0["a1"]), float(theta0["a2"]), float(theta0["a3"])]

    if method == "EXP0":
        return torch.tensor(base, dtype=torch.float32, device=device)

    if method == "EXP1":
        d = float(theta0.get("delta", 0.0))
        d = max(min(d, 0.799999), -0.799999)
        sp = (d + 0.8) / 1.6
        return torch.tensor(base + [math.log(sp / (1.0 - sp))], dtype=torch.float32, device=device)

    if method == "EXP2":
        d = float(theta0.get("delta", 0.0))
        d = max(min(d, 0.799999), -0.799999)
        sp = (d + 0.8) / 1.6
        g = float(theta0.get("gamma", 1.0))
        g = min(max(g, 0.600001), 1.399999)
        spg = (g - 0.6) / 0.8
        return torch.tensor(base + [math.log(sp / (1.0 - sp)), math.log(spg / (1.0 - spg))], dtype=torch.float32, device=device)

    # EXP3
    d = float(theta0.get("delta", 0.0))
    d = max(min(d, 0.799999), -0.799999)
    sp = (d + 0.8) / 1.6
    tau = float(theta0.get("tau", 0.0))
    tau = max(min(tau, 0.999999), -0.999999)
    spt = (tau + 1.0) / 2.0
    return torch.tensor(base + [math.log(sp / (1.0 - sp)), math.log(spt / (1.0 - spt))], dtype=torch.float32, device=device)

def theta_from_q(method: str, q: torch.Tensor) -> Dict[str, torch.Tensor]:
    if method == "Paper":
        return {"b": torch.exp(q[0]), "c": torch.exp(q[1])}
    if method == "M1":
        return {"k": torch.exp(q[0]), "b": torch.exp(q[1])}
    if method == "M2":
        return {"lambda": torch.exp(q[0]), "B": torch.exp(q[1])}
    if method == "M2pro":
        lam = torch.exp(q[0]); B = torch.exp(q[1])
        sp = sigmoid_torch(q[2])
        p = torch.tensor(M2PRO_PMIN, device=q.device) + (torch.tensor(M2PRO_PMAX, device=q.device) - torch.tensor(M2PRO_PMIN, device=q.device)) * sp
        return {"lambda": lam, "B": B, "p": p}

    lam = torch.exp(q[0]); B = torch.exp(q[1])
    a0, a1, a2, a3 = q[2], q[3], q[4], q[5]
    th = {"lambda": lam, "B": B, "a0": a0, "a1": a1, "a2": a2, "a3": a3}
    if method == "EXP1":
        sp = sigmoid_torch(q[6]); th["delta"] = -0.8 + 1.6 * sp
    elif method == "EXP2":
        sp = sigmoid_torch(q[6]); th["delta"] = -0.8 + 1.6 * sp
        spg = sigmoid_torch(q[7]); th["gamma"] = 0.6 + 0.8 * spg
    elif method == "EXP3":
        sp = sigmoid_torch(q[6]); th["delta"] = -0.8 + 1.6 * sp
        spt = sigmoid_torch(q[7]); th["tau"] = -1.0 + 2.0 * spt
    return th

def basis_torch(method: str, th: Dict[str, torch.Tensor], T: torch.Tensor) -> torch.Tensor:
    if method == "Paper":
        b = th["b"]; c = th["c"]
        z = clamp_pow_arg(T / (c + 1e-12))
        pow2 = torch.pow(torch.tensor(2.0, device=T.device), z)
        return 1.0 / (pow2 + b)

    x = torch.clamp((T - TMIN_NORM) / (TMAX_NORM - TMIN_NORM), 0.0, 1.0)

    if method == "M1":
        k = th["k"]; b = th["b"]
        z = clamp_pow_arg(k * x)
        pow2 = torch.pow(torch.tensor(2.0, device=T.device), z)
        return 1.0 / (pow2 + b)

    if method == "M2":
        lam = th["lambda"]; B = th["B"]
        u = x * x
        z = clamp_pow_arg(lam * u)
        return 1.0 / (torch.exp(z) + B)

    if method == "M2pro":
        lam = th["lambda"]; B = th["B"]; p = th["p"]
        u = torch.pow(x, p)
        z = clamp_pow_arg(lam * u)
        return 1.0 / (torch.exp(z) + B)

    # EXP*
    lam = th["lambda"]; B = th["B"]
    g = th["a0"] + th["a1"] * x + th["a2"] * x * x + th["a3"] * x * x * x
    pbase = torch.tensor(EXP_PMIN, device=T.device) + (torch.tensor(EXP_PMAX, device=T.device) - torch.tensor(EXP_PMIN, device=T.device)) * sigmoid_torch(g)

    if method == "EXP0":
        ptag = pbase
    elif method == "EXP1":
        ptag = pbase + th["delta"]
    elif method == "EXP2":
        ptag = th["gamma"] * pbase + th["delta"]
    else:  # EXP3
        ptag = pbase + th["delta"] + th["tau"] * (x - 0.5)

    ptag = torch.clamp(ptag, EXP_PMIN, EXP_PMAX)
    u = torch.pow(x, ptag)
    z = clamp_pow_arg(lam * u)
    return 1.0 / (torch.exp(z) + B

    )

def schemeB_adapt_theta(method: str,
                        theta0: Dict[str, float],
                        train_tags: List[Dict[str, Any]],
                        mu_log_s: float,
                        sig_log_s: float,
                        hp: Dict[str, Any],
                        T0: float,
                        y0: float,
                        device: torch.device) -> Tuple[Dict[str, float], float]:
    q0 = q0_from_theta0(method, theta0, device=device)
    q = torch.nn.Parameter(q0.clone())
    opt = torch.optim.Adam([q], lr=ADAPT_LR)

    alpha_train = float(hp["alpha_train"])
    alpha_theta = float(hp["alpha_theta"])
    alpha_sprior = float(hp["alpha_sprior"])

    mu = torch.tensor(float(mu_log_s), dtype=torch.float32, device=device)
    sig = torch.tensor(float(max(sig_log_s, 1e-6)), dtype=torch.float32, device=device)
    T0_t = torch.tensor(float(T0), dtype=torch.float32, device=device)
    y0_t = torch.tensor(float(y0), dtype=torch.float32, device=device)

    train_Ts = [torch.tensor(tg["T"], dtype=torch.float32, device=device) for tg in train_tags]
    train_ys = [torch.tensor(tg["y"], dtype=torch.float32, device=device) for tg in train_tags]

    for _ in range(ADAPT_ITERS):
        opt.zero_grad()
        th = theta_from_q(method, q)

        train_sse = torch.tensor(0.0, device=device)
        for Ttr, ytr in zip(train_Ts, train_ys):
            b = basis_torch(method, th, Ttr)
            b = torch.clamp(b, 1e-12, 1e12)
            s_i = torch.dot(b, ytr) / (torch.dot(b, b) + 1e-12)
            r = ytr - s_i * b
            train_sse = train_sse + torch.sum(r * r)

        bas0 = basis_torch(method, th, T0_t.view(1)).view(())
        bas0 = torch.clamp(bas0, 1e-12, 1e12)
        s_hat = torch.clamp(y0_t / bas0, 1e-12, 1e18)

        loss = alpha_train * train_sse + alpha_theta * torch.sum((q - q0) * (q - q0)) + alpha_sprior * torch.pow((torch.log(s_hat) - mu) / sig, 2.0)

        if not torch.isfinite(loss).item():
            break

        loss.backward()
        opt.step()

    th_f = theta_from_q(method, q.detach())
    theta_adapt = {k: float(v.detach().cpu().item()) for k, v in th_f.items()}
    s_hat_f = float(y0) / float(method_basis_np(method, theta_adapt, float(T0)))
    return theta_adapt, float(s_hat_f)





# =========================
# ===== BATCH HELPERS =====
# =========================

RAW_POINT_FILTER_REPORT: List[Dict[str, Any]] = []


def maybe_float_fmt(v: Any) -> Any:
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(float(v)):
            return ""
        return f"{float(v):.10g}"
    return v


def write_rows(path: str, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    ensure_dir(os.path.dirname(path))
    if fields is None:
        fields, seen = [], set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    fields.append(k)
                    seen.add(k)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: maybe_float_fmt(v) for k, v in r.items()})


def normalize_colname(x: Any) -> str:
    s = str(x)
    s = s.replace("\ufeff", "")
    s = s.replace("锘縏", "")
    s = s.replace("ï»¿", "")
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    return s.lower()


def find_column(df: pd.DataFrame, target: str) -> str:
    target_norm = normalize_colname(target)
    for c in df.columns:
        if normalize_colname(c) == target_norm:
            return c

    # Time 列乱码兜底：例如 锘縏ime
    if target_norm == "time":
        for c in df.columns:
            cc = normalize_colname(c)
            if cc.endswith("time") or "time" in cc:
                return c

    raise RuntimeError(f"找不到列 {target}，当前列名: {list(df.columns)}")


def find_optional_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        try:
            return find_column(df, c)
        except Exception:
            pass
    return None


def parse_datetime_series(s: pd.Series, prefer_temp_format: bool = False) -> pd.Series:
    ss = s.astype(str).str.strip()
    if prefer_temp_format:
        dt = pd.to_datetime(ss, format="%y-%m-%d %H:%M:%S", errors="coerce")
        if dt.notna().mean() > 0.6:
            return dt
    dt = pd.to_datetime(ss, errors="coerce")
    if dt.notna().mean() < 0.6:
        dt2 = pd.to_datetime(ss, format="%y-%m-%d %H:%M:%S", errors="coerce")
        if dt2.notna().sum() > dt.notna().sum():
            return dt2
    return dt


def read_temperature_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, engine="python", encoding="utf-8-sig")
    time_col = find_column(df, TEMP_TIME_COL)
    ch1_col = find_column(df, TEMP_CH1_COL)
    ch2_col = find_column(df, TEMP_CH2_COL)

    dt = parse_datetime_series(df[time_col], prefer_temp_format=True)
    ch1 = pd.to_numeric(df[ch1_col], errors="coerce")
    ch2 = pd.to_numeric(df[ch2_col], errors="coerce")

    valid = dt.notna() & ch1.notna() & ch2.notna() & (ch1 > 0) & (ch2 > 0)
    if valid.sum() < 10:
        raise RuntimeError(f"温度 CSV 有效行太少: {valid.sum()}")

    sec = dt[valid].astype("int64").to_numpy(dtype=np.float64) / 1e9
    val = ((ch1[valid].to_numpy(dtype=np.float64) + ch2[valid].to_numpy(dtype=np.float64)) / 2.0)

    order = np.argsort(sec)
    return sec[order], val[order]


def parse_burst_sub_times(x: Any) -> List[float]:
    """
    从 Burst_Details 中解析子放电时间。

    兼容格式示例：
        0.7840(RSSI=-51.0dBm;dphi=0.70;w=0.22;src=reader)
        0.785,0.786,0.784,0.783,0.785

    规则：优先取“数字 + 左括号”前面的数字，避免把 RSSI、dphi、w 误识别成放电时间。
    """
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return []
    text = str(x).strip()
    if not text:
        return []

    vals = re.findall(r"(?<![A-Za-z0-9_.-])([0-9]+(?:\.[0-9]+)?|\.[0-9]+)\s*\(", text)
    out = []
    for v in vals:
        try:
            fv = float(v)
            if np.isfinite(fv):
                out.append(fv)
        except Exception:
            pass

    if len(out) > 0:
        return out

    # 兜底：按分隔符切开，取每段开头数字
    for part in re.split(r"[,;\s]+", text):
        m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?|\.[0-9]+)", part)
        if m:
            try:
                fv = float(m.group(1))
                if np.isfinite(fv):
                    out.append(fv)
            except Exception:
                pass
    return out


def judge_burst_details(sub_times: List[float], fused_t: float) -> Tuple[bool, str, Dict[str, Any]]:
    """
    判断一条记录的 Burst_Details 是否合格。
    返回 keep, reason, debug。
    """
    debug = {
        "sub_count": len(sub_times),
        "sub_range": "",
        "sub_median": "",
        "max_one_vs_others_abs": "",
        "max_one_vs_others_z": "",
        "fused_vs_sub_median": "",
    }

    if len(sub_times) < MIN_SUB_TIMES_REQUIRED:
        return False, "子放电时间数量不足", debug

    arr = np.asarray(sub_times, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < MIN_SUB_TIMES_REQUIRED:
        debug["sub_count"] = int(arr.size)
        return False, "有效子放电时间数量不足", debug

    if np.any(arr < SUB_TIME_MIN) or np.any(arr > SUB_TIME_MAX):
        return False, "子放电时间超出范围", debug

    sub_median = float(np.median(arr))
    sub_range = float(np.max(arr) - np.min(arr))
    debug["sub_median"] = sub_median
    debug["sub_range"] = sub_range

    if sub_range > SUB_RANGE_TH:
        return False, "子放电时间内部极差过大", debug

    max_abs = 0.0
    max_z = 0.0
    for i in range(arr.size):
        others = np.delete(arr, i)
        med_o = float(np.median(others))
        mad_o = robust_mad(others)
        abs_diff = abs(float(arr[i]) - med_o)
        z = abs(z_robust_scalar(float(arr[i]), med_o, mad_o))
        max_abs = max(max_abs, float(abs_diff))
        max_z = max(max_z, float(z))

    debug["max_one_vs_others_abs"] = max_abs
    debug["max_one_vs_others_z"] = max_z

    if max_abs > SUB_ONE_VS_OTHERS_ABS_TH:
        return False, "某个子放电时间相对其他值偏差过大", debug

    if max_z > SUB_ONE_VS_OTHERS_Z_TH:
        return False, "某个子放电时间robust_z过大", debug

    if not np.isfinite(fused_t):
        return False, "融合放电时间无效", debug

    fused_diff = abs(float(fused_t) - sub_median)
    debug["fused_vs_sub_median"] = fused_diff
    if fused_diff > FUSED_VS_SUB_MEDIAN_TH:
        return False, "融合放电时间与子放电中位数差距过大", debug

    return True, "", debug


def reject_discharge_outliers(sec: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    对已经通过基础筛选/Burst/RSSI 的点，再做 Fused_T(s) 局部异常筛选。
    只在内存中筛掉，不改原始 CSV。
    """
    sec = np.asarray(sec, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if not ENABLE_FUSED_LOCAL_OUTLIER_FILTER:
        return sec, y, {
            "n_removed_by_local_z": 0,
            "n_removed_by_local_absdiff": 0,
        }

    if y.size < max(5, LOCAL_WINDOW):
        return sec, y, {
            "n_removed_by_local_z": 0,
            "n_removed_by_local_absdiff": 0,
        }

    keep_local = np.ones(y.size, dtype=bool)
    removed_by_z = np.zeros(y.size, dtype=bool)
    removed_by_abs = np.zeros(y.size, dtype=bool)

    for i in range(y.size):
        left = max(0, i - LOCAL_WINDOW)
        right = min(y.size, i + LOCAL_WINDOW + 1)
        idx = np.arange(left, right)
        idx = idx[idx != i]
        neigh = y[idx]
        neigh = neigh[np.isfinite(neigh)]
        if neigh.size < 3:
            continue

        med = float(np.median(neigh))
        mad = robust_mad(neigh)
        z = abs(z_robust_scalar(float(y[i]), med, mad))
        abs_diff = abs(float(y[i]) - med)

        bad_z = bool(np.isfinite(z) and z > LOCAL_Z_TH)
        bad_abs = bool(np.isfinite(abs_diff) and abs_diff > LOCAL_ABS_DIFF_TH)

        if bad_z or bad_abs:
            keep_local[i] = False
            removed_by_z[i] = bad_z
            removed_by_abs[i] = bad_abs

    return sec[keep_local], y[keep_local], {
        "n_removed_by_local_z": int(removed_by_z.sum()),
        "n_removed_by_local_absdiff": int(removed_by_abs.sum()),
    }


def read_discharge_csv(discharge_csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取单个标签的放电 CSV，并在误差验证前做强筛选。

    注意：这里不会改 DISCHARGE_DIR 里的原始文件；只是返回筛选后的 sec/y 给后续误差计算。
    """
    df = pd.read_csv(discharge_csv_file, engine="python", encoding="utf-8-sig")
    time_col = find_column(df, RFID_TIME_COL)
    y_col = find_column(df, RFID_PTIME_COL)
    rssi_col = find_optional_column(df, RSSI_COL_CANDIDATES)
    burst_col = find_optional_column(df, [BURST_DETAILS_COL])

    dt = parse_datetime_series(df[time_col], prefer_temp_format=False)
    y = pd.to_numeric(df[y_col], errors="coerce")

    n_total = int(len(df))

    valid = dt.notna() & y.notna() & np.isfinite(y)
    n_basic_valid = int(valid.sum())
    n_removed_basic = int(n_total - n_basic_valid)

    # 1) Fused_T(s) 范围筛选
    n_removed_by_fused_range = 0
    if ENABLE_FUSED_RANGE_FILTER:
        before = int(valid.sum())
        keep_range = (y >= FUSED_TIME_MIN) & (y <= FUSED_TIME_MAX)
        valid = valid & keep_range
        after = int(valid.sum())
        n_removed_by_fused_range = int(before - after)

    # 2) Burst_Details 子融合数据筛选
    n_removed_by_burst = 0
    burst_filter_used = bool(ENABLE_BURST_DETAILS_FILTER and burst_col is not None)
    burst_reason_counts: Dict[str, int] = {}

    if burst_filter_used:
        burst_keep = np.ones(n_total, dtype=bool)
        for i in range(n_total):
            if not bool(valid.iloc[i]):
                continue
            fused_t = float(y.iloc[i]) if np.isfinite(y.iloc[i]) else np.nan
            sub_times = parse_burst_sub_times(df[burst_col].iloc[i])
            keep, reason, _debug = judge_burst_details(sub_times, fused_t)
            if not keep:
                burst_keep[i] = False
                burst_reason_counts[reason] = burst_reason_counts.get(reason, 0) + 1

        before = int(valid.sum())
        valid = valid & pd.Series(burst_keep, index=df.index)
        after = int(valid.sum())
        n_removed_by_burst = int(before - after)

    # 3) RSSI/SNR 筛选，可选
    n_removed_by_rssi = 0
    rssi_filter_used = bool(ENABLE_RSSI_FILTER and rssi_col is not None)

    if rssi_filter_used:
        rssi = pd.to_numeric(df[rssi_col], errors="coerce")
        rssi_valid = rssi.notna() & np.isfinite(rssi)
        if RSSI_REJECT_MODE == "reject_gt":
            rssi_keep = (~rssi_valid) | (rssi <= RSSI_THRESHOLD)
        elif RSSI_REJECT_MODE == "reject_lt":
            rssi_keep = (~rssi_valid) | (rssi >= RSSI_THRESHOLD)
        else:
            raise RuntimeError("RSSI_REJECT_MODE 只能是 reject_gt 或 reject_lt")
        before = int(valid.sum())
        valid = valid & rssi_keep
        after = int(valid.sum())
        n_removed_by_rssi = int(before - after)

    if valid.sum() < 5:
        RAW_POINT_FILTER_REPORT.append({
            "file": discharge_csv_file,
            "n_total": n_total,
            "n_basic_valid": n_basic_valid,
            "n_removed_basic": n_removed_basic,
            "n_removed_by_fused_range": n_removed_by_fused_range,
            "burst_col": burst_col or "",
            "burst_filter_used": burst_filter_used,
            "n_removed_by_burst": n_removed_by_burst,
            "burst_reason_counts": json.dumps(burst_reason_counts, ensure_ascii=False),
            "rssi_col": rssi_col or "",
            "rssi_filter_used": rssi_filter_used,
            "rssi_threshold": RSSI_THRESHOLD if rssi_filter_used else "",
            "rssi_reject_mode": RSSI_REJECT_MODE if rssi_filter_used else "",
            "n_removed_by_rssi": n_removed_by_rssi,
            "n_removed_by_local_z": "",
            "n_removed_by_local_absdiff": "",
            "n_final": 0,
            "note": "too_few_after_basic_fused_burst_or_rssi",
        })
        raise RuntimeError(f"{discharge_csv_file} 可用放电数据太少")

    sec = dt[valid].astype("int64").to_numpy(dtype=np.float64) / 1e9
    yy = y[valid].to_numpy(dtype=np.float64)

    order = np.argsort(sec)
    sec, yy = sec[order], yy[order]

    sec2, yy2, rep = reject_discharge_outliers(sec, yy)

    RAW_POINT_FILTER_REPORT.append({
        "file": discharge_csv_file,
        "n_total": n_total,
        "n_basic_valid": n_basic_valid,
        "n_removed_basic": n_removed_basic,
        "n_removed_by_fused_range": n_removed_by_fused_range,
        "burst_col": burst_col or "",
        "burst_filter_used": burst_filter_used,
        "n_removed_by_burst": n_removed_by_burst,
        "burst_reason_counts": json.dumps(burst_reason_counts, ensure_ascii=False),
        "rssi_col": rssi_col or "",
        "rssi_filter_used": rssi_filter_used,
        "rssi_threshold": RSSI_THRESHOLD if rssi_filter_used else "",
        "rssi_reject_mode": RSSI_REJECT_MODE if rssi_filter_used else "",
        "n_removed_by_rssi": n_removed_by_rssi,
        "n_removed_by_local_z": rep.get("n_removed_by_local_z", ""),
        "n_removed_by_local_absdiff": rep.get("n_removed_by_local_absdiff", ""),
        "n_final": int(sec2.size),
        "note": "",
    })

    if sec2.size < 5:
        raise RuntimeError(f"{discharge_csv_file} 异常过滤后数据太少")

    return sec2, yy2

def interp_temperature_at_rfid(rfid_sec: np.ndarray, temp_sec: np.ndarray, temp_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q = np.asarray(rfid_sec, dtype=np.float64)
    vals = np.interp(q, temp_sec, temp_val, left=np.nan, right=np.nan)

    pos = np.searchsorted(temp_sec, q)
    dist = np.full(q.size, np.inf, dtype=np.float64)

    mask_l = pos > 0
    dist[mask_l] = np.minimum(dist[mask_l], np.abs(q[mask_l] - temp_sec[pos[mask_l] - 1]))

    mask_r = pos < temp_sec.size
    dist[mask_r] = np.minimum(dist[mask_r], np.abs(q[mask_r] - temp_sec[pos[mask_r]]))

    good = np.isfinite(vals) & (dist <= MAX_TEMP_INTERP_GAP_SEC)
    vals[~good] = np.nan
    return vals, good


def possible_discharge_filenames(tag: str) -> List[str]:
    """
    严格 EPC 同名匹配。

    MAPPING_DATA_FILE 里是 C250，就只找 C250.csv / C250.CSV；
    里是 0001，就只找 0001.csv / 0001.CSV。

    不做 0001 -> C201，不自动加 200，不猜 EPC。
    """
    tag = str(tag).strip().upper()
    return [f"{tag}.csv", f"{tag}.CSV"]


def normalize_discharge_dirs(discharge_dirs) -> List[str]:
    """
    兼容单个目录字符串或多个目录列表。
    """
    if discharge_dirs is None:
        return []
    if isinstance(discharge_dirs, (list, tuple)):
        return [str(x) for x in discharge_dirs]
    return [str(discharge_dirs)]


def find_discharge_file(discharge_dirs, tag: str) -> Optional[str]:
    """
    严格 EPC 同名匹配，但可以在多个放电目录中查找。
    不做任何 EPC 映射。
    """
    for d in normalize_discharge_dirs(discharge_dirs):
        for name in possible_discharge_filenames(tag):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return None


def load_raw_eval_for_tag(reg_label: str, discharge_dir, temp_sec: np.ndarray, temp_val: np.ndarray) -> Dict[str, Any]:
    f = find_discharge_file(discharge_dir, reg_label)
    if f is None:
        raise RuntimeError(f"找不到标签 {reg_label} 对应放电 CSV")

    actual_epc = os.path.splitext(os.path.basename(f))[0].strip().upper()

    rfid_sec, y = read_discharge_csv(f)
    Ttrue, good_temp = interp_temperature_at_rfid(rfid_sec, temp_sec, temp_val)
    good = good_temp & np.isfinite(Ttrue) & np.isfinite(y)

    if good.sum() < 5:
        raise RuntimeError(f"标签 {reg_label} -> {actual_epc} 同步温度后有效点太少: {good.sum()}")

    return {
        "REG_LABEL": str(reg_label).upper(),
        "EPC": actual_epc,
        "file": f,
        "rfid_sec": rfid_sec[good],
        "T_true": Ttrue[good],
        "y_obs": y[good],
    }


def fmt_time(sec: float) -> str:
    try:
        return pd.to_datetime(float(sec), unit="s").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def compute_train_s_for_method(method: str, theta0: Dict[str, float], train_tags: List[Dict[str, Any]]) -> np.ndarray:
    ss = []
    for tg in train_tags:
        T = np.asarray(tg["T"], dtype=np.float64)
        y = np.asarray(tg["y"], dtype=np.float64)
        b = np.array([method_basis_np(method, theta0, float(t)) for t in T], dtype=np.float64)
        ok = np.isfinite(b) & np.isfinite(y)
        b = b[ok]
        y = y[ok]
        if b.size < 2:
            continue
        s_i = float(np.dot(b, y) / (np.dot(b, b) + 1e-12))
        if np.isfinite(s_i) and s_i > 0:
            ss.append(s_i)
    return np.asarray(ss, dtype=np.float64)


def prepare_method_contexts(params: Dict[str, Any], epc2tag: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    methods = list(params.get("methods", {}).keys())
    for m in methods:
        info = params["methods"].get(m, {})
        if not info.get("available", False):
            continue

        theta0 = info["theta0"]
        hp = info["hp"]
        prior = info.get("prior_log_s", {})
        kept_epcs = info.get("kept_epcs", [])

        train_tags = []
        for e in kept_epcs:
            key = str(e).upper()
            if key in epc2tag:
                train_tags.append(epc2tag[key])
            elif str(e) in epc2tag:
                train_tags.append(epc2tag[str(e)])

        s_train = compute_train_s_for_method(m, theta0, train_tags)
        med_s = float(np.median(s_train)) if s_train.size else 1.0
        mad_s = robust_mad(s_train) if s_train.size else 1.0

        out[m] = {
            "theta0": theta0,
            "hp": hp,
            "train_tags": train_tags,
            "s_train": s_train,
            "med_s": med_s,
            "mad_s": mad_s,
            "mu_log_s": float(prior.get("mu_log_s", 0.0)),
            "sig_log_s": float(prior.get("sig_log_s", 1.0)),
            "z_thresh_s": float(hp.get("z_thresh_s", 5.5)),
            "z_thresh_y": float(hp.get("z_thresh_y", 5.5)),
            "beta": float(hp.get("beta", 0.5)),
        }
    return out


def compute_error_metrics(T_hat: np.ndarray, T_true: np.ndarray) -> Dict[str, Any]:
    T_hat = np.asarray(T_hat, dtype=np.float64)
    T_true = np.asarray(T_true, dtype=np.float64)
    valid = np.isfinite(T_hat) & np.isfinite(T_true)

    n_total = int(T_true.size)
    n_valid = int(valid.sum())

    if n_valid == 0:
        return {
            "nEvalPts": n_total,
            "nPredValid": 0,
            "validPredRatio": 0.0,
            "MAE": "",
            "RMSE": "",
            "MedianAE": "",
            "P95AE": "",
            "MaxAE": "",
            "Bias": "",
        }

    err = T_hat[valid] - T_true[valid]
    ae = np.abs(err)

    return {
        "nEvalPts": n_total,
        "nPredValid": n_valid,
        "validPredRatio": float(n_valid / max(n_total, 1)),
        "MAE": float(np.mean(ae)),
        "RMSE": float(np.sqrt(np.mean(err * err))),
        "MedianAE": float(np.median(ae)),
        "P95AE": float(np.percentile(ae, 95)),
        "MaxAE": float(np.max(ae)),
        "Bias": float(np.mean(err)),
    }


def add_tempbin_rows(rows_bin: List[Dict[str, Any]], base_row: Dict[str, Any],
                     T_hat: np.ndarray, T_true: np.ndarray) -> None:
    valid = np.isfinite(T_hat) & np.isfinite(T_true)
    if valid.sum() == 0:
        return

    Th = T_hat[valid]
    Tt = T_true[valid]

    tmin = math.floor(float(np.nanmin(Tt)) / TEMP_BIN_WIDTH) * TEMP_BIN_WIDTH
    tmax = math.ceil(float(np.nanmax(Tt)) / TEMP_BIN_WIDTH) * TEMP_BIN_WIDTH

    edges = np.arange(tmin, tmax + TEMP_BIN_WIDTH + 1e-9, TEMP_BIN_WIDTH)

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (Tt >= lo) & (Tt < hi if i < len(edges) - 2 else Tt <= hi)
        if mask.sum() == 0:
            continue
        met = compute_error_metrics(Th[mask], Tt[mask])
        r = dict(base_row)
        r.update({
            "temp_bin": f"{lo:g}_{hi:g}",
            "bin_start": float(lo),
            "bin_end": float(hi),
        })
        r.update(met)
        rows_bin.append(r)


def collect_param_fields(rows_param: List[Dict[str, Any]]) -> List[str]:
    base = ["EPC", "REG_LABEL", "method", "reg_idx", "T0", "y0", "status", "reason", "zS", "zY", "s_hat"]
    seen = set(base)
    extra = []
    for r in rows_param:
        for k in r.keys():
            if k not in seen:
                extra.append(k)
                seen.add(k)
    return base + extra


def evaluate_one_tag(reg_tag: Dict[str, Any], raw_eval: Dict[str, Any], method_ctx: Dict[str, Any],
                     T_eval_min: float, T_eval_max: float, device: torch.device):
    reg_label = str(reg_tag["EPC"]).upper()
    epc = str(raw_eval.get("EPC", reg_label)).upper()

    Treg = np.asarray(reg_tag["T"], dtype=np.float64)
    yreg = np.asarray(reg_tag["y"], dtype=np.float64)

    T_raw = np.asarray(raw_eval["T_true"], dtype=np.float64)
    y_raw = np.asarray(raw_eval["y_obs"], dtype=np.float64)
    time_raw = np.asarray(raw_eval["rfid_sec"], dtype=np.float64)

    base_mask = np.isfinite(T_raw) & np.isfinite(y_raw) & (T_raw >= T_eval_min) & (T_raw <= T_eval_max)

    rows_err, rows_sum, rows_param, rows_bin, rows_detail = [], [], [], [], []

    for m, ctx in method_ctx.items():
        theta0 = ctx["theta0"]
        hp = ctx["hp"]
        train_tags = ctx["train_tags"]
        s_train = ctx["s_train"]
        med_s = ctx["med_s"]
        mad_s = ctx["mad_s"]
        beta = ctx["beta"]

        n_total = 0
        n_accept = 0
        mae_list, rmse_list, max_list = [], [], []
        best = None

        for j in range(Treg.size):
            n_total += 1
            T0 = float(Treg[j])
            y0 = float(yreg[j])

            mask_eval = base_mask.copy()
            if EXCLUDE_REG_TEMP_WINDOW_C and EXCLUDE_REG_TEMP_WINDOW_C > 0:
                mask_eval &= (np.abs(T_raw - T0) > EXCLUDE_REG_TEMP_WINDOW_C)

            T_true = T_raw[mask_eval]
            y_obs = y_raw[mask_eval]
            time_eval = time_raw[mask_eval]
            nEvalPts = int(T_true.size)

            base_row = {
                "EPC": epc,
                "REG_LABEL": reg_label,
                "method": m,
                "reg_idx": j + 1,
                "T0": T0,
                "y0": y0,
                "raw_file": raw_eval.get("file", ""),
            }

            def reject_row(reason, zS="", zY=""):
                r = dict(base_row)
                r.update({
                    "status": "REJECT",
                    "reason": reason,
                    "zS": zS,
                    "zY": zY,
                    "nEvalPts": nEvalPts,
                    "nPredValid": 0,
                    "validPredRatio": 0,
                    "MAE": "",
                    "RMSE": "",
                    "MedianAE": "",
                    "P95AE": "",
                    "MaxAE": "",
                    "Bias": "",
                })
                return r

            if nEvalPts == 0:
                rows_err.append(reject_row("no_raw_eval_points"))
                continue

            try:
                theta_adapt, s_hat = schemeB_adapt_theta(
                    m, theta0, train_tags, ctx["mu_log_s"], ctx["sig_log_s"], hp, T0, y0, device
                )
            except Exception as e:
                rows_err.append(reject_row(f"adapt_fail:{e}"))
                rows_param.append({**base_row, "status": "REJECT", "reason": "adapt_fail", "s_hat": ""})
                continue

            rr = dict(base_row)
            rr.update({"s_hat": float(s_hat) if np.isfinite(s_hat) else ""})
            rr.update(theta_adapt)

            if (not np.isfinite(s_hat)) or s_hat <= 0:
                rows_err.append(reject_row("invalid_s"))
                rr.update({"status": "REJECT", "reason": "invalid_s"})
                rows_param.append(rr)
                continue

            zS = z_robust_scalar(float(s_hat), med_s, mad_s)

            bas0 = method_basis_np(m, theta0, T0)
            if (not np.isfinite(bas0)) or bas0 <= 0 or s_train.size < 2:
                rows_err.append(reject_row("basis0_invalid", zS=zS))
                rr.update({"status": "REJECT", "reason": "basis0_invalid", "zS": zS, "zY": ""})
                rows_param.append(rr)
                continue

            y_train0 = s_train * bas0
            zY = z_robust_scalar(float(y0), float(np.median(y_train0)), robust_mad(y_train0))

            if abs(zS) > ctx["z_thresh_s"] or abs(zY) > ctx["z_thresh_y"]:
                rows_err.append(reject_row("z_outlier", zS=zS, zY=zY))
                rr.update({"status": "REJECT", "reason": "z_outlier", "zS": zS, "zY": zY})
                rows_param.append(rr)
                continue

            T_hat = method_invert_np(m, theta_adapt, float(s_hat), y_obs)
            met = compute_error_metrics(T_hat, T_true)

            if met["nPredValid"] == 0:
                r = dict(base_row)
                r.update({"status": "REJECT", "reason": "invert_fail", "zS": zS, "zY": zY})
                r.update(met)
                rows_err.append(r)
                rr.update({"status": "REJECT", "reason": "invert_fail", "zS": zS, "zY": zY})
                rows_param.append(rr)
                continue

            n_accept += 1
            mae_list.append(float(met["MAE"]))
            rmse_list.append(float(met["RMSE"]))
            max_list.append(float(met["MaxAE"]))

            obj = beta * float(met["MAE"]) + (1.0 - beta) * float(met["MaxAE"])
            if best is None or obj < best["obj"]:
                best = {"obj": obj, "reg_idx": j + 1, "T0": T0, "y0": y0, **met}

            r = dict(base_row)
            r.update({"status": "ACCEPT", "reason": "", "zS": zS, "zY": zY})
            r.update(met)
            rows_err.append(r)

            rr.update({"status": "ACCEPT", "reason": "", "zS": zS, "zY": zY})
            rows_param.append(rr)

            add_tempbin_rows(rows_bin, {**base_row, "status": "ACCEPT"}, T_hat, T_true)

            if WRITE_RAW_DETAIL:
                valid_pred = np.isfinite(T_hat) & np.isfinite(T_true)
                for k in np.where(valid_pred)[0]:
                    rows_detail.append({
                        "EPC": epc,
                        "REG_LABEL": reg_label,
                        "method": m,
                        "reg_idx": j + 1,
                        "T0": T0,
                        "y0": y0,
                        "raw_idx": int(k + 1),
                        "rfid_time": fmt_time(float(time_eval[k])),
                        "T_true": float(T_true[k]),
                        "y_obs": float(y_obs[k]),
                        "T_hat": float(T_hat[k]),
                        "absErr": float(abs(T_hat[k] - T_true[k])),
                        "signedErr": float(T_hat[k] - T_true[k]),
                    })

        rows_sum.append({
            "EPC": epc,
            "REG_LABEL": reg_label,
            "method": m,
            "beta": beta,
            "reg_points_total": n_total,
            "reg_points_ACCEPT": n_accept,
            "ACCEPT_rate": float(n_accept / max(n_total, 1)),
            "MAE_over_ACCEPT_mean": "" if n_accept == 0 else float(np.mean(mae_list)),
            "RMSE_over_ACCEPT_mean": "" if n_accept == 0 else float(np.mean(rmse_list)),
            "MaxAE_over_ACCEPT_mean": "" if n_accept == 0 else float(np.mean(max_list)),
            "best_reg_idx": "" if best is None else best["reg_idx"],
            "best_T0": "" if best is None else best["T0"],
            "best_y0": "" if best is None else best["y0"],
            "best_MAE": "" if best is None else best["MAE"],
            "best_RMSE": "" if best is None else best["RMSE"],
            "best_MedianAE": "" if best is None else best["MedianAE"],
            "best_P95AE": "" if best is None else best["P95AE"],
            "best_MaxAE": "" if best is None else best["MaxAE"],
            "best_Bias": "" if best is None else best["Bias"],
            "best_objective": "" if best is None else best["obj"],
        })

    return rows_err, rows_sum, rows_param, rows_bin, rows_detail


def _to_float_or_nan(v: Any) -> float:
    try:
        if v is None or v == "":
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _safe_mean(vals: List[Any]) -> Any:
    arr = np.array([_to_float_or_nan(v) for v in vals], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return "" if arr.size == 0 else float(np.mean(arr))


def _safe_median(vals: List[Any]) -> Any:
    arr = np.array([_to_float_or_nan(v) for v in vals], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return "" if arr.size == 0 else float(np.median(arr))


def _safe_sum(vals: List[Any]) -> float:
    arr = np.array([_to_float_or_nan(v) for v in vals], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.sum(arr)) if arr.size else 0.0


def _weighted_mean(vals: List[Any], weights: List[Any]) -> Any:
    v = np.array([_to_float_or_nan(x) for x in vals], dtype=np.float64)
    w = np.array([_to_float_or_nan(x) for x in weights], dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return ""
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def _is_accept_status(v: Any) -> bool:
    return str(v).strip().upper() in {"ACCEPT", "ACCEPTED", "OK", "TRUE", "1", "YES", "PASS", "KEPT"}


def build_overall_summary(all_err_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    methods = sorted(set(str(r.get("method", "")) for r in all_err_rows if r.get("method", "") != ""))

    metric_names = ["MAE", "RMSE", "MedianAE", "P95AE", "MaxAE", "Bias"]

    for m in methods:
        g_all = [r for r in all_err_rows if str(r.get("method", "")) == m]
        g_valid = [r for r in g_all if np.isfinite(_to_float_or_nan(r.get("MAE", "")))]

        epcs_all = sorted(set(str(r.get("EPC", "")) for r in g_all if str(r.get("EPC", "")) != ""))
        epcs_valid = sorted(set(str(r.get("EPC", "")) for r in g_valid if str(r.get("EPC", "")) != ""))

        n_reg_total = len(g_all)
        n_reg_with_valid_error = len(g_valid)
        n_accept = sum(1 for r in g_all if _is_accept_status(r.get("status", "")))
        n_reject = n_reg_total - n_accept

        by_tag_all: Dict[str, List[Dict[str, Any]]] = {}
        by_tag_valid: Dict[str, List[Dict[str, Any]]] = {}

        for r in g_all:
            epc = str(r.get("EPC", "")).strip()
            if epc:
                by_tag_all.setdefault(epc, []).append(r)

        for r in g_valid:
            epc = str(r.get("EPC", "")).strip()
            if epc:
                by_tag_valid.setdefault(epc, []).append(r)

        row: Dict[str, Any] = {
            "method": m,
            "n_tags": len(epcs_valid),
            "n_tags_total_seen": len(epcs_all),
            "n_reg_total": n_reg_total,
            "n_reg_with_valid_error": n_reg_with_valid_error,
            "n_accept": n_accept,
            "n_reject": n_reject,
            "accept_rate": "" if n_reg_total == 0 else float(n_accept / n_reg_total),
            "reject_rate": "" if n_reg_total == 0 else float(n_reject / n_reg_total),
            "n_eval_points_sum": _safe_sum([r.get("nEvalPts", "") for r in g_valid]),
            "n_pred_valid_sum": _safe_sum([r.get("nPredValid", "") for r in g_valid]),
        }

        tag_accept_rates = []
        for _, rs in by_tag_all.items():
            if len(rs) > 0:
                tag_accept_rates.append(sum(_is_accept_status(x.get("status", "")) for x in rs) / len(rs))
        row["accept_rate_mean_over_tags"] = _safe_mean(tag_accept_rates)
        row["accept_rate_median_over_tags"] = _safe_median(tag_accept_rates)

        # 所有注册点直接平均
        for metric in metric_names:
            row[f"{metric}_mean_over_all_reg_points"] = _safe_mean([r.get(metric, "") for r in g_valid])
            row[f"{metric}_median_over_all_reg_points"] = _safe_median([r.get(metric, "") for r in g_valid])

        # 按原始评价点数加权
        weights = [r.get("nPredValid", r.get("nEvalPts", "")) for r in g_valid]
        for metric in metric_names:
            row[f"{metric}_weighted_by_eval_points"] = _weighted_mean([r.get(metric, "") for r in g_valid], weights)

        # 每个标签先平均，再对标签平均
        tag_metric_means: Dict[str, List[float]] = {metric: [] for metric in metric_names}
        for _, rs in by_tag_valid.items():
            for metric in metric_names:
                v = _safe_mean([r.get(metric, "") for r in rs])
                if v != "":
                    tag_metric_means[metric].append(float(v))

        for metric in metric_names:
            row[f"{metric}_mean_over_tags"] = _safe_mean(tag_metric_means[metric])
            row[f"{metric}_median_over_tags"] = _safe_median(tag_metric_means[metric])

        # best / worst
        best_rows, worst_rows = [], []
        for _, rs in by_tag_valid.items():
            rs2 = [r for r in rs if np.isfinite(_to_float_or_nan(r.get("MAE", "")))]
            if not rs2:
                continue
            best_rows.append(min(rs2, key=lambda x: _to_float_or_nan(x.get("MAE", ""))))
            worst_rows.append(max(rs2, key=lambda x: _to_float_or_nan(x.get("MAE", ""))))

        for metric in metric_names:
            row[f"best_{metric}_mean_over_tags"] = _safe_mean([r.get(metric, "") for r in best_rows])
            row[f"best_{metric}_median_over_tags"] = _safe_median([r.get(metric, "") for r in best_rows])
            row[f"worst_{metric}_mean_over_tags"] = _safe_mean([r.get(metric, "") for r in worst_rows])
            row[f"worst_{metric}_median_over_tags"] = _safe_median([r.get(metric, "") for r in worst_rows])

        rows.append(row)

    return rows


def build_overall_summary_cn(overall_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cn_map = {
        "method": "方法",
        "n_tags": "有效标签数量",
        "n_tags_total_seen": "读取到的标签数量",
        "n_reg_total": "注册实验总数",
        "n_reg_with_valid_error": "有有效误差的注册实验数",
        "n_accept": "接受注册点数",
        "n_reject": "拒绝注册点数",
        "accept_rate": "注册点总体接受率",
        "reject_rate": "注册点总体拒绝率",
        "accept_rate_mean_over_tags": "各标签接受率平均",
        "accept_rate_median_over_tags": "各标签接受率中位数",
        "n_eval_points_sum": "原始评价点总数",
        "n_pred_valid_sum": "有效反算温度点总数",
    }

    # 自动翻译各误差列
    name_map = {
        "MAE": "MAE",
        "RMSE": "RMSE",
        "MedianAE": "中位绝对误差",
        "P95AE": "P95绝对误差",
        "MaxAE": "最大绝对误差",
        "Bias": "平均偏差",
    }
    suffix_map = {
        "mean_over_all_reg_points": "所有注册点平均",
        "median_over_all_reg_points": "所有注册点中位数",
        "weighted_by_eval_points": "按原始评价点数加权",
        "mean_over_tags": "先按标签平均再平均",
        "median_over_tags": "标签中位数",
    }

    for metric, cname in name_map.items():
        for suffix, sname in suffix_map.items():
            cn_map[f"{metric}_{suffix}"] = f"{cname}_{sname}"
        cn_map[f"best_{metric}_mean_over_tags"] = f"best_{cname}_每标签最佳注册点平均"
        cn_map[f"best_{metric}_median_over_tags"] = f"best_{cname}_每标签最佳注册点中位数"
        cn_map[f"worst_{metric}_mean_over_tags"] = f"worst_{cname}_每标签最差注册点平均"
        cn_map[f"worst_{metric}_median_over_tags"] = f"worst_{cname}_每标签最差注册点中位数"

    out = []
    for r in overall_rows:
        nr = {}
        for k, v in r.items():
            nr[cn_map.get(k, k)] = v
        out.append(nr)
    return out


def parse_epc_code(epc: str) -> Tuple[str, int]:
    """
    把 C250 解析成 (C, 250)。
    只用于分组筛选，不用于文件名映射。
    """
    s = str(epc).strip().upper()
    m = re.match(r"^([A-Z]+)(\d+)$", s)
    if not m:
        return s, -1
    return m.group(1), int(m.group(2))


def epc_in_range(epc: str, start_epc: str, end_epc: str) -> bool:
    prefix, num = parse_epc_code(epc)
    prefix_s, num_s = parse_epc_code(start_epc)
    prefix_e, num_e = parse_epc_code(end_epc)

    if prefix != prefix_s or prefix != prefix_e:
        return False
    if num < 0 or num_s < 0 or num_e < 0:
        return False
    return num_s <= num <= num_e


def epc_in_group(epc: str, ranges: List[Tuple[str, str]]) -> bool:
    epc = str(epc).strip().upper()
    for start_epc, end_epc in ranges:
        if epc_in_range(epc, start_epc, end_epc):
            return True
    return False


def filter_reg_tags_by_group(reg_tags: List[Dict[str, Any]], ranges: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for tg in reg_tags:
        epc = str(tg.get("EPC", "")).strip().upper()
        if epc_in_group(epc, ranges):
            out.append(tg)
    return out


def range_to_text(ranges: List[Tuple[str, str]]) -> str:
    return ", ".join([f"{a}-{b}" for a, b in ranges])


def run_one_eval_group(
    group_name: str,
    group_ranges: List[Tuple[str, str]],
    reg_tags_all: List[Dict[str, Any]],
    method_ctx: Dict[str, Any],
    temp_sec: np.ndarray,
    temp_val: np.ndarray,
    T_eval_min: float,
    T_eval_max: float,
    device: torch.device,
) -> None:
    """
    跑一个标签组，并把结果写到 OUT_BASE_DIR/group_name。
    """
    global RAW_POINT_FILTER_REPORT
    RAW_POINT_FILTER_REPORT = []

    group_out_dir = os.path.join(OUT_BASE_DIR, group_name)
    ensure_dir(group_out_dir)

    reg_tags = filter_reg_tags_by_group(reg_tags_all, group_ranges)

    print("\n" + "#" * 90)
    print(f"[GROUP] {group_name}")
    print(f"[GROUP] EPC ranges: {range_to_text(group_ranges)}")
    print(f"[GROUP] register tags in group: {len(reg_tags)}")
    print(f"[GROUP] output dir: {group_out_dir}")
    print("#" * 90)

    if len(reg_tags) == 0:
        print(f"[GROUP-SKIP] {group_name}: MAPPING_DATA_FILE 中没有这个分组的标签。")
        write_rows(os.path.join(group_out_dir, "all_tags_errors_points.csv"), [])
        write_rows(os.path.join(group_out_dir, "all_tags_errors_summary.csv"), [])
        write_rows(os.path.join(group_out_dir, "all_tags_errors_by_tempbin.csv"), [])
        write_rows(os.path.join(group_out_dir, "all_tags_params_points.csv"), [])
        write_rows(os.path.join(group_out_dir, "all_methods_overall_summary.csv"), [])
        write_rows(os.path.join(group_out_dir, "all_methods_overall_summary_cn.csv"), [])
        write_rows(os.path.join(group_out_dir, "raw_point_filter_report.csv"), [])
        return

    all_err, all_sum, all_param, all_bin = [], [], [], []

    raw_detail_dir = os.path.join(group_out_dir, "raw_detail")
    param_root = os.path.join(group_out_dir, "params")
    ensure_dir(param_root)
    if WRITE_RAW_DETAIL:
        ensure_dir(raw_detail_dir)

    for idx, reg_tag in enumerate(reg_tags, start=1):
        reg_label = str(reg_tag["EPC"]).upper()
        print(f"\n========== [{group_name} {idx}/{len(reg_tags)}] REG={reg_label} ==========")

        try:
            raw_eval = load_raw_eval_for_tag(reg_label, DISCHARGE_DIRS, temp_sec, temp_val)
            epc = str(raw_eval.get("EPC", reg_label)).upper()
            print(f"[RAW] REG={reg_label} EPC={epc} file={raw_eval['file']} eval_points={len(raw_eval['T_true'])}")
        except Exception as e:
            print(f"[SKIP] {reg_label}: {e}")
            all_sum.append({"EPC": "", "REG_LABEL": reg_label, "method": "ALL", "error": str(e)})
            continue

        rows_err, rows_sum, rows_param, rows_bin, rows_detail = evaluate_one_tag(
            reg_tag, raw_eval, method_ctx, T_eval_min, T_eval_max, device
        )

        all_err.extend(rows_err)
        all_sum.extend(rows_sum)
        all_param.extend(rows_param)
        all_bin.extend(rows_bin)

        if WRITE_PER_TAG_FILES:
            epc_dir = os.path.join(group_out_dir, epc)
            epc_param_dir = os.path.join(param_root, epc)
            ensure_dir(epc_dir)
            ensure_dir(epc_param_dir)

            write_rows(os.path.join(epc_dir, "errors_points.csv"), rows_err)
            write_rows(os.path.join(epc_dir, "errors_summary.csv"), rows_sum)
            write_rows(os.path.join(epc_dir, "errors_by_tempbin.csv"), rows_bin)
            write_rows(os.path.join(epc_param_dir, "params_points.csv"), rows_param, collect_param_fields(rows_param))

        if WRITE_RAW_DETAIL and rows_detail:
            write_rows(os.path.join(raw_detail_dir, f"{epc}_raw_detail.csv"), rows_detail)

    print(f"\n========== WRITE GROUP OUTPUTS: {group_name} ==========")

    write_rows(os.path.join(group_out_dir, "all_tags_errors_points.csv"), all_err)
    write_rows(os.path.join(group_out_dir, "all_tags_errors_summary.csv"), all_sum)
    write_rows(os.path.join(group_out_dir, "all_tags_errors_by_tempbin.csv"), all_bin)
    write_rows(os.path.join(group_out_dir, "all_tags_params_points.csv"), all_param, collect_param_fields(all_param))

    overall = build_overall_summary(all_err)
    write_rows(os.path.join(group_out_dir, "all_methods_overall_summary.csv"), overall)
    write_rows(os.path.join(group_out_dir, "all_methods_overall_summary_cn.csv"), build_overall_summary_cn(overall))

    write_rows(os.path.join(group_out_dir, "raw_point_filter_report.csv"), RAW_POINT_FILTER_REPORT)

    print("Saved group:", group_name)
    print(" ", os.path.join(group_out_dir, "all_tags_errors_points.csv"))
    print(" ", os.path.join(group_out_dir, "all_tags_errors_summary.csv"))
    print(" ", os.path.join(group_out_dir, "all_methods_overall_summary.csv"))
    print(" ", os.path.join(group_out_dir, "all_methods_overall_summary_cn.csv"))
    print(" ", os.path.join(group_out_dir, "all_tags_params_points.csv"))
    print(" ", os.path.join(group_out_dir, "all_tags_errors_by_tempbin.csv"))
    print(" ", os.path.join(group_out_dir, "raw_point_filter_report.csv"))


def main():
    ensure_dir(OUT_BASE_DIR)

    with open(FORMULA_PARAMS_PATH, "r", encoding="utf-8") as f:
        params = json.load(f)

    # 固定使用脚本顶部手动设置的评价温区，忽略 JSON 中同名字段。
    T_eval_min = float(DEFAULT_T_EVAL_MIN)
    T_eval_max = float(DEFAULT_T_EVAL_MAX)
    print(f"[CONFIG] fixed eval temperature range: {T_eval_min} ~ {T_eval_max} C (JSON T_EVAL_MIN/T_EVAL_MAX ignored)")

    print("[LOAD] training raw data:", RAW_DATA_PATH)
    train_tags = load_tags_from_txt(RAW_DATA_PATH)
    epc2tag = {str(tg["EPC"]).upper(): tg for tg in train_tags}
    epc2tag.update({str(tg["EPC"]): tg for tg in train_tags})
    print("[LOAD] train tags:", len(train_tags))

    print("[LOAD] register points:", REG_POINTS_FILE)
    reg_tags_all = load_tags_from_txt(REG_POINTS_FILE)
    print("[LOAD] register tags total:", len(reg_tags_all))

    print("[LOAD] discharge dirs:")
    for d in normalize_discharge_dirs(DISCHARGE_DIRS):
        print("   ", d)

    print("[LOAD] temperature csv:", TEMP_CSV_FILE)
    temp_sec, temp_val = read_temperature_csv(TEMP_CSV_FILE)

    device = torch.device("cpu")
    print("[DEVICE]", device)

    method_ctx = prepare_method_contexts(params, epc2tag)
    if not method_ctx:
        raise RuntimeError("没有可用方法，请检查 formula_params JSON。")

    group_overview_rows = []

    for g in EVAL_GROUPS:
        group_name = str(g["name"])
        group_ranges = [(str(a).upper(), str(b).upper()) for a, b in g["ranges"]]
        n_reg = len(filter_reg_tags_by_group(reg_tags_all, group_ranges))
        group_overview_rows.append({
            "group": group_name,
            "ranges": range_to_text(group_ranges),
            "n_register_tags_in_mapping": n_reg,
            "out_dir": os.path.join(OUT_BASE_DIR, group_name),
        })
        run_one_eval_group(
            group_name=group_name,
            group_ranges=group_ranges,
            reg_tags_all=reg_tags_all,
            method_ctx=method_ctx,
            temp_sec=temp_sec,
            temp_val=temp_val,
            T_eval_min=T_eval_min,
            T_eval_max=T_eval_max,
            device=device,
        )

    write_rows(os.path.join(OUT_BASE_DIR, "groups_overview.csv"), group_overview_rows)

    print("\n" + "=" * 90)
    print("全部分组完成。输出根目录：", OUT_BASE_DIR)
    print("已生成 5 个结果文件夹：")
    for g in EVAL_GROUPS:
        print(" ", os.path.join(OUT_BASE_DIR, str(g["name"])))
    print("分组概览：", os.path.join(OUT_BASE_DIR, "groups_overview.csv"))
    print("每个文件夹内重点看：all_methods_overall_summary_cn.csv")
    print("=" * 90)


if __name__ == "__main__":
    main()
