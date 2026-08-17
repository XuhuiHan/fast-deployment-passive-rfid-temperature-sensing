# -*- coding: utf-8 -*-
"""
gpu_tune_filter_study_schemeB.py  (方案B：调参 + 学习theta0 + 过滤 + 评估)

输出：
- results_schemeB.json   (每个方法的最优：hp + theta0 + prior(log s) + kept_epcs + eval summary)

说明（方案B）：
- 过滤：同你之前思路（先学theta0 -> per-tag RMSE -> robust z剔除，迭代）
- 注册点适配：对每个注册点，优化 theta 使：
    alpha_train * TrainSSE(theta)  +  alpha_sprior * ((log(s_hat)-mu)/sig)^2  + alpha_theta * ||q-q0||^2
  其中 s_hat = y0 / basis(theta, T0)
- 误差计算：只和该标签原始采样点比（含注册点），且仅统计 T_EVAL_MIN~T_EVAL_MAX 范围内点

依赖：
- numpy, torch
"""

import os
import re
import json
import math
from typing import Dict, Any, List, Tuple

import numpy as np
import torch


# =========================
# ======= PATHS ===========
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "training", "studydata.txt")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "offline_training")
RESULT_JSON = os.path.join(OUT_DIR, "results_schemeB.json")


# =========================
# ======= SETTINGS ========
# =========================
SEED = 1234

# 固定方法种子。不要再使用 Python 内置 hash(method.name)，
# 因为 hash 在不同进程中默认会变化，导致八个独立项目无法严格复现单文件结果。
METHOD_SEED_OFFSETS = {
    "Paper": 101,
    "M1": 202,
    "M2": 303,
    "M2pro": 404,
    "EXP0": 505,
    "EXP1": 606,
    "EXP2": 707,
    "EXP3": 808,
}

N_TRIALS_PER_METHOD = 30
PRINT_EVERY = 1

T_EVAL_MIN = 20.0
T_EVAL_MAX = 80.0

TMIN_NORM = -50.0
TMAX_NORM = 100.0

EXP_PMIN = 0.8
EXP_PMAX = 4.0

M2PRO_PMIN = 1.2
M2PRO_PMAX = 3.0

# 适配步数/学习率（固定，不作为trial变量）
ADAPT_ITERS = 80
ADAPT_LR = 0.06

# 数值安全
EXP_CLIP = 60.0  # exp(60)很大，足够当“无穷”避免溢出


# =========================
# ======= UTILS ===========
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

def robust_z_scores(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return np.full_like(v, np.nan)
    med = np.median(vv)
    mad = robust_mad(v)
    z = 0.6745 * (v - med) / (mad if mad > 1e-12 else 1e-12)
    return z

def z_robust_scalar(x: float, med: float, mad: float) -> float:
    mad = mad if (mad is not None and np.isfinite(mad) and mad > 1e-12) else 1e-12
    return 0.6745 * (x - med) / mad


# =========================
# ======= PARSER ==========
# =========================
# 同时兼容旧 EPC（如 0001）和新 EPC（如 C201）。
# 允许 EPC、Temp_data、Time_data 行之间存在空格或空行，
# 并兼容数组末尾有无分号的情况。
_TAG_PATTERN = re.compile(
    r"(?P<epc>[A-Za-z0-9]{4})[ \t]*\r?\n"
    r"\s*Temp_data\s*=\s*\[(?P<temp>[^\]]+)\]\s*;?"
    r"\s*\r?\n"
    r"\s*Time_data\s*=\s*\[(?P<time>[^\]]+)\]\s*;?",
    re.MULTILINE
)

def _parse_num_list(s: str) -> np.ndarray:
    """解析逗号或空格分隔的数值数组。"""
    s = s.replace(",", " ")
    values = [x for x in s.split() if x.strip()]
    return np.array([float(x) for x in values], dtype=np.float64)

def load_tags_from_txt(path: str) -> List[Dict[str, Any]]:
    """从文本中读取标签数据，兼容 EPC=0001 和 EPC=C201 两种格式。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据文件不存在：{path}")

    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        txt = f.read()

    matches = list(_TAG_PATTERN.finditer(txt))
    if not matches:
        preview = txt[:800].replace("\r", "\\r").replace("\n", "\\n\n")
        raise RuntimeError(
            "No tags parsed. Check file format.\n"
            f"数据文件：{path}\n"
            "当前解析器支持如下格式：\n"
            "C201\n"
            "Temp_data = [20, 30, 40];\n"
            "Time_data = [0.8, 0.7, 0.6];\n"
            f"文件前 800 个字符：\n{preview}"
        )

    tags: List[Dict[str, Any]] = []
    seen_epcs = set()

    for match in matches:
        epc = match.group("epc").strip().upper()
        temp_text = match.group("temp")
        time_text = match.group("time")

        T = _parse_num_list(temp_text)
        y = _parse_num_list(time_text)

        if T.size == 0 or y.size == 0:
            raise ValueError(
                f"EPC={epc} 的 Temp_data 或 Time_data 为空："
                f"Temp={T.size}, Time={y.size}"
            )

        if T.size != y.size:
            raise ValueError(
                f"EPC={epc} 的温度点和放电时间点数量不一致："
                f"Temp_data={T.size}, Time_data={y.size}"
            )

        if not np.all(np.isfinite(T)):
            raise ValueError(f"EPC={epc} 的 Temp_data 中存在非有限数值。")

        if not np.all(np.isfinite(y)):
            raise ValueError(f"EPC={epc} 的 Time_data 中存在非有限数值。")

        if epc in seen_epcs:
            raise ValueError(f"发现重复 EPC：{epc}")

        seen_epcs.add(epc)
        tags.append({"EPC": epc, "T": T, "y": y})

    print(f"成功解析标签数量：{len(tags)}")
    print("前几个 EPC：", [tag["EPC"] for tag in tags[:10]])

    return tags


# =========================
# ======= METHODS =========
# =========================
class Method:
    name: str

    def q0(self) -> torch.Tensor:
        raise NotImplementedError

    def theta_from_q(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def theta_to_float(self, th: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {k: float(v.detach().cpu().item()) for k, v in th.items()}

    def basis_torch(self, th: Dict[str, torch.Tensor], T: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def basis_np(self, th: Dict[str, float], T: float) -> float:
        raise NotImplementedError

    def invert_np(self, th: Dict[str, float], s: float, yobs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def profiled_s_np(self, th: Dict[str, float], T: np.ndarray, y: np.ndarray) -> float:
        b = np.array([self.basis_np(th, float(t)) for t in T], dtype=np.float64)
        den = float(np.dot(b, b) + 1e-12)
        return float(np.dot(b, y) / den)

    def s_from_one_point_np(self, th: Dict[str, float], T0: float, y0: float) -> float:
        bas = self.basis_np(th, T0)
        if (not np.isfinite(bas)) or bas <= 0:
            return float("nan")
        return float(y0 / bas)


class MethodPaper(Method):
    name = "Paper"

    def q0(self) -> torch.Tensor:
        return torch.tensor([math.log(10.0), math.log(15.0)], dtype=torch.float32)

    def theta_from_q(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"b": torch.exp(q[0]), "c": torch.exp(q[1])}

    def basis_torch(self, th: Dict[str, torch.Tensor], T: torch.Tensor) -> torch.Tensor:
        b = th["b"]; c = th["c"]
        z = clamp_pow_arg(T / (c + 1e-12))
        pow2 = torch.pow(torch.tensor(2.0, device=T.device), z)
        return 1.0 / (pow2 + b)

    def basis_np(self, th: Dict[str, float], T: float) -> float:
        b = float(th["b"]); c = float(th["c"])
        return 1.0 / (safe_pow2(T / c) + b)

    def invert_np(self, th: Dict[str, float], s: float, yobs: np.ndarray) -> np.ndarray:
        b = float(th["b"]); c = float(th["c"])
        y = np.asarray(yobs, dtype=np.float64)
        Tout = np.full_like(y, np.nan, dtype=np.float64)
        if s <= 0 or b <= 0 or c <= 0:
            return Tout
        v = s / y - b
        ok = np.isfinite(v) & (v >= 1.0) & np.isfinite(y) & (y > 0)
        Tout[ok] = c * (np.log(v[ok]) / np.log(2.0))
        return Tout


class MethodM1(Method):
    name = "M1"

    def q0(self) -> torch.Tensor:
        return torch.tensor([math.log(6.0), math.log(1.0)], dtype=torch.float32)

    def theta_from_q(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"k": torch.exp(q[0]), "b": torch.exp(q[1])}

    def basis_torch(self, th: Dict[str, torch.Tensor], T: torch.Tensor) -> torch.Tensor:
        k = th["k"]; b = th["b"]
        x = torch.clamp((T - TMIN_NORM) / (TMAX_NORM - TMIN_NORM), 0.0, 1.0)
        z = clamp_pow_arg(k * x)
        pow2 = torch.pow(torch.tensor(2.0, device=T.device), z)
        return 1.0 / (pow2 + b)

    def basis_np(self, th: Dict[str, float], T: float) -> float:
        k = float(th["k"]); b = float(th["b"])
        x = x_norm_np(T)
        return 1.0 / (safe_pow2(k * x) + b)

    def invert_np(self, th: Dict[str, float], s: float, yobs: np.ndarray) -> np.ndarray:
        k = float(th["k"]); b = float(th["b"])
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


class MethodM2(Method):
    name = "M2"

    def q0(self) -> torch.Tensor:
        return torch.tensor([math.log(3.0), math.log(10.0)], dtype=torch.float32)

    def theta_from_q(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"lambda": torch.exp(q[0]), "B": torch.exp(q[1])}

    def basis_torch(self, th: Dict[str, torch.Tensor], T: torch.Tensor) -> torch.Tensor:
        lam = th["lambda"]; B = th["B"]
        x = torch.clamp((T - TMIN_NORM) / (TMAX_NORM - TMIN_NORM), 0.0, 1.0)
        u = x * x
        z = clamp_pow_arg(lam * u)
        return 1.0 / (torch.exp(z) + B)

    def basis_np(self, th: Dict[str, float], T: float) -> float:
        lam = float(th["lambda"]); B = float(th["B"])
        x = x_norm_np(T)
        return 1.0 / (safe_exp(lam * (x * x)) + B)

    def invert_np(self, th: Dict[str, float], s: float, yobs: np.ndarray) -> np.ndarray:
        lam = float(th["lambda"]); B = float(th["B"])
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


class MethodM2pro(Method):
    name = "M2pro"

    def q0(self) -> torch.Tensor:
        return torch.tensor([math.log(3.0), math.log(10.0), 0.0], dtype=torch.float32)

    def theta_from_q(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        lam = torch.exp(q[0]); B = torch.exp(q[1])
        sp = sigmoid_torch(q[2])
        p = torch.tensor(M2PRO_PMIN, device=q.device) + (torch.tensor(M2PRO_PMAX, device=q.device) - torch.tensor(M2PRO_PMIN, device=q.device)) * sp
        return {"lambda": lam, "B": B, "p": p}

    def basis_torch(self, th: Dict[str, torch.Tensor], T: torch.Tensor) -> torch.Tensor:
        lam = th["lambda"]; B = th["B"]; p = th["p"]
        x = torch.clamp((T - TMIN_NORM) / (TMAX_NORM - TMIN_NORM), 0.0, 1.0)
        u = torch.pow(x, p)
        z = clamp_pow_arg(lam * u)
        return 1.0 / (torch.exp(z) + B)

    def basis_np(self, th: Dict[str, float], T: float) -> float:
        lam = float(th["lambda"]); B = float(th["B"]); p = float(th["p"])
        x = x_norm_np(T)
        return 1.0 / (safe_exp(lam * (x ** p)) + B)

    def invert_np(self, th: Dict[str, float], s: float, yobs: np.ndarray) -> np.ndarray:
        lam = float(th["lambda"]); B = float(th["B"]); p = float(th["p"])
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


class MethodEXP(Method):
    def __init__(self, mode: str):
        self.name = mode
        self.mode = mode

    def q0(self) -> torch.Tensor:
        base = [math.log(2.0), math.log(1.0), 0.0, 0.0, 0.0, 0.0]
        if self.mode == "EXP0":
            return torch.tensor(base, dtype=torch.float32)
        if self.mode == "EXP1":
            return torch.tensor(base + [0.0], dtype=torch.float32)
        if self.mode == "EXP2":
            return torch.tensor(base + [0.0, 0.0], dtype=torch.float32)
        if self.mode == "EXP3":
            return torch.tensor(base + [0.0, 0.0], dtype=torch.float32)
        return torch.tensor(base, dtype=torch.float32)

    def theta_from_q(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        lam = torch.exp(q[0]); B = torch.exp(q[1])
        a0, a1, a2, a3 = q[2], q[3], q[4], q[5]
        th = {"lambda": lam, "B": B, "a0": a0, "a1": a1, "a2": a2, "a3": a3}
        if self.mode == "EXP1":
            sp = sigmoid_torch(q[6]); th["delta"] = -0.8 + 1.6 * sp
        elif self.mode == "EXP2":
            sp = sigmoid_torch(q[6]); th["delta"] = -0.8 + 1.6 * sp
            spg = sigmoid_torch(q[7]); th["gamma"] = 0.6 + 0.8 * spg
        elif self.mode == "EXP3":
            sp = sigmoid_torch(q[6]); th["delta"] = -0.8 + 1.6 * sp
            spt = sigmoid_torch(q[7]); th["tau"] = -1.0 + 2.0 * spt
        return th

    def _pbase_torch(self, th: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        g = th["a0"] + th["a1"] * x + th["a2"] * x * x + th["a3"] * x * x * x
        return torch.tensor(EXP_PMIN, device=x.device) + (torch.tensor(EXP_PMAX, device=x.device) - torch.tensor(EXP_PMIN, device=x.device)) * sigmoid_torch(g)

    def _ptag_torch(self, th: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        pb = self._pbase_torch(th, x)
        if self.mode == "EXP0":
            pt = pb
        elif self.mode == "EXP1":
            pt = pb + th["delta"]
        elif self.mode == "EXP2":
            pt = th["gamma"] * pb + th["delta"]
        elif self.mode == "EXP3":
            pt = pb + th["delta"] + th["tau"] * (x - 0.5)
        else:
            pt = pb
        return torch.clamp(pt, EXP_PMIN, EXP_PMAX)

    def basis_torch(self, th: Dict[str, torch.Tensor], T: torch.Tensor) -> torch.Tensor:
        lam = th["lambda"]; B = th["B"]
        x = torch.clamp((T - TMIN_NORM) / (TMAX_NORM - TMIN_NORM), 0.0, 1.0)
        ptag = self._ptag_torch(th, x)
        u = torch.pow(x, ptag)
        z = clamp_pow_arg(lam * u)
        return 1.0 / (torch.exp(z) + B)

    def _pbase_np(self, th: Dict[str, float], x: float) -> float:
        a0 = float(th["a0"]); a1 = float(th["a1"]); a2 = float(th["a2"]); a3 = float(th["a3"])
        g = a0 + a1 * x + a2 * x * x + a3 * x * x * x
        return float(EXP_PMIN + (EXP_PMAX - EXP_PMIN) * sigmoid_np(g))

    def _ptag_np(self, th: Dict[str, float], x: float) -> float:
        pb = self._pbase_np(th, x)
        delta = float(th.get("delta", 0.0))
        gamma = float(th.get("gamma", 1.0))
        tau = float(th.get("tau", 0.0))
        if self.mode == "EXP0":
            pt = pb
        elif self.mode == "EXP1":
            pt = pb + delta
        elif self.mode == "EXP2":
            pt = gamma * pb + delta
        elif self.mode == "EXP3":
            pt = pb + delta + tau * (x - 0.5)
        else:
            pt = pb
        pt = min(max(pt, EXP_PMIN), EXP_PMAX)
        return float(pt)

    def basis_np(self, th: Dict[str, float], T: float) -> float:
        lam = float(th["lambda"]); B = float(th["B"])
        x = x_norm_np(T)
        ptag = self._ptag_np(th, x)
        return 1.0 / (safe_exp(lam * (x ** ptag)) + B)

    def invert_np(self, th: Dict[str, float], s: float, yobs: np.ndarray) -> np.ndarray:
        y = np.asarray(yobs, dtype=np.float64)
        Tout = np.full_like(y, np.nan, dtype=np.float64)
        if s <= 0:
            return Tout
        lam = float(th["lambda"]); B = float(th["B"])
        if lam <= 0 or B <= 0:
            return Tout

        def y_model(xx: float) -> float:
            xx = min(max(xx, 0.0), 1.0)
            ptag = self._ptag_np(th, xx)
            den = safe_exp(lam * (xx ** ptag)) + B
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


def build_methods() -> List[Method]:
    return [
        MethodPaper(),
        MethodM1(),
        MethodM2(),
        MethodM2pro(),
        MethodEXP("EXP0"),
        MethodEXP("EXP1"),
        MethodEXP("EXP2"),
        MethodEXP("EXP3"),
    ]


# =========================
# ======= LEARN theta0 =====
# =========================
def learn_theta0_profiled(method: Method,
                          tags: List[Dict[str, Any]],
                          idx_list: List[int],
                          hp: Dict[str, Any],
                          device: torch.device) -> Dict[str, float]:
    q = torch.nn.Parameter(method.q0().to(device))
    opt = torch.optim.Adam([q], lr=float(hp["learn_lr"]))

    Ts = []
    ys = []
    for idx in idx_list:
        Ts.append(torch.tensor(tags[idx]["T"], dtype=torch.float32, device=device))
        ys.append(torch.tensor(tags[idx]["y"], dtype=torch.float32, device=device))

    alpha_p = float(hp.get("alpha_p", 0.0))
    alpha_smooth = float(hp.get("alpha_smooth", 0.0))

    for _ in range(int(hp["learn_iters"])):
        opt.zero_grad()
        th = method.theta_from_q(q)

        sse = torch.tensor(0.0, device=device)
        for T, y in zip(Ts, ys):
            b = method.basis_torch(th, T)
            b = torch.clamp(b, 1e-12, 1e12)
            s = torch.dot(b, y) / (torch.dot(b, b) + 1e-12)
            r = y - s * b
            sse = sse + torch.sum(r * r)

        if isinstance(method, MethodM2pro) and alpha_p > 0:
            p = th["p"]
            sse = sse + alpha_p * (p - 2.0) * (p - 2.0)

        if isinstance(method, MethodEXP) and alpha_smooth > 0:
            xg = torch.linspace(0.0, 1.0, steps=60, device=device)
            pb = method._pbase_torch(th, xg)
            d2 = pb[2:] - 2 * pb[1:-1] + pb[:-2]
            sse = sse + alpha_smooth * torch.sum(d2 * d2)

        if not torch.isfinite(sse).item():
            break

        sse.backward()
        opt.step()

    th_final = method.theta_from_q(q.detach())
    return method.theta_to_float(th_final)


def compute_tag_rmse(method: Method,
                     theta0: Dict[str, float],
                     tags: List[Dict[str, Any]],
                     idx_list: List[int]) -> np.ndarray:
    rmse = np.full((len(idx_list),), np.nan, dtype=np.float64)
    for k, idx in enumerate(idx_list):
        T = tags[idx]["T"]
        y = tags[idx]["y"]
        b = np.array([method.basis_np(theta0, float(t)) for t in T], dtype=np.float64)
        den = float(np.dot(b, b) + 1e-12)
        s = float(np.dot(b, y) / den)
        r = y - s * b
        rmse[k] = float(np.sqrt(np.mean(r * r)))
    return rmse


def filter_tags(method: Method,
                tags: List[Dict[str, Any]],
                hp: Dict[str, Any],
                device: torch.device) -> Tuple[List[int], Dict[str, float]]:
    min_points = int(hp["min_points"])

    ok = []
    for i, tg in enumerate(tags):
        T = tg["T"]; y = tg["y"]
        if T.size >= min_points and T.size == y.size and np.all(np.isfinite(T)) and np.all(np.isfinite(y)):
            ok.append(i)

    kept = ok[:]
    if len(kept) < 4:
        theta0 = learn_theta0_profiled(method, tags, kept, hp, device)
        return kept, theta0

    for _ in range(int(hp["filter_iter"])):
        theta0 = learn_theta0_profiled(method, tags, kept, hp, device)
        rmse = compute_tag_rmse(method, theta0, tags, kept)
        z = robust_z_scores(rmse)
        bad = (np.abs(z) > float(hp["z_rmse_th"])) | (rmse > float(hp["rmse_abs_max"])) | (~np.isfinite(rmse))
        new_kept = [kept[i] for i in range(len(kept)) if not bad[i]]
        if len(new_kept) < 4 or len(new_kept) == len(kept):
            kept = new_kept
            break
        kept = new_kept

    theta0 = learn_theta0_profiled(method, tags, kept, hp, device)
    rmse = compute_tag_rmse(method, theta0, tags, kept)
    z = robust_z_scores(rmse)
    bad = (np.abs(z) > float(hp["z_rmse_th"])) | (rmse > float(hp["rmse_abs_max"])) | (~np.isfinite(rmse))
    kept = [kept[i] for i in range(len(kept)) if not bad[i]]
    if len(kept) < 2:
        kept = ok[:]

    return kept, theta0


# =========================
# ======= SchemeB adapt =====
# =========================
def schemeB_adapt_theta(method: Method,
                        theta0: Dict[str, float],
                        train_tags: List[Dict[str, Any]],
                        mu_log_s: float,
                        sig_log_s: float,
                        hp: Dict[str, Any],
                        T0: float,
                        y0: float,
                        device: torch.device) -> Tuple[Dict[str, float], float]:
    """
    方案B：在单点注册时，不只调 s，而是让该点参与更新 theta（用训练标签拟合约束住theta）
    Loss = alpha_train*TrainSSE(theta) + alpha_sprior*( (log(s_hat)-mu)/sig )^2 + alpha_theta*||q-q0||^2
    """
    # build q0 from theta0
    if method.name == "Paper":
        q0 = torch.tensor([math.log(theta0["b"]), math.log(theta0["c"])], dtype=torch.float32, device=device)
    elif method.name == "M1":
        q0 = torch.tensor([math.log(theta0["k"]), math.log(theta0["b"])], dtype=torch.float32, device=device)
    elif method.name == "M2":
        q0 = torch.tensor([math.log(theta0["lambda"]), math.log(theta0["B"])], dtype=torch.float32, device=device)
    elif method.name == "M2pro":
        p = float(theta0["p"])
        p = min(max(p, M2PRO_PMIN + 1e-6), M2PRO_PMAX - 1e-6)
        sp = (p - M2PRO_PMIN) / (M2PRO_PMAX - M2PRO_PMIN)
        qp = math.log(sp / (1.0 - sp))
        q0 = torch.tensor([math.log(theta0["lambda"]), math.log(theta0["B"]), qp], dtype=torch.float32, device=device)
    else:
        # EXP*
        base = [math.log(theta0["lambda"]), math.log(theta0["B"]),
                float(theta0["a0"]), float(theta0["a1"]), float(theta0["a2"]), float(theta0["a3"])]
        if method.name == "EXP0":
            q0 = torch.tensor(base, dtype=torch.float32, device=device)
        elif method.name == "EXP1":
            d = float(theta0.get("delta", 0.0))
            d = max(min(d, 0.799999), -0.799999)
            sp = (d + 0.8) / 1.6
            q0 = torch.tensor(base + [math.log(sp / (1.0 - sp))], dtype=torch.float32, device=device)
        elif method.name == "EXP2":
            d = float(theta0.get("delta", 0.0))
            d = max(min(d, 0.799999), -0.799999)
            sp = (d + 0.8) / 1.6
            g = float(theta0.get("gamma", 1.0))
            g = min(max(g, 0.600001), 1.399999)
            spg = (g - 0.6) / 0.8
            q0 = torch.tensor(base + [math.log(sp / (1.0 - sp)), math.log(spg / (1.0 - spg))], dtype=torch.float32, device=device)
        elif method.name == "EXP3":
            d = float(theta0.get("delta", 0.0))
            d = max(min(d, 0.799999), -0.799999)
            sp = (d + 0.8) / 1.6
            tau = float(theta0.get("tau", 0.0))
            tau = max(min(tau, 0.999999), -0.999999)
            spt = (tau + 1.0) / 2.0
            q0 = torch.tensor(base + [math.log(sp / (1.0 - sp)), math.log(spt / (1.0 - spt))], dtype=torch.float32, device=device)
        else:
            q0 = torch.tensor(base, dtype=torch.float32, device=device)

    q = torch.nn.Parameter(q0.clone())
    opt = torch.optim.Adam([q], lr=ADAPT_LR)

    alpha_train = float(hp["alpha_train"])
    alpha_theta = float(hp["alpha_theta"])
    alpha_sprior = float(hp["alpha_sprior"])

    mu = torch.tensor(float(mu_log_s), dtype=torch.float32, device=device)
    sig = torch.tensor(float(max(sig_log_s, 1e-6)), dtype=torch.float32, device=device)
    T0_t = torch.tensor(float(T0), dtype=torch.float32, device=device)
    y0_t = torch.tensor(float(y0), dtype=torch.float32, device=device)

    # pack train tensors once
    train_Ts = [torch.tensor(tg["T"], dtype=torch.float32, device=device) for tg in train_tags]
    train_ys = [torch.tensor(tg["y"], dtype=torch.float32, device=device) for tg in train_tags]

    alpha_p = float(hp.get("alpha_p", 0.0))
    alpha_smooth = float(hp.get("alpha_smooth", 0.0))

    for _ in range(ADAPT_ITERS):
        opt.zero_grad()
        th = method.theta_from_q(q)

        train_sse = torch.tensor(0.0, device=device)
        for Ttr, ytr in zip(train_Ts, train_ys):
            b = method.basis_torch(th, Ttr)
            b = torch.clamp(b, 1e-12, 1e12)
            s_i = torch.dot(b, ytr) / (torch.dot(b, b) + 1e-12)
            r = ytr - s_i * b
            train_sse = train_sse + torch.sum(r * r)

        bas0 = method.basis_torch(th, T0_t.view(1)).view(())
        bas0 = torch.clamp(bas0, 1e-12, 1e12)
        s_hat = torch.clamp(y0_t / bas0, 1e-12, 1e18)

        loss_train = alpha_train * train_sse
        loss_theta = alpha_theta * torch.sum((q - q0) * (q - q0))
        loss_s = alpha_sprior * torch.pow((torch.log(s_hat) - mu) / sig, 2.0)
        loss = loss_train + loss_theta + loss_s

        if isinstance(method, MethodM2pro) and alpha_p > 0:
            p = th["p"]
            loss = loss + alpha_p * (p - 2.0) * (p - 2.0)

        if isinstance(method, MethodEXP) and alpha_smooth > 0:
            xg = torch.linspace(0.0, 1.0, steps=60, device=device)
            pb = method._pbase_torch(th, xg)
            d2 = pb[2:] - 2 * pb[1:-1] + pb[:-2]
            loss = loss + alpha_smooth * torch.sum(d2 * d2)

        if not torch.isfinite(loss).item():
            break

        loss.backward()
        opt.step()

    th_final = method.theta_from_q(q.detach())
    theta_adapt = method.theta_to_float(th_final)
    s_hat_f = method.s_from_one_point_np(theta_adapt, float(T0), float(y0))
    return theta_adapt, float(s_hat_f)


# =========================
# ======= EVALUATION ======
# =========================
def evaluate_one_method_schemeB(method: Method,
                                tags: List[Dict[str, Any]],
                                kept_idx: List[int],
                                theta0: Dict[str, float],
                                hp: Dict[str, Any],
                                device: torch.device) -> Tuple[float, Dict[str, Any]]:
    """
    LOTO：每个标签轮流当新标签；每个温度点轮流当注册点。
    训练集：kept_idx 中排除当前 test 标签（如果在其中）
    """
    beta = float(hp["beta"])
    penalty_accept = float(hp.get("penalty_accept", 5.0))

    n_total = 0
    n_accept = 0
    mean_errs = []
    max_errs = []

    for test_i, tg in enumerate(tags):
        T = tg["T"]; y = tg["y"]
        if T.size < int(hp["min_points"]) or T.size != y.size:
            continue

        mask_eval = (T >= T_EVAL_MIN) & (T <= T_EVAL_MAX)
        T_true = T[mask_eval]
        y_obs = y[mask_eval]
        if T_true.size < 1:
            continue

        train_idx = [i for i in kept_idx if i != test_i]
        if len(train_idx) < 2:
            train_idx = [i for i in kept_idx]  # 兜底

        # 训练集 tag list
        train_tags = [tags[i] for i in train_idx]

        # 用训练集计算 log(s) 先验
        s_train = []
        for i in train_idx:
            s_i = method.profiled_s_np(theta0, tags[i]["T"], tags[i]["y"])
            if np.isfinite(s_i) and s_i > 0:
                s_train.append(s_i)
        s_train = np.asarray(s_train, dtype=np.float64)
        if s_train.size < 2:
            mu_log_s, sig_log_s = 0.0, 1.0
        else:
            ls = np.log(s_train)
            mu_log_s = float(np.mean(ls))
            sig_log_s = float(np.std(ls) + 1e-6)

        med_s = float(np.median(s_train)) if s_train.size else 1.0
        mad_s = robust_mad(s_train) if s_train.size else 1.0

        for j in range(T.size):
            T0 = float(T[j]); y0 = float(y[j])
            n_total += 1

            # adapt theta using schemeB
            try:
                theta_adapt, s_hat = schemeB_adapt_theta(method, theta0, train_tags, mu_log_s, sig_log_s, hp, T0, y0, device)
            except Exception:
                continue

            if (not np.isfinite(s_hat)) or s_hat <= 0:
                continue

            # zS using training s distribution
            zS = z_robust_scalar(s_hat, med_s, mad_s)

            # zY using y_train0 = s_train * basis(theta0, T0)
            bas0 = method.basis_np(theta0, T0)
            if (not np.isfinite(bas0)) or bas0 <= 0 or s_train.size < 2:
                continue
            y_train0 = s_train * bas0
            med_y0 = float(np.median(y_train0))
            mad_y0 = robust_mad(y_train0)
            zY = z_robust_scalar(y0, med_y0, mad_y0)

            if abs(zS) > float(hp["z_thresh_s"]) or abs(zY) > float(hp["z_thresh_y"]):
                continue

            # error vs existing points
            T_hat = method.invert_np(theta_adapt, float(s_hat), y_obs)
            err = np.abs(T_hat - T_true)
            err = err[np.isfinite(err)]
            if err.size == 0:
                continue

            n_accept += 1
            mean_errs.append(float(np.mean(err)))
            max_errs.append(float(np.max(err)))

    if n_accept == 0:
        return 1e9, {"reason": "no_accept", "n_total": n_total, "n_accept": n_accept}

    mean_mean = float(np.mean(mean_errs))
    mean_max = float(np.mean(max_errs))
    acc_rate = float(n_accept / max(n_total, 1))
    obj = beta * mean_mean + (1.0 - beta) * mean_max + penalty_accept * (1.0 - acc_rate)

    if (not np.isfinite(obj)) or (obj != obj):
        obj = 1e9

    return float(obj), {
        "n_total": n_total,
        "n_accept": n_accept,
        "accept_rate": acc_rate,
        "meanErr_mean": mean_mean,
        "maxErr_mean": mean_max,
    }


# =========================
# ======= HP SAMPLER ======
# =========================
def sample_hp(rs: np.random.RandomState, method_name: str) -> Dict[str, Any]:
    hp = {
        "min_points": int(rs.randint(5, 9)),
        "rmse_abs_max": float(rs.uniform(0.04, 0.12)),
        "z_rmse_th": float(rs.uniform(2.0, 4.5)),
        "filter_iter": int(rs.randint(1, 4)),
        "learn_iters": int(rs.randint(200, 901)),
        "learn_lr": float(10 ** rs.uniform(math.log10(0.01), math.log10(0.15))),
        "z_thresh_s": float(rs.uniform(2.5, 4.8)),
        "z_thresh_y": float(rs.uniform(2.5, 4.8)),
        "beta": float(rs.uniform(0.2, 0.9)),
        # SchemeB: 三个关键权重
        "alpha_train": float(10 ** rs.uniform(math.log10(0.1), math.log10(10.0))),
        "alpha_theta": float(10 ** rs.uniform(math.log10(0.01), math.log10(5.0))),
        "alpha_sprior": float(10 ** rs.uniform(math.log10(0.1), math.log10(20.0))),
        "penalty_accept": 5.0,
    }
    if method_name == "M2pro":
        hp["alpha_p"] = float(rs.uniform(0.0, 200.0))
    if method_name.startswith("EXP"):
        hp["alpha_smooth"] = float(rs.uniform(0.0, 50.0))
    return hp


# =========================
# ======= TUNING LOOP ======
# =========================
def tune_one_method(method: Method, tags: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    rs = np.random.RandomState(SEED + METHOD_SEED_OFFSETS[method.name])

    best_obj = 1e18
    best_pack = None

    print(f"\n=== Tuning method={method.name} (SchemeB) ===")
    for t in range(N_TRIALS_PER_METHOD):
        hp = sample_hp(rs, method.name)

        try:
            kept_idx, theta0 = filter_tags(method, tags, hp, device)
            obj, summary = evaluate_one_method_schemeB(method, tags, kept_idx, theta0, hp, device)
            if (not np.isfinite(obj)) or (obj != obj):
                obj = 1e9
        except Exception:
            obj, summary = 1e9, {"reason": "exception"}
            kept_idx, theta0 = [], {}

        if obj < best_obj:
            best_obj = obj
            kept_epcs = [tags[i]["EPC"] for i in kept_idx] if kept_idx else []
            # prior(log s)基于kept集（用于后续导出/评估脚本）
            s_kept = []
            for i in kept_idx:
                s_i = method.profiled_s_np(theta0, tags[i]["T"], tags[i]["y"])
                if np.isfinite(s_i) and s_i > 0:
                    s_kept.append(s_i)
            s_kept = np.asarray(s_kept, dtype=np.float64)
            if s_kept.size >= 2:
                ls = np.log(s_kept)
                mu_log_s = float(np.mean(ls))
                sig_log_s = float(np.std(ls) + 1e-6)
            else:
                mu_log_s, sig_log_s = 0.0, 1.0

            best_pack = {
                "best_objective": float(best_obj),
                "kept_count": int(len(kept_idx)),
                "kept_epcs": kept_epcs,
                "theta0": theta0,
                "prior_log_s": {"mu_log_s": mu_log_s, "sig_log_s": sig_log_s},
                "hp": hp,
                "eval": summary,
            }

        if (t + 1) % PRINT_EVERY == 0:
            print(f"[{method.name}] trial={t+1}/{N_TRIALS_PER_METHOD} obj={obj:.6g} best={best_obj:.6g} kept={len(kept_idx)}")

    return best_pack if best_pack is not None else {
        "best_objective": 1e9,
        "kept_count": 0,
        "kept_epcs": [],
        "theta0": {},
        "prior_log_s": {"mu_log_s": 0.0, "sig_log_s": 1.0},
        "hp": {},
        "eval": {"reason": "no_success"},
    }


def main():
    # 八个项目并行运行时，每个进程使用一个 PyTorch CPU 线程。
    # 如只单独运行一个项目，可把 1 改为合适的线程数。
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ensure_dir(OUT_DIR)

    device = torch.device("cpu")
    tags = load_tags_from_txt(RAW_DATA_PATH)
    print(f"Loaded dataset=rraw path={RAW_DATA_PATH}")
    print(f"  tags={len(tags)}  device={device}")

    methods = build_methods()

    results = {
        "scheme": "B",
        "raw_data_path": RAW_DATA_PATH,
        "seed": SEED,
        "n_trials_per_method": N_TRIALS_PER_METHOD,
        "device": str(device),
        "TMIN_NORM": TMIN_NORM,
        "TMAX_NORM": TMAX_NORM,
        "T_EVAL_MIN": T_EVAL_MIN,
        "T_EVAL_MAX": T_EVAL_MAX,
        "methods": {}
    }

    for m in methods:
        pack = tune_one_method(m, tags, device)
        results["methods"][m.name] = pack

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n[SAVED]", RESULT_JSON)


if __name__ == "__main__":
    main()
