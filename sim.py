#!/usr/bin/env python3
"""h001 有限样本敏感性 Monte Carlo。

复现命令：python3 sim.py

DGP（一维等距设计，A2 的特例）：
  x, u 独立的空间 LTU 过程，协方差 K(c)=exp(-c|i-j|)/(2c)（A1, A5）；
  y = 1 + x + u（β0=1, α=0）；位置 s_i=i（A2）。
  用 AR(1) 递推精确生成（O(n)），不依赖 Cholesky。
估计量：
  β̂(a) = x̃'Σ(a)^+ỹ / (x̃'Σ(a)^+x̃)，Σ(a)=M K(a) M。
  a=0 是 LBM-GLS；a=c 是 oracle；a→∞ 是 OLS（naive）。
结果：
  results.csv · d1.csv · d2.csv · d3.csv · d4.csv · negative.csv
  coverage.png · bias.png · rate.png · local_expansion.png · qq.png
  图内不写标题（图题归论文）。
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.linalg import eigh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

BETA0 = 1.0
SIGMA_X = 1.0
SIGMA_U = 1.0
MASTER_SEED = 20261022

N_GRID = [25, 50, 100, 200, 400]
C_GRID = [0.05, 0.10, 0.20]
C_SUP = [0.05 + 0.025 * i for i in range(7)]   # 7 点网格近似 C=[0.05,0.20] 的 sup
A_RATIOS = [0.0, 0.5, 1.0]          # a = ratio * c
R = 1000


# ── DGP：一维 AR(1) 精确生成 ────────────────────────────────────────────
def ar1(n: int, c: float, rng) -> np.ndarray:
    """协方差 exp(-c|i-j|)/(2c) 的平稳 AR(1)。"""
    phi = np.exp(-c)
    var = 1.0 / (2.0 * c)
    z = np.empty(n)
    z[0] = rng.normal(0.0, np.sqrt(var))
    eps = rng.normal(0.0, 1.0, n)
    innov = np.sqrt(var * (1.0 - phi**2))
    for i in range(1, n):
        z[i] = phi * z[i - 1] + innov * eps[i]
    return z


def locations(n: int) -> np.ndarray:
    return np.arange(n, dtype=float)


@lru_cache(maxsize=256)
def Kmat(n: int, c: float) -> np.ndarray:
    s = locations(n)
    d = np.abs(s[:, None] - s[None, :])
    return np.exp(-c * d) / (2.0 * c)


@lru_cache(maxsize=512)
def sigma(n: int, c: float) -> np.ndarray:
    M = np.eye(n) - np.ones((n, n)) / n
    if c <= 0.0:
        s = locations(n)
        d = np.abs(s[:, None] - s[None, :])
        return -0.5 * M @ d @ M
    return M @ Kmat(n, c) @ M


def pinv_spd(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    w, V = eigh(A)
    mx = max(float(np.max(np.abs(w))), 1.0)
    mask = w > tol * mx
    return (V[:, mask] / w[mask]) @ V[:, mask].T


@lru_cache(maxsize=32)
def basis_perp(n: int) -> np.ndarray:
    rng = np.random.default_rng(12345)
    A = np.column_stack([np.ones(n), rng.normal(size=(n, n - 1))])
    Q, _ = np.linalg.qr(A)
    return Q[:, 1:]


def kappa_n(n: int, a: float, c: float, Q: np.ndarray | None = None) -> float:
    if Q is None:
        Q = basis_perp(n)
    Sa = sigma(n, a)
    Sc = sigma(n, c)
    Sa_r = Q.T @ Sa @ Q
    Sc_r = Q.T @ Sc @ Q
    w = eigh(Sc_r, Sa_r, eigvals_only=True)
    w = np.sort(w)
    mu1 = float(np.mean(w))
    mu2 = float(np.mean(w**2))
    return mu2 / (mu1**2)


def vb_n(n: int, a: float, Q: np.ndarray | None = None) -> float:
    if Q is None:
        Q = basis_perp(n)
    s = locations(n)
    d = np.abs(s[:, None] - s[None, :])
    dK = -np.exp(-a * d) * (a * d + 1.0) / (2.0 * a**2)
    M = np.eye(n) - np.ones((n, n)) / n
    Sa = M @ (np.exp(-a * d) / (2.0 * a)) @ M
    dSa = M @ dK @ M
    Sa_r = Q.T @ Sa @ Q
    dSa_r = Q.T @ dSa @ Q
    w, V = eigh(Sa_r)
    wh = (V / np.sqrt(w)) @ V.T
    B = wh @ dSa_r @ wh
    lam = np.linalg.eigvalsh((B + B.T) / 2.0)
    return float(np.mean(lam**2) - np.mean(lam) ** 2)


def beta_hat_vec(X: np.ndarray, Y: np.ndarray, Sa_plus: np.ndarray) -> np.ndarray:
    """X,Y: (R,n) → β̂(a) 的 R 个值。"""
    Xt = X - X.mean(axis=1, keepdims=True)
    Yt = Y - Y.mean(axis=1, keepdims=True)
    W = Xt @ Sa_plus
    num = np.einsum("ij,ij->i", W, Yt)
    den = np.einsum("ij,ij->i", W, Xt)
    return num / den


def vn_vec(X: np.ndarray, c: float, Sa_plus: np.ndarray, Sc: np.ndarray) -> np.ndarray:
    """条件方差 V_n(a,c;x) 的 R 个值。"""
    Xt = X - X.mean(axis=1, keepdims=True)
    W = Xt @ Sa_plus
    num = np.einsum("ij,ij->i", W, W @ Sc)
    den = np.einsum("ij,ij->i", W, Xt)
    return SIGMA_U**2 * num / (den**2)


# ── D1：协方差展开 ──────────────────────────────────────────────────────
def run_d1() -> list[dict]:
    rows = []
    for n in [50, 100, 200, 400, 800]:
        s = locations(n)
        d = np.abs(s[:, None] - s[None, :])
        M = np.eye(n) - np.ones((n, n)) / n
        SL = -0.5 * M @ d @ M
        Delta = 0.25 * M @ (d**2) @ M
        for c in [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]:
            Sc = M @ (np.exp(-c * d) / (2.0 * c)) @ M
            err = float(np.linalg.norm(Sc - SL - c * Delta, 2))
            rows.append({"n": n, "c": c, "abs_error": err,
                         "abs_over_c2": err / (c**2),
                         "rel_error": err / max(float(np.linalg.norm(SL, 2)), 1e-12)})
    return rows


# ── D3：局部展开 ────────────────────────────────────────────────────────
def run_d3() -> list[dict]:
    n, a = 200, 0.10
    Q = basis_perp(n)
    base = kappa_n(n, a, a, Q)
    vb = vb_n(n, a, Q)
    rows = []
    for delta in [0.005, 0.01, 0.02, 0.04, 0.08, 0.12]:
        k = kappa_n(n, a, a + delta, Q)
        pred = 1.0 + delta**2 * vb
        rows.append({"n": n, "a": a, "delta": delta, "kappa": k,
                     "residual": k - pred, "vb": vb, "base": base})
    return rows


# ── D4：minimax 诊断 ────────────────────────────────────────────────────
def run_d4() -> list[dict]:
    rows = []
    a_grid = np.linspace(0.0, 0.40, 9)
    for n in [50, 100, 200, 400]:
        Q = basis_perp(n)
        kappas = np.array([[kappa_n(n, a, c, Q) for c in C_SUP] for a in a_grid])
        sup = kappas.max(axis=1)
        i_pos = int(np.argmin(sup[1:])) + 1
        sup0 = float(sup[0])
        rows.append({"n": n, "A": "positive_grid_{0.05,...,0.40}",
                     "a_star": float(a_grid[i_pos]), "sup_star": float(sup[i_pos]),
                     "sup_0": sup0, "diff": float(sup[i_pos] - sup0)})
    return rows


# ── D2/D5/D6：主 Monte Carlo + 条件方差 ─────────────────────────────────
def run_mc() -> tuple[list[dict], list[dict], dict]:
    ss = np.random.SeedSequence(MASTER_SEED)
    child = ss.spawn(len(N_GRID) * len(C_GRID))
    idx = 0
    rows, d2_rows = [], []
    qq_data = None

    for n in N_GRID:
        for c in C_GRID:
            rng = np.random.default_rng(child[idx]); idx += 1
            X = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_X
            U = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_U
            Y = BETA0 + X + U
            Scs = {cp: sigma(n, cp) for cp in C_SUP}

            for ratio in A_RATIOS:
                a = ratio * c
                Sa = sigma(n, a)
                Sa_plus = pinv_spd(Sa)
                beta = beta_hat_vec(X, Y, Sa_plus)
                V = vn_vec(X, c, Sa_plus, Scs[c])
                Vbar = np.maximum.reduce([vn_vec(X, cp, Sa_plus, Scs[cp]) for cp in C_SUP])
                bias = float(np.mean(beta) - BETA0)
                bias_se = float(np.std(beta, ddof=1) / np.sqrt(R))
                rmse = float(np.sqrt(np.mean((beta - BETA0) ** 2)))
                rmse_se = float(np.std((beta - BETA0) ** 2, ddof=1) / (2 * rmse * np.sqrt(R))) if rmse > 0 else 0.0
                cover = float(np.mean(np.abs(beta - BETA0) <= 1.959963984540054 * np.sqrt(Vbar)))
                cov_se = float(np.sqrt(cover * (1 - cover) / R))
                # 逐点区间：把 a 当作真值（assumed）与用真值（oracle）两版
                Va = vn_vec(X, a, Sa_plus, sigma(n, a))
                cov_point = float(np.mean(np.abs(beta - BETA0) <= 1.959963984540054 * np.sqrt(Va)))
                cov_point_se = float(np.sqrt(cov_point * (1 - cov_point) / R))
                cov_oracle = float(np.mean(np.abs(beta - BETA0) <= 1.959963984540054 * np.sqrt(V)))
                len_point = float(np.mean(2 * 1.959963984540054 * np.sqrt(Va)))
                len_oracle = float(np.mean(2 * 1.959963984540054 * np.sqrt(V)))
                len_robust = float(np.mean(2 * 1.959963984540054 * np.sqrt(Vbar)))
                vr = Va / V
                ratio_mean = float(np.mean(vr))
                ratio_q05 = float(np.quantile(vr, 0.05))
                ratio_q95 = float(np.quantile(vr, 0.95))
                rows.append({"n": n, "c": c, "a": a, "ratio_a": ratio,
                             "bias": bias, "bias_se": bias_se,
                             "rmse": rmse, "rmse_se": rmse_se,
                             "coverage": cover, "cov_se": cov_se,
                             "mean_V": float(np.mean(V)), "mean_Vbar": float(np.mean(Vbar)),
                             "mean_Va": float(np.mean(Va)),
                             "cov_point": cov_point, "cov_point_se": cov_point_se,
                             "cov_oracle": cov_oracle,
                             "len_point": len_point, "len_oracle": len_oracle,
                             "len_robust": len_robust,
                             "ratio_mean": ratio_mean, "ratio_q05": ratio_q05,
                             "ratio_q95": ratio_q95})

            # D2：固定 x，只重抽 u
            rng_x = np.random.default_rng(MASTER_SEED + 7 * n + int(1000 * c))
            for ratio in [0.0, 0.5, 1.0]:
                a = ratio * c
                Sa_plus = pinv_spd(sigma(n, a))
                ratios = []
                std_beta = []
                for j in range(10):
                    x = ar1(n, c, rng_x) * SIGMA_X
                    uu = np.array([ar1(n, c, rng_x) for _ in range(300)]) * SIGMA_U
                    yy = BETA0 + x + uu
                    xx = np.tile(x, (len(uu), 1))
                    bb = beta_hat_vec(xx, yy, Sa_plus)
                    vv = float(vn_vec(xx[:1], c, Sa_plus, Scs[c])[0])
                    ratios.append(float(np.var(bb, ddof=1) / vv))
                    std_beta.extend(((bb - BETA0) / np.sqrt(vv)).tolist())
                d2_rows.append({"n": n, "c": c, "a": a, "ratio_a": ratio,
                                "mean_ratio": float(np.mean(ratios)),
                                "se_ratio": float(np.std(ratios, ddof=1) / np.sqrt(len(ratios)))})
                if n == 200 and abs(c - 0.10) < 1e-12 and ratio == 0.0:
                    z = np.array(std_beta)
                    qq_data = {"n": n, "c": c, "a": a,
                               "skew": float(stats.skew(z)),
                               "kurtosis": float(stats.kurtosis(z)),
                               "std": float(np.std(z, ddof=1)),
                               "ks_stat": float(stats.kstest(z, "norm").statistic),
                               "ks_p": float(stats.kstest(z, "norm").pvalue),
                               "sample": z[:2000]}
    return rows, d2_rows, qq_data or {}


# ── 负例：违反 A1（u 非高斯，t5）──────────────────────────────────────
def run_negative() -> list[dict]:
    """负例：违反 A1 的 x⊥u（内生回归量）。"""
    rows = []
    n, c, rho = 200, 0.10, 0.5
    rng = np.random.default_rng(MASTER_SEED + 91)
    X = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_X
    V = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_U
    U = rho * X + np.sqrt(1 - rho**2) * V
    Y = BETA0 + X + U
    Scs = {cp: sigma(n, cp) for cp in C_SUP}
    for ratio in [0.0, 1.0]:
        a = ratio * c
        Sa_plus = pinv_spd(sigma(n, a))
        beta = beta_hat_vec(X, Y, Sa_plus)
        Vbar = np.maximum.reduce([vn_vec(X, cp, Sa_plus, Scs[cp]) for cp in C_SUP])
        cover = float(np.mean(np.abs(beta - BETA0) <= 1.959963984540054 * np.sqrt(Vbar)))
        rows.append({"n": n, "c": c, "a": a, "rho": rho,
                     "bias": float(np.mean(beta) - BETA0),
                     "coverage": cover,
                     "cov_se": float(np.sqrt(cover * (1 - cover) / R))})
    return rows


def run_pointwise_panel() -> list[dict]:
    """D8：逐点区间在 gap 两侧的表现（a 相对 c 可大、可小）。"""
    rows = []
    for i, n in enumerate([100, 200]):
        for j, c in enumerate(C_GRID):
            rng = np.random.default_rng(MASTER_SEED + 7300 + 17 * i + j)
            X = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_X
            U = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_U
            Y = BETA0 + X + U
            for k, ratio in enumerate([0.5, 1.0, 2.0]):
                a = ratio * c
                Sa_plus = pinv_spd(sigma(n, a))
                beta = beta_hat_vec(X, Y, Sa_plus)
                Va = vn_vec(X, a, Sa_plus, sigma(n, a))     # 假定 a 为真
                Vt = vn_vec(X, c, Sa_plus, sigma(n, c))     # 真方差
                cover = float(np.mean(np.abs(beta - BETA0) <= 1.959963984540054 * np.sqrt(Va)))
                vr = Va / Vt
                rows.append({"n": n, "c": c, "a": a, "ratio_a": ratio,
                             "coverage": cover,
                             "cov_se": float(np.sqrt(cover * (1 - cover) / R)),
                             "mean_ratio": float(np.mean(vr)),
                             "ratio_q05": float(np.quantile(vr, 0.05)),
                             "ratio_q95": float(np.quantile(vr, 0.95)),
                             "len": float(np.mean(2 * 1.959963984540054 * np.sqrt(Va)))})
    return rows


def run_minimax(d4: list[dict]) -> list[dict]:
    """D7：在 minimax 权重 a*_n 上报告覆盖与区间长度（design-level a*）。"""
    a_star = {int(r["n"]): float(r["a_star"]) for r in d4}
    rows = []
    for i, n in enumerate(N_GRID):
        for j, c in enumerate(C_GRID):
            rng = np.random.default_rng(MASTER_SEED + 9100 + 31 * i + j)
            X = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_X
            U = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_U
            Y = BETA0 + X + U
            a = a_star.get(n, 0.05)
            Sa_plus = pinv_spd(sigma(n, a))
            beta = beta_hat_vec(X, Y, Sa_plus)
            V = vn_vec(X, c, Sa_plus, sigma(n, c))
            Vbar = np.maximum.reduce([vn_vec(X, cp, Sa_plus, sigma(n, cp)) for cp in C_SUP])
            cover = float(np.mean(np.abs(beta - BETA0) <= 1.959963984540054 * np.sqrt(Vbar)))
            rows.append({"n": n, "c": c, "a_star": a, "coverage": cover,
                         "cov_se": float(np.sqrt(cover * (1 - cover) / R)),
                         "mean_Vbar": float(np.mean(Vbar)),
                         "len": float(np.mean(2 * 1.959963984540054 * np.sqrt(Vbar)))})
    return rows


def run_ols() -> list[dict]:
    """naive（下界）：OLS（a→∞ 的极限，完全忽略空间协方差）。"""
    rows = []
    for n in N_GRID:
        for c in C_GRID:
            rng = np.random.default_rng(MASTER_SEED + 3000 + n + int(1000 * c))
            X = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_X
            U = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_U
            Y = BETA0 + X + U
            Xt = X - X.mean(axis=1, keepdims=True)
            Yt = Y - Y.mean(axis=1, keepdims=True)
            den = np.einsum("ij,ij->i", Xt, Xt)
            beta = np.einsum("ij,ij->i", Xt, Yt) / den
            Scs = {cp: sigma(n, cp) for cp in C_SUP}
            Vbar = np.maximum.reduce([
                SIGMA_U**2 * np.einsum("ij,ij->i", Xt, Xt @ Scs[cp]) / (den**2)
                for cp in C_SUP
            ])
            cover = float(np.mean(np.abs(beta - BETA0) <= 1.959963984540054 * np.sqrt(Vbar)))
            bias = float(np.mean(beta) - BETA0)
            rmse = float(np.sqrt(np.mean((beta - BETA0) ** 2)))
            rows.append({"n": n, "c": c, "estimator": "OLS",
                         "bias": bias,
                         "bias_se": float(np.std(beta, ddof=1) / np.sqrt(R)),
                         "rmse": rmse,
                         "coverage": cover,
                         "cov_se": float(np.sqrt(cover * (1 - cover) / R))})
    return rows


def run_hac() -> list[dict]:
    """可行对照：OLS + 空间 HAC（Bartlett 核）方差。

    与 run_ols 使用同一批抽样（同种子流），因此两表可直接对比。
    带宽：b=n//4（随 n 增长的主规则）与 b=5（短带宽敏感性）。
    """
    Z = 1.959963984540054
    rows = []
    for n in N_GRID:
        for c in C_GRID:
            rng = np.random.default_rng(MASTER_SEED + 3000 + n + int(1000 * c))
            X = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_X
            U = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_U
            Y = BETA0 + X + U
            Xt = X - X.mean(axis=1, keepdims=True)
            Yt = Y - Y.mean(axis=1, keepdims=True)
            den = np.einsum("ij,ij->i", Xt, Xt)
            beta = np.einsum("ij,ij->i", Xt, Yt) / den
            e = Yt - beta[:, None] * Xt
            for label, b in [("n/4", max(1, n // 4)), ("5", 5)]:
                b = min(b, n - 1)
                s = np.einsum("ij,ij->i", Xt * e, Xt * e)
                for lag in range(1, b + 1):
                    w = 1.0 - lag / (b + 1.0)
                    s += 2.0 * w * np.einsum(
                        "ij,ij->i", Xt[:, lag:] * e[:, lag:], Xt[:, :-lag] * e[:, :-lag]
                    )
                V = s / den**2
                cover = float(np.mean(np.abs(beta - BETA0) <= Z * np.sqrt(V)))
                rows.append({
                    "n": n, "c": c, "estimator": "OLS-HAC", "bw": label, "b": b,
                    "bias": float(np.mean(beta) - BETA0),
                    "bias_se": float(np.std(beta, ddof=1) / np.sqrt(R)),
                    "rmse": float(np.sqrt(np.mean((beta - BETA0) ** 2))),
                    "coverage": cover,
                    "cov_se": float(np.sqrt(cover * (1 - cover) / R)),
                    "mean_len": float(np.mean(2.0 * Z * np.sqrt(V))),
                })
    return rows


def run_gls_hac() -> list[dict]:
    """可行对照 II：GLS(a=0) + 空间 HAC 方差（与 OLS-HAC 同一批抽样）。

    与 OLS-HAC 的区别只在估计量：这里用与 benchmark 相同的 GLS 权重，
    因此把「估计量低效」与「方差估计失败」分开。临界值用 t_{n-1}。
    """
    rows = []
    for n in N_GRID:
        for c in C_GRID:
            rng = np.random.default_rng(MASTER_SEED + 3000 + n + int(1000 * c))
            X = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_X
            U = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_U
            Y = BETA0 + X + U
            Xt = X - X.mean(axis=1, keepdims=True)
            Yt = Y - Y.mean(axis=1, keepdims=True)
            W = pinv_spd(sigma(n, 0.0))
            H = Xt @ W
            den = np.einsum("ij,ij->i", H, Xt)
            beta = np.einsum("ij,ij->i", H, Yt) / den
            u = Yt - beta[:, None] * Xt
            score = H * u
            tcrit = float(stats.t.ppf(0.975, n - 1))
            for label, b in [("5", 5), ("n/4", max(1, n // 4))]:
                b = min(b, n - 1)
                s = np.einsum("ij,ij->i", score, score)
                for lag in range(1, b + 1):
                    w = 1.0 - lag / (b + 1.0)
                    s += 2.0 * w * np.einsum("ij,ij->i", score[:, lag:], score[:, :-lag])
                V = s / den**2
                rmse = float(np.sqrt(np.mean((beta - BETA0) ** 2)))
                cover = float(np.mean(np.abs(beta - BETA0) <= tcrit * np.sqrt(V)))
                cover_true = float(np.mean(np.abs(beta - BETA0) <= tcrit * rmse))
                tstat = (beta - BETA0) / np.sqrt(V)
                rows.append({
                    "n": n, "c": c, "estimator": "GLS-HAC", "bw": label, "b": b,
                    "bias": float(np.mean(beta) - BETA0),
                    "bias_se": float(np.std(beta, ddof=1) / np.sqrt(R)),
                    "rmse": rmse,
                    "coverage": cover,
                    "cov_se": float(np.sqrt(cover * (1 - cover) / R)),
                    "coverage_true_sd": cover_true,
                    "mean_len": float(np.mean(2.0 * tcrit * np.sqrt(V))),
                    "mean_se_over_rmse": float(np.mean(np.sqrt(V)) / rmse),
                    "t_skew": float(stats.skew(tstat)),
                    "t_exkurt": float(stats.kurtosis(tstat)),
                })
    return rows


def run_single_draw() -> list[dict]:
    """Section 6 illustration: one fixed-seed synthetic realization at n=200, c=0.10, a=0."""
    n, c, a = 200, 0.10, 0.0
    rng = np.random.default_rng(MASTER_SEED + 777)
    x = ar1(n, c, rng) * SIGMA_X
    u = ar1(n, c, rng) * SIGMA_U
    y = BETA0 + x + u
    Sa_plus = pinv_spd(sigma(n, a))
    Scs = {cp: sigma(n, cp) for cp in C_SUP}
    beta = float(beta_hat_vec(x[None, :], y[None, :], Sa_plus)[0])
    Vbar = float(np.maximum.reduce([vn_vec(x[None, :], cp, Sa_plus, Scs[cp]) for cp in C_SUP])[0])
    se = float(np.sqrt(Vbar))
    z = 1.959963984540054
    return [{"n": n, "c": c, "a": a, "seed": MASTER_SEED + 777,
             "beta_hat": beta, "se": se,
             "ci_lo": beta - z * se, "ci_hi": beta + z * se}]


def run_coverage_recheck() -> list[dict]:
    """对 R=1000 下最低覆盖的格子做 R=5000 复算（不是调参，只是降 MC 误差）。"""
    n, c, a, Rbig = 400, 0.05, 0.05, 5000
    Scs = {cp: sigma(n, cp) for cp in C_SUP}
    Sa_plus = pinv_spd(sigma(n, a))
    rng = np.random.default_rng(MASTER_SEED + 405)
    X = np.array([ar1(n, c, rng) for _ in range(Rbig)]) * SIGMA_X
    U = np.array([ar1(n, c, rng) for _ in range(Rbig)]) * SIGMA_U
    Y = BETA0 + X + U
    beta = beta_hat_vec(X, Y, Sa_plus)
    Vbar = np.maximum.reduce([vn_vec(X, cp, Sa_plus, Scs[cp]) for cp in C_SUP])
    cover = float(np.mean(np.abs(beta - BETA0) <= 1.959963984540054 * np.sqrt(Vbar)))
    return [{"n": n, "c": c, "a": a, "R": Rbig, "coverage": cover,
             "cov_se": float(np.sqrt(cover * (1 - cover) / Rbig))}]


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (RES / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def _log_n_axis(ax, xs) -> None:
    """Log x-axis with plain integer labels at the observed n values.

    The default log minor formatter renders overlapping mathtext labels on
    this grid (25/50/100/200/400); fixed major ticks avoid that.
    """
    ticks = sorted({int(x) for x in xs})
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{int(v)}"))
    ax.xaxis.set_minor_locator(NullLocator())


def plot_all(results, d1, d3, qq) -> None:
    # coverage vs n（c=0.1）
    fig, ax = plt.subplots(figsize=(6, 4))
    for ratio, label in [(0.0, "a=0 (LBM-GLS)"), (0.5, "a=c/2"), (1.0, "a=c (oracle)")]:
        xs = sorted({r["n"] for r in results if abs(r["c"] - 0.10) < 1e-12 and r["ratio_a"] == ratio})
        ys = [next(r["coverage"] for r in results if r["n"] == n and abs(r["c"] - 0.10) < 1e-12 and r["ratio_a"] == ratio) for n in xs]
        ax.plot(xs, ys, marker="o", label=label)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    _log_n_axis(ax, [r["n"] for r in results])
    ax.set_xlabel("n"); ax.set_ylabel("coverage"); ax.legend()
    fig.savefig(RES / "coverage.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # bias vs n（c=0.1）
    fig, ax = plt.subplots(figsize=(6, 4))
    for ratio, label in [(0.0, "a=0 (LBM-GLS)"), (0.5, "a=c/2"), (1.0, "a=c (oracle)")]:
        xs = sorted({r["n"] for r in results if abs(r["c"] - 0.10) < 1e-12 and r["ratio_a"] == ratio})
        ys = [next(r["bias"] for r in results if r["n"] == n and abs(r["c"] - 0.10) < 1e-12 and r["ratio_a"] == ratio) for n in xs]
        ax.plot(xs, ys, marker="o", label=label)
    _log_n_axis(ax, [r["n"] for r in results])
    ax.set_xlabel("n"); ax.set_ylabel("bias"); ax.legend()
    fig.savefig(RES / "bias.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # D1 log-log
    fig, ax = plt.subplots(figsize=(6, 4))
    for n in sorted({r["n"] for r in d1}):
        rs = [r for r in d1 if r["n"] == n]
        ax.loglog([r["c"] for r in rs], [r["rel_error"] for r in rs], marker="o", label=f"n={n}")
    ax.set_xlabel("c"); ax.set_ylabel("relative error"); ax.legend()
    fig.savefig(RES / "rate.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # D3 log-log residual
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog([r["delta"] for r in d3], [abs(r["residual"]) for r in d3], marker="o")
    ax.set_xlabel("delta"); ax.set_ylabel("|residual|")
    fig.savefig(RES / "local_expansion.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # QQ
    if qq:
        z = np.array(qq["sample"])
        fig, ax = plt.subplots(figsize=(5, 5))
        (osm, osr), _fit = stats.probplot(z, dist="norm", fit=True)
        ax.plot(osm, osr, "o", markersize=3)
        lo = float(min(np.min(osm), np.min(osr))); hi = float(max(np.max(osm), np.max(osr)))
        ax.plot([lo, hi], [lo, hi], color="red", linewidth=1)
        ax.set_xlabel("theoretical quantiles"); ax.set_ylabel("sample quantiles")
        fig.savefig(RES / "qq.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def sigma_deriv(n: int, c: float) -> np.ndarray:
    """d/dc of the demeaned exponential-kernel covariance, c>0."""
    s = locations(n)
    d = np.abs(s[:, None] - s[None, :])
    dK = -np.exp(-c * d) * (c * d + 1.0) / (2.0 * c * c)
    M = np.eye(n) - np.ones((n, n)) / n
    return M @ dK @ M


def run_r1_check() -> list[dict]:
    """Deterministic check of R1: exact pointwise ratio against 1+delta*rho."""
    c0, a = 0.10, 0.10
    rows = []
    for n in [100, 200, 400]:
        rng = np.random.default_rng(MASTER_SEED + 4242 + n)
        x = ar1(n, c0, rng)
        Sa_plus = pinv_spd(sigma(n, a))
        denom = float(x @ Sa_plus @ x)
        rho = float(x @ Sa_plus @ sigma_deriv(n, a) @ Sa_plus @ x / denom)
        for delta in [0.005, 0.01, 0.02, 0.04, 0.08, 0.12]:
            c = a + delta
            ratio = float(x @ Sa_plus @ sigma(n, c) @ Sa_plus @ x / denom)
            pred = 1.0 + delta * rho
            rows.append({"n": n, "c0": c0, "a": a, "delta": delta,
                         "ratio_exact": ratio, "ratio_first_order": pred,
                         "remainder": ratio - pred,
                         "remainder_over_delta2": (ratio - pred) / delta ** 2,
                         "rho": rho})
    return rows


def run_strong_persistence() -> list[dict]:
    """Two strong-persistence cells with the truth interior to C."""
    configs = [
        (200, 0.05, [0.02 + 0.01 * i for i in range(7)]),
        (400, 0.005, [0.002 + 0.001 * i for i in range(7)]),
    ]
    rows = []
    for k, (n, c, C) in enumerate(configs):
        rng = np.random.default_rng(MASTER_SEED + 5150 + k)
        X = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_X
        U = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_U
        Y = BETA0 + X + U
        Scs = {cp: sigma(n, cp) for cp in C}
        for ratio in A_RATIOS:
            a = ratio * c
            Sa_plus = pinv_spd(sigma(n, a))
            beta = beta_hat_vec(X, Y, Sa_plus)
            Vbar = np.maximum.reduce([vn_vec(X, cp, Sa_plus, Scs[cp]) for cp in C])
            cover = float(np.mean(np.abs(beta - BETA0) <= 1.959963984540054 * np.sqrt(Vbar)))
            rows.append({"n": n, "c": c, "a": a, "ratio_a": ratio,
                         "c_times_diam": c * (n - 1),
                         "bias": float(np.mean(beta) - BETA0),
                         "rmse": float(np.sqrt(np.mean((beta - BETA0) ** 2))),
                         "coverage": cover,
                         "cov_se": float(np.sqrt(cover * (1 - cover) / R)),
                         "len_robust": float(np.mean(2 * 1.959963984540054 * np.sqrt(Vbar)))})
    return rows


def run_misspecified_c() -> list[dict]:
    """Boundary demonstration: truth outside the stated uncertainty set C.

    Two cells: truth above C (conservative by L4) and truth below C (the
    coverage bound can fail). A4 is violated by construction; the cells show
    what the inclusion/exclusion of the truth does to coverage.
    """
    configs = [
        (200, 0.50, [0.05 + 0.025 * i for i in range(7)]),
        (200, 0.05, [0.50 + 0.05 * i for i in range(11)]),
    ]
    rows = []
    for k, (n, c, C) in enumerate(configs):
        rng = np.random.default_rng(MASTER_SEED + 5250 + k)
        X = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_X
        U = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_U
        Y = BETA0 + X + U
        Scs = {cp: sigma(n, cp) for cp in C}
        for ratio in A_RATIOS:
            a = ratio * c
            Sa_plus = pinv_spd(sigma(n, a))
            beta = beta_hat_vec(X, Y, Sa_plus)
            Vbar = np.maximum.reduce([vn_vec(X, cp, Sa_plus, Scs[cp]) for cp in C])
            cover = float(np.mean(np.abs(beta - BETA0) <= 1.959963984540054 * np.sqrt(Vbar)))
            rows.append({"n": n, "c": c, "a": a, "ratio_a": ratio,
                         "C_lo": C[0], "C_hi": C[-1],
                         "bias": float(np.mean(beta) - BETA0),
                         "rmse": float(np.sqrt(np.mean((beta - BETA0) ** 2))),
                         "coverage": cover,
                         "cov_se": float(np.sqrt(cover * (1 - cover) / R)),
                         "len_robust": float(np.mean(2 * 1.959963984540054 * np.sqrt(Vbar)))})
    return rows


def run_minimax_fine() -> list[dict]:
    """Deterministic sensitivity of the grid minimiser below 0.05."""
    grid = [0.01 * i for i in range(1, 16)]
    rows = []
    for n in N_GRID:
        sup = {a: max(kappa_n(n, a, cp) for cp in C_SUP) for a in grid}
        a_star = min(grid, key=lambda a: sup[a])
        rows.append({"n": n, "a_star_fine": a_star,
                     "sup_kappa": sup[a_star], "grid": "0.01:0.01:0.15"})
    return rows



def _scpc_critical(n: int, qmax: int = 10, M: int = 3000):
    """Plain SCPC location-model critical value, calibrated for c >= 0.05."""
    from scipy.special import gammaln
    s = locations(n)
    d = np.abs(s[:, None] - s[None, :])
    off = d[np.triu_indices(n, 1)]
    rho_max = float(np.mean(np.exp(-0.05 * off)))
    M1 = np.eye(n) - np.ones((n, n)) / n
    Smin = M1 @ np.exp(-0.05 * d) @ M1
    w, V = eigh(Smin)
    idx = np.argsort(w)[::-1][:qmax]
    R = V[:, idx] * np.sqrt(n)
    rng = np.random.default_rng(MASTER_SEED + 7700 + n)
    cgrid = [0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]
    cv = np.zeros(qmax)
    for c in cgrid:
        S = np.exp(-c * d)
        L = np.linalg.cholesky(S + 1e-12 * np.eye(n))
        U = rng.standard_normal((M, n)) @ L.T
        h = U @ R / np.sqrt(n)
        num = U.sum(axis=1) / np.sqrt(n)
        var = np.cumsum(h * h, axis=1) / np.arange(1, qmax + 1)
        tau = np.abs(num[:, None]) / np.sqrt(var)
        cv = np.maximum(cv, np.quantile(tau, 0.95, axis=0))
    qs = np.arange(1, qmax + 1)
    el = qs ** (-0.5) * cv * np.exp(gammaln((qs + 1) / 2) - gammaln(qs / 2))
    q = int(qs[np.argmin(el)])
    return q, float(cv[q - 1]), rho_max, R[:, :q]


def run_scpc() -> list[dict]:
    """Plain SCPC (location-model critical value) on the OLS estimator and draws."""
    rows = []
    for n in N_GRID:
        q, cv, rho_max, Rmat = _scpc_critical(n)
        for c in C_GRID:
            rng = np.random.default_rng(MASTER_SEED + 3000 + n + int(1000 * c))
            X = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_X
            U = np.array([ar1(n, c, rng) for _ in range(R)]) * SIGMA_U
            Y = BETA0 + X + U
            cov = 0
            lens = []
            for j in range(R):
                x = X[j]; y = Y[j]
                xm = x - x.mean(); ym = y - y.mean()
                sxx = float(np.mean(xm * xm))
                beta = float(np.sum(xm * ym) / np.sum(xm * xm))
                e = ym - beta * xm
                uhat = xm * e
                h = Rmat.T @ uhat / np.sqrt(n)
                s2 = float(np.mean(h * h))
                se = np.sqrt(s2) / (sxx * np.sqrt(n))
                cov += abs(beta - BETA0) <= cv * se
                lens.append(2 * cv * se)
            cover = cov / R
            rows.append({"n": n, "c": c, "q": q, "cv": cv, "rho_max": rho_max,
                         "coverage": cover,
                         "cov_se": float(np.sqrt(cover * (1 - cover) / R)),
                         "mean_len": float(np.mean(lens))})
    return rows

def main() -> int:
    d1 = run_d1()
    d3 = run_d3()
    d4 = run_d4()
    results, d2, qq = run_mc()
    neg = run_negative()
    recheck = run_coverage_recheck()
    ols = run_ols()
    single = run_single_draw()
    minimax = run_minimax(d4)
    pointwise = run_pointwise_panel()
    hac = run_hac()
    gls_hac = run_gls_hac()
    r1_check = run_r1_check()
    strong = run_strong_persistence()
    misspecified = run_misspecified_c()
    minimax_fine = run_minimax_fine()
    scpc = run_scpc()

    write_csv("results.csv", results)
    write_csv("coverage_recheck.csv", recheck)
    write_csv("ols.csv", ols)
    write_csv("single_draw.csv", single)
    write_csv("d1.csv", d1)
    write_csv("d2.csv", d2)
    write_csv("d3.csv", d3)
    write_csv("d4.csv", d4)
    write_csv("negative.csv", neg)
    write_csv("minimax.csv", minimax)
    write_csv("d8.csv", pointwise)
    write_csv("hac.csv", hac)
    write_csv("gls_hac.csv", gls_hac)
    write_csv("r1_check.csv", r1_check)
    write_csv("strong_persistence.csv", strong)
    write_csv("misspecified_c.csv", misspecified)
    write_csv("minimax_fine.csv", minimax_fine)
    write_csv("scpc.csv", scpc)
    plot_all(results, d1, d3, qq)

    # 摘要（stdout 可直接抄进 plan.md ⑦）
    summary = {"n_cells": len(results), "d1": d1, "d3": d3, "d4": d4,
               "d2": d2, "negative": neg, "recheck": recheck, "ols": ols, "single": single,
               "minimax": minimax, "pointwise": pointwise, "hac": hac, "gls_hac": gls_hac,
               "r1_check": r1_check, "strong_persistence": strong,
               "misspecified_c": misspecified, "minimax_fine": minimax_fine, "scpc": scpc,
               "qq": {k: v for k, v in qq.items() if k != "sample"}}
    (RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✓ 结果已写入", RES)
    for r in results:
        print(f"  n={r['n']:4d} c={r['c']:.2f} a={r['a']:.3f} "
              f"bias={r['bias']:+.4f}±{r['bias_se']:.4f} "
              f"rmse={r['rmse']:.4f} cov={r['coverage']:.3f}±{r['cov_se']:.3f}")
    print("D3 residual slope:", float(np.polyfit(np.log([r['delta'] for r in d3]),
                                                 np.log([abs(r['residual']) for r in d3]), 1)[0]))
    if qq:
        print("QQ skew/kurtosis:", qq["skew"], qq["kurtosis"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
