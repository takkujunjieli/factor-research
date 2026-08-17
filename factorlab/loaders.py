"""数据加载:OpenAP 信号、月度收益、以及自带的合成 panel(用于离线验证 IC 引擎)。

真实数据(全市场横截面,不放 dashboard,单独下载到 data/):
  - 信号:OpenAP(openassetpricing.com)"Firm Level Characteristics" 宽表
    signed_predictors_dl_wide.csv —— 列 = permno, yyyymm, <~210 个信号>。文件很大,
    只按需读取所选信号列(见 load_openap_signals 的 usecols)。
  - 收益:月度个股收益 [permno, yyyymm, ret]。这是唯一需要你补的一块:
      · 有院校/WRDS 权限 → CRSP 月度收益(最佳,含退市收益,无幸存者偏差)。
      · 无 → 免费近似:按 ticker 映射 Stooq/yfinance 月度收益(有幸存者偏差、无退市收益,
        仅供探索)。permno↔ticker 映射随时间变,注意对齐。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def to_period(yyyymm) -> pd.PeriodIndex:
    """yyyymm(int 200501 或可转字符串)→ 月度 Period。"""
    s = pd.Series(list(yyyymm)).astype(int) if not hasattr(yyyymm, "astype") else pd.Series(yyyymm).astype(int)
    return pd.PeriodIndex(pd.to_datetime(s, format="%Y%m"), freq="M")


def load_openap_signals(path: str, signal: str, id_col="permno", ym_col="yyyymm"):
    """只读所需信号列(宽表 210 列全读会很慢/占内存)。返回 [permno, period, <signal>]。"""
    if str(path).endswith((".parquet", ".pq")):
        df = pd.read_parquet(path, columns=[id_col, ym_col, signal])
    else:
        df = pd.read_csv(path, usecols=[id_col, ym_col, signal])
    df["period"] = to_period(df[ym_col])
    return df[[id_col, "period", signal]].rename(columns={id_col: "permno"}), signal


def load_returns(path: str, id_col="permno", ym_col="yyyymm", ret_col="ret"):
    """月度个股收益 → [permno, period, ret]。"""
    if str(path).endswith((".parquet", ".pq")):
        df = pd.read_parquet(path, columns=[id_col, ym_col, ret_col])
    else:
        df = pd.read_csv(path, usecols=[id_col, ym_col, ret_col])
    df["period"] = to_period(df[ym_col])
    return df[[id_col, "period", ret_col]].rename(columns={id_col: "permno", ret_col: "ret"})


def make_synthetic(n_stocks=300, n_months=180, decay=0.6, b=0.06, noise=0.05,
                   max_lag=12, seed=42, signal_name="synthetic_alpha"):
    """合成 panel,IC 已知按几何衰减(IC_h ∝ decay^(h-1)),用来验证 IC 引擎是否正确。

    构造:signal iid N(0,1);ret_t = b·Σ_{j≥1} decay^(j-1)·signal_{t-j} + 噪声。
    则 corr(signal_t, ret_{t+h}) ∝ decay^(h-1) —— IC_1 最大、每加一期乘 decay。
    """
    rng = np.random.default_rng(seed)
    permnos = np.arange(10001, 10001 + n_stocks)
    periods = pd.period_range("2005-01", periods=n_months, freq="M")
    S = rng.standard_normal((n_months, n_stocks))
    R = np.zeros((n_months, n_stocks))
    for s in range(n_months):
        acc = np.zeros(n_stocks)
        for j in range(1, min(s, max_lag) + 1):
            acc += (decay ** (j - 1)) * S[s - j]
        R[s] = b * acc + noise * rng.standard_normal(n_stocks)
    ti, si = np.meshgrid(np.arange(n_months), np.arange(n_stocks), indexing="ij")
    df = pd.DataFrame({
        "permno": permnos[si.ravel()],
        "period": periods[ti.ravel()],
        signal_name: S.ravel(),
        "ret": R.ravel(),
    })
    signals = df[["permno", "period", signal_name]]
    returns = df[["permno", "period", "ret"]]
    return signals, returns, signal_name
