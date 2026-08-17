"""IC(信息系数)与 IC 衰减。

rank IC_h = 每个横截面月 t 上,signal_t 与 h 期后单期收益 ret_{t+h} 的 Spearman 秩相关,
再对 t 求均值。IC_h 随 h 增大而衰减,反映信号预测力的持续性/半衰期。

前视收益用「按 (permno, period) 键 join」而非行内 shift —— 对有缺月的真实数据也正确。
Spearman 用 pandas 秩相关实现(不依赖 scipy)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _xs_ic(g: pd.DataFrame, xcol: str, ycol: str, min_n: int):
    sub = g[[xcol, ycol]].dropna()
    if len(sub) < min_n:
        return np.nan
    return sub[xcol].rank().corr(sub[ycol].rank())   # 秩 + Pearson = Spearman


def ic_decay(signals: pd.DataFrame, returns: pd.DataFrame, signal_col: str,
             horizons=range(1, 13), min_stocks=20):
    """返回 (summary_df 按 horizon, per_month dict[h]→月度 IC 序列)。

    signals: [permno, period, <signal_col>];returns: [permno, period, ret]。
    """
    base = signals[["permno", "period", signal_col]].dropna(subset=[signal_col])
    ret = returns[["permno", "period", "ret"]]
    rows, per_month = [], {}
    for h in horizons:
        r = ret.rename(columns={"ret": "fwd"}).copy()
        r["period"] = r["period"] - h          # 把 t+h 的收益贴回到信号所在的 t 上
        p = base.merge(r, on=["permno", "period"], how="left")
        ic = (p.groupby("period")[[signal_col, "fwd"]]
                .apply(lambda g: _xs_ic(g, signal_col, "fwd", min_stocks))
                .dropna())
        per_month[h] = ic
        n = int(len(ic))
        mean = float(ic.mean()) if n else np.nan
        std = float(ic.std(ddof=1)) if n > 1 else np.nan
        rows.append({
            "horizon": h,
            "mean_IC": mean,
            "IC_std": std,
            "IC_IR": mean / std if std else np.nan,          # 单月口径的信息比
            "t_stat": mean / std * np.sqrt(n) if std and n else np.nan,
            "pos_rate": float((ic > 0).mean()) if n else np.nan,
            "n_months": n,
        })
    return pd.DataFrame(rows).set_index("horizon"), per_month
