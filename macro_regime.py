#!/usr/bin/env python3
"""FRED 宏观状态变量 + regime 分类(独立研究模块,不属于 dashboard)。

拉一组利率/信用/波动/金融状况序列(FRED 无 key 的公开 CSV 端点),重采样到周频,
算 **point-in-time** 滚动 z-score(不用全样本 → 无前视偏差,可直接用于回测/调参),
合成一个 stress 复合指标,分出 risk-on / neutral / risk-off 三态 + 若干离散 flag。

输出 macro_regimes.csv/parquet:date × [原序列, 衍生特征, z, stress, regime, flags]。
把它按 date join 到你的因子/信号 panel,就能做**条件化 / regime 切换**。

用法: python3 macro_regime.py [--start 1997-01-01] [--zwin 156] [--out macro_regimes]
可选: 装了 statsmodels 时 --markov 用 2 态 Markov switching 出平滑 regime 概率。
注:严格回测请按发布时滞 lag 月频序列(NFCI ~1 周、CPI ~2 周);利率/VIX 当日可得。
"""
import argparse
import numpy as np
import pandas as pd

# FRED series_id → (简称, 方向:+1=值越大越 risk-off)
SERIES = {
    "DGS10":        ("10Y 国债收益率", 0),
    "DGS2":         ("2Y 国债收益率", 0),
    "T10Y2Y":       ("期限利差 10Y-2Y", -1),   # 越低/倒挂 = 越 risk-off
    "BAMLH0A0HYM2": ("高收益 OAS(信用压力)", +1),
    "BAMLC0A0CM":   ("投资级 OAS", +1),
    "VIXCLS":       ("VIX", +1),
    "NFCI":         ("芝加哥联储金融状况(>0=偏紧)", +1),
    "STLFSI4":      ("圣路易斯联储金融压力", +1),
    "T10YIE":       ("10Y 盈亏平衡通胀", 0),
}
STRESS_KEYS = ["VIXCLS", "BAMLH0A0HYM2", "NFCI", "T10Y2Y"]   # 合成 stress 用这几个


def fred_series(sid: str, start: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
    s = pd.read_csv(url, index_col=0, parse_dates=True, na_values=["."]).iloc[:, 0]
    s.name = sid
    s.index.name = "date"
    return s


def load_panel(start: str, freq: str) -> pd.DataFrame:
    cols = {}
    for sid in SERIES:
        try:
            cols[sid] = fred_series(sid, start)
        except Exception as e:  # noqa: BLE001
            print(f"跳过 {sid}: {e}")
    df = pd.concat(cols.values(), axis=1).sort_index()
    df = df.resample(freq).last().ffill()   # freq=B(日)/W-FRI(周);前填把周/月频序列铺开
    return df


def roll_z(s: pd.Series, win: int, minp: int = 52) -> pd.Series:
    """trailing 滚动 z-score(point-in-time,不含未来)。"""
    m = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return (s - m) / sd


def build(start="1997-01-01", freq="W-FRI", zwin=None, markov=False) -> pd.DataFrame:
    df = load_panel(start, freq)
    daily = freq.upper().startswith(("B", "D"))

    # PIT 发布时滞:周/月频序列(NFCI/金融压力 ~1周延迟)后移,避免用当天还没公布的数
    for c, k in {"NFCI": 5 if daily else 1, "STLFSI4": 5 if daily else 1}.items():
        if c in df.columns:
            df[c] = df[c].shift(k)
    out = df.copy()

    # 窗口按频率取(日/周):z 窗 ~3y、min ~1y、信用变化 ~1季、VIX 变化 ~1月
    if zwin is None:
        zwin = 756 if daily else 156
    minp = 252 if daily else 52
    hy_win, vix_win = (63, 21) if daily else (13, 4)

    # 衍生:z-score(逐序列)+ 压力变化动量
    for sid in df.columns:
        out[f"z_{sid}"] = roll_z(df[sid], zwin, minp)
    out["hy_chg"] = df["BAMLH0A0HYM2"].diff(hy_win)
    out["vix_chg"] = df["VIXCLS"].diff(vix_win)

    # stress 复合:各分量按方向取 z 后平均(高 = risk-off)
    parts = []
    for sid in STRESS_KEYS:
        d = SERIES[sid][1]
        if d != 0 and f"z_{sid}" in out:
            parts.append(np.sign(d) * out[f"z_{sid}"])
    out["stress"] = pd.concat(parts, axis=1).mean(axis=1)

    # 三态 regime(阈值可调):stress>0.5 risk-off,< -0.5 risk-on,否则 neutral
    out["regime"] = pd.cut(out["stress"], [-np.inf, -0.5, 0.5, np.inf],
                           labels=["risk_on", "neutral", "risk_off"])

    # 离散 flags(直觉可读)
    out["curve_inverted"] = (df["T10Y2Y"] < 0).astype("Int64")
    out["vix_elevated"]   = (df["VIXCLS"] > 20).astype("Int64")
    out["credit_widening"] = ((out["hy_chg"] > 0) & (out["z_BAMLH0A0HYM2"] > 0.5)).astype("Int64")
    out["nfci_tight"]     = (df["NFCI"] > 0).astype("Int64")

    if markov:
        try:
            from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
            y = out["stress"].dropna()
            mod = MarkovRegression(y, k_regimes=2, switching_variance=True).fit(disp=False)
            hi = int(np.argmax(mod.params[[f"const[{i}]" for i in range(2)]].values))  # 高均值=risk-off 态
            out.loc[y.index, "mkv_riskoff_prob"] = mod.smoothed_marginal_probabilities[hi].values
        except Exception as e:  # noqa: BLE001
            print(f"Markov 跳过(需 statsmodels): {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1997-01-01")
    ap.add_argument("--freq", default="D", choices=["D", "W"], help="D=日频(B) / W=周频(W-FRI)")
    ap.add_argument("--zwin", type=int, default=None, help="滚动 z 窗口(缺省:日756/周156)")
    ap.add_argument("--out", default="macro_regimes")
    ap.add_argument("--markov", action="store_true")
    a = ap.parse_args()
    freq = "B" if a.freq == "D" else "W-FRI"
    df = build(a.start, freq, a.zwin, a.markov)
    df.to_csv(f"{a.out}.csv")
    try:
        df.to_parquet(f"{a.out}.parquet")
    except Exception:  # noqa: BLE001
        pass
    valid = df.dropna(subset=["stress"])
    unit = "日" if a.freq == "D" else "周"
    print(f"\n宏观 regime 表: {valid.index.min().date()} → {valid.index.max().date()} · {len(valid)} {unit}")
    print("regime 分布:", valid["regime"].value_counts().to_dict())
    print(f"\n最近 8 {unit}:")
    show = valid.tail(8)[["VIXCLS", "T10Y2Y", "BAMLH0A0HYM2", "NFCI", "stress", "regime"]]
    print(show.round(2).to_string())


if __name__ == "__main__":
    main()
