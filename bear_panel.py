#!/usr/bin/env python3
"""熊市预警研究面板(月频,~1970 起;独立研究,不属于 dashboard)。

目标 = "未来最差":forward 6/12 月最差累计收益(熊市严重度),而非普通 forward return。
自动从市场指数检测熊市(≥20% 回撤)→ 给 leave-one-bear-out 的 episode 标注(warning 窗 / 在熊中),
供"跨不同成因熊市"的稳健性检验(见 README 讨论)。

特征三块(都尽量取长历史,月频):
  传导/市场内部:曲线(10Y-3M 长短)、信用(Moody's BAA-AAA 长 + HY OAS 短)、波动(VIX 短 + 已实现)、
                金融状况(NFCI)、市场趋势/回撤/动量、防御-周期轮动、行业广度(49 行业 %在10月线上)、EBP。
  实体经济:失业率、Sahm、CFNAI、初请、CPI 同比。
  额外:EBP 模型自带的 Fed 衰退概率 est_prob(基准对照)。

PIT:真宏观(失业/CFNAI/Sahm/CPI/EBP)按发布滞后 shift 1 月;市场可观测(曲线/VIX/利差/NFCI/收益)不 shift。
数据源全公开(French / FRED / Fed EBP CSV),MB 级。
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from build_panel import load_french_factors, load_french_industries
from macro_regime import fred_series

START = "1970-01-01"
ME = "ME"   # 月末
ROOT = Path(__file__).resolve().parent
# Welch-Goyal PredictorData(GKX 用的宏观集);Google Sheets 导出,缓存到 gitignored 的 data/
GOYAL_URL = "https://docs.google.com/spreadsheets/d/17mw_IpaiLFDrGnrPRQ2o1ugV5nJsZuD1/export?format=xlsx"
GOYAL_LOCAL = ROOT / "data" / "goyal_predictors.xlsx"


def load_goyal() -> pd.DataFrame:
    """Welch-Goyal 月频:返回 bm(账面市值比)、ntis(净股本发行)。首次下载缓存到本地。"""
    if not GOYAL_LOCAL.exists():
        GOYAL_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(GOYAL_URL, timeout=60)
        r.raise_for_status()
        GOYAL_LOCAL.write_bytes(r.content)
    m = pd.read_excel(GOYAL_LOCAL, sheet_name="Monthly")
    m = m[pd.to_numeric(m["yyyymm"], errors="coerce").notna()].copy()
    idx = pd.to_datetime(m["yyyymm"].astype(int).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    out = pd.DataFrame(index=idx)
    out["bm"] = pd.to_numeric(m["b/m"].values, errors="coerce")
    out["ntis"] = pd.to_numeric(m["ntis"].values, errors="coerce")
    return out


def m_ret(daily: pd.Series) -> pd.Series:
    return (1 + daily).resample(ME).prod() - 1


def fred_m(sid: str, how: str = "last") -> pd.Series:
    s = fred_series(sid, START)
    return s.resample(ME).last() if how == "last" else s.resample(ME).mean()


def load_ebp() -> pd.DataFrame:
    r = requests.get("https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv", timeout=30)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()
    d.index = d.index.to_period("M").to_timestamp(how="end").normalize()
    return d[["gz_spread", "ebp", "est_prob"]]


def detect_bears(P: pd.Series, thresh: float = 0.15):
    """月频总收益指数上检测严重回撤段(峰后跌 ≥thresh;谷=恢复前最低;恢复到前峰后重置)。
    注:用总收益指数(含股息),回撤比价格浅,故阈值取 0.15 ≈ 传统 ~20% 价格熊市 + 准熊级回调。"""
    bears = []
    peak = P.iloc[0]; peak_dt = P.index[0]; in_bear = False; trough = peak; trough_dt = peak_dt
    for dt, p in P.items():
        if not in_bear and p > peak:
            peak, peak_dt = p, dt
        elif not in_bear and p <= peak * (1 - thresh):
            in_bear, trough, trough_dt = True, p, dt
        elif in_bear:
            if p < trough:
                trough, trough_dt = p, dt
            if p >= peak:
                bears.append((peak_dt, trough_dt)); in_bear = False; peak, peak_dt = p, dt
    if in_bear:
        bears.append((peak_dt, trough_dt))
    return bears


def fwd_min(P: pd.Series, H: int) -> pd.Series:
    """未来 H 月内最差累计收益(相对今天):min_{1..H}(P_{t+h}/P_t) - 1。"""
    fut = pd.concat([P.shift(-h) for h in range(1, H + 1)], axis=1).min(axis=1)
    return fut / P - 1


def main():
    print("拉 French 市场/行业 + FRED + EBP ...")
    fac = load_french_factors()
    mkt_m = m_ret(fac["Mkt-RF"] + fac["RF"])
    P = (1 + mkt_m).cumprod()

    feat = pd.DataFrame(index=P.index)
    feat["mkt_ret_1m"] = mkt_m
    feat["mkt_mom_12m"] = P / P.shift(12) - 1
    feat["mkt_dist_10ma"] = P / P.rolling(10).mean() - 1
    feat["mkt_dd"] = P / P.cummax() - 1
    feat["mkt_rvol_3m"] = mkt_m.rolling(3).std() * np.sqrt(12)

    # 防御-周期轮动(12 行业,12 月相对表现)
    ind12 = m_ret(load_french_industries(12))
    dfn = ["ind_Utils", "ind_Hlth", "ind_NoDur"]
    cyc = ["ind_Durbl", "ind_Manuf", "ind_Enrgy", "ind_BusEq", "ind_Shops"]
    cum = lambda s: (1 + s).rolling(12).apply(np.prod, raw=True) - 1
    feat["def_minus_cyc_12m"] = cum(ind12[dfn].mean(axis=1)) - cum(ind12[cyc].mean(axis=1))

    # 广度:49 行业中 %在自身 10 月均线上(缺失月按 0 收益近似)
    ind49_d = load_french_industries(49).fillna(0.0)
    idx49 = (1 + ind49_d).cumprod().resample(ME).last()
    feat["breadth_pct_above_10ma"] = (idx49 > idx49.rolling(10).mean()).mean(axis=1)

    # 市场可观测(不 shift):曲线/信用/波动/金融状况
    mkt_obs = pd.DataFrame(index=feat.index)
    mkt_obs["term_10y3m_long"] = fred_m("GS10") - fred_m("TB3MS")
    mkt_obs["term_10y3m"] = fred_m("T10Y3M")
    mkt_obs["term_10y2y"] = fred_m("T10Y2Y")
    mkt_obs["credit_baa_aaa"] = fred_m("BAA") - fred_m("AAA")   # 长历史信用(HY OAS 的 fredgraph 端点只回~3年,弃用;信用维度由此 + gz/ebp 覆盖)
    mkt_obs["vix"] = fred_m("VIXCLS", "mean")
    mkt_obs["nfci"] = fred_m("NFCI")
    goyal = load_goyal()
    mkt_obs["bm"] = goyal["bm"]        # 市场账面市值比(用现价,近实时,不 shift)

    # 真宏观(shift 1 月做 PIT):失业/活动/通胀/EBP
    macro = pd.DataFrame(index=feat.index)
    macro["unrate"] = fred_m("UNRATE")
    macro["sahm"] = fred_m("SAHMREALTIME")
    macro["cfnai"] = fred_m("CFNAI")
    macro["claims"] = fred_m("ICSA", "mean")
    cpi = fred_m("CPIAUCSL")
    macro["cpi_yoy"] = cpi / cpi.shift(12) - 1
    macro["ntis"] = goyal["ntis"]               # 净股本发行(12月滚动,发布有滞后 → 归入 macro 后移)
    macro = macro.join(load_ebp())              # gz_spread, ebp, est_prob(Fed 衰退概率)
    macro = macro.shift(1)                       # 发布滞后

    # 目标 + 熊市标注
    tgt = pd.DataFrame(index=P.index)
    tgt["y_fwd_minret_6m"] = fwd_min(P, 6)
    tgt["y_fwd_minret_12m"] = fwd_min(P, 12)
    bears = detect_bears(P)
    nb = pd.Series(0, index=P.index, dtype=int)   # 未来12月内即将到来的熊市 id(warning 窗)
    ib = pd.Series(0, index=P.index, dtype=int)   # 正处于第 i 段熊市
    for i, (pk, tr) in enumerate(bears, 1):
        nb[(P.index > pk - pd.DateOffset(months=12)) & (P.index <= pk)] = i
        ib[(P.index >= pk) & (P.index <= tr)] = i
    tgt["next_bear_id"] = nb        # 供 leave-one-bear-out:留出某 id 的 warning 窗
    tgt["in_bear_id"] = ib
    tgt["y_bear12"] = (nb > 0).astype(int)   # 未来12月内是否有熊市顶(二元预警)

    panel = feat.join(mkt_obs).join(macro).join(tgt)
    panel = panel[panel.index >= "1971-01-01"]   # 预热(12月动量/均线)后
    panel.to_parquet("bear_panel.parquet")

    fcols = [c for c in panel.columns if not c.startswith(("y_", "next_bear", "in_bear"))]
    print(f"\nbear_panel.parquet: {panel.index.min().date()} → {panel.index.max().date()} · "
          f"{len(panel)} 月 · 特征 {len(fcols)}")
    print(f"\n检测到 {len(bears)} 段严重回撤(总收益 ≥15%,≈传统熊市 + 准熊级回调):")
    for i, (pk, tr) in enumerate(bears, 1):
        dd = P[tr] / P[pk] - 1
        print(f"  #{i}  峰 {pk.date()} → 谷 {tr.date()}  {dd*100:5.0f}%")
    print(f"\ny_bear12=1 占比: {panel['y_bear12'].mean()*100:.0f}%  "
          f"(warning 月 {int(panel['y_bear12'].sum())} / {len(panel)})")
    print("最近 6 月(部分列):")
    show = panel.tail(6)[["mkt_dd", "term_10y3m", "credit_baa_aaa", "sahm", "ebp", "est_prob", "y_bear12"]]
    print(show.round(3).to_string())


if __name__ == "__main__":
    main()
