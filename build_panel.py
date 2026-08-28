#!/usr/bin/env python3
"""日频特征面板装配器(独立研究,不属于 dashboard)。

拼:Ken French 日频因子(FF5+Mom)+ 12 行业组合日收益 + FRED 宏观/regime(macro_regime),
按业务日对齐 → 加 forward-N-day 目标(市场 & 各行业未来收益/方向)。产出 panel_daily.parquet。
用于「因子层」非线性建模(判市场/行业趋势),不做个股打分。纯公开数据、MB 级。

对齐/PIT:French 因子/行业与 FRED 同为日频;NFCI 等慢序列已在 macro_regime 里按发布时滞后移。
目标是 forward(未来)收益,最后 N 行目标为 NaN(无未来),训练时 dropna(subset=目标)。
"""
import io
import re
import zipfile

import numpy as np
import pandas as pd
import requests

import macro_regime

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def french_csv(zipname: str) -> str:
    r = requests.get(FRENCH + zipname, timeout=40)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return z.read(z.namelist()[0]).decode("latin1")


def parse_french_daily(text: str) -> pd.DataFrame:
    """解析 French 日频 CSV 的第一个数据块(行业文件=价值加权 VW;因子文件=唯一块)。
    值为百分数→转小数;哨兵 -99.99/-999→NaN。"""
    lines = text.splitlines()
    di = next(i for i, l in enumerate(lines) if re.match(r"^\s*\d{8}\s*,", l))  # 首个数据行
    hi = di - 1
    while hi >= 0 and not lines[hi].strip().startswith(","):
        hi -= 1
    cols = [c.strip() for c in lines[hi].split(",")][1:]   # 首列是日期(空名)
    dates, rows = [], []
    for l in lines[di:]:
        if not re.match(r"^\s*\d{8}\s*,", l):
            break
        parts = [p.strip() for p in l.split(",")]
        dates.append(parts[0])
        rows.append([float(x) for x in parts[1:len(cols) + 1]])
    df = pd.DataFrame(rows, index=pd.to_datetime(dates, format="%Y%m%d"), columns=cols)
    df.index.name = "date"
    return df.mask(df <= -99, np.nan) / 100.0


def load_french_factors() -> pd.DataFrame:
    ff5 = parse_french_daily(french_csv("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"))
    mom = parse_french_daily(french_csv("F-F_Momentum_Factor_daily_CSV.zip"))
    mom.columns = ["Mom"]
    return ff5.join(mom, how="left")


def load_french_industries(n: int = 12) -> pd.DataFrame:
    ind = parse_french_daily(french_csv(f"{n}_Industry_Portfolios_daily_CSV.zip"))
    ind.columns = [f"ind_{c.strip()}" for c in ind.columns]
    return ind


def fwd_ret(r: pd.Series, N: int) -> pd.Series:
    """未来 N 日累积收益(t 行 = t+1..t+N 复利);末 N 行为 NaN。"""
    return np.expm1(np.log1p(r).rolling(N).sum().shift(-N))


def main():
    print("拉 French 因子/行业 + FRED 宏观 ...")
    fac = load_french_factors()                 # Mkt-RF,SMB,HML,RMW,CMA,RF,Mom
    ind = load_french_industries(12)            # 12 行业日收益
    mac = macro_regime.build(freq="B")

    mac_cols = [c for c in mac.columns if c.startswith("z_")] + \
        ["stress", "regime", "curve_inverted", "vix_elevated", "credit_widening", "nfci_tight"]
    mac = mac[mac_cols].copy()
    mac["regime_code"] = mac["regime"].map({"risk_on": -1, "neutral": 0, "risk_off": 1}).astype("float")

    panel = fac.join(ind, how="inner").join(mac, how="inner")
    panel = panel.dropna(subset=["stress"])     # 从 regime 有效期开始(z 窗预热后)

    # 目标:市场(总收益=Mkt-RF+RF)未来 5/20 日 + 各行业未来 20 日;及方向
    mkt = panel["Mkt-RF"] + panel["RF"]
    for N in (5, 20):
        y = fwd_ret(mkt, N)
        panel[f"y_mkt_fwd{N}"] = y
        panel[f"y_mkt_dir{N}"] = (y > 0).astype("Int64").where(y.notna())
    for c in [c for c in panel.columns if c.startswith("ind_")]:
        y = fwd_ret(panel[c], 20)
        panel[f"y_{c}_fwd20"] = y
        panel[f"y_{c}_dir20"] = (y > 0).astype("Int64").where(y.notna())

    panel.to_parquet("panel_daily.parquet")
    feat = [c for c in panel.columns if not c.startswith("y_")]
    tgt = [c for c in panel.columns if c.startswith("y_")]
    print(f"\npanel_daily.parquet: {panel.index.min().date()} → {panel.index.max().date()} · "
          f"{len(panel)} 交易日 · 特征 {len(feat)} · 目标 {len(tgt)}")
    print("特征组: FF因子(7) + 行业(12) + 宏观 z/stress/regime/flags")
    print("目标示例:", tgt[:6], "...")
    print("\n最近 3 行(部分列):")
    show = panel.tail(3)[["Mkt-RF", "Mom", "ind_Hlth", "stress", "regime", "y_mkt_fwd20"]]
    print(show.round(4).to_string())


if __name__ == "__main__":
    main()
