#!/usr/bin/env python3
"""把 bear_panel 序列化成 topic-scoped 的 research_bearbull.json,供 dashboard 的 research.html
在浏览器里跑模型(approach A)。写进 stock-dashboard/data/(公开、无隐私)。

schema(为 topic / 方向(熊-牛) / 多实体(大盘-行业-个股)三层扩展预留):
  {
    topic:"bearbull", freq, dates:[...],
    macro_features:[...],          # 共享、entity 无关
    coincident_features:[...],     # 价格/波动同步项(供 leading/full 切换)
    data:{ <feature>:[...] },       # 所有特征列存一次(按列名)
    directions:{
      bear:{ label,
        entities:{ market:{ label, side, price:[...], tech_features:[...],
                            episodes:[{id,peak,trough,dd,class}],
                            targets:{y_fwd_minret_12m,y_bear12,next_bear_id,in_bear_id} } } }
      # bull:{...} 以后;entities 里以后加 ind_Semis / stock_NVDA
    }
  }
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from bear_panel import detect_bears, load_french_factors, m_ret

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "ai-dashboard" / "data" / "research_bearbull.json"   # 项目 2026-08 改名 stock-dashboard→ai-dashboard

# 识别度高的大盘基准(公开、无 key):Yahoo 日频收盘 → 月末,只作图用不进模型
YF = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=0&period2={end}&interval=1d"


def load_index_monthly(sym: str, index: pd.DatetimeIndex) -> pd.Series:
    """Yahoo 日频收盘 → 月末 last,reindex 到 panel。失败则整列 None(前端自动跳过)。"""
    try:
        u = YF.format(sym=sym.replace("^", "%5E"), end=int(time.time()))
        r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=40)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        cl = pd.Series(res["indicators"]["quote"][0]["close"],
                       index=pd.to_datetime(res["timestamp"], unit="s")).dropna()
        m = cl.resample("ME").last()
        m.index = m.index.normalize()
        return m.reindex(index)
    except Exception as e:
        print(f"  ⚠️ {sym} 拉取失败({e}),benchmark 置空")
        return pd.Series(index=index, dtype=float)

MACRO = ["term_10y3m", "term_10y3m_long", "term_10y2y", "credit_baa_aaa", "nfci", "vix",
         "unrate", "sahm", "cfnai", "claims", "cpi_yoy", "bm", "ntis", "gz_spread", "ebp", "est_prob"]
MKT_TECH = ["mkt_dd", "mkt_ret_1m", "mkt_mom_12m", "mkt_dist_10ma", "mkt_rvol_3m",
            "def_minus_cyc_12m", "breadth_pct_above_10ma"]
COINCIDENT = ["mkt_dd", "mkt_ret_1m", "mkt_mom_12m", "mkt_dist_10ma", "mkt_rvol_3m", "vix"]
PREDICT = {1966: "policy", 1968: "endogenous", 1972: "policy", 1980: "policy",
           1987: "exogenous", 1990: "exogenous", 1998: "endogenous", 2000: "endogenous",
           2007: "endogenous", 2020: "exogenous", 2021: "policy"}


def _col(s):
    return [None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 6) for v in s]


def main():
    panel = pd.read_parquet(ROOT / "bear_panel.parquet")
    fac = load_french_factors()
    Pfull = (1 + m_ret(fac["Mkt-RF"] + fac["RF"])).cumprod()
    P = Pfull.reindex(panel.index)
    bears = detect_bears(Pfull)

    episodes = [{"id": i, "peak": pk.strftime("%Y-%m-%d"), "trough": tr.strftime("%Y-%m-%d"),
                 "dd": round(float(Pfull[tr] / Pfull[pk] - 1), 4), "class": PREDICT.get(pk.year, "unknown")}
                for i, (pk, tr) in enumerate(bears, 1)]

    sp500 = load_index_monthly("^GSPC", panel.index)
    nasdaq = load_index_monthly("^IXIC", panel.index)

    out = {
        "topic": "bearbull",
        "freq": "monthly",
        "dates": [d.strftime("%Y-%m-%d") for d in panel.index],
        "macro_features": [c for c in MACRO if c in panel.columns],
        "coincident_features": COINCIDENT,
        "data": {c: _col(panel[c]) for c in MACRO + MKT_TECH if c in panel.columns},
        "benchmarks": {"sp500": _col(sp500), "nasdaq": _col(nasdaq)},
        "directions": {
            "bear": {
                "label": "熊市(下行)预警",
                "predictability": {str(k): v for k, v in PREDICT.items()},
                "entities": {
                    "market": {
                        "label": "美股大盘",
                        "side": "long",
                        "price": _col(P),
                        "tech_features": [c for c in MKT_TECH if c in panel.columns],
                        "episodes": episodes,
                        "targets": {
                            "y_fwd_minret_12m": _col(panel["y_fwd_minret_12m"]),
                            "y_bear12": _col(panel["y_bear12"]),
                            "next_bear_id": _col(panel["next_bear_id"]),
                            "in_bear_id": _col(panel["in_bear_id"]),
                        },
                    }
                },
            }
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"research_bearbull.json → {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"  {len(out['dates'])} 月 · macro {len(out['macro_features'])} · tech {len(MKT_TECH)} · {len(episodes)} 段熊市")
    print("  可预测性:", {c: sum(e['class'] == c for e in episodes) for c in sorted(set(e['class'] for e in episodes))})


if __name__ == "__main__":
    main()
