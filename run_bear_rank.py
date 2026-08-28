#!/usr/bin/env python3
"""熊市预警信号排名:对 bear_panel.parquet 的每个特征,算连续目标 IC + leave-one-bear-out AUC,
按跨熊市稳健度排序。产物写 out/。

例:
  python3 bear_panel.py            # 先生成 bear_panel.parquet
  python3 run_bear_rank.py          # 排名
  python3 run_bear_rank.py --panel bear_panel.parquet --target y_fwd_minret_6m
"""
import argparse
from pathlib import Path

import pandas as pd

from factorlab import bear, plot

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(ROOT / "bear_panel.parquet"))
    ap.add_argument("--target", default="y_fwd_minret_12m", help="连续目标(算有符号 IC)")
    ap.add_argument("--outdir", default=str(ROOT / "out"))
    a = ap.parse_args()

    panel = pd.read_parquet(a.panel)
    feat_cols = [c for c in panel.columns
                 if not c.startswith(("y_", "next_bear", "in_bear"))]
    n_bears = int(panel["next_bear_id"].max())
    print(f"面板 {panel.index.min().date()}→{panel.index.max().date()} · "
          f"{len(panel)} 月 · 特征 {len(feat_cols)} · 熊市 {n_bears} 段 · 目标 {a.target}")

    rk = bear.rank_signals(panel, feat_cols, cont_target=a.target)

    pd_opts = dict(float_format=lambda v: f"{v:.3f}")
    print("\n=== 熊市预警信号排名(按 lobo_auc_mean 降序)===")
    print("ic_fwd12:与未来12月最差收益的秩相关(负=高值预警);lobo_*:留一熊市的 held-out AUC")
    print(rk.to_string(**pd_opts))

    outdir = Path(a.outdir)
    outdir.mkdir(exist_ok=True)
    csv = outdir / "bear_rank.csv"
    svg = outdir / "bear_rank.svg"
    rk.to_csv(csv)
    top = rk["lobo_auc_mean"].dropna().head(15)[::-1]   # 画前15,从下到上
    plot.bar_svg(top, str(svg), "熊市预警信号 · LOBO held-out AUC(越高越稳健)")
    print(f"\n表 → {csv}\n图 → {svg}")
    print("\n提醒:lobo_auc_min 低 / n_folds 少的信号,别只看 mean——可能只在某几次熊市灵。")


if __name__ == "__main__":
    main()
