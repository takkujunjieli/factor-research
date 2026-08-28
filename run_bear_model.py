#!/usr/bin/env python3
"""稳健轻量宏观预警模型:对 bear_panel.parquet 拟合 L2-logistic(无 NN),leave-one-bear-out 评估,
输出跨熊市样本外 AUC、标准化系数(哪些信号在推高/压低预警)、以及当前预警概率。产物写 out/。

  python3 run_bear_model.py                 # 默认 target=y_bear12, lam=10
  python3 run_bear_model.py --lam 20        # 更强正则(更稳、更钝)
"""
import argparse
from pathlib import Path

import pandas as pd

from factorlab import bear, model

ROOT = Path(__file__).resolve().parent
# 市场自身的价格/波动同步项("已经在跌/慌"的结果,非前兆);--leading 时剔除做纯领先模型
COINCIDENT = ["mkt_dd", "mkt_ret_1m", "mkt_dist_10ma", "mkt_rvol_3m", "mkt_mom_12m", "vix"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(ROOT / "bear_panel.parquet"))
    ap.add_argument("--lam", type=float, default=10.0, help="L2 强度(大=更稳更钝)")
    ap.add_argument("--leading", action="store_true", help="剔除市场价格/波动同步项,做纯领先模型")
    ap.add_argument("--outdir", default=str(ROOT / "out"))
    a = ap.parse_args()

    panel = pd.read_parquet(a.panel)
    feats = [c for c in panel.columns if not c.startswith(("y_", "next_bear", "in_bear"))]
    if a.leading:
        feats = [f for f in feats if f not in COINCIDENT]
        print(f"【纯领先】剔除同步项 {COINCIDENT}")
    print(f"面板 {panel.index.min().date()}→{panel.index.max().date()} · {len(panel)} 月 · "
          f"特征 {len(feats)} · λ={a.lam}")

    res = model.lobo_model(panel, feats, lam=a.lam)

    # 对照:最佳单信号
    rk = bear.rank_signals(panel, feats)
    best = rk.iloc[0]

    print("\n=== 组合模型(L2-logistic)leave-one-bear-out ===")
    print(f"  样本外 AUC: mean {res['auc_mean']:.3f} · min {res['auc_min']:.3f} · "
          f"hit {res['hit']*100:.0f}% ({sum(v>0.5 for v in res['auc_by_bear'].values())}/{len(res['auc_by_bear'])} 次熊市)")
    print(f"  对照·最佳单信号 {rk.index[0]}: mean {best['lobo_auc_mean']:.3f} · min {best['lobo_auc_min']:.3f} · hit {best['lobo_hit']*100:.0f}%")

    print("\n=== 逐熊市样本外 AUC ===")
    for k, v in sorted(res["auc_by_bear"].items()):
        print(f"  bear#{k}: {v:.3f}")

    print("\n=== 终模型标准化系数(+推高预警 / −压低;|大|=影响大)===")
    for name, c in res["coef"].head(12).items():
        print(f"  {name:<24} {c:+.3f}")

    prob = res["prob"].dropna()
    print(f"\n=== 当前预警概率(全样本模型)===")
    print(prob.tail(6).round(3).to_string())
    print(f"  历史分位: 当前 {prob.iloc[-1]:.3f} 处于历史 {(prob < prob.iloc[-1]).mean()*100:.0f}% 分位")

    outdir = Path(a.outdir)
    outdir.mkdir(exist_ok=True)
    res["prob"].to_frame("warn_prob").to_csv(outdir / "bear_model_prob.csv")
    res["coef"].to_frame("coef_std").to_csv(outdir / "bear_model_coef.csv")
    print(f"\n概率序列 → {outdir/'bear_model_prob.csv'} · 系数 → {outdir/'bear_model_coef.csv'}")
    print("提醒:样本外只有 ~11 次熊市;λ 越小越容易过拟合,先看 min/hit 是否稳,再谈 mean。")


if __name__ == "__main__":
    main()
