#!/usr/bin/env python3
"""跑一个信号的 IC 衰减。默认合成数据(离线可跑,验证引擎);接 OpenAP 真实数据见 --source openap。

例:
  python3 run_ic.py                                   # 合成 demo
  python3 run_ic.py --source openap \\
      --signals data/signed_predictors_dl_wide.csv \\
      --returns data/crsp_monthly_ret.csv \\
      --signal Mom12m --max-horizon 18
"""
import argparse
from pathlib import Path

from factorlab import ic as ic_mod
from factorlab import loaders, plot

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "openap"], default="synthetic")
    ap.add_argument("--signals", help="OpenAP 信号宽表路径(csv/parquet)")
    ap.add_argument("--returns", help="月度收益路径 [permno,yyyymm,ret]")
    ap.add_argument("--signal", default=None, help="信号列名(openap 必填)")
    ap.add_argument("--max-horizon", type=int, default=12)
    ap.add_argument("--min-stocks", type=int, default=20)
    ap.add_argument("--outdir", default=str(ROOT / "out"))
    a = ap.parse_args()

    if a.source == "synthetic":
        signals, returns, signal = loaders.make_synthetic()
        print(f"[合成数据] {signals['permno'].nunique()} 只 × {signals['period'].nunique()} 月;"
              f"信号={signal}(设计为几何衰减 IC,用于验证引擎)")
    else:
        if not (a.signals and a.returns and a.signal):
            ap.error("openap 需要 --signals --returns --signal")
        signals, signal = loaders.load_openap_signals(a.signals, a.signal)
        returns = loaders.load_returns(a.returns)
        print(f"[OpenAP] 信号={signal};{signals['permno'].nunique()} permno × "
              f"{signals['period'].nunique()} 月;收益 {len(returns):,} 行")

    horizons = range(1, a.max_horizon + 1)
    summary, _ = ic_mod.ic_decay(signals, returns, signal, horizons=horizons, min_stocks=a.min_stocks)

    pd_opts = dict(float_format=lambda v: f"{v:.4f}")
    print("\n=== IC 衰减 ===")
    print(summary.to_string(**pd_opts))
    half = _half_life(summary["mean_IC"].tolist())
    if half:
        print(f"\nIC 半衰期 ≈ {half:.1f} 个月(mean_IC 跌到 h=1 的一半处)")

    outdir = Path(a.outdir)
    outdir.mkdir(exist_ok=True)
    csv = outdir / f"ic_decay_{signal}.csv"
    svg = outdir / f"ic_decay_{signal}.svg"
    summary.to_csv(csv)
    plot.ic_decay_svg(summary, str(svg), f"IC decay · {signal} ({a.source})")
    print(f"\n表 → {csv}\n图 → {svg}")


def _half_life(ics):
    if not ics or ics[0] <= 0:
        return None
    target = ics[0] / 2
    for h, v in enumerate(ics, 1):
        if v <= target:
            return h
    return None


if __name__ == "__main__":
    main()
