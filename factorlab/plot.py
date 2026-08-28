"""零依赖 SVG 出图(matplotlib 没装也能用)。IC 衰减折线 + 0 线 + ±1 标准误带。"""
from __future__ import annotations
import pandas as pd


def ic_decay_svg(summary: pd.DataFrame, out_path: str, title: str):
    hs = list(summary.index)
    ics = summary["mean_IC"].tolist()
    ses = (summary["IC_std"] / summary["n_months"] ** 0.5).tolist()   # 均值的标准误
    W, H, pl, pr, pt, pb = 640, 320, 54, 18, 40, 44
    iw, ih = W - pl - pr, H - pt - pb
    lo = min([i - s for i, s in zip(ics, ses)] + [0.0])
    hi = max([i + s for i, s in zip(ics, ses)] + [0.0])
    span = (hi - lo) or 1.0
    x = lambda k: pl + (k - hs[0]) / (max(hs) - hs[0] or 1) * iw
    y = lambda v: pt + (hi - v) / span * ih
    band = " ".join(f"{x(h):.1f},{y(i + s):.1f}" for h, i, s in zip(hs, ics, ses)) + " " + \
           " ".join(f"{x(h):.1f},{y(i - s):.1f}" for h, i, s in reversed(list(zip(hs, ics, ses))))
    line = " ".join(f"{x(h):.1f},{y(i):.1f}" for h, i in zip(hs, ics))
    dots = "".join(f'<circle cx="{x(h):.1f}" cy="{y(i):.1f}" r="3" fill="#2563eb"><title>h={h}: IC={i:.4f}</title></circle>'
                   for h, i in zip(hs, ics))
    yticks = ""
    for k in range(5):
        v = lo + span * k / 4
        yy = y(v)
        yticks += f'<line x1="{pl}" y1="{yy:.1f}" x2="{pl + iw}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-width="1"/>' \
                  f'<text x="{pl - 6}" y="{yy + 3:.1f}" text-anchor="end" font-size="10" fill="#6b7280">{v:.3f}</text>'
    xticks = "".join(f'<text x="{x(h):.1f}" y="{H - pb + 16}" text-anchor="middle" font-size="10" fill="#6b7280">{h}</text>'
                     for h in hs)
    zero = f'<line x1="{pl}" y1="{y(0):.1f}" x2="{pl + iw}" y2="{y(0):.1f}" stroke="#9ca3af" stroke-width="1" stroke-dasharray="4 3"/>'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">
<rect width="{W}" height="{H}" fill="white"/>
<text x="{W/2}" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#111827">{title}</text>
{yticks}{zero}
<polygon points="{band}" fill="#2563eb" fill-opacity="0.12"/>
<polyline points="{line}" fill="none" stroke="#2563eb" stroke-width="2"/>
{dots}{xticks}
<text x="{pl + iw/2}" y="{H - 6}" text-anchor="middle" font-size="11" fill="#374151">horizon h(月)</text>
<text x="14" y="{pt + ih/2}" text-anchor="middle" font-size="11" fill="#374151" transform="rotate(-90 14 {pt + ih/2})">mean rank IC</text>
</svg>'''
    with open(out_path, "w") as f:
        f.write(svg)
    return out_path


def bar_svg(series, out_path: str, title: str, ref: float = 0.5):
    """横向条形图(排名用)。series: index=名称, 值=分数;ref 画参考线(如 AUC=0.5)。"""
    items = list(series.items())
    n = len(items)
    W, rowh, pl, pr, pt, pb = 660, 22, 210, 56, 40, 16
    H = pt + pb + n * rowh
    vals = [v for _, v in items]
    lo = min(vals + [ref])
    hi = max(vals + [ref])
    span = (hi - lo) or 1.0
    x = lambda v: pl + (v - lo) / span * (W - pl - pr)
    bars = ""
    for i, (name, v) in enumerate(items):
        yy = pt + i * rowh
        col = "#2563eb" if v >= ref else "#9ca3af"
        bars += f'<text x="{pl - 6}" y="{yy + rowh/2 + 3:.1f}" text-anchor="end" font-size="10" fill="#374151">{name}</text>'
        bars += f'<rect x="{x(min(v, ref)):.1f}" y="{yy + 3:.1f}" width="{abs(x(v) - x(ref)):.1f}" height="{rowh - 6}" fill="{col}"><title>{name}: {v:.3f}</title></rect>'
        bars += f'<text x="{x(v) + (4 if v >= ref else -4):.1f}" y="{yy + rowh/2 + 3:.1f}" text-anchor="{"start" if v >= ref else "end"}" font-size="9" fill="#6b7280">{v:.3f}</text>'
    refline = f'<line x1="{x(ref):.1f}" y1="{pt}" x2="{x(ref):.1f}" y2="{pt + n*rowh}" stroke="#ef4444" stroke-width="1" stroke-dasharray="4 3"/><text x="{x(ref):.1f}" y="{pt - 4}" text-anchor="middle" font-size="9" fill="#ef4444">{ref:g}</text>'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">
<rect width="{W}" height="{H}" fill="white"/>
<text x="{W/2}" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#111827">{title}</text>
{refline}{bars}
</svg>'''
    with open(out_path, "w") as fh:
        fh.write(svg)
    return out_path
