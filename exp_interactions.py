import pandas as pd, numpy as np
from factorlab import model, bear
p = pd.read_parquet("bear_panel.parquet")
base = [c for c in p.columns if not c.startswith(("y_","next_bear","in_bear"))]

def ez(s, minp=60):  # PIT expanding z-score(只用过去)
    return (s - s.expanding(minp).mean()) / s.expanding(minp).std()

pairs = [("vix","bm"),("mkt_rvol_3m","bm"),("vix","ntis"),("mkt_rvol_3m","ntis")]
inter = {f"{a}_x_{b}": ez(p[a])*ez(p[b]) for a,b in pairs}
pI = p.join(pd.DataFrame(inter, index=p.index))
ic = list(inter)

def run(feats, name):
    r = model.lobo_model(pI, feats, lam=10.0)
    print(f"  {name:<26} mean {r['auc_mean']:.3f} · min {r['auc_min']:.3f} · hit {r['hit']*100:3.0f}% · nfeat {len(feats)}")

print("=== 组合模型 LOBO 对比(加不加交互)===")
run(base,                                   "A 基线(仅主效应)")
run(base+ic,                                "B 主效应 + 4交互")
run(["bm","ntis","vix","mkt_rvol_3m"]+ic,   "C 交互为主(主+交互)")
run(ic,                                     "D 只有交互项")
print("\n=== 4 个交互项 单独 LOBO 排名(对照单主效应)===")
r2 = bear.rank_signals(pI, ic + ["vix","bm","ntis","mkt_rvol_3m"])
print(r2.round(3).to_string())
