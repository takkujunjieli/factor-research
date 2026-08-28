"""熊市预警信号排名(时间序列 · leave-one-bear-out)。

和 ic.py(个股横截面)不同:这里是市场层时间序列。对 bear_panel 的每个特征评两件事——
  1) ic_fwd12:与连续目标 y_fwd_minret_12m 的有符号秩相关(负=特征越高、未来回撤越惨,即预警)。
  2) LOBO AUC:留出某一次熊市,方向用"其它熊市"标定,再看能否把留出那次的 warning 窗
     从 calm 月里分出来。逐次熊市求 AUC → mean / min / hit(>0.5 的比例)。
LOBO 直接回答"信号能否泛化到没见过的、成因不同的熊市"(见 README 讨论),而非拟合少数几次。
Spearman/AUC 都零依赖(rank 实现),短历史特征只在其覆盖的熊市上计入(看 n_folds)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _auc(score: pd.Series, label: pd.Series) -> float:
    """Mann-Whitney AUC(rank 实现,无 sklearn)。label 为 0/1。"""
    df = pd.DataFrame({"s": score, "y": label}).dropna()
    npos = int((df["y"] == 1).sum())
    nneg = int((df["y"] == 0).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = df["s"].rank()
    return float((r[df["y"] == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def signed_ic(feature: pd.Series, target: pd.Series, min_n: int = 24) -> float:
    sub = pd.concat([feature, target], axis=1).dropna()
    if len(sub) < min_n:
        return np.nan
    return float(sub.iloc[:, 0].rank().corr(sub.iloc[:, 1].rank()))


def rank_signals(panel: pd.DataFrame, feat_cols, cont_target="y_fwd_minret_12m",
                 warn_col="next_bear_id", inbear_col="in_bear_id") -> pd.DataFrame:
    """返回按 lobo_auc_mean 降序的排名表。
    calm = 非 warning 且 非熊中;warning_k = 第 k 段熊市的前瞻窗(next_bear_id==k)。
    """
    calm = (panel[warn_col] == 0) & (panel[inbear_col] == 0)
    bears = sorted(int(b) for b in panel[warn_col].unique() if b > 0)
    rows = []
    for f in feat_cols:
        x = panel[f]
        aucs = []
        for k in bears:
            others = [b for b in bears if b != k]
            pos_o = panel[warn_col].isin(others)          # 其它熊市 warning
            m_o = pos_o | calm
            auc_o = _auc(x[m_o], pos_o.astype(int)[m_o])   # 用其它熊市定方向
            d = 1.0 if (auc_o == auc_o and auc_o >= 0.5) else -1.0
            pos_k = panel[warn_col] == k                   # 留出的第 k 段
            m_k = pos_k | calm
            a = _auc((d * x)[m_k], pos_k.astype(int)[m_k])
            if a == a:
                aucs.append(a)
        rows.append({
            "feature": f,
            "ic_fwd12": signed_ic(x, panel[cont_target]),
            "lobo_auc_mean": float(np.mean(aucs)) if aucs else np.nan,
            "lobo_auc_min": float(np.min(aucs)) if aucs else np.nan,
            "lobo_hit": float(np.mean([a > 0.5 for a in aucs])) if aucs else np.nan,
            "n_folds": len(aucs),
        })
    return (pd.DataFrame(rows).set_index("feature")
            .sort_values("lobo_auc_mean", ascending=False))
