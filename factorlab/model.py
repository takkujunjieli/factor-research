"""稳健轻量宏观预警模型:L2 正则 logistic(numpy IRLS 自实现,无 sklearn/NN)。

为什么这样选:aggregate 月频只有 ~660 行、~11 次熊市 → 深网/大模型必过拟合。
L2 logistic + 标准化 + leave-one-bear-out 是"少事件、低信噪比"下的稳健基线:
系数可解释、强正则压噪、LOBO 给诚实的跨熊市样本外。

标准化只用训练折的均值/方差(无前视),缺失标准化后填 0(=训练均值,中性)。
评估口径与单信号排名一致(warning vs calm,方向由拟合自动决定),便于对比"组合 vs 最佳单信号"。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .bear import _auc


def ridge_logistic(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 60) -> np.ndarray:
    """L2 正则 logistic(IRLS);截距不罚。返回 w[0]=截距, w[1:]=各特征系数。"""
    n, p = X.shape
    Xb = np.column_stack([np.ones(n), X])
    w = np.zeros(p + 1)
    P = np.eye(p + 1) * lam
    P[0, 0] = 0.0
    for _ in range(iters):
        mu = np.clip(1.0 / (1.0 + np.exp(-(Xb @ w))), 1e-6, 1 - 1e-6)
        Wd = mu * (1 - mu)
        z = Xb @ w + (y - mu) / Wd
        A = Xb.T @ (Xb * Wd[:, None]) + P
        b = Xb.T @ (Wd * z)
        w_new = np.linalg.solve(A, b)
        if np.max(np.abs(w_new - w)) < 1e-9:
            w = w_new
            break
        w = w_new
    return w


def _fit_predict(train_df, train_y, score_df, lam):
    mean = train_df.mean()
    std = train_df.std().replace(0, 1.0)
    Ztr = ((train_df - mean) / std).fillna(0.0).values
    w = ridge_logistic(Ztr, train_y.values.astype(float), lam)
    Zsc = ((score_df - mean) / std).fillna(0.0).values
    prob = 1.0 / (1.0 + np.exp(-(np.column_stack([np.ones(len(Zsc)), Zsc]) @ w)))
    return prob, w, mean, std


def lobo_model(panel, feat_cols, target="y_bear12", warn_col="next_bear_id",
               inbear_col="in_bear_id", lam=10.0, embargo=12):
    """留一熊市评估 L2-logistic(purged + embargoed)。

    修正两处会抬高 AUC 的泄漏:
    ① 每个 calm 月按"离哪次熊市峰最近"唯一归属某一 fold —— 只在自己那折当负样本被评分,
       绝不同时进训练。避免"负基线被拟合过"的泄漏。
    ② embargo:留出第 k 段时,把它 [预警窗起-embargo, 谷底+embargo] 时间禁区内的所有月
       从训练集删掉(含邻近 calm / 其它熊市尾巴),隔断自相关泄漏。
    返回 per-bear AUC、mean/min/hit、终模型系数(标准化)、全样本预警概率序列。"""
    idx = panel.index
    calm = (panel[warn_col] == 0) & (panel[inbear_col] == 0)
    bears = sorted(int(b) for b in panel[warn_col].unique() if b > 0)
    y = panel[target].astype(float)
    # 每次熊市的峰(=预警窗最后一月)与整段跨度(预警窗起→谷底),用于归属与 embargo
    peak = {k: idx[panel[warn_col] == k].max() for k in bears}
    span = {k: (idx[panel[warn_col] == k].min(),
                idx[panel[inbear_col] == k].max() if (panel[inbear_col] == k).any() else peak[k])
            for k in bears}
    # calm 月唯一归属:离哪个峰最近算谁的(只在那折做负样本)
    calm_idx = idx[calm]
    owner = pd.Series({t: min(bears, key=lambda k: abs((t - peak[k]).days)) for t in calm_idx})
    aucs = {}
    for k in bears:
        lo = span[k][0] - pd.DateOffset(months=embargo)
        hi = span[k][1] + pd.DateOffset(months=embargo)
        embargoed = (idx >= lo) & (idx <= hi)            # 禁区:训练一律不碰
        test_calm = idx.isin(owner.index[owner == k])    # 只有归属 k 的 calm 进第 k 折评分
        score = (panel[warn_col] == k) | test_calm
        others = panel[warn_col].isin([b for b in bears if b != k])
        train_calm = calm & ~test_calm                   # 其余 calm 可训练…
        train = (others | train_calm) & ~embargoed       # …但都要避开禁区
        prob, *_ = _fit_predict(panel.loc[train, feat_cols], y[train], panel.loc[score, feat_cols], lam)
        lab = (panel.loc[score, warn_col] == k).astype(int).values
        a = _auc(pd.Series(prob), pd.Series(lab))
        if a == a:
            aucs[k] = a
    # 终模型:全部 warning ∪ calm 拟合 → 系数 + 当前概率
    full = (panel[warn_col] > 0) | calm
    prob_all, w, mean, std = _fit_predict(panel.loc[full, feat_cols], y[full], panel[feat_cols], lam)
    coef = pd.Series(w[1:], index=feat_cols).sort_values(key=abs, ascending=False)
    return {
        "auc_by_bear": aucs,
        "auc_mean": float(np.mean(list(aucs.values()))) if aucs else np.nan,
        "auc_min": float(np.min(list(aucs.values()))) if aucs else np.nan,
        "hit": float(np.mean([v > 0.5 for v in aucs.values()])) if aucs else np.nan,
        "coef": coef,
        "prob": pd.Series(prob_all, index=panel.index),
    }
