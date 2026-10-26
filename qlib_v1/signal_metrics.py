"""Signal-only evaluation. No holdings, allocation, costs or equity simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def correlation(frame, min_observations):
    paired = frame[["score", "target"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(paired) < min_observations or paired.score.nunique() < 2 or paired.target.nunique() < 2:
        return {"ic": None, "rank_ic": None, "observations": len(paired)}
    return {"ic": float(paired.score.corr(paired.target)),
            "rank_ic": float(paired.score.rank(method="average").corr(paired.target.rank(method="average"))),
            "observations": len(paired)}


def summarize(series):
    result = {"periods": len(series)}
    for key, ratio in (("ic", "icir"), ("rank_ic", "rank_icir")):
        values = np.array([row[key] for row in series if row[key] is not None], dtype=float)
        mean = float(values.mean()) if len(values) else None
        std = float(values.std(ddof=1)) if len(values) >= 2 else None
        result.update({key: mean, f"{key}_std": std,
                       ratio: mean / std if std is not None and std > 1e-12 else None,
                       f"{key}_periods": len(values)})
    return result


def evaluate_signal(scores, targets, config, horizon):
    """CS: one correlation across stocks per date; TS: one per stock per window.

    ICIR is mean(IC)/sample_std(IC), unannualized. Fixed, disjoint 63-session
    signal-date windows define TS stability; labels can overlap within windows.
    """
    if scores.empty:
        index = pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])
        scores = pd.Series(index=index, dtype=float)
    paired = pd.DataFrame({"score": scores, "target": targets.reindex(scores.index)})
    paired = paired.replace([np.inf, -np.inf], np.nan)
    cs = []
    for day, frame in paired.groupby(level="datetime", sort=True):
        value = correlation(frame, config["min_candidates"])
        cs.append({"date": str(day.date()), **value})
    valid_cs = [row for row in cs if row["ic"] is not None]
    cs_summary = summarize(valid_cs)
    cs_summary.update(signal_dates=len(cs), invalid_dates=len(cs) - len(valid_cs),
                      paired_observations=int(paired.dropna().shape[0]),
                      missing_labels=int(paired.target.isna().sum()),
                      series=cs)
    dates = pd.DatetimeIndex(sorted(paired.index.get_level_values("datetime").unique()))
    window = config["timing_window"]
    # Global signal-date boundaries, shared by all stocks; no per-stock date
    # compression around missing labels and no overlapping correlation windows.
    blocks = [dates[i:i + window] for i in range(0, len(dates) - window + 1, window)]
    ts = []
    for symbol, frame in paired.groupby(level="instrument", sort=True):
        series = []
        for days in blocks:
            sample = frame[frame.index.get_level_values("datetime").isin(days)]
            value = correlation(sample, config["timing_min_observations"])
            series.append({"start": str(days[0].date()), "end": str(days[-1].date()), **value})
        summary = summarize(series)
        ts.append({"symbol": symbol, "overall": correlation(frame, config["timing_min_observations"]),
                   "windows": summary, "series": series})
    return {"schema": "signal-ic-v1", "status": "research_only" if valid_cs else "insufficient_history",
            "horizon": horizon, "cross_sectional": cs_summary, "time_series": ts,
            "sufficient_oos": len(valid_cs) >= config["min_oos_dates"],
            "definitions": {"ic": "Pearson(score, forward price return)",
                            "rank_ic": "Spearman(score, forward price return), average ranks for ties",
                            "icir": "mean(IC) / sample_std(IC), ddof=1; unannualized",
                            "selection": "one cross-sectional IC per signal date",
                            "timing": f"per-stock correlations in disjoint {window}-signal-session windows; ICIR across windows",
                            "label": f"t+1 open to t+{horizon} close, split-adjusted price return",
                            "dependence": "Forward returns overlap; ICIR is descriptive stability, not a t-statistic or proof of significance"}}
