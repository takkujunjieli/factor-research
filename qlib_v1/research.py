"""Causal daily features, Qlib Ridge and purged chronological evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["momentum_5", "momentum_20", "momentum_60", "volatility_20",
            "range_20", "volume_ratio", "distance_ma20", "gap", "intraday"]


def make_panel(bars, sessions, config):
    observed = sessions[sessions <= bars.date.max()]
    chunks = []
    for symbol, raw in bars.groupby("symbol"):
        b = raw.set_index("date").reindex(observed)
        close = b.close
        ret = close.pct_change(fill_method=None)
        f = pd.DataFrame(index=observed)
        for n in (5, 20, 60):
            f[f"momentum_{n}"] = close / close.shift(n) - 1
        f["volatility_20"] = ret.rolling(20).std()
        f["range_20"] = ((b.high - b.low) / close).rolling(20).mean()
        f["volume_ratio"] = b.volume / b.volume.rolling(20).mean() - 1
        f["distance_ma20"] = close / close.rolling(20).mean() - 1
        f["gap"] = b.open / close.shift(1) - 1
        f["intraday"] = close / b.open - 1
        quality = close.rolling(61).count().eq(61)
        f["eligible"] = quality if config.get("screen_current_only") else (
            (close >= config["min_price"])
            & ((close * b.volume).rolling(20).mean() >= config["min_dollar_volume"])
            & quality)
        for h in config["horizons"]:
            # Signal after t close -> entry t+1 open -> exit t+h close.
            complete = b.close.notna().iloc[::-1].rolling(h).sum().iloc[::-1].shift(-1).eq(h)
            f[f"label_{h}"] = (close.shift(-h) / b.open.shift(-1) - 1).where(complete)
        f["symbol"] = symbol
        chunks.append(f)
    if not chunks:
        index = pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])
        return pd.DataFrame(index=index, columns=FEATURES + ["eligible"] + [f"label_{h}" for h in config["horizons"]])
    panel = pd.concat(chunks).reset_index(names="datetime").set_index(["datetime", "symbol"]).sort_index()
    panel.index.names = ["datetime", "instrument"]
    return panel.replace([np.inf, -np.inf], np.nan)


def fit_predict(train, test, horizon, alpha):
    """Use Qlib's real DatasetH/DataHandlerLP/LinearModel, not a substitute model."""
    from qlib.contrib.model.linear import LinearModel
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    mean, scale = train[FEATURES].mean(), train[FEATURES].std().replace(0, 1).fillna(1)
    combined = pd.concat([train, test]).sort_index()
    x = ((combined[FEATURES] - mean) / scale).clip(-10, 10)
    y = combined[[f"label_{horizon}"]].rename(columns={f"label_{horizon}": "return"})
    frame = pd.concat({"feature": x, "label": y}, axis=1)
    handler = DataHandlerLP(data_loader=StaticDataLoader(frame), infer_processors=[], learn_processors=[])
    dataset = DatasetH(handler=handler, segments={
        "train": (train.index.get_level_values(0).min(), train.index.get_level_values(0).max()),
        "test": (test.index.get_level_values(0).min(), test.index.get_level_values(0).max())})
    model = LinearModel(estimator="ridge", alpha=alpha, fit_intercept=True)
    model.fit(dataset)
    predictions = model.predict(dataset)
    artifact = {"feature_names": FEATURES, "mean": mean.to_dict(), "scale": scale.to_dict(),
                "coefficients": dict(zip(FEATURES, np.asarray(model.coef_).reshape(-1).tolist())),
                "intercept": float(model.intercept_)}
    return predictions, artifact


def rolling_predictions(panel, sessions, config, horizon):
    dates = panel.index.get_level_values(0)
    usable = panel[panel.eligible & panel[FEATURES].notna().all(axis=1)]
    unique = pd.DatetimeIndex(sorted(usable.index.get_level_values(0).unique()))
    positions = {day: i for i, day in enumerate(sessions)}
    predictions, folds = [], []
    cursor = config["min_train_sessions"] + horizon
    while cursor < len(unique):
        test_days = unique[cursor:cursor + config["refit_sessions"]]
        first = test_days[0]
        # All training outcomes must be observed strictly before the test signal day.
        end_position = positions[first] - horizon - 1
        train_days = sessions[max(0, end_position - config["train_sessions"] + 1):end_position + 1]
        train = usable[usable.index.get_level_values(0).isin(train_days)].dropna(subset=[f"label_{horizon}"])
        test = usable[usable.index.get_level_values(0).isin(test_days)]
        if train.index.get_level_values(0).nunique() < config["min_train_sessions"] or test.empty:
            cursor += config["refit_sessions"]
            continue
        pred, _ = fit_predict(train, test, horizon, config["ridge_alpha"])
        predictions.append(pred)
        last_train = train.index.get_level_values(0).max()
        folds.append({"train_start": str(train.index.get_level_values(0).min().date()),
                      "train_end": str(last_train.date()),
                      "latest_training_outcome": str(sessions[positions[last_train] + horizon].date()),
                      "test_start": str(first.date()), "test_end": str(test_days[-1].date()),
                      "train_rows": len(train), "test_rows": len(test)})
        cursor += config["refit_sessions"]
    scores = pd.concat(predictions).sort_index() if predictions else pd.Series(dtype=float)
    return scores, folds


def current_predictions(panel, sessions, config, horizon, asof):
    usable = panel[panel.eligible & panel[FEATURES].notna().all(axis=1)]
    # Match the OOS refit cadence: daily inference uses the last scheduled model,
    # rather than silently evaluating monthly refits but deploying daily refits.
    unique = pd.DatetimeIndex(sorted(usable.index.get_level_values(0).unique()))
    first_test = config["min_train_sessions"] + horizon
    if len(unique) <= first_test or asof not in unique:
        return pd.Series(dtype=float), None
    cursor = first_test + ((unique.get_loc(asof) - first_test) // config["refit_sessions"]) * config["refit_sessions"]
    fit_asof = unique[cursor]
    pos = sessions.get_loc(fit_asof)
    end_position = pos - horizon - 1
    train_days = sessions[max(0, end_position - config["train_sessions"] + 1):end_position + 1]
    train = usable[usable.index.get_level_values(0).isin(train_days)].dropna(subset=[f"label_{horizon}"])
    test = usable[usable.index.get_level_values(0) == asof]
    if train.index.get_level_values(0).nunique() < config["min_train_sessions"] or test.empty:
        return pd.Series(dtype=float), None
    pred, artifact = fit_predict(train, test, horizon, config["ridge_alpha"])
    artifact.update(train_start=str(train.index.get_level_values(0).min().date()),
                    train_end=str(train.index.get_level_values(0).max().date()), train_rows=len(train),
                    fit_asof=str(fit_asof.date()),
                    latest_training_outcome=str(sessions[sessions.get_loc(train.index.get_level_values(0).max()) + horizon].date()))
    return pred, artifact


def evaluate(scores, panel, config, horizon):
    from .signal_metrics import evaluate_signal
    return evaluate_signal(scores, panel[f"label_{horizon}"], config, horizon)
