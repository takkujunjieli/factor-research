"""SQLite append-only forecasts and separately appended settlements."""
from __future__ import annotations

import json
import sqlite3

import pandas as pd

from .data import canonical_hash


def connect(path):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS forecasts(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS settlements(id TEXT PRIMARY KEY REFERENCES forecasts(id), payload TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS forecasts_no_update BEFORE UPDATE ON forecasts BEGIN SELECT RAISE(ABORT, 'immutable forecast'); END;
        CREATE TRIGGER IF NOT EXISTS forecasts_no_delete BEFORE DELETE ON forecasts BEGIN SELECT RAISE(ABORT, 'immutable forecast'); END;
        CREATE TRIGGER IF NOT EXISTS settlements_no_update BEFORE UPDATE ON settlements BEGIN SELECT RAISE(ABORT, 'immutable settlement'); END;
        CREATE TRIGGER IF NOT EXISTS settlements_no_delete BEFORE DELETE ON settlements BEGIN SELECT RAISE(ABORT, 'immutable settlement'); END;
    """)
    return db


def append_forecasts(db, rows):
    with db:
        for row in rows:
            identity = {key: row[key] for key in ("asof", "symbol", "horizon", "model_id", "universe_id")}
            row = {**row, "id": canonical_hash(identity)}
            existing = db.execute("SELECT payload FROM forecasts WHERE id=?", (row["id"],)).fetchone()
            if existing:
                old = json.loads(existing[0])
                # Rerun timestamp may change; the issued numerical prediction must not.
                if any(old[k] != row[k] for k in ("score", "entry_date", "target_date", "rank")):
                    raise ValueError("Forecast identity collision: attempted history rewrite")
                continue
            db.execute("INSERT INTO forecasts VALUES (?, ?)", (row["id"], json.dumps(row, allow_nan=False)))


def settle(db, bars, asof, data_hash):
    prices = bars.set_index(["date", "symbol"])
    existing = {row[0] for row in db.execute("SELECT id FROM settlements")}
    with db:
        for ident, payload in db.execute("SELECT id, payload FROM forecasts").fetchall():
            row = json.loads(payload)
            if ident in existing or row["target_date"] > str(asof.date()):
                continue
            import pandas as pd
            try:
                # Same adjustment vintage for entry and exit; never compare a frozen
                # pre-split price with a freshly split-adjusted exit.
                entry = float(prices.loc[(pd.Timestamp(row["entry_date"]), row["symbol"]), "open"])
                exit_price = float(prices.loc[(pd.Timestamp(row["target_date"]), row["symbol"]), "close"])
            except KeyError:
                continue
            result = {"id": ident, "entry_price": entry, "exit_price": exit_price,
                      "actual_return": exit_price / entry - 1, "data_hash": data_hash,
                      "settled_asof": str(asof.date())}
            db.execute("INSERT INTO settlements VALUES (?, ?)", (ident, json.dumps(result, allow_nan=False)))


def summary(db, asof, config=None):
    from .signal_metrics import evaluate_signal
    config = config or {"min_candidates": 3, "timing_window": 63, "timing_min_observations": 20, "min_oos_dates": 252}
    forecasts = [json.loads(r[0]) for r in db.execute("SELECT payload FROM forecasts ORDER BY rowid")]
    settled = {r[0]: json.loads(r[1]) for r in db.execute("SELECT id, payload FROM settlements")}
    horizons = {}
    for horizon in sorted({r["horizon"] for r in forecasts}):
        rows = [r for r in forecasts if r["horizon"] == horizon]
        pairs = [(r, settled[r["id"]]) for r in rows if r["id"] in settled]
        # Keep every issued version in the audit log. Score only the latest
        # pre-entry cohort per signal date so revisions cannot multiply samples.
        cohorts = {}
        for r in rows:
            key = (r["universe_id"], r["asof"])
            if key not in cohorts or r["generated_at"] > cohorts[key][0]:
                cohorts[key] = (r["generated_at"], r["model_id"])
        latest = [r for r in rows if r["model_id"] == cohorts[(r["universe_id"], r["asof"])][1]]
        # Different stock universes are separate experiments, never pooled.
        evaluations = {}
        for universe in sorted({r["universe_id"] for r in latest}):
            sample = [r for r in latest if r["universe_id"] == universe]
            index = pd.MultiIndex.from_tuples([(pd.Timestamp(r["asof"]), r["symbol"]) for r in sample], names=["datetime", "instrument"])
            scores = pd.Series([r["score"] for r in sample], index=index, dtype=float)
            targets = pd.Series([settled.get(r["id"], {}).get("actual_return", float("nan")) for r in sample], index=index, dtype=float)
            evaluations[universe] = evaluate_signal(scores, targets, config, horizon)
        horizons[str(horizon)] = {
            "issued": len(rows), "settled": len(pairs),
            "independent_signal_dates": len({r[0]["asof"] for r in pairs}),
            "overdue_missing_prices": sum(r["target_date"] <= str(asof.date()) and r["id"] not in settled for r in rows),
            "signal_evaluations": evaluations,
            "cohort_policy": "latest pre-entry model per universe/signal date; all older versions retained in audit",
        }
    recent = []
    for row in forecasts[-500:][::-1]:
        outcome = settled.get(row["id"])
        recent.append({**row, "actual_return": outcome["actual_return"] if outcome else None,
                       "status": "settled" if outcome else "missing_prices" if row["target_date"] <= str(asof.date()) else "pending"})
    return {"total": len(forecasts), "settled": len(settled), "horizons": horizons, "recent": recent}
