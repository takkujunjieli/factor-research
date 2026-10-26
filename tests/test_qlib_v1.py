"""Behavioral checks for temporal leakage, execution and immutable predictions."""
from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from qlib_v1.data import calendar, validate_bars
from qlib_v1.ledger import append_forecasts, connect, settle, summary
from qlib_v1.research import FEATURES, current_predictions, evaluate, make_panel, rolling_predictions
from run_qlib import run

ROOT = Path(__file__).resolve().parents[1]


def fixture(n=390, symbols=("AAA", "BBB", "CCC", "SPY")):
    sessions, _ = calendar("2022-01-01", "2026-10-25T02:00:00Z")
    sessions = sessions[:n + 70]
    rng = np.random.default_rng(42)
    rows = []
    for j, symbol in enumerate(symbols):
        close = 100 * np.exp(np.cumsum(rng.normal(.0004, .012, n)))
        for i, day in enumerate(sessions[:n]):
            opening = close[i] * (1 + .002 * np.sin(i + j))
            rows.append(dict(date=day, symbol=symbol, open=opening, high=max(opening, close[i]) * 1.01,
                             low=min(opening, close[i]) * .99, close=close[i], volume=2000000))
    config = json.loads((ROOT / "config/qlib-v1.json").read_text())
    config.update(symbols=list(symbols[:-1]), start="2022-01-01", horizons=[5, 20], min_train_sessions=80,
                  train_sessions=120, refit_sessions=40, display_limit=10, min_candidates=3,
                  min_oos_dates=40, min_dollar_volume=1, timing_window=63, timing_min_observations=20)
    return pd.DataFrame(rows), sessions, config


class CausalFeatures(unittest.TestCase):
    def test_provider_bad_ohlc_is_quarantined_without_repair_or_horizon_compression(self):
        bars, sessions, config = fixture()
        bad = (bars.symbol == "AAA") & (bars.date == sessions[103])
        bars.loc[bad, "low"] = bars.loc[bad, ["open", "close"]].max(axis=1) + .23
        with self.assertRaisesRegex(ValueError, "Inconsistent OHLC.*AAA"):
            validate_bars(bars, sessions, sessions[389])
        cleaned = validate_bars(bars, sessions, sessions[389], quarantine_ohlc=True)
        self.assertEqual(len(cleaned), len(bars) - 1)
        self.assertIn("AAA", set(cleaned.symbol))
        self.assertEqual(cleaned.attrs["quarantined_bars"][0]["symbol"], "AAA")
        self.assertEqual(cleaned.attrs["quarantined_bars"][0]["date"], str(sessions[103].date()))
        panel = make_panel(cleaned, sessions, config)
        self.assertTrue(np.isnan(panel.loc[(sessions[100], "AAA"), "label_5"]))
        self.assertFalse(panel.loc[(sessions[104], "AAA"), "eligible"])

    def test_future_changes_do_not_change_past_features(self):
        bars, sessions, config = fixture()
        before = make_panel(bars, sessions, config)
        mutated = bars.copy()
        day = sessions[150]
        mutated.loc[mutated.date > day, ["open", "high", "low", "close", "volume"]] *= 3
        after = make_panel(mutated, sessions, config)
        pd.testing.assert_frame_equal(before.loc[:day, FEATURES], after.loc[:day, FEATURES])

    def test_label_uses_next_open_and_exact_session_exit(self):
        bars, sessions, config = fixture()
        panel = make_panel(bars, sessions, config)
        b = bars[bars.symbol == "AAA"].set_index("date")
        expected = b.loc[sessions[105], "close"] / b.loc[sessions[101], "open"] - 1
        self.assertAlmostEqual(panel.loc[(sessions[100], "AAA"), "label_5"], expected)
        self.assertTrue(np.isnan(panel.loc[(sessions[389], "AAA"), "label_5"]))

    def test_missing_session_does_not_shorten_target_horizon(self):
        bars, sessions, config = fixture()
        bars = bars[~((bars.symbol == "AAA") & (bars.date == sessions[103]))]
        panel = make_panel(bars, sessions, config)
        self.assertTrue(np.isnan(panel.loc[(sessions[100], "AAA"), "label_5"]))
        self.assertFalse(panel.loc[(sessions[104], "AAA"), "eligible"])

    def test_duplicates_bad_prices_and_weekends_rejected(self):
        bars, sessions, _ = fixture()
        for broken in (pd.concat([bars, bars.iloc[:1]]), bars.assign(close=-1), bars.assign(date=pd.Timestamp("2022-01-08"))):
            with self.assertRaises(ValueError):
                validate_bars(broken, sessions, sessions[389])

    def test_calendar_excludes_partial_session_and_handles_holiday(self):
        _, before_close = calendar("2026-01-01", "2026-10-23T19:00:00Z")
        _, after_close = calendar("2026-01-01", "2026-10-23T20:21:00Z")
        self.assertEqual(str(before_close.date()), "2026-10-22")
        self.assertEqual(str(after_close.date()), "2026-10-23")
        sessions, _ = calendar("2026-01-01", "2026-07-06T22:00:00Z")
        self.assertNotIn(pd.Timestamp("2026-07-03"), sessions)


class Ledger(unittest.TestCase):
    def test_forward_ic_uses_latest_cohort_without_multiplying_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = connect(Path(tmp) / "ledger.sqlite")
            rows = []
            for version, stamp in (("old", "2026-10-24T10:00:00Z"), ("new", "2026-10-24T11:00:00Z")):
                for rank, symbol in enumerate(("AAA", "BBB", "CCC"), 1):
                    rows.append(dict(asof="2026-10-23", symbol=symbol, horizon=5, model_id=version,
                                     universe_id="test", score=rank if version == "new" else -rank,
                                     rank=rank, entry_date="2026-10-26", target_date="2026-10-30", generated_at=stamp))
            append_forecasts(db, rows)
            for ident, payload in db.execute("SELECT id,payload FROM forecasts").fetchall():
                r = json.loads(payload)
                db.execute("INSERT INTO settlements VALUES (?,?)", (ident, json.dumps({"actual_return": r["rank"] * .01})))
            db.commit()
            report = summary(db, pd.Timestamp("2026-10-30"))
            self.assertEqual(report["total"], 6)
            cs = report["horizons"]["5"]["signal_evaluations"]["test"]["cross_sectional"]
            self.assertEqual(cs["paired_observations"], 3)
            self.assertEqual(cs["ic_periods"], 1)
            self.assertAlmostEqual(cs["rank_ic"], 1)
            self.assertIsNone(cs["icir"])
            db.close()

    def test_immutable_idempotent_and_separate_settlement(self):
        bars, sessions, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            db = connect(Path(tmp) / "ledger.sqlite")
            row = dict(asof=str(sessions[100].date()), symbol="AAA", horizon=5, model_id="model1", universe_id="test",
                       score=.02, rank=1, entry_date=str(sessions[101].date()), target_date=str(sessions[105].date()), generated_at="original")
            append_forecasts(db, [row])
            append_forecasts(db, [{**row, "generated_at": "rerun"}])
            self.assertEqual(summary(db, sessions[104])["total"], 1)
            settle(db, bars, sessions[104], "snapshot1")
            self.assertEqual(summary(db, sessions[104])["settled"], 0)
            settle(db, bars, sessions[105], "snapshot2")
            before = summary(db, sessions[105])
            settle(db, bars.assign(close=1000), sessions[106], "revised")
            self.assertEqual(before["recent"][0]["actual_return"], summary(db, sessions[106])["recent"][0]["actual_return"])
            with self.assertRaises(ValueError):
                append_forecasts(db, [{**row, "score": .99}])
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("DELETE FROM forecasts")
            self.assertEqual(before["recent"][0]["generated_at"], "original")
            db.close()


class QlibIntegration(unittest.TestCase):
    def test_real_qlib_folds_are_purged_and_future_changes_do_not_change_predictions(self):
        bars, sessions, config = fixture(n=230)
        panel = make_panel(bars[bars.symbol != "SPY"], sessions, config)
        scores, folds = rolling_predictions(panel, sessions, config, 20)
        self.assertGreater(len(scores), 0)
        for fold in folds:
            self.assertLess(fold["latest_training_outcome"], fold["test_start"])
        live, artifact = current_predictions(panel, sessions, config, 20, sessions[229])
        pd.testing.assert_series_equal(live, scores.loc[[sessions[229]]])
        self.assertEqual(artifact["fit_asof"], folds[-1]["test_start"])
        changed = bars.copy()
        boundary = sessions[190]
        changed.loc[changed.date > boundary, ["open", "close", "high", "low"]] *= 1.5
        other, _ = rolling_predictions(make_panel(changed[changed.symbol != "SPY"], sessions, config), sessions, config, 20)
        pd.testing.assert_series_equal(scores.loc[:boundary], other.loc[:boundary])

    def test_cli_export_demo_has_no_live_ledger_and_repeated_run_is_stable(self):
        from argparse import Namespace
        bars, sessions, config = fixture(n=230)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config_file, prices = tmp / "config.json", tmp / "prices.csv"
            config_file.write_text(json.dumps(config))
            bars.to_csv(prices, index=False)
            args = Namespace(config=config_file, state=tmp / "state", output=tmp / "report.json",
                             prices=prices, fetch=None, asof=str(sessions[229].date()), demo=True)
            report = run(args)
            self.assertEqual(report["mode"], "synthetic_demo")
            self.assertEqual(report["ledger"]["total"], 0)
            self.assertFalse((args.state / "forecast-ledger.sqlite").exists())
            self.assertEqual(len(report["horizons"]["5"]["candidates"]), 3)
            self.assertNotIn("selected", report["horizons"]["5"]["candidates"][0])
            self.assertNotIn("metrics", report["horizons"]["5"]["evaluation"])
            self.assertNotIn("curves", report["horizons"]["5"]["evaluation"])
            rerun = run(args)
            self.assertEqual(report["horizons"]["5"]["evaluation"], rerun["horizons"]["5"]["evaluation"])
            self.assertEqual(report["horizons"]["5"]["model"]["model_id"], rerun["horizons"]["5"]["model"]["model_id"])
            # Invalid refresh leaves previously good output intact.
            saved = args.output.read_bytes()
            broken = copy.copy(args)
            broken.asof = "2099-01-01"
            with self.assertRaises(ValueError):
                run(broken)
            self.assertEqual(saved, args.output.read_bytes())


if __name__ == "__main__":
    unittest.main()
