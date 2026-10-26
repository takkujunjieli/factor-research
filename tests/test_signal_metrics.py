import unittest

import numpy as np
import pandas as pd

from qlib_v1.signal_metrics import correlation, evaluate_signal, summarize

CONFIG = {"min_candidates": 3, "timing_window": 6, "timing_min_observations": 3, "min_oos_dates": 5}


class SignalMetricsTest(unittest.TestCase):
    def test_no_predictions_is_explicitly_insufficient(self):
        value = evaluate_signal(pd.Series(dtype=float), pd.Series(dtype=float), CONFIG, 5)
        self.assertEqual(value["status"], "insufficient_history")
        self.assertIsNone(value["cross_sectional"]["icir"])

    def test_pearson_and_spearman_have_distinct_correct_definitions(self):
        frame = pd.DataFrame({"score": [1, 2, 3, 4], "target": [1, 2, 3, 100]})
        value = correlation(frame, 3)
        self.assertAlmostEqual(value["rank_ic"], 1)
        self.assertAlmostEqual(value["ic"], frame.score.corr(frame.target))
        self.assertLess(value["ic"], .9)

    def test_icir_uses_sample_std_without_annualization_and_handles_zero_variance(self):
        rows = [{"ic": x, "rank_ic": x} for x in [-.1, .2, .4]]
        result = summarize(rows)
        self.assertAlmostEqual(result["icir"], np.mean([-.1, .2, .4]) / np.std([-.1, .2, .4], ddof=1))
        self.assertIsNone(summarize([{"ic": 1., "rank_ic": 1.}] * 3)["icir"])
        self.assertIsNone(summarize([])["rank_ic"])

    def test_selection_and_timing_are_distinct_not_pooled_stock_date_correlation(self):
        days = pd.bdate_range("2026-01-01", periods=18)
        index = pd.MultiIndex.from_product([days, ["A", "B", "C"]], names=["datetime", "instrument"])
        # Each day ranks stocks perfectly, but within each stock the score
        # is constant, so temporal correlation is undefined.
        scores = pd.Series(np.tile([1., 2., 3.], len(days)), index=index)
        target = scores + np.repeat(np.arange(len(days)), 3)
        result = evaluate_signal(scores, target, CONFIG, 5)
        self.assertAlmostEqual(result["cross_sectional"]["rank_ic"], 1)
        self.assertEqual(result["cross_sectional"]["ic_periods"], 18)
        for stock in result["time_series"]:
            self.assertIsNone(stock["overall"]["ic"])
            self.assertEqual(stock["windows"]["ic_periods"], 0)
            self.assertEqual(len(stock["series"]), 3)
        self.assertNotIn("curves", result)
        self.assertNotIn("metrics", result)

    def test_pending_labels_and_small_cross_sections_are_not_zero_ic(self):
        days = pd.bdate_range("2026-01-01", periods=3)
        index = pd.MultiIndex.from_product([days, ["A", "B", "C"]], names=["datetime", "instrument"])
        scores = pd.Series(np.tile([1., 2., 3.], 3), index=index)
        target = scores.copy()
        target.iloc[-4:] = np.nan
        result = evaluate_signal(scores, target, CONFIG, 20)
        self.assertEqual(result["cross_sectional"]["ic_periods"], 1)
        self.assertEqual(result["cross_sectional"]["invalid_dates"], 2)
        self.assertEqual(result["cross_sectional"]["missing_labels"], 4)
        self.assertIsNone(result["cross_sectional"]["icir"])


if __name__ == "__main__":
    unittest.main()
