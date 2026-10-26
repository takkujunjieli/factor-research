import copy
import json
import unittest
from pathlib import Path

import pandas as pd

from qlib_v1.universe import resolve, validate_settings, fingerprint
from qlib_v1.screen import screen_current
from qlib_v1.research import make_panel, FEATURES, current_predictions, rolling_predictions, evaluate
from test_qlib_v1 import fixture

ROOT = Path(__file__).resolve().parents[1]


class UniverseTest(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((ROOT / 'config/qlib-v1.json').read_text())
        self.catalog = json.loads((ROOT / 'config/qlib-industry-catalog.json').read_text())
        self.settings = json.loads((ROOT / 'config/qlib-universes.json').read_text())

    def test_theme_expansion_include_exclude_and_no_base_pool(self):
        p = self.settings['profiles']['default']
        p.update(include_base=False, groups=['cryptocurrency','semiconductors'], include=[' purr ', 'NVDA'], exclude=['STX'])
        config = resolve(self.base, self.settings, self.catalog)
        expected = {s for group in self.catalog['groups'] if group['id'] in p['groups'] for s in group['symbols']} - {'STX'}
        self.assertEqual(config['symbols'], sorted(expected))
        self.assertTrue(config['screen_current_only'])
        self.assertEqual(config['min_candidates'], 3)
        self.assertEqual(len(self.base['symbols']), 103)

    def test_merged_categories_keep_all_members_and_migrate_saved_selection(self):
        groups = {g['id']:g for g in self.catalog['groups']}
        self.assertEqual(set(groups['cryptocurrency']['symbols']), {'PURR','MSTR','ASST','SMLR','BMNR','SBET','BTBT','DFDV','UPXI'})
        self.assertEqual(len(groups['semiconductors']['symbols']),33)
        self.assertFalse(any(g.startswith('chip-') or g in ['hype-treasury','btc-treasury','eth-treasury','digital-asset-treasury'] for g in groups))
        self.settings['profiles']['default']['groups'] = ['chip-memory','chip-design','hype-treasury','btc-treasury']
        cleaned = validate_settings(self.settings,self.catalog)
        self.assertEqual(cleaned['profiles']['default']['groups'],['cryptocurrency','semiconductors'])

    def test_invalid_settings_do_not_pass_to_runner(self):
        for update in ({'include':['AAPL;rm -rf /']}, {'groups':['unknown']},
                       {'filters':{'min_price':float('nan')}}, {'filters':{'min_market_cap':100,'max_market_cap':10}},
                       {'filters':{'volume_window':0}}, {'filters':{'min_history':30}}, {'command':'echo'}):
            settings = copy.deepcopy(self.settings)
            settings['profiles']['default'].update(update)
            with self.assertRaises(ValueError):
                validate_settings(settings, self.catalog)

    def test_current_cap_price_liquidity_filters_and_no_historical_reapplication(self):
        bars, sessions, config = fixture(n=390)
        asof = sessions[389]
        filters = {'min_price':50,'min_dollar_volume':100,'volume_window':20,
                   'min_market_cap':1e9,'max_market_cap':3e9,'min_history':61}
        selected = screen_current(bars, ['AAA','BBB','CCC'], sessions, asof, filters,
                                  {'AAA':2e9,'BBB':4e9})
        self.assertEqual([r['symbol'] for r in selected], ['AAA'])
        self.assertNotIn('excluded', selected[0])
        # A past low price is permitted when today's screen selected the stock.
        old = bars.copy()
        old.loc[(old.symbol=='AAA') & (old.date < sessions[100]), ['open','high','low','close']] *= .01
        config.update(screen_current_only=True)
        panel = make_panel(old[old.symbol=='AAA'], sessions, config)
        self.assertTrue(panel.loc[(sessions[80],'AAA'),'eligible'])

    def test_zero_or_one_stock_pool_does_not_crash_feature_builder(self):
        bars, sessions, config = fixture()
        panel = make_panel(bars.iloc[:0], sessions, config)
        self.assertTrue(panel.empty)
        self.assertTrue(set(FEATURES).issubset(panel.columns))
        scores, folds = rolling_predictions(panel, sessions, config, 5)
        self.assertTrue(scores.empty)
        self.assertEqual(folds, [])
        live, artifact = current_predictions(panel, sessions, config, 5, sessions[389])
        self.assertTrue(live.empty)
        self.assertIsNone(artifact)
        self.assertEqual(evaluate(scores, panel, config, 5)['status'], 'insufficient_history')

    def test_screened_membership_changes_ledger_identity(self):
        config = resolve(self.base, self.settings, self.catalog)
        self.assertNotEqual(fingerprint(config,['PURR']),fingerprint(config,['PURR','NVDA']))


if __name__ == '__main__':
    unittest.main()
