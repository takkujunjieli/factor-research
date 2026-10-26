import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from qlib_v1.observations import realized_statistics
import run_market_observations as runner
from test_qlib_v1 import fixture


class ObservationsTest(unittest.TestCase):
    def test_exact_trading_day_momentum_and_annualized_realized_volatility(self):
        bars,sessions,_=fixture(n=100)
        selected=[{'symbol':'AAA','price':float(bars[bars.symbol=='AAA'].close.iloc[-1])}]
        row=realized_statistics(bars,selected,sessions,sessions[99])[0]
        close=bars[bars.symbol=='AAA'].close.reset_index(drop=True)
        for n in (5,20,60):
            expected_momentum=close.iloc[-1]/close.iloc[-n-1]-1
            expected_vol=close.iloc[-n-1:].pct_change().iloc[1:].std(ddof=1)*np.sqrt(252)
            self.assertAlmostEqual(row['periods'][str(n)]['momentum'],expected_momentum)
            self.assertAlmostEqual(row['periods'][str(n)]['volatility'],expected_vol)

    def test_missing_dates_are_not_compressed_and_future_prices_are_ignored(self):
        bars,sessions,_=fixture(n=100)
        bars=bars[~((bars.symbol=='AAA')&(bars.date==sessions[50]))]
        selected=[{'symbol':'AAA'}]
        row=realized_statistics(bars,selected,sessions,sessions[99])[0]
        self.assertIsNone(row['periods']['60']['momentum'])
        self.assertIsNotNone(row['periods']['20']['volatility'])
        shorter=bars[bars.date<=sessions[90]]
        self.assertEqual(realized_statistics(bars,selected,sessions,sessions[90]),realized_statistics(shorter,selected,sessions,sessions[90]))

    def test_flat_prices_mean_zero_not_missing(self):
        bars,sessions,_=fixture(n=100)
        bars['close']=100.
        value=realized_statistics(bars,[{'symbol':'AAA'}],sessions,sessions[99])[0]
        self.assertEqual(value['periods']['60'],{'momentum':0.,'volatility':0.})

    def test_cached_cli_builds_current_pool_without_importing_qlib_or_creating_ledger(self):
        bars,sessions,base=fixture(n=100)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config').mkdir();state=root/'state';state.mkdir()
            base.update(symbols=['AAA','BBB','CCC'])
            (root/'config/qlib-v1.json').write_text(json.dumps(base))
            settings={'version':1,'active':'a','profiles':{'a':{'name':'观察池','include_base':True,'groups':[],
                'include':[],'exclude':[],'filters':{'min_price':0,'min_dollar_volume':0,'volume_window':20,
                'min_market_cap':0,'max_market_cap':None,'min_history':61}}}}
            (root/'config/qlib-universes.json').write_text(json.dumps(settings))
            (root/'config/qlib-industry-catalog.json').write_text(json.dumps({'groups':[]}))
            bars.to_csv(state/'prices.csv',index=False)
            args=Namespace(state=state,fetch=None,output=root/'observations.json')
            with patch.object(runner,'ROOT',root),patch.object(runner,'calendar',return_value=(sessions,sessions[99])),patch.dict('sys.modules',{'qlib':None}):
                report=runner.run(args)
            self.assertEqual(len(report['rows']),3)
            self.assertIsNone(report['definitions']['model'])
            self.assertFalse(list(root.rglob('*.sqlite')))
            self.assertNotIn('score',report['rows'][0])
            self.assertTrue(args.output.exists())


if __name__=='__main__':
    unittest.main()
