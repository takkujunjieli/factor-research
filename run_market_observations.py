#!/usr/bin/env python3
"""Update current stock-pool momentum/volatility without importing a model."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path

import pandas as pd

from qlib_v1.data import atomic_json, calendar, fetch_alpaca, fetch_yahoo, validate_bars
from qlib_v1.observations import realized_statistics
from qlib_v1.screen import market_caps, screen_current
from qlib_v1.universe import resolve, fingerprint

ROOT = Path(__file__).resolve().parent


def build_report(config, bars, sessions, asof, caps=None, revision=None, provider='cache'):
    selected = screen_current(bars, config['symbols'], sessions, asof, config['current_screen'], caps)
    return {
        'schema': 'market-observations-v1', 'generated_at': datetime.now(timezone.utc).isoformat(),
        'asof': str(asof.date()), 'provider': provider, 'settings_revision': revision,
        'universe_id': fingerprint(config, [s['symbol'] for s in selected]),
        'universe_name': config['universe_name'], 'filters': config['current_screen'],
        'rows': realized_statistics(bars, selected, sessions, asof),
        'definitions': {'momentum': 'close(t) / close(t-n) - 1',
                        'volatility': 'sample standard deviation of last n daily simple returns × sqrt(252), ddof=1',
                        'return_basis': 'split-adjusted price return, dividends excluded',
                        'model': None},
    }


def run(args):
    raw = (ROOT / 'config/qlib-universes.json').read_bytes()
    config = resolve(json.loads((ROOT / 'config/qlib-v1.json').read_text()), json.loads(raw),
                     json.loads((ROOT / 'config/qlib-industry-catalog.json').read_text()))
    sessions, asof = calendar(config['start'])
    args.state.mkdir(parents=True, exist_ok=True)
    cache = args.state / 'prices.csv'
    # The existing research cache is read-only here. Updating observations never
    # replaces model data, fits models or appends to the forecast ledger.
    source = cache if cache.exists() else ROOT / 'data/qlib-v1/prices.csv'
    if args.fetch:
        lookback = max(61, config['current_screen']['min_history'], config['current_screen']['volume_window'])
        start = (asof - pd.Timedelta(days=lookback * 2 + 60)).date().isoformat()
        if args.fetch == 'alpaca':
            bars = fetch_alpaca(config['symbols'], start, asof)
            provider = 'alpaca_sip'
        else:
            bars, _ = fetch_yahoo(config['symbols'], start, asof)
            provider = 'yahoo_exploratory'
        bars = validate_bars(bars, sessions, asof)
        if bars.date.max() != asof:
            raise ValueError('最新完整交易日行情尚未覆盖')
        cache.write_text(bars.to_csv(index=False, date_format='%Y-%m-%d'))
        atomic_json(args.state / 'source.json', {'provider': provider})
    else:
        if not source.exists():
            raise ValueError('请先更新行情')
        bars = validate_bars(pd.read_csv(source), sessions, asof)
        if not set(config['symbols']).issubset(set(bars.symbol)) or bars.date.max() != asof:
            raise ValueError('当前股票池或日期未被缓存覆盖，请更新行情')
        meta = args.state / 'source.json' if source == cache else ROOT / 'data/qlib-v1/prices.meta.json'
        provider = json.loads(meta.read_text()).get('provider', 'cache') if meta.exists() else 'cache'
    filters = config['current_screen']
    caps = market_caps(config['symbols'], args.state) if filters['min_market_cap'] > 0 or filters['max_market_cap'] is not None else {}
    result = build_report(config, bars, sessions, asof, caps, hashlib.sha256(raw).hexdigest(), provider)
    atomic_json(args.output, result)
    print(f"Observed {len(result['rows'])} stocks through {result['asof']}; no prediction/model run", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', choices=['alpaca', 'yahoo'])
    parser.add_argument('--state', type=Path, default=ROOT / 'data/market-observations')
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'noetic-dashboard/data/market_observations.json')
    args = parser.parse_args()
    args.state.mkdir(parents=True, exist_ok=True)
    with (args.state / '.run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)
        except Exception as exc:
            print(f'行情统计更新失败 ({type(exc).__name__})；保留上次结果', flush=True)
            if isinstance(exc, (ValueError, FileNotFoundError)):
                print(str(exc), flush=True)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
