"""Realized price statistics only. No models, labels or forecasts."""
import numpy as np


def realized_statistics(bars, selected, sessions, asof):
    result = []
    dates = sessions[sessions <= asof]
    for stock in selected:
        symbol = stock['symbol']
        close = bars[bars.symbol == symbol].set_index('date').close.reindex(dates)
        values = {}
        for days in (60, 20, 5):
            sample = close.iloc[-days - 1:]
            complete = len(sample) == days + 1 and sample.notna().all()
            returns = sample.pct_change(fill_method=None).iloc[1:]
            values[str(days)] = {
                'momentum': float(sample.iloc[-1] / sample.iloc[0] - 1) if complete else None,
                'volatility': float(returns.std(ddof=1) * np.sqrt(252)) if complete else None,
            }
        result.append({**stock, 'periods': values})
    return result
