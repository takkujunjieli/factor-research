"""Apply user thresholds once at the current snapshot, never historically."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math

from .data import atomic_json


def market_caps(symbols, state, refresh=False):
    import yfinance as yf
    yf.set_tz_cache_location(str(state / "yahoo-cache"))
    today = datetime.now(timezone.utc).date().isoformat()
    path = state / "market_caps.json"
    cached = json.loads(path.read_text()) if path.exists() else {}
    values = dict(cached.get("values", {})) if cached.get("asof") == today and not refresh else {}
    for symbol in symbols:
        if symbol in values:
            continue
        try:
            value = float(yf.Ticker(symbol.replace(".", "-")).fast_info["market_cap"])
            values[symbol] = value if math.isfinite(value) and value > 0 else None
        except Exception:
            values[symbol] = None
    atomic_json(path, {"asof": today, "values": values, "source": "yahoo_fast_info"})
    if symbols and not any(values.get(symbol) for symbol in symbols):
        raise ValueError("当前市值数据暂不可用，请稍后刷新")
    return values


def screen_current(bars, symbols, sessions, asof, filters, caps=None):
    caps = caps or {}
    eligible = []
    for symbol in symbols:
        rows = bars[bars.symbol == symbol].set_index("date").reindex(sessions[sessions <= asof])
        if rows.empty or rows.close.iloc[-filters["min_history"]:].count() < filters["min_history"]:
            continue
        price = rows.close.iloc[-1]
        daily_dollars = (rows.close * rows.volume).iloc[-filters["volume_window"]:]
        if daily_dollars.count() < filters["volume_window"]:
            continue
        dollar_volume = daily_dollars.mean()
        if not math.isfinite(price) or not math.isfinite(dollar_volume):
            continue
        if price < filters["min_price"] or dollar_volume < filters["min_dollar_volume"]:
            continue
        cap = caps.get(symbol)
        if filters["min_market_cap"] > 0 or filters["max_market_cap"] is not None:
            if cap is None or not math.isfinite(cap) or cap <= 0:
                continue
            if cap < filters["min_market_cap"] or (filters["max_market_cap"] is not None and cap > filters["max_market_cap"]):
                continue
        eligible.append({"symbol": symbol, "price": float(price), "market_cap": cap,
                         "average_dollar_volume": float(dollar_volume), "asof": str(asof.date())})
    return eligible
