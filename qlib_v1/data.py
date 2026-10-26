"""Full-refresh price snapshots: never splice incompatible adjustment vintages."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import requests


def calendar(start, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        raise ValueError("now must include timezone")
    cal = xcals.get_calendar("XNYS", start=start, end=now.date() + pd.Timedelta(days=100))
    sessions = cal.sessions.tz_localize(None)
    completed = [s for s in sessions if cal.session_close(s) + pd.Timedelta(minutes=20) <= now]
    if not completed:
        raise ValueError("No completed XNYS sessions")
    return sessions, completed[-1]


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def get_json(session, url, **kwargs):
    for attempt in range(4):
        response = session.get(url, timeout=60, **kwargs)
        if response.status_code not in (429, 500, 502, 503, 504):
            response.raise_for_status()
            return response.json()
        time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Data provider unavailable (HTTP {response.status_code})")


def fetch_alpaca(symbols, start, asof):
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise ValueError("Alpaca requires ALPACA_API_KEY / ALPACA_API_SECRET (or APCA equivalents)")
    session = requests.Session()
    session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    out = []
    for offset in range(0, len(symbols), 25):
        params = {"symbols": ",".join(symbols[offset:offset + 25]), "timeframe": "1Day",
                  "start": f"{start}T00:00:00Z",
                  "end": (asof + pd.Timedelta(days=1)).strftime("%Y-%m-%dT05:00:00Z"),
                  "adjustment": "split", "feed": "sip", "sort": "asc", "limit": 10000}
        while True:
            payload = get_json(session, "https://data.alpaca.markets/v2/stocks/bars", params=params)
            for symbol, bars in payload.get("bars", {}).items():
                for b in bars:
                    day = pd.Timestamp(b["t"]).tz_convert("America/New_York").date().isoformat()
                    out.append(dict(date=day, symbol=symbol, open=b["o"], high=b["h"],
                                    low=b["l"], close=b["c"], volume=b["v"]))
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
    return pd.DataFrame(out)


def fetch_yahoo(symbols, start, asof):
    """Explicit exploratory alternative. Yahoo OHLC are split-adjusted; dividends excluded."""
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    out, errors = [], []
    for symbol in symbols:
        params = {"period1": int(pd.Timestamp(start, tz="America/New_York").timestamp()),
                  "period2": int((asof.tz_localize("America/New_York") + pd.Timedelta(days=1)).timestamp()),
                  "interval": "1d"}
        try:
            payload = get_json(session, f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.replace('.', '-')}", params=params)
            result = payload["chart"]["result"][0]
            quotes = result["indicators"]["quote"][0]
            for i, ts in enumerate(result.get("timestamp", [])):
                row = {field: quotes[field][i] for field in ("open", "high", "low", "close", "volume")}
                if any(value is None for value in row.values()):
                    continue
                out.append(dict(date=pd.Timestamp(ts, unit="s", tz="UTC").tz_convert("America/New_York").date().isoformat(), symbol=symbol, **row))
        except (requests.RequestException, KeyError, IndexError, TypeError, RuntimeError):
            errors.append(symbol)
        time.sleep(0.15)
    return pd.DataFrame(out), errors


def validate_bars(frame, sessions, asof, *, quarantine_ohlc=False):
    required = ["date", "symbol", "open", "high", "low", "close", "volume"]
    if any(c not in frame for c in required):
        raise ValueError("Prices require columns: " + ", ".join(required))
    frame = frame[required].copy()
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    if frame.date.dt.tz is not None:
        raise ValueError("Daily price dates must be timezone-free session dates")
    if not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError("Daily price dates must be normalized")
    if not frame.symbol.map(lambda s: isinstance(s, str) and bool(re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", s))).all():
        raise ValueError("Invalid symbol")
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate symbol/session rows")
    for col in required[2:]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    values = frame[required[2:]].to_numpy()
    if not np.isfinite(values).all() or (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame.volume < 0).any():
        raise ValueError("Non-finite, non-positive price or negative volume")
    inconsistent = (frame.high < frame[["open", "close", "low"]].max(axis=1)) | (frame.low > frame[["open", "close", "high"]].min(axis=1))
    quarantined = []
    if inconsistent.any():
        bad = frame.loc[inconsistent].iloc[0]
        if not quarantine_ohlc:
            raise ValueError(f"Inconsistent OHLC: {int(inconsistent.sum())} bars; {bad.symbol} {bad.date.date()} "
                             f"open={bad.open:.12g} high={bad.high:.12g} low={bad.low:.12g} close={bad.close:.12g}")
        quarantined = [{"symbol": r.symbol, "date": str(r.date.date()), "reason": "inconsistent_ohlc",
                        **{field: float(r[field]) for field in ("open", "high", "low", "close", "volume")}}
                       for _, r in frame.loc[inconsistent & (frame.date <= asof)].iterrows()]
        # Keep the instrument and its calendar gap. Never repair prices or
        # compress the remaining dates into a shorter forward-return horizon.
        frame = frame.loc[~inconsistent].copy()
    # Reject non-session dates; never quietly turn weekend observations into training data.
    if not frame.date.isin(sessions).all():
        raise ValueError("Prices contain non-XNYS session dates")
    frame = frame[frame.date <= asof].sort_values(["date", "symbol"])
    if frame.empty:
        raise ValueError("No completed daily bars")
    frame.attrs["quarantined_bars"] = quarantined
    return frame


def quality_report(frame, symbols, sessions, asof):
    rows = []
    for symbol in symbols:
        dates = pd.DatetimeIndex(frame.loc[frame.symbol == symbol, "date"])
        expected = sessions[(sessions >= dates.min()) & (sessions <= asof)] if len(dates) else []
        rows.append({"symbol": symbol, "bars": len(dates),
                     "last_date": dates.max().date().isoformat() if len(dates) else None,
                     "missing_sessions": len(set(expected) - set(dates)),
                     "current": bool(len(dates) and dates.max() == asof)})
    return rows
