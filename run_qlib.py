#!/usr/bin/env python3
"""Run the Qlib v1 research pipeline and export a public, account-free dashboard."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from qlib_v1.data import (atomic_json, calendar, canonical_hash, fetch_alpaca,
                          fetch_yahoo, quality_report, validate_bars)
from qlib_v1.ledger import append_forecasts, connect, settle, summary
from qlib_v1.research import (FEATURES, current_predictions, evaluate, make_panel,
                             rolling_predictions)
from qlib_v1.universe import resolve, fingerprint
from qlib_v1.screen import screen_current, market_caps

ROOT = Path(__file__).resolve().parent


def validate_config(c):
    if c.get("version") != 1 or c.get("universe_type") != "static_current_survivors":
        raise ValueError("Unsupported config/universe type")
    if not c["symbols"] or len(set(c["symbols"])) != len(c["symbols"]) or c["benchmark"] in c["symbols"]:
        raise ValueError("Universe must be nonempty, unique and exclude benchmark")
    for key in ("train_sessions", "min_train_sessions", "refit_sessions", "display_limit", "min_candidates", "min_oos_dates", "timing_window", "timing_min_observations"):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if c["train_sessions"] < c["min_train_sessions"] or c["timing_min_observations"] > c["timing_window"] or c["min_candidates"] < 3:
        raise ValueError("Inconsistent window/candidate settings")
    if not c["horizons"] or len(set(c["horizons"])) != len(c["horizons"]) or any(type(h) is not int or not 1 <= h <= 60 for h in c["horizons"]):
        raise ValueError("Horizons must be unique integers between 1 and 60")
    if c["ridge_alpha"] <= 0:
        raise ValueError("Invalid regularization")


def run(args):
    config = json.loads(args.config.read_text())
    settings_path = ROOT / "config/qlib-universes.json"
    # Explicit test/research configs keep their own universe. The application
    # default uses the user-editable profiles saved by the local dashboard.
    if args.config.resolve() == (ROOT / "config/qlib-v1.json").resolve() and settings_path.exists():
        config = resolve(config, json.loads(settings_path.read_text()),
                         json.loads((ROOT / "config/qlib-industry-catalog.json").read_text()))
    validate_config(config)
    now = datetime.now(timezone.utc)
    sessions, market_asof = calendar(config["start"], now)
    asof = pd.Timestamp(args.asof) if args.asof else market_asof
    if asof not in sessions or asof > market_asof:
        raise ValueError("asof must be a completed XNYS session")
    symbols = config["symbols"] + [config["benchmark"]]
    args.state.mkdir(parents=True, exist_ok=True)
    errors = []
    quarantined = []
    if args.prices:
        bars = pd.read_csv(args.prices)
        provider = "csv"
    elif args.fetch == "alpaca":
        bars = fetch_alpaca(symbols, config["start"], asof)
        provider = "alpaca_sip"
    elif args.fetch == "yahoo":
        bars, errors = fetch_yahoo(symbols, config["start"], asof)
        provider = "yahoo_exploratory"
    else:
        cached = args.state / "prices.csv"
        metadata = args.state / "prices.meta.json"
        if not cached.exists() or not metadata.exists():
            raise ValueError("No prices. Run --fetch alpaca, --fetch yahoo, or --prices FILE first")
        bars = pd.read_csv(cached)
        meta = json.loads(metadata.read_text())
        quarantined = meta.get("quarantined_bars", [])
        provider = meta["provider"]
        if not set(symbols).issubset(set(bars.symbol)):
            raise ValueError("股票池包含尚未下载的股票，请刷新行情")
        if hashlib.sha256(cached.read_bytes()).hexdigest() != meta["data_hash"]:
            raise ValueError("Cached prices checksum mismatch")
    try:
        bars = validate_bars(bars, sessions, asof, quarantine_ohlc=bool(args.fetch))
    except ValueError:
        # Keep the rejected public-price batch for diagnosis; never replace the
        # last valid cache or dashboard with unvalidated observations.
        bars.to_csv(args.state / "rejected-prices.csv", index=False)
        raise
    quarantined = bars.attrs.get("quarantined_bars") or quarantined
    if quarantined:
        atomic_json(args.state / "quarantined-bars.json", quarantined)
        print(f"Data quality: quarantined {len(quarantined)} inconsistent OHLC bars; instruments retained", flush=True)
    bars = bars[bars.symbol.isin(symbols)]
    if config["benchmark"] not in set(bars.symbol):
        raise ValueError("Benchmark prices missing")
    quality = quality_report(bars, symbols, sessions, asof)
    # Do not silently backdate an apparently current run if every feed is stale.
    if bars.date.max() != asof or not next(r for r in quality if r["symbol"] == config["benchmark"])["current"]:
        raise ValueError("Requested session is missing; previous successful output remains unchanged")
    price_bytes = bars.to_csv(index=False, date_format="%Y-%m-%d").encode()
    data_hash = hashlib.sha256(price_bytes).hexdigest()
    (args.state / "prices.csv").write_bytes(price_bytes)
    atomic_json(args.state / "prices.meta.json", {"provider": provider, "universe_hash": canonical_hash(symbols),
                                                "data_hash": data_hash, "fetched_at": now.isoformat(),
                                                "quarantined_bars": quarantined})
    screened = None
    if config.get("screen_current_only"):
        filters = config["current_screen"]
        caps = market_caps(config["symbols"], args.state) if filters["min_market_cap"] > 0 or filters["max_market_cap"] is not None else {}
        screened = screen_current(bars, config["symbols"], sessions, asof, filters, caps)
        config["requested_symbols"] = config["symbols"]
        config["symbols"] = [row["symbol"] for row in screened]
        config["universe_id"] = fingerprint(config, config["symbols"])
        quality = [row for row in quality if row["symbol"] in config["symbols"]]
    # Code fingerprint makes experiment identities change when implementation changes.
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in [Path(__file__), *sorted((ROOT / "qlib_v1").glob("*.py"))])).hexdigest()
    import qlib
    # Native Qlib static loader consumes the canonical panel without a second binary
    # price store. Disable expression/dataset caches; no home-directory writes.
    qlib.init(provider_uri=str(args.state.resolve()), region="us", expression_cache=None,
              dataset_cache=None, logging_level=40)
    panel = make_panel(bars[bars.symbol.isin(config["symbols"])], sessions, config)
    report = {"schema": "qlib-research-v1", "generated_at": now.isoformat(), "asof": str(asof.date()),
              "provider": provider, "mode": "research_only", "config": config,
              "data_hash": data_hash, "code_hash": code_hash, "quality": quality,
              "versions": {name: importlib.metadata.version(name) for name in ("pyqlib", "pandas", "numpy", "exchange-calendars")},
              "warnings": ["固定的当前股票池，历史存在幸存者及事后选池偏差，结果仅供研究。",
                           "IC 标签为拆股调整后的价格收益，不包含分红。",
                           "排名分数不是上涨概率，不直接用于 Kelly 或实盘下单。",
                           "日线数据不能验证盘中止损成交、期权收益或业务论点的因果关系。"],
              "horizons": {}}
    report["current_universe"] = {"name": config.get("universe_name", config["universe_id"]),
                                  "screening": "current_snapshot_only" if screened is not None else "legacy",
                                  "filters": config.get("current_screen"), "stocks": screened or [],
                                  "asof": str(asof.date()), "count": len(config["symbols"])}
    report["data_quality"] = {"quarantined_bars": quarantined, "quarantined_count": len(quarantined)}
    if quarantined:
        samples = ", ".join(f"{r['symbol']} {r['date']}" for r in quarantined[:5])
        report["warnings"].append(f"已隔离 {len(quarantined)} 条 OHLC 异常日线（{samples}）；未改写价格，相关特征及标签按缺失处理。")
    if screened is not None:
        report["warnings"][0] = "历史 IC / ICIR 基于当前条件筛出的股票池；筛选条件不按历史日期回放。"
    if errors:
        report["warnings"].append("抓取失败的标的：" + ", ".join(errors))
    if provider == "yahoo_exploratory":
        report["warnings"].append("Yahoo 数据仅用于探索；上线前应以同口径 SIP 数据复核。")
    forecasts = []
    import exchange_calendars as xcals
    next_session = sessions[sessions.get_loc(asof) + 1]
    can_issue = asof == market_asof and pd.Timestamp(now) < xcals.get_calendar("XNYS").session_open(next_session) and not args.demo
    report["forward_issuance"] = "enabled" if can_issue else "disabled_historical_or_after_entry_or_demo"
    for horizon in config["horizons"]:
        print(f"Qlib: {horizon}-session purged walk-forward", flush=True)
        identity = canonical_hash({"data": data_hash, "config": config, "code": code_hash, "horizon": horizon})
        archive = args.state / "experiments" / identity
        archive.mkdir(parents=True, exist_ok=True)
        cached_result = archive / "evaluation.json"
        if cached_result.exists():
            evaluation = json.loads(cached_result.read_text())
        else:
            scores, folds = rolling_predictions(panel, sessions, config, horizon)
            evaluation = evaluate(scores, panel, config, horizon)
            evaluation["folds"] = folds
            scores.rename("score").to_csv(archive / "oos_predictions.csv")
            atomic_json(cached_result, evaluation)
        live, artifact = current_predictions(panel, sessions, config, horizon, asof)
        candidates = []
        if artifact:
            artifact["training_data_hash"] = hashlib.sha256(panel.loc[(slice(artifact["train_start"], artifact["train_end"]), slice(None)), FEATURES + [f"label_{horizon}"]].to_csv().encode()).hexdigest()
            model_id = canonical_hash({"artifact": artifact, "config": config, "code": code_hash})
            artifact.update(model_id=model_id, qlib_version=report["versions"]["pyqlib"])
            atomic_json(archive / "model.json", artifact)
            ranked = live.droplevel(0).sort_values(ascending=False, kind="stable")
            for rank, (symbol, score) in enumerate(ranked.items(), 1):
                feature = panel.loc[(asof, symbol)]
                row = {"symbol": symbol, "rank": rank, "score": float(score), "horizon": horizon,
                       "asof": str(asof.date()), "entry_date": str(next_session.date()),
                       "target_date": str(sessions[sessions.get_loc(asof) + horizon].date()),
                       "model_id": model_id, "universe_id": config["universe_id"],
                       "generated_at": now.isoformat(), "data_hash": data_hash,
                       "features": {key: float(feature[key]) for key in ("momentum_5", "momentum_20", "volatility_20")}}
                candidates.append(row)
                if can_issue:
                    forecasts.append(row)
        report["horizons"][str(horizon)] = {"status": "research_only" if candidates else "insufficient_history",
                                           "experiment_id": identity, "evaluation": evaluation,
                                           "model": artifact, "candidates": candidates}
        atomic_json(archive / "manifest.json", {"config": config, "data_hash": data_hash, "code_hash": code_hash,
                                               "generated_at": now.isoformat(), "provider": provider,
                                               "versions": report["versions"]})
    # Only forward-issued rows enter this ledger. Historical OOS predictions stay
    # in their experiment folders and are never presented as paper-track records.
    ledger_path = args.state / ("demo-ledger.sqlite" if args.demo else "forecast-ledger.sqlite")
    with connect(ledger_path) as db:
        append_forecasts(db, forecasts)
        settle(db, bars, asof, data_hash)
        report["ledger"] = summary(db, asof, config=config)
    if args.demo:
        report["mode"] = "synthetic_demo"
        report["warnings"].insert(0, "合成数据演示：不是市场行情、投资信号或真实绩效。")
    atomic_json(args.output, report)
    print(f"Exported {args.output}; {len(forecasts)} forward forecasts; {report['ledger']['settled']} settled", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/qlib-v1.json")
    parser.add_argument("--state", type=Path, default=ROOT / "data/qlib-v1")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "noetic-dashboard/data/qlib_research.json")
    parser.add_argument("--fetch", choices=["alpaca", "yahoo"])
    parser.add_argument("--prices", type=Path, help="Split-adjusted OHLCV CSV (same return basis as Alpaca adjustment=split)")
    parser.add_argument("--asof", help="Historical completed session; disables forward issuance")
    parser.add_argument("--demo", action="store_true", help="Explicit synthetic/demo namespace; never records live forecasts")
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    if args.fetch and args.prices:
        parser.error("Choose --fetch or --prices")
    args.state.mkdir(parents=True, exist_ok=True)
    with (args.state / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, "Another Qlib run owns this state directory\n")
        try:
            run(args)
            atomic_json(args.state / "refresh-result.json", {"started_at": started_at, "exit_code": 0, "error": None})
        except Exception as exc:
            # Avoid printing credential-bearing request headers or provider payloads.
            print(f"Qlib run failed ({type(exc).__name__}); existing dashboard output was preserved.", file=sys.stderr)
            if isinstance(exc, (ValueError, FileNotFoundError, ImportError)):
                print(str(exc), file=sys.stderr)
            message = "研究运行失败，请检查本地 refresh.log"
            if isinstance(exc, ValueError) and str(exc).startswith("Inconsistent OHLC"):
                message = "日线 OHLC 价格不一致：" + str(exc).removeprefix("Inconsistent OHLC: ")
            elif isinstance(exc, ValueError) and str(exc).startswith("Requested session is missing"):
                message = "行情数据缺少最近完整交易日，请稍后重试"
            atomic_json(args.state / "refresh-result.json", {"started_at": started_at, "exit_code": 1, "error": message})
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
