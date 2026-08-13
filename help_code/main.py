#!/usr/bin/env python3
"""Backtest driver over a ticker+date range."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from brain import TradeDecision, decide
from loaders import (
    FUNDAMENTALS_DIR,
    MARKETS_DIR,
    NEWS_DIR,
    PROJECT_ROOT,
    build_snapshot_from_row,
    load_fundamentals,
    load_markets_df,
    load_news_for_date,
)
from trader import ClosedTrade, Portfolio, log_trade

CAPITAL = 10_000.0
MARGIN = 10_000.0
TRADES_DIR = PROJECT_ROOT / "trades_data"
ORDERS_PATH = TRADES_DIR / "orders.jsonl"


def _parse_yyyymmdd(s: str) -> date:
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"Date must be YYYYMMDD, got: {s}")
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def _date_str(d: date) -> str:
    return d.strftime("%Y%m%d")


def _trading_days(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon=0 .. Fri=4
            yield d
        d += timedelta(days=1)


def _ensure_markets(tickers: list[str], start: date, end: date) -> None:
    """Download markets for tickers that have no data in range."""
    need = []
    for t in tickers:
        df = load_markets_df(t, start, end)
        if df is None or df.empty:
            need.append(t)
    if not need:
        print("[Data] Markets OK (using existing)", flush=True)
        return
    tickers_str = ",".join(need)
    start_s = _date_str(start)
    end_s = _date_str(end)
    print(f"[Data] Downloading markets for {tickers_str} ({start_s}–{end_s})...", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "markets" / "download_markets.py"),
            "--tickers",
            tickers_str,
            "--start",
            start_s,
            "--end",
            end_s,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _ensure_fundamentals(tickers: list[str]) -> None:
    """Download fundamentals for tickers that have no JSON."""
    need = [t for t in tickers if not (FUNDAMENTALS_DIR / f"{t.upper()}.json").exists()]
    if not need:
        print("[Data] Fundamentals OK (using existing)", flush=True)
        return
    tickers_str = ",".join(need)
    print(f"[Data] Downloading fundamentals for {tickers_str}...", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "fundamentals" / "download_fundamentals.py"),
            "--tickers",
            tickers_str,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _ensure_news(tickers: list[str], start: date, end: date, source: str = "google_news") -> None:
    """Fetch news for any missing trading day. source: google_news|benzinga|finviz."""
    missing_days = []
    for d in _trading_days(start, end):
        if not (NEWS_DIR / f"{_date_str(d)}.json").exists():
            missing_days.append(d)
    if not missing_days:
        print("[Data] News OK (using existing)", flush=True)
        return
    tickers_str = ",".join(tickers)
    start_s = _date_str(start)
    end_s = _date_str(end)
    print(f"[Data] Downloading news for {tickers_str} ({start_s}–{end_s}), {len(missing_days)} days missing (source={source})...", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "news&socials" / "crawl_google_news_rss.py"),
            "--tickers",
            tickers_str,
            "--from-date",
            start_s,
            "--to-date",
            end_s,
            "--out-dir",
            str(NEWS_DIR),
            "--source",
            source,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def ensure_data(tickers: list[str], start: date, end: date, news_source: str = "google_news") -> None:
    """Ensure markets, fundamentals, and news exist for tickers and range; download if missing."""
    if not tickers:
        return
    print("[Data] Checking markets, fundamentals, news...", flush=True)
    _ensure_markets(tickers, start, end)
    _ensure_fundamentals(tickers)
    _ensure_news(tickers, start, end, source=news_source)
    print("[Data] Ready.", flush=True)


def _log_order(entry: dict) -> None:
    TRADES_DIR.mkdir(parents=True, exist_ok=True)
    with open(ORDERS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _check_tp_sl(
    portfolio: Portfolio,
    ticker: str,
    day_high: float,
    day_low: float,
    day_close: float,
    date_s: str,
) -> None:
    """If position in ticker hits TP or SL, close at TP or SL (use close as proxy if intraday not available)."""
    pos = portfolio.positions.get(ticker)
    if not pos:
        return
    exit_price = None
    reason = ""
    if day_high >= pos.take_profit:
        exit_price = pos.take_profit
        reason = "TP"
    elif day_low <= pos.stop_loss:
        exit_price = pos.stop_loss
        reason = "SL"
    if exit_price is None:
        return
    dec = TradeDecision(
        action="SELL",
        ticker=ticker,
        quantity=pos.quantity,
        take_profit_price=pos.take_profit,
        stop_loss_price=pos.stop_loss,
        short_reason=reason,
        reasoning="",
    )
    closed = portfolio.apply_decision(dec, exit_price, margin_allowance=0)
    if closed:
        print(f"  [CLOSE] {date_s} {ticker} x{closed.quantity} {reason} pnl={closed.pnl:.2f}", flush=True)
        log_trade(closed)
        _log_order({
            "date": date_s,
            "ticker": ticker,
            "action": "CLOSE",
            "quantity": closed.quantity,
            "entry_price": closed.entry_price,
            "exit_price": closed.exit_price,
            "pnl": closed.pnl,
            "reason": reason,
        })


def _portfolio_value(portfolio: Portfolio, ticker_closes: dict[str, float]) -> float:
    v = portfolio.cash
    for ticker, pos in portfolio.positions.items():
        v += pos.quantity * ticker_closes.get(ticker, pos.entry_price)
    return v


def _unrealized_pnl(portfolio: Portfolio, ticker_closes: dict[str, float]) -> float:
    """Unrealized P/L from open positions: (current_price - entry_price) * quantity."""
    total = 0.0
    for ticker, pos in portfolio.positions.items():
        px = ticker_closes.get(ticker, pos.entry_price)
        total += (px - pos.entry_price) * pos.quantity
    return total


def _position_summary(portfolio: Portfolio, ticker_closes: dict[str, float]) -> str:
    """One-line summary of current position for LLM: ticker, qty, entry, current price, TP, SL, unrealized P/L."""
    if not portfolio.positions:
        return "No open position."
    parts = []
    for ticker, pos in portfolio.positions.items():
        px = ticker_closes.get(ticker, pos.entry_price)
        u = (px - pos.entry_price) * pos.quantity
        parts.append(
            f"{ticker} {pos.quantity} shares @ entry {pos.entry_price:.2f} | "
            f"current price {px:.2f} | TP {pos.take_profit:.2f} SL {pos.stop_loss:.2f} | unrealized P/L ${u:.2f}"
        )
    return "Open position: " + "; ".join(parts)


def _spy_equity_curve(all_dates: list[date], start: date, end: date) -> list[float] | None:
    """Benchmark: SPY equity curve starting with CAPITAL."""
    df = load_markets_df("SPY", start, end)
    if df is None or df.empty:
        return None
    # Filter to match backtest dates (df.index.date is ndarray; use boolean mask)
    ad_set = set(all_dates)
    df = df[[pd.Timestamp(d).date() in ad_set for d in df.index]]
    if df.empty:
        return None

    first_price = float(df.iloc[0]["Close"])
    if first_price <= 0:
        return None

    shares = CAPITAL / first_price
    curve = []
    curve.append(CAPITAL)

    for d in all_dates:
        try:
            row = df.loc[df.index.date == d].iloc[0]
            price = float(row["Close"])
            curve.append(shares * price)
        except (IndexError, KeyError):
            # If SPY missing for a day, use last known value
            if curve:
                curve.append(curve[-1])
            else:
                curve.append(CAPITAL)
    return curve


def compute_metrics(equity_curve: list[float]) -> dict:
    """Cumulative return, annualized return, Sharpe ratio, max drawdown. All values real floats for JSON."""
    if not equity_curve or len(equity_curve) < 2:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }
    start_val = float(equity_curve[0])
    end_val = float(equity_curve[-1])
    cumulative_return = (end_val / start_val - 1.0) if start_val else 0.0
    daily_returns = []
    for i in range(1, len(equity_curve)):
        prev = float(equity_curve[i - 1])
        if prev > 0:
            daily_returns.append((float(equity_curve[i]) / prev) - 1.0)
        else:
            daily_returns.append(0.0)
    n = len(daily_returns)
    annualized_return = 0.0
    sharpe_ratio = 0.0
    if n > 0:
        # 252 trading days per year; avoid complex when (1+cumulative_return) <= 0
        base = 1.0 + cumulative_return
        if base <= 0:
            annualized_return = -1.0
        else:
            annualized_return = float(base ** (252 / n) - 1.0)
        mean_ret = sum(daily_returns) / n
        variance = sum((r - mean_ret) ** 2 for r in daily_returns) / max(n - 1, 1)
        variance = max(0.0, float(variance))  # avoid complex from sqrt(negative)
        std = variance ** 0.5
        sharpe_ratio = float(mean_ret / std * (252 ** 0.5)) if std > 0 else 0.0
    peak = float(equity_curve[0])
    max_dd = 0.0
    for v in equity_curve:
        v = float(v)
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
    return {
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_dd),
    }


def run_backtest(tickers: list[str], start: date, end: date, mode: str = "medium",
                 use_tqdm: bool = True, news_source: str = "google_news") -> None:
    ensure_data(tickers, start, end, news_source=news_source)
    if load_markets_df("SPY", start, end) is None:
        print("[Backtest] Fetching SPY for benchmark...", flush=True)
        _ensure_markets(["SPY"], start, end)

    print("[Backtest] Loading market data...", flush=True)
    markets = {}
    for t in tickers:
        df = load_markets_df(t, start, end)
        if df is None or df.empty:
            print(f"[WARN] No markets data for {t}, skip.", flush=True)
            continue
        markets[t] = df

    if not markets:
        print("No markets data for any ticker. Exiting.")
        return

    portfolio = Portfolio(cash=CAPITAL)
    equity_curve = [CAPITAL]
    all_dates = sorted(_trading_days(start, end))
    n_days = len(all_dates)
    print(f"[Backtest] Start: {n_days} trading days, tickers: {list(markets.keys())}, mode: {mode}", flush=True)

    last_ticker_closes: dict[str, float] = {}
    bt_t0 = time.perf_counter()
    day_iter = tqdm(all_dates, desc="Backtest days", unit="day") if use_tqdm else all_dates
    for day_idx, d in enumerate(day_iter, 1):
        date_s = _date_str(d)
        if use_tqdm:
            day_iter.set_postfix_str(date_s)
        else:
            print(f"[Backtest] Day {day_idx}/{n_days} {date_s}", flush=True)
        forced_today = False  # aggressive: at most one forced trade per day
        ticker_closes = {}
        for ticker, df in markets.items():
            try:
                row = df.loc[df.index.date == d].iloc[0]
            except (IndexError, KeyError):
                continue
            ticker_closes[ticker] = float(row.get("Close", row.get("Open", 0)))
            day_high = float(row.get("High", ticker_closes[ticker]))
            day_low = float(row.get("Low", ticker_closes[ticker]))
            day_open = float(row.get("Open", ticker_closes[ticker]))

            _check_tp_sl(
                portfolio, ticker, day_high, day_low, ticker_closes[ticker], date_s
            )

            snap = build_snapshot_from_row(ticker, row)
            news = load_news_for_date(ticker, date_s)
            position_summary = _position_summary(portfolio, ticker_closes)
            buying_power = portfolio.cash + MARGIN
            print(f"  [LLM] {date_s} {ticker} (price={day_open:.2f})...", flush=True)
            dec = decide(
                snap,
                cash=buying_power,
                feed_context=news,
                position_summary=position_summary,
                trading_mode=mode,
            )
            # Aggressive mode: force at most one trade per day when no position and LLM said HOLD
            did_force = False
            if mode == "aggressive" and not forced_today and dec.action == "HOLD" and dec.quantity == 0:
                has_pos = portfolio.positions.get(ticker) is not None
                if not has_pos and buying_power >= day_open:
                    min_qty = max(1, min(5, int(buying_power / day_open)))  # cap forced size to limit drawdown
                    atr = max(getattr(snap, "atr14", day_open * 0.01), 0.01)
                    dec = TradeDecision(
                        action="BUY",
                        ticker=ticker,
                        quantity=min_qty,
                        take_profit_price=day_open + 2 * atr,
                        stop_loss_price=day_open - 1.5 * atr,
                        short_reason="aggressive: forced daily trade",
                        reasoning="",
                    )
                    forced_today = True
                    did_force = True
                    print(f"  [LLM] {date_s} {ticker} -> BUY qty={dec.quantity} (forced aggressive)", flush=True)
            if not did_force:
                print(f"  [LLM] {date_s} {ticker} -> {dec.action} qty={dec.quantity}", flush=True)
            if dec.action == "BUY" and dec.quantity > 0:
                # One position only: allow BUY only if no position or same ticker (add)
                existing = list(portfolio.positions.keys())
                if existing and dec.ticker not in portfolio.positions:
                    print(f"  [SKIP] {date_s} BUY {dec.ticker} (already hold {existing[0]}; one position at a time)", flush=True)
                else:
                    closed = portfolio.apply_decision(
                        dec, day_open, margin_allowance=MARGIN
                    )
                    if portfolio.positions.get(dec.ticker):
                        pos = portfolio.positions[dec.ticker]
                        print(f"  [OPEN] {date_s} {dec.ticker} x{pos.quantity} @{day_open:.2f} TP={pos.take_profit:.2f} SL={pos.stop_loss:.2f}", flush=True)
                        _log_order({
                            "date": date_s,
                            "ticker": dec.ticker,
                            "action": "OPEN",
                            "quantity": dec.quantity,
                            "price": day_open,
                            "take_profit": dec.take_profit_price,
                            "stop_loss": dec.stop_loss_price,
                        })
            elif dec.action == "SELL" and dec.quantity > 0:
                closed = portfolio.apply_decision(dec, day_open, margin_allowance=0)
                if closed:
                    print(f"  [CLOSE] {date_s} {closed.ticker} x{closed.quantity} pnl={closed.pnl:.2f} ({closed.reason})", flush=True)
                    log_trade(closed)
                    _log_order({
                        "date": date_s,
                        "ticker": closed.ticker,
                        "action": "CLOSE",
                        "quantity": closed.quantity,
                        "entry_price": closed.entry_price,
                        "exit_price": closed.exit_price,
                        "pnl": closed.pnl,
                        "reason": closed.reason,
                    })

        val = _portfolio_value(portfolio, ticker_closes)
        equity_curve.append(val)
        unrealized = _unrealized_pnl(portfolio, ticker_closes)
        last_ticker_closes = dict(ticker_closes)
        print(f"  [EOD] {date_s} portfolio=${val:.0f} unrealized_pnl=${unrealized:.2f}", flush=True)

    metrics = compute_metrics(equity_curve)
    realized_pnl = portfolio.total_pnl
    final_unrealized = _unrealized_pnl(portfolio, last_ticker_closes)
    metrics["realized_pnl"] = float(realized_pnl)
    metrics["unrealized_pnl"] = float(final_unrealized)

    spy_curve = _spy_equity_curve(all_dates, start, end)
    if spy_curve is not None:
        spy_metrics = compute_metrics(spy_curve)
        metrics["spy"] = spy_metrics
        metrics["excess"] = {
            "cumulative_return": metrics["cumulative_return"] - spy_metrics["cumulative_return"],
            "annualized_return": metrics["annualized_return"] - spy_metrics["annualized_return"],
            "sharpe_ratio": metrics["sharpe_ratio"] - spy_metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"] - spy_metrics["max_drawdown"],
        }

    TRADES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRADES_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("--- Backtest done ---", flush=True)
    print(f"Capital: ${CAPITAL:.0f} + Margin: ${MARGIN:.0f}", flush=True)
    print(f"Period: {start} to {end}", flush=True)
    print(f"Realized P/L: ${realized_pnl:.2f}", flush=True)
    print(f"Unrealized P/L (open positions): ${final_unrealized:.2f}", flush=True)
    print("", flush=True)
    print("--- Strategy vs SPY (same period) ---", flush=True)
    if spy_curve is not None:
        spy_metrics = metrics["spy"]
        exc = metrics["excess"]
        print(f"                  Strategy    SPY      Excess", flush=True)
        print(f"Cum Ret:          {metrics['cumulative_return']:>8.2%}   {spy_metrics['cumulative_return']:>6.2%}   {exc['cumulative_return']:>+8.2%}", flush=True)
        print(f"Ann. Ret:         {metrics['annualized_return']:>8.2%}   {spy_metrics['annualized_return']:>6.2%}   {exc['annualized_return']:>+8.2%}", flush=True)
        print(f"Sharpe:           {metrics['sharpe_ratio']:>8.2f}   {spy_metrics['sharpe_ratio']:>6.2f}   {exc['sharpe_ratio']:>+8.2f}", flush=True)
        print(f"Max Drawdown:     {metrics['max_drawdown']:>8.2%}   {spy_metrics['max_drawdown']:>6.2%}   {exc['max_drawdown']:>+8.2%}", flush=True)
    else:
        print(f"Cum Ret:          {metrics['cumulative_return']:>8.2%}", flush=True)
        print(f"Ann. Ret:         {metrics['annualized_return']:>8.2%}", flush=True)
        print(f"Sharpe:           {metrics['sharpe_ratio']:>8.2f}", flush=True)
        print(f"Max Drawdown:     {metrics['max_drawdown']:>8.2%}", flush=True)
    print(f"Orders: {ORDERS_PATH}", flush=True)
    print(f"Decisions: {TRADES_DIR / 'decisions' / 'decision_log.jsonl'}", flush=True)
    print(f"Backtest wall time: {time.perf_counter() - bt_t0:.1f}s", flush=True)


def _first_trading_day_with_data(ticker: str, start: date, end: date) -> date | None:
    df = load_markets_df(ticker, start, end)
    if df is None or df.empty:
        return None
    for d in sorted(_trading_days(start, end)):
        try:
            df.loc[df.index.date == d].iloc[0]
            return d
        except (IndexError, KeyError):
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest from local markets/fundamentals/news.")
    ap.add_argument("--tickers", required=True, help="Comma-separated e.g. AAPL,TSLA,NVDA")
    ap.add_argument("--start", required=True, help="Start YYYYMMDD")
    ap.add_argument("--end", required=True, help="End YYYYMMDD")
    ap.add_argument(
        "--mode",
        choices=["aggressive", "medium", "conservative"],
        default="medium",
        help="aggressive=force trade every day; medium=trade when >50%% edge; conservative=only high conviction",
    )
    ap.add_argument(
        "--no-backtest",
        action="store_true",
        help="Skip full backtest (use with --attack-eval for attack study only)",
    )
    ap.add_argument(
        "--attack-eval",
        action="store_true",
        help="Run attack ASR study over the prompt corpus vs benign baseline",
    )
    ap.add_argument("--attack-repeats", type=int, default=10, help="Repeats per attack prompt (default 10)")
    ap.add_argument(
        "--eval-date",
        default="",
        help="YYYYMMDD snapshot for attack eval; default first trading day with data for first ticker",
    )
    ap.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    ap.add_argument(
        "--output-base",
        type=Path,
        default=None,
        help="If set, create {output_base}/{run_label}_{timestamp}/ and save logs + copies of outputs",
    )
    ap.add_argument(
        "--run-label",
        default="main",
        help="Run folder prefix when --output-base is set",
    )
    ap.add_argument(
        "--no-tee",
        action="store_true",
        help="With --output-base, do not mirror console to logs/console.txt",
    )
    ap.add_argument(
        "--news-source",
        choices=["google_news", "benzinga", "finviz"],
        default="google_news",
        help="News feed provider. benzinga needs BENZINGA_API_KEY; finviz uses public scrape "
             "unless FINVIZ_API_TOKEN is set (default: google_news, free).",
    )
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    start = _parse_yyyymmdd(args.start)
    end = _parse_yyyymmdd(args.end)

    if args.no_backtest and not args.attack_eval:
        print("Error: --no-backtest requires --attack-eval.", flush=True)
        sys.exit(1)

    run_dir: Path | None = None
    if args.output_base is not None:
        from run_output import make_run_directory, tee_stdio_to, write_readme, write_run_meta

        args.output_base.mkdir(parents=True, exist_ok=True)
        run_dir = make_run_directory(args.output_base, args.run_label)
        write_readme(
            run_dir,
            "main.py run\nsummary/: run_meta.json, metrics.json, attack_asr.json\nlogs/: attack_study_log.jsonl, console.txt\n",
        )
        meta = {
            "script": "main.py",
            "tickers": tickers,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "mode": args.mode,
            "attack_eval": args.attack_eval,
            "run_directory": str(run_dir.resolve()),
        }
        write_run_meta(run_dir, meta)

    def _body() -> None:
        if not args.no_backtest:
            run_backtest(tickers, start, end, mode=args.mode, use_tqdm=not args.no_tqdm,
                         news_source=args.news_source)
            if run_dir is not None:
                (run_dir / "summary").mkdir(exist_ok=True)
                m = TRADES_DIR / "metrics.json"
                if m.exists():
                    shutil.copy(m, run_dir / "summary" / "metrics.json")
                o = TRADES_DIR / "orders.jsonl"
                if o.exists():
                    shutil.copy(o, run_dir / "summary" / "orders.jsonl")

        if args.attack_eval:
            from attacks.attack_eval import print_asr_table, run_attack_evaluation, save_asr_json

            ensure_data(tickers, start, end, news_source=args.news_source)
            eval_ticker = tickers[0]
            if args.eval_date:
                eval_d = _parse_yyyymmdd(args.eval_date)
            else:
                eval_d = _first_trading_day_with_data(eval_ticker, start, end)
            if eval_d is None:
                print(f"[Attack eval] No market data for {eval_ticker} in range; abort.", flush=True)
                sys.exit(1)
            print(f"[Attack eval] Using {eval_ticker} @ {eval_d} | repeats={args.attack_repeats}", flush=True)
            rows, elapsed = run_attack_evaluation(
                eval_ticker,
                eval_d,
                args.mode,
                repeats=args.attack_repeats,
                buying_power=CAPITAL + MARGIN,
                attack_log_path=run_dir / "logs" / "attack_study_log.jsonl" if run_dir else None,
            )
            print_asr_table(rows, elapsed, label=eval_ticker)
            if run_dir is not None:
                save_asr_json(rows, elapsed, output_path=run_dir / "summary" / "attack_asr.json", ticker=eval_ticker)
                print(f"Attack log: {run_dir / 'logs' / 'attack_study_log.jsonl'}", flush=True)
            else:
                save_asr_json(rows, elapsed, ticker=eval_ticker)
                print(f"Full prompt/response log: {TRADES_DIR / 'attack_study_log.jsonl'}", flush=True)

    if run_dir is not None and not args.no_tee:
        from run_output import tee_stdio_to

        with tee_stdio_to(run_dir):
            print(f"[main] Run directory: {run_dir}", flush=True)
            print(f"[main] Console log: {run_dir / 'logs' / 'console.txt'}", flush=True)
            _body()
        print(f"[main] Outputs saved under: {run_dir}", flush=True)
    else:
        _body()
        if run_dir is not None:
            print(f"[main] Outputs saved under: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
