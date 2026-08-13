#!/usr/bin/env python3
"""Download OHLCV plus technical indicators per ticker."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


DATA_DIR = Path(__file__).parent / "data"


def parse_yyyymmdd(s: str) -> date:
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"Date must be YYYYMMDD, got: {s}")
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if len(series) < period + 1:
        return pd.Series(np.nan, index=series.index)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty or any(c not in df.columns for c in ("High", "Low", "Close")):
        return pd.Series(np.nan, index=df.index)
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def download_one_ticker(ticker: str, start: date, end: date, interval: str = "1d") -> Path | None:
    t = yf.Ticker(ticker)
    # yfinance end is exclusive, so add one day
    df = t.history(start=start, end=end, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        print(f"[WARN] No data for {ticker} between {start} and {end}")
        return None

    close = df["Close"]
    df["rsi14"] = calc_rsi(close, 14)
    df["ema20"] = calc_ema(close, 20)
    df["ema50"] = calc_ema(close, 50)
    df["ema100"] = calc_ema(close, 100)
    df["ema200"] = calc_ema(close, 200)
    df["atr14"] = calc_atr(df, 14)

    out_dir = DATA_DIR / ticker.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")
    out_path = out_dir / f"{ticker.upper()}_{start_s}_{end_s}.csv"
    df.to_csv(out_path)
    print(f"[OK] markets {ticker} -> {out_path}")
    return out_path


def parse_tickers(raw: str) -> list[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Download markets OHLCV + TA to local CSVs.")
    ap.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. AAPL,TSLA,NVDA")
    ap.add_argument("--start", required=True, help="Start date YYYYMMDD")
    ap.add_argument("--end", required=True, help="End date YYYYMMDD (inclusive)")
    ap.add_argument("--interval", default="1d", choices=["1d", "1wk", "1mo"])
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    tickers = parse_tickers(args.tickers)
    start = parse_yyyymmdd(args.start)
    end = parse_yyyymmdd(args.end)

    for t in tickers:
        download_one_ticker(t, start, end, interval=args.interval)


if __name__ == "__main__":
    main()

