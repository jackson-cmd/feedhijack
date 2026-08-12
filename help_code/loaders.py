"""
Load data from markets/, fundamentals/, news&socials/ for backtest.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from data import IntradaySnapshot

PROJECT_ROOT = Path(__file__).resolve().parent
MARKETS_DIR = PROJECT_ROOT / "markets" / "data"
FUNDAMENTALS_DIR = PROJECT_ROOT / "fundamentals" / "data"
NEWS_DIR = PROJECT_ROOT / "news&socials" / "headlines_crawler" / "output" / "google_news_rss"


def _parse_yyyymmdd(s: str) -> date:
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"Date must be YYYYMMDD, got: {s}")
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def load_markets_df(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    """Load one ticker's CSV from markets/data/TICKER/ that overlaps [start, end]. Concat if multiple files."""
    ticker_dir = MARKETS_DIR / ticker.upper()
    if not ticker_dir.exists():
        return None
    frames = []
    for p in sorted(ticker_dir.glob("*.csv")):
        try:
            df = pd.read_csv(p, index_col=0, parse_dates=True)
        except Exception:
            continue
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index, utc=True)
        frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames).sort_index()
    combined.index = pd.to_datetime(combined.index, utc=True)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.loc[(combined.index.date >= start) & (combined.index.date <= end)]
    return combined if not combined.empty else None


def build_snapshot_from_row(ticker: str, row: pd.Series) -> IntradaySnapshot:
    """Build IntradaySnapshot from one row of markets CSV (has OHLCV + rsi14, ema*, atr14)."""
    def f(k: str, default: float = 0.0) -> float:
        v = row.get(k, default)
        if pd.isna(v):
            return default
        return float(v)

    price = f("Open", row.get("Close", 0))
    last = f("Close", price)
    return IntradaySnapshot(
        ticker=ticker.upper(),
        price=price,
        last_price=last,
        vwap=(f("High") + f("Low") + f("Close")) / 3,
        rsi=f("rsi14", 50.0),
        macd=0.0,
        macd_signal=0.0,
        macd_hist=0.0,
        bb_upper=last,
        bb_mid=last,
        bb_lower=last,
        support=f("Low"),
        resistance=f("High"),
        volume=int(f("Volume", 0)),
        ema20=f("ema20", last),
        ema50=f("ema50", last),
        ema100=f("ema100", last),
        ema200=f("ema200", last),
        atr14=f("atr14", last * 0.01),
    )


def load_fundamentals(ticker: str) -> dict[str, Any]:
    """Load fundamentals/data/TICKER.json if exists."""
    path = FUNDAMENTALS_DIR / f"{ticker.upper()}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_news_for_date(ticker: str, date_str: str) -> str:
    """Load news for one date from news&socials/.../google_news_rss/YYYYMMDD.json, filter by ticker."""
    path = NEWS_DIR / f"{date_str}.json"
    if not path.exists():
        return "No news for this date."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "No news for this date."
    items = data.get("items", [])
    parts = []
    for it in items:
        if it.get("ticker", "").upper() != ticker.upper():
            continue
        title = (it.get("title") or "").strip()
        source = (it.get("source") or "").strip()
        if title:
            parts.append(f"  - {title} ({source})")
    if not parts:
        return "No news for this ticker on this date."
    return "News:\n" + "\n".join(parts[:10])
