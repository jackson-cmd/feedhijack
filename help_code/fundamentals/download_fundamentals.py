#!/usr/bin/env python3
"""Download fundamentals & earnings per ticker to fundamentals/data/{ticker}.json."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


DATA_DIR = Path(__file__).parent / "data"


@dataclass
class EarningsSnapshot:
    ticker: str
    last_eps: float | None
    last_revenue: float | None
    last_report_date: str | None
    next_earnings_date: str | None
    eps_surprise: float | None
    quarterly_earnings: list[dict[str, Any]]


@dataclass
class FundamentalSnapshot:
    ticker: str
    market_cap: float | None
    pe: float | None
    pb: float | None
    trailing_eps: float | None
    sector: str
    industry: str
    debt_to_equity: float | None
    profit_margin: float | None
    created_at: str
    earnings: EarningsSnapshot | None


def fetch_earnings_snapshot(ticker: str, t: yf.Ticker) -> EarningsSnapshot | None:
    # yfinance still routes quarterly_earnings through deprecated Ticker.earnings internally
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        q = t.quarterly_earnings
    if q is None or q.empty:
        return None

    last_row = q.iloc[-1]
    last_eps = float(last_row.get("Earnings")) if "Earnings" in last_row else None
    last_rev = float(last_row.get("Revenue")) if "Revenue" in last_row else None
    last_date = str(q.index[-1].date()) if not q.index.empty else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        edates = t.earnings_dates
    next_date = None
    surprise = None
    if isinstance(edates, pd.DataFrame) and not edates.empty:
        row = edates.sort_index().iloc[-1]
        next_date = row.name.strftime("%Y-%m-%d")
        if "surprise" in row:
            try:
                surprise = float(row["surprise"])
            except Exception:
                surprise = None

    quarterly_list: list[dict[str, Any]] = []
    for idx, r in q.iterrows():
        quarterly_list.append(
            {
                "report_date": str(idx.date()),
                "eps": float(r.get("Earnings")) if "Earnings" in r else None,
                "revenue": float(r.get("Revenue")) if "Revenue" in r else None,
            }
        )

    return EarningsSnapshot(
        ticker=ticker,
        last_eps=last_eps,
        last_revenue=last_rev,
        last_report_date=last_date,
        next_earnings_date=next_date,
        eps_surprise=surprise,
        quarterly_earnings=quarterly_list,
    )


def fetch_fundamentals(ticker: str) -> FundamentalSnapshot | None:
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    earnings = fetch_earnings_snapshot(ticker, t)

    return FundamentalSnapshot(
        ticker=ticker,
        market_cap=info.get("marketCap"),
        pe=info.get("trailingPE"),
        pb=info.get("priceToBook"),
        trailing_eps=info.get("trailingEps"),
        sector=info.get("sector", "") or "",
        industry=info.get("industry", "") or "",
        debt_to_equity=info.get("debtToEquity"),
        profit_margin=info.get("profitMargins"),
        created_at=datetime.now(timezone.utc).isoformat(),
        earnings=earnings,
    )


def save_fundamentals(snapshot: FundamentalSnapshot) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{snapshot.ticker.upper()}.json"

    def encode(obj: Any):
        if isinstance(obj, FundamentalSnapshot) or isinstance(obj, EarningsSnapshot):
            return asdict(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    with path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, default=encode)
    print(f"[OK] fundamentals {snapshot.ticker} -> {path}")
    return path


def parse_tickers(raw: str) -> list[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Download fundamentals & earnings to local JSONs.")
    ap.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. AAPL,TSLA,NVDA")
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    tickers = parse_tickers(args.tickers)
    for tkr in tickers:
        snap = fetch_fundamentals(tkr)
        if snap:
            save_fundamentals(snap)
        else:
            print(f"[WARN] No fundamentals for {tkr}")


if __name__ == "__main__":
    main()

