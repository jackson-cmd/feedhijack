"""OHLCV + RSI, MACD, Bollinger Bands data ingestion."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf


@dataclass
class IntradaySnapshot:
    """Condensed market snapshot for LLM decision."""
    ticker: str
    # price = day's open (agent anchor); last_price = latest observed
    price: float
    last_price: float
    vwap: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    support: float
    resistance: float
    volume: int
    # Trend / volatility
    ema20: float
    ema50: float
    ema100: float
    ema200: float
    atr14: float


def _calc_vwap(df: pd.DataFrame) -> float:
    """Volume-weighted average price."""
    if df.empty or "Volume" not in df.columns or "High" not in df.columns:
        return 0.0
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    v = df["Volume"].astype(float)
    if v.sum() <= 0:
        return float(df["Close"].iloc[-1])
    return float((typical * v).sum() / v.sum())


def _calc_rsi(series: pd.Series, period: int = 14) -> float:
    """RSI using pandas (no TA-Lib dependency)."""
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _calc_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float, float, float]:
    """MACD line, signal line, histogram."""
    if len(series) < slow + signal:
        return 0.0, 0.0, 0.0
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    return float(macd_line.iloc[-1]), float(sig_line.iloc[-1]), float(hist.iloc[-1])


def _calc_bollinger(series: pd.Series, period: int = 20, std: float = 2.0) -> tuple[float, float, float]:
    """Bollinger Bands: upper, mid, lower."""
    if len(series) < period:
        return float(series.iloc[-1]), float(series.iloc[-1]), float(series.iloc[-1])
    mid = series.rolling(period).mean().iloc[-1]
    std_val = series.rolling(period).std().iloc[-1]
    if pd.isna(std_val) or std_val == 0:
        std_val = series.std() or 0.01
    upper = mid + std * float(std_val)
    lower = mid - std * float(std_val)
    return float(upper), float(mid), float(lower)


def _calc_support_resistance(df: pd.DataFrame, lookback: int = 20) -> tuple[float, float]:
    """Simple support/resistance from recent high/low."""
    if df.empty or len(df) < lookback:
        return 0.0, 0.0
    window = df.tail(lookback)
    low = float(window["Low"].min())
    high = float(window["High"].max())
    return low, high


def _calc_ema(series: pd.Series, span: int) -> float:
    """Exponential moving average."""
    if len(series) < span:
        return float(series.iloc[-1])
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range (ATR)."""
    if df.empty or any(c not in df.columns for c in ("High", "Low", "Close")):
        return 0.0
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
    if len(tr) < period:
        return float(tr.mean())
    return float(tr.rolling(period).mean().iloc[-1])


def fetch_ohlcv(ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def build_snapshot(ticker: str, df_5m: Optional[pd.DataFrame] = None) -> Optional[IntradaySnapshot]:
    """Build snapshot with indicators; uses 5m data if provided, else fetches."""
    if df_5m is None or df_5m.empty:
        df_5m = fetch_ohlcv(ticker, interval="5m", period="5d")
    if df_5m.empty:
        df_5m = fetch_ohlcv(ticker, interval="1d", period="1mo")

    if df_5m.empty:
        return None

    close = df_5m["Close"]
    last_price = float(close.iloc[-1])

    # Use today's session open as the agent's "price" anchor.
    open_price = last_price
    if "Open" in df_5m.columns and not df_5m["Open"].empty:
        try:
            last_ts = df_5m.index[-1]
            session_ts = last_ts
            if getattr(last_ts, "tzinfo", None) is not None:
                # Prefer US market timezone if index is tz-aware.
                session_ts = last_ts.tz_convert("America/New_York")
            session_date = session_ts.date()

            idx = df_5m.index
            if getattr(idx, "tz", None) is not None:
                idx_cmp = idx.tz_convert("America/New_York")
            else:
                idx_cmp = idx
            mask = pd.Series([t.date() == session_date for t in idx_cmp], index=df_5m.index)
            day_df = df_5m.loc[mask]
            if not day_df.empty:
                open_price = float(day_df["Open"].iloc[0])
            else:
                open_price = float(df_5m["Open"].iloc[-1])
        except Exception:
            open_price = float(df_5m["Open"].iloc[-1])

    price = open_price
    vwap = _calc_vwap(df_5m)
    rsi = _calc_rsi(close)
    macd_val, macd_sig, macd_hist = _calc_macd(close)
    bb_u, bb_m, bb_l = _calc_bollinger(close)
    support, resistance = _calc_support_resistance(df_5m)
    volume = int(df_5m["Volume"].iloc[-1]) if "Volume" in df_5m.columns else 0

    ema20 = _calc_ema(close, 20)
    ema50 = _calc_ema(close, 50)
    ema100 = _calc_ema(close, 100)
    ema200 = _calc_ema(close, 200)
    atr14 = _calc_atr(df_5m if len(df_5m) >= 20 else fetch_ohlcv(ticker, interval="1d", period="6mo"), 14)

    return IntradaySnapshot(
        ticker=ticker,
        price=price,
        last_price=last_price,
        vwap=vwap,
        rsi=rsi,
        macd=macd_val,
        macd_signal=macd_sig,
        macd_hist=macd_hist,
        bb_upper=bb_u,
        bb_mid=bb_m,
        bb_lower=bb_l,
        support=support,
        resistance=resistance,
        volume=volume,
        ema20=ema20,
        ema50=ema50,
        ema100=ema100,
        ema200=ema200,
        atr14=atr14,
    )


async def fetch_snapshots(tickers: list[str]) -> dict[str, Optional[IntradaySnapshot]]:
    """Concurrent fetch of snapshots for multiple tickers."""
    loop = asyncio.get_event_loop()

    def _fetch(t: str) -> tuple[str, Optional[IntradaySnapshot]]:
        return t, build_snapshot(t)

    tasks = [loop.run_in_executor(None, _fetch, t) for t in tickers]
    results = await asyncio.gather(*tasks)
    return dict(results)
