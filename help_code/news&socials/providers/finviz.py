"""Finviz news provider (Elite CSV if FINVIZ_API_TOKEN, else public scrape)."""
from __future__ import annotations

import csv
import io
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

_ELITE_ENDPOINT = "https://elite.finviz.com/news_export.ashx"
_PUBLIC_QUOTE = "https://finviz.com/quote.ashx"
_UA = "Mozilla/5.0 (compatible; FeedHijack-Backtest/1.0)"


def _fetch_elite(tickers: list[str], day: date, timeout: int) -> list[dict]:
    token = os.getenv("FINVIZ_API_TOKEN", "").strip()
    params = {"v": "3", "auth": token, "t": ",".join(t.upper() for t in tickers)}
    url = _ELITE_ENDPOINT + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = r.read().decode("utf-8", errors="replace")
    out: list[dict] = []
    for row in csv.DictReader(io.StringIO(text)):
        ts = (row.get("Date") or "").strip()
        try:
            pub_day = datetime.strptime(ts.split(" ")[0], "%Y-%m-%d").date()
        except ValueError:
            continue
        if pub_day != day:
            continue
        sym = (row.get("Ticker") or "").strip().upper()
        title = (row.get("Title") or "").strip()
        if not sym or not title:
            continue
        out.append({
            "ticker": sym,
            "title": title,
            "source": (row.get("Source") or "Finviz").strip(),
            "url": (row.get("Url") or "").strip(),
            "published_at": ts,
        })
    return out


_ROW_RE = re.compile(
    r'<tr[^>]*class="[^"]*(?:news_table-row|cursor-pointer)[^"]*"[^>]*>\s*'
    r'<td[^>]*>(?P<ts>.*?)</td>.*?'
    r'<a[^>]*class="[^"]*tab-link-news[^"]*"[^>]*href="(?P<href>[^"]+)"[^>]*>'
    r'(?P<title>.*?)</a>'
    r'(?:.*?<span[^>]*>\(?(?P<src>[^)<]+)\)?</span>)?',
    re.S,
)


def _parse_public_ts(txt: str) -> date | None:
    """Finviz cell is 'Nov-08-25 04:32PM' or, for continuation rows, just '04:32PM'.
    Return None on the time-only case so the caller inherits the previous row's date.
    """
    txt = txt.strip().replace("\xa0", " ")
    if not txt:
        return None
    parts = txt.split()
    if len(parts) == 1:
        return None
    try:
        return datetime.strptime(parts[0], "%b-%d-%y").date()
    except ValueError:
        return None


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def _fetch_public_one(ticker: str, day: date, timeout: int) -> list[dict]:
    url = _PUBLIC_QUOTE + "?" + urllib.parse.urlencode({"t": ticker.upper()})
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            html = r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        print(f"[finviz] {ticker} HTTP {e.code}", flush=True)
        return []

    out: list[dict] = []
    last_date: date | None = None
    for m in _ROW_RE.finditer(html):
        parsed = _parse_public_ts(_strip_html(m.group("ts")))
        if parsed is not None:
            last_date = parsed
        row_date = last_date
        if row_date != day:
            continue
        title = _strip_html(m.group("title"))
        source = (m.group("src") or "Finviz").strip()
        href = m.group("href").strip()
        if not title:
            continue
        out.append({
            "ticker": ticker.upper(),
            "title": title,
            "source": source,
            "url": href,
            "published_at": row_date.isoformat(),
        })
    return out


def fetch(tickers: list[str], day: date, per_ticker_sleep: float = 0.6,
          timeout: int = 30) -> list[dict]:
    """Finviz news for `tickers` on `day`. Uses Elite feed if FINVIZ_API_TOKEN is set."""
    if os.getenv("FINVIZ_API_TOKEN"):
        try:
            return _fetch_elite(tickers, day, timeout)
        except Exception as e:
            print(f"[finviz] Elite fetch failed ({e}); falling back to public scrape.", flush=True)

    if (date.today() - day) > timedelta(days=10):
        print(
            f"[finviz] public page only reliably surfaces ~1 week of history; "
            f"day={day} may return 0 items. Set FINVIZ_API_TOKEN for full history.",
            flush=True,
        )

    out: list[dict] = []
    for i, t in enumerate(tickers):
        out.extend(_fetch_public_one(t, day, timeout))
        if i < len(tickers) - 1:
            time.sleep(per_ticker_sleep)
    return out
