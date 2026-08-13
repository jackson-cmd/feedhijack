"""Google News RSS provider (no key)."""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

_BASE = "https://news.google.com/rss/search"
_UA = "Mozilla/5.0 (compatible; FeedHijack-Backtest/1.0; +https://example.com/bot)"


def _fmt(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _clean_title(raw: str) -> tuple[str, str]:
    """Split Google News' 'Title - Source' suffix."""
    if not raw:
        return "", ""
    m = re.match(r"^(.*)\s+-\s+([^-]+)$", raw.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return raw.strip(), ""


def _fetch(query: str, day: date, timeout: int = 20) -> str:
    # after: is inclusive, before: is exclusive — one day at a time keeps items scoped.
    after = _fmt(day)
    before = _fmt(date.fromordinal(day.toordinal() + 1))
    q = f'{query} after:{after} before:{before}'
    params = {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    url = _BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _parse(xml_text: str, ticker: str, day: date) -> list[dict]:
    out: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    channel = root.find("channel")
    if channel is None:
        return out
    for it in channel.findall("item"):
        title_raw = (it.findtext("title") or "").strip()
        title, src_from_title = _clean_title(title_raw)
        link = (it.findtext("link") or "").strip()
        pub_raw = (it.findtext("pubDate") or "").strip()
        # <source url="...">NAME</source>
        src_el = it.find("source")
        source = (src_el.text if src_el is not None and src_el.text else src_from_title).strip()
        pub_iso = ""
        if pub_raw:
            try:
                pub_iso = parsedate_to_datetime(pub_raw).isoformat()
            except Exception:
                pub_iso = pub_raw
        out.append({
            "ticker": ticker.upper(),
            "title": title,
            "source": source,
            "url": link,
            "published_at": pub_iso,
        })
    return out


def fetch(tickers: list[str], day: date, per_ticker_sleep: float = 0.5) -> list[dict]:
    """Fetch headlines for each ticker on `day`. Small sleep between tickers to be polite."""
    items: list[dict] = []
    for i, t in enumerate(tickers):
        try:
            xml_text = _fetch(f'"{t}" stock', day)
            items.extend(_parse(xml_text, t, day))
        except Exception as e:
            print(f"[google_news_rss] {t} {day} failed: {e}", flush=True)
        if i < len(tickers) - 1:
            time.sleep(per_ticker_sleep)
    return items
