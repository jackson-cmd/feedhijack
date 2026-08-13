"""Benzinga News API provider (needs BENZINGA_API_KEY)."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

_ENDPOINT = "https://api.benzinga.com/api/v2/news"
_UA = "FeedHijack-Backtest/1.0"


def _get_key() -> str:
    key = os.getenv("BENZINGA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "BENZINGA_API_KEY missing. Get one at benzinga.com and set it in .env "
            "(or export BENZINGA_API_KEY=...)."
        )
    return key


def _fetch_page(tickers: list[str], day: date, page: int, page_size: int, timeout: int) -> list[dict]:
    params = {
        "token": _get_key(),
        "tickers": ",".join(t.upper() for t in tickers),
        "dateFrom": day.strftime("%Y-%m-%d"),
        "dateTo": day.strftime("%Y-%m-%d"),
        "pageSize": str(page_size),
        "page": str(page),
        "displayOutput": "headline",
    }
    url = _ENDPOINT + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def _normalize(record: dict) -> list[dict]:
    """One Benzinga record can attach to N tickers; emit one item per ticker."""
    title = (record.get("title") or "").strip()
    url = (record.get("url") or "").strip()
    pub = (record.get("created") or record.get("updated") or "").strip()
    stocks = record.get("stocks") or []
    out = []
    for s in stocks:
        sym = (s.get("name") or "").strip().upper()
        if not sym or not title:
            continue
        out.append({
            "ticker": sym,
            "title": title,
            "source": "Benzinga",
            "url": url,
            "published_at": pub,
        })
    return out


def fetch(tickers: list[str], day: date, page_size: int = 100, max_pages: int = 5,
          timeout: int = 30, page_sleep: float = 0.3) -> list[dict]:
    """Fetch Benzinga news for `tickers` on `day`. Paginates until an empty page or max_pages."""
    all_items: list[dict] = []
    for page in range(max_pages):
        try:
            records = _fetch_page(tickers, day, page, page_size, timeout)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
            print(f"[benzinga] {day} page={page} HTTP {e.code}: {body}", flush=True)
            break
        except Exception as e:
            print(f"[benzinga] {day} page={page} failed: {e}", flush=True)
            break
        if not records:
            break
        for rec in records:
            all_items.extend(_normalize(rec))
        if len(records) < page_size:
            break
        time.sleep(page_sleep)
    return all_items
