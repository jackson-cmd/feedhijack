#!/usr/bin/env python3
"""Fetch daily news headlines and write one JSON per day to --out-dir.

Sources (pick with --source or the NEWS_SOURCE env var):
  google_news  (default, free, no key)
  benzinga     (requires BENZINGA_API_KEY)
  finviz       (public scrape by default; FINVIZ_API_TOKEN uses Elite export)

Output shape (per YYYYMMDD.json), matching help_code/loaders.py:
  {
    "date":   "YYYY-MM-DD",
    "source": "google_news",
    "items":  [{"ticker","title","source","url","published_at"}, ...]
  }

Name kept as `crawl_google_news_rss.py` for backward-compat with help_code/main.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# make the sibling `providers` package importable without needing a parent package
# (this file lives inside `news&socials/` whose name isn't a valid Python identifier)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from providers import benzinga, finviz, google_news_rss  # noqa: E402

_SOURCES = {
    "google_news": google_news_rss.fetch,
    "google": google_news_rss.fetch,
    "benzinga": benzinga.fetch,
    "finviz": finviz.fetch,
}


def _parse_yyyymmdd(s: str) -> date:
    if len(s) != 8 or not s.isdigit():
        raise argparse.ArgumentTypeError(f"Date must be YYYYMMDD, got: {s}")
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def _each_day(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def _write_day(out_dir: Path, day: date, source: str, items: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{day.strftime('%Y%m%d')}.json"
    payload = {"date": day.isoformat(), "source": source, "items": items}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch daily headlines from a news feed provider.")
    ap.add_argument("--tickers", required=True, help="Comma-separated tickers")
    ap.add_argument("--from-date", required=True, help="Start YYYYMMDD (inclusive)")
    ap.add_argument("--to-date", required=True, help="End YYYYMMDD (inclusive)")
    ap.add_argument("--out-dir", required=True, help="Output directory for YYYYMMDD.json files")
    ap.add_argument(
        "--source",
        default=os.getenv("NEWS_SOURCE", "google_news"),
        choices=sorted(_SOURCES.keys()),
        help="Which provider to use (env NEWS_SOURCE also honored; default google_news)",
    )
    ap.add_argument("--overwrite", action="store_true", help="Re-fetch even if output already exists")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("No tickers.", file=sys.stderr)
        sys.exit(2)

    start = _parse_yyyymmdd(args.from_date)
    end = _parse_yyyymmdd(args.to_date)
    if end < start:
        print("--to-date must be >= --from-date", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out_dir)
    fetch = _SOURCES[args.source]
    canonical_source = "google_news" if args.source in ("google_news", "google") else args.source

    print(f"[news] source={canonical_source} tickers={tickers} {start}..{end}", flush=True)
    total_days = 0
    total_items = 0
    for day in _each_day(start, end):
        target = out_dir / f"{day.strftime('%Y%m%d')}.json"
        if target.exists() and not args.overwrite:
            print(f"[news] {day} exists, skip.", flush=True)
            continue
        try:
            items = fetch(tickers, day)
        except Exception as e:
            print(f"[news] {day} fetch failed: {e}", flush=True)
            items = []
        path = _write_day(out_dir, day, canonical_source, items)
        total_days += 1
        total_items += len(items)
        print(f"[news] {day} -> {path}  ({len(items)} items)", flush=True)

    print(f"[news] Done. Days written: {total_days}, items: {total_items}", flush=True)


if __name__ == "__main__":
    main()
