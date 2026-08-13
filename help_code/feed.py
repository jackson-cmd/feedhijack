"""News & social feed loader for decision context."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class NewsItem:
    ticker: str
    source: str
    headline: str
    summary: str
    sentiment: str
    relevance_score: float
    published_at: str


@dataclass
class SocialItem:
    ticker: str
    platform: str
    content: str
    sentiment: str
    engagement: int
    posted_at: str


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_news(base_dir: Optional[Path] = None) -> list[NewsItem]:
    """Load news.json, return list of NewsItem."""
    base = base_dir or Path(__file__).parent
    data = _load_json(base / "news.json")
    items = data.get("items", [])
    return [
        NewsItem(
            ticker=it.get("ticker", ""),
            source=it.get("source", ""),
            headline=it.get("headline", ""),
            summary=it.get("summary", ""),
            sentiment=it.get("sentiment", "neutral"),
            relevance_score=float(it.get("relevance_score", 0.5)),
            published_at=it.get("published_at", ""),
        )
        for it in items
    ]


def load_social(base_dir: Optional[Path] = None) -> list[SocialItem]:
    """Load social.json, return list of SocialItem."""
    base = base_dir or Path(__file__).parent
    data = _load_json(base / "social.json")
    items = data.get("items", [])
    return [
        SocialItem(
            ticker=it.get("ticker", ""),
            platform=it.get("platform", ""),
            content=it.get("content", ""),
            sentiment=it.get("sentiment", "neutral"),
            engagement=int(it.get("engagement", 0)),
            posted_at=it.get("posted_at", ""),
        )
        for it in items
    ]


def get_feed_for_ticker(
    ticker: str,
    news: list[NewsItem],
    social: list[SocialItem],
) -> tuple[list[NewsItem], list[SocialItem]]:
    """Filter news and social by ticker, sorted by relevance/engagement."""
    n = [x for x in news if x.ticker.upper() == ticker.upper()]
    s = [x for x in social if x.ticker.upper() == ticker.upper()]
    n.sort(key=lambda x: x.relevance_score, reverse=True)
    s.sort(key=lambda x: x.engagement, reverse=True)
    return n[:5], s[:5]


def format_feed_for_prompt(news: list[NewsItem], social: list[SocialItem]) -> str:
    """Condensed string for LLM prompt."""
    parts = []
    if news:
        parts.append("News:")
        for x in news[:3]:
            parts.append(f"  - [{x.sentiment}] {x.headline} ({x.source})")
    if social:
        parts.append("Social:")
        for x in social[:3]:
            c = x.content[:80] + ("..." if len(x.content) > 80 else "")
        parts.append(f"  - [{x.sentiment}] {c} (eng:{x.engagement})")
    return "\n".join(parts) if parts else "No news/social data."
