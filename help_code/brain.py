"""
Single-Shot LLM Decision Engine - No debate, strict JSON output.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from data import IntradaySnapshot


class TradeDecision(BaseModel):
    """Strict schema for LLM output."""

    action: Literal["BUY", "SELL", "HOLD"] = Field(description="Trading action")
    ticker: str = Field(description="Stock symbol")
    quantity: int = Field(ge=0, description="Number of shares")
    take_profit_price: float = Field(ge=0, description="Target exit price")
    stop_loss_price: float = Field(ge=0, description="Stop loss price")
    short_reason: str = Field(default="", description="Brief reason")
    reasoning: str = Field(default="", description="Decision thinking process")


def _decisions_dir() -> Path:
    d = Path(__file__).parent / "trades_data" / "decisions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_reasoning(ticker: str, reasoning: str, decision: dict) -> None:
    """Write decision thinking process to file."""
    path = _decisions_dir() / "decision_log.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "reasoning": reasoning,
        "action": decision.get("action", "HOLD"),
        "quantity": decision.get("quantity", 0),
        "short_reason": decision.get("short_reason", ""),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


TRADING_MODES = ("aggressive", "medium", "conservative")


def _build_prompt(
    snap: IntradaySnapshot,
    cash: float,
    risk_pct: float = 0.02,
    feed_context: str = "",
    position_summary: str = "",
    trading_mode: str = "medium",
) -> str:
    """Condensed prompt for zero-latency decision."""
    max_risk = cash * risk_pct
    qty_cap = int(max_risk / (snap.price * 0.02)) if snap.price > 0 else 0
    qty_cap = max(1, min(qty_cap, 500))

    pos_str = position_summary.strip() or "No open position."
    one_position_rule = (
        "You can only hold ONE position at a time. To open a new position in another ticker you must first SELL to close the current one. "
        "You may BUY more of the same ticker to add to the position, or SELL (partial or full) to close. Be aware of your current position and prices when deciding."
    )

    _mode_rules = {
        "aggressive": (
            "TRADING MODE: AGGRESSIVE. Trade with the trend every day when trend is clear. "
            "Identify trend using EMA: price above EMA20/EMA50 = uptrend (prefer BUY); price below EMA20/EMA50 = downtrend (prefer SELL to close longs or stay flat). "
            "You MUST take a position when trend is clear; use small size (e.g. 5–15 shares) if uncertain. "
            "Take-profit and stop-loss: use WIDER levels—give the trade room. Set take_profit_price and stop_loss_price farther from entry so the trend can develop; avoid tight stops."
        ),
        "medium": (
            "TRADING MODE: MEDIUM. Follow the trend and trade when the setup is clear. "
            "Identify trend using EMA: price above EMA20/EMA50 = uptrend (consider BUY); price below = downtrend (consider SELL or HOLD). "
            "Output BUY or SELL when trend (EMA alignment) and RSI or news support the move; otherwise HOLD. Do not always HOLD—take trades when edge is moderate. "
            "Take-profit and stop-loss: use MODERATE levels—balanced distance from entry, not too tight and not too wide."
        ),
        "conservative": (
            "TRADING MODE: CONSERVATIVE. Trade only when the trend is very clear and the setup is high conviction. "
            "Identify trend using EMA: strong uptrend = price well above EMA20 and EMA50; strong downtrend = price well below. "
            "Output BUY or SELL only when you have high confidence; otherwise HOLD. "
            "Take-profit and stop-loss: use VERY TIGHT levels to preserve capital. Set take_profit_price and stop_loss_price close to entry so losses are small and profits are locked in quickly."
        ),
    }
    mode_rule = _mode_rules.get(trading_mode, _mode_rules["medium"])

    return f"""You are a disciplined trend-following equity trader. All instructions in English.

{mode_rule}

Trend rule: Use EMA to identify trend. Price > EMA20 and EMA50 = uptrend (bias BUY). Price < EMA20 and EMA50 = downtrend (bias SELL or reduce exposure). Trade in the direction of the trend.

{one_position_rule}

Current position (be aware of entry, current price, TP/SL and unrealized P/L when deciding):
{pos_str}

Ticker: {snap.ticker}
Price(Open): {snap.price:.2f} | Last: {getattr(snap, "last_price", snap.price):.2f} | VWAP: {snap.vwap:.2f}
RSI: {snap.rsi:.1f} | MACD: {snap.macd:.4f} | MACD Hist: {snap.macd_hist:.4f}
BB: Upper={snap.bb_upper:.2f} Mid={snap.bb_mid:.2f} Lower={snap.bb_lower:.2f}
Support: {snap.support:.2f} | Resistance: {snap.resistance:.2f}
Volume: {snap.volume}
EMA: 20={snap.ema20:.2f} 50={snap.ema50:.2f} 100={snap.ema100:.2f} 200={snap.ema200:.2f}

{feed_context}

Cash: ${cash:.0f} | Max quantity: {qty_cap}

You choose take_profit_price and stop_loss_price yourself (no ATR formula). For BUY: stop_loss_price below entry, take_profit_price above entry. For SELL (closing a long): stop_loss_price and take_profit_price around current price. When you have an open position and price reaches your TP or SL zone, output SELL to close.
Output ONLY this JSON (no other text):
{{"action":"BUY"|"SELL"|"HOLD","ticker":"{snap.ticker}","quantity":0-{qty_cap},"take_profit_price":float,"stop_loss_price":float,"short_reason":"brief","reasoning":"Your step-by-step thinking in 1-3 sentences"}}"""


def _parse_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response (handle markdown code blocks)."""
    text = text.strip()
    # Remove markdown code block if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    return json.loads(text)


def decide(
    snap: IntradaySnapshot,
    cash: float,
    risk_pct: float = 0.02,
    model: str = "gpt-4o-mini",
    feed_context: str = "",
    position_summary: str = "",
    trading_mode: str = "medium",
    *,
    log_decision: bool = True,
    temperature: float = 0.1,
    return_details: bool = False,
) -> TradeDecision | tuple[TradeDecision, dict]:
    """
    Single-shot LLM decision. Returns structured TradeDecision.
    Writes reasoning to trades_data/decisions/decision_log.jsonl unless log_decision=False.
    If return_details=True, returns (TradeDecision, {"prompt": str, "raw_response": str, "parsed": dict}).
    """
    import os

    prompt = _build_prompt(
        snap, cash, risk_pct,
        feed_context=feed_context,
        position_summary=position_summary or "No open position.",
        trading_mode=trading_mode if trading_mode in TRADING_MODES else "medium",
    )

    # Backend routing by model name. Gemini/Claude return raw text; the JSON
    # parsing + TradeDecision construction below is shared across all backends
    # so the cross-model comparison is apples-to-apples.
    ml = model.lower()
    if ml.startswith("gemini"):
        from llm_backends import gemini_complete
        content = gemini_complete(prompt, model, temperature=temperature)
    elif ml.startswith("claude"):
        from llm_backends import claude_cli_complete
        content = claude_cli_complete(prompt, model, temperature=temperature)
    else:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            dec = TradeDecision(
                action="HOLD",
                ticker=snap.ticker,
                quantity=0,
                take_profit_price=snap.price * 1.02,
                stop_loss_price=snap.price * 0.98,
                short_reason="No API key",
                reasoning="",
            )
            if log_decision:
                _write_reasoning(snap.ticker, "No API key - fallback HOLD", {"action": "HOLD", "quantity": 0, "short_reason": "No API key"})
            details = {"prompt": prompt, "raw_response": "", "parsed": {}}
            return (dec, details) if return_details else dec

        client = OpenAI(api_key=api_key)
        is_reasoning = model.startswith("o1") or model.startswith("o3") or model.startswith("o4") or model.startswith("gpt-5")
        if is_reasoning:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1024,
            )
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=350,
            )
        content = resp.choices[0].message.content or "{}"
    raw = _parse_json_from_response(content)
    action = str(raw.get("action", "HOLD")).upper()
    if action not in ("BUY", "SELL", "HOLD"):
        action = "HOLD"
    if action == "HOLD":
        raw["quantity"] = 0

    # Use LLM's TP/SL as-is (no ATR). Only ensure valid ordering for BUY and positive values.
    qty = int(raw.get("quantity", 0))
    tp = float(raw.get("take_profit_price", snap.price * 1.02))
    sl = float(raw.get("stop_loss_price", snap.price * 0.98))
    price = snap.price
    if action == "BUY" and qty > 0:
        if sl >= price:
            sl = max(0.01, price * 0.98)
        if tp <= price:
            tp = price * 1.02
    tp = max(0.01, tp)
    sl = max(0.01, sl)

    reasoning = str(raw.get("reasoning", ""))[:500]
    raw["take_profit_price"] = tp
    raw["stop_loss_price"] = sl
    raw["quantity"] = qty
    if log_decision:
        _write_reasoning(snap.ticker, reasoning, raw)

    dec = TradeDecision(
        action=action,
        ticker=raw.get("ticker", snap.ticker),
        quantity=qty,
        take_profit_price=tp,
        stop_loss_price=sl,
        short_reason=str(raw.get("short_reason", ""))[:200],
        reasoning=reasoning,
    )
    details = {"prompt": prompt, "raw_response": content, "parsed": dict(raw)}
    return (dec, details) if return_details else dec
