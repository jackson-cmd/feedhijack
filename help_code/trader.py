"""Execution and P/L tracking."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from brain import TradeDecision


@dataclass
class Position:
    """Open position."""
    ticker: str
    quantity: int
    entry_price: float
    take_profit: float
    stop_loss: float
    entry_time: str


@dataclass
class ClosedTrade:
    """Closed trade with P/L."""
    ticker: str
    quantity: int
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_time: str
    reason: str


@dataclass
class Portfolio:
    """Local portfolio state."""
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    total_pnl: float = 0.0

    def apply_decision(
        self,
        decision: TradeDecision,
        current_price: float,
        margin_allowance: float = 0.0,
    ) -> Optional[ClosedTrade]:
        """Apply BUY/SELL/HOLD (BUY may use margin). Returns ClosedTrade on close."""
        if decision.action == "HOLD" or decision.quantity <= 0:
            return None

        buying_power = self.cash + margin_allowance
        if decision.action == "BUY":
            cost = decision.quantity * current_price
            if cost > buying_power:
                return None
            self.cash -= cost
            existing = self.positions.get(decision.ticker)
            if existing:
                # Add to position: weighted average entry, update TP/SL from decision
                new_qty = existing.quantity + decision.quantity
                new_entry = (existing.entry_price * existing.quantity + current_price * decision.quantity) / new_qty
                self.positions[decision.ticker] = Position(
                    ticker=decision.ticker,
                    quantity=new_qty,
                    entry_price=new_entry,
                    take_profit=decision.take_profit_price,
                    stop_loss=decision.stop_loss_price,
                    entry_time=existing.entry_time,
                )
            else:
                self.positions[decision.ticker] = Position(
                    ticker=decision.ticker,
                    quantity=decision.quantity,
                    entry_price=current_price,
                    take_profit=decision.take_profit_price,
                    stop_loss=decision.stop_loss_price,
                    entry_time=datetime.now().isoformat(),
                )
            return None

        if decision.action == "SELL":
            pos = self.positions.get(decision.ticker)
            if not pos:
                return None
            qty = min(decision.quantity, pos.quantity)
            proceeds = qty * current_price
            self.cash += proceeds
            pnl = (current_price - pos.entry_price) * qty
            pnl_pct = (current_price / pos.entry_price - 1) * 100 if pos.entry_price else 0

            closed = ClosedTrade(
                ticker=decision.ticker,
                quantity=qty,
                entry_price=pos.entry_price,
                exit_price=current_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_time=datetime.now().isoformat(),
                reason=decision.short_reason or "LLM signal",
            )
            self.closed_trades.append(closed)
            self.total_pnl += pnl

            if qty >= pos.quantity:
                del self.positions[decision.ticker]
            else:
                self.positions[decision.ticker] = Position(
                    ticker=pos.ticker,
                    quantity=pos.quantity - qty,
                    entry_price=pos.entry_price,
                    take_profit=pos.take_profit,
                    stop_loss=pos.stop_loss,
                    entry_time=pos.entry_time,
                )
            return closed

        return None


def _data_dir() -> Path:
    d = Path(__file__).parent / "trades_data"
    d.mkdir(exist_ok=True)
    return d


def save_portfolio(portfolio: Portfolio) -> None:
    """Persist portfolio state to JSON."""
    path = _data_dir() / "portfolio.json"
    data = {
        "cash": portfolio.cash,
        "total_pnl": portfolio.total_pnl,
        "positions": {
            k: {
                "ticker": v.ticker,
                "quantity": v.quantity,
                "entry_price": v.entry_price,
                "take_profit": v.take_profit,
                "stop_loss": v.stop_loss,
                "entry_time": v.entry_time,
            }
            for k, v in portfolio.positions.items()
        },
        "closed_count": len(portfolio.closed_trades),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_portfolio(initial_cash: float = 100_000.0) -> Portfolio:
    """Load portfolio from disk or create new."""
    path = _data_dir() / "portfolio.json"
    if not path.exists():
        return Portfolio(cash=initial_cash)

    with open(path) as f:
        data = json.load(f)

    positions = {}
    for k, v in data.get("positions", {}).items():
        positions[k] = Position(
            ticker=v["ticker"],
            quantity=v["quantity"],
            entry_price=v["entry_price"],
            take_profit=v["take_profit"],
            stop_loss=v["stop_loss"],
            entry_time=v["entry_time"],
        )

    p = Portfolio(
        cash=data.get("cash", initial_cash),
        positions=positions,
        total_pnl=data.get("total_pnl", 0),
    )
    log_path = _data_dir() / "trades_log.jsonl"
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    try:
                        t = json.loads(line)
                        p.closed_trades.append(ClosedTrade(
                            ticker=t["ticker"],
                            quantity=t["quantity"],
                            entry_price=t["entry_price"],
                            exit_price=t["exit_price"],
                            pnl=t["pnl"],
                            pnl_pct=t["pnl_pct"],
                            exit_time=t["exit_time"],
                            reason=t.get("reason", ""),
                        ))
                    except Exception:
                        pass
    return p


def log_trade(closed: ClosedTrade) -> None:
    """Append closed trade to log file."""
    path = _data_dir() / "trades_log.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps({
            "ticker": closed.ticker,
            "quantity": closed.quantity,
            "entry_price": closed.entry_price,
            "exit_price": closed.exit_price,
            "pnl": closed.pnl,
            "pnl_pct": closed.pnl_pct,
            "exit_time": closed.exit_time,
            "reason": closed.reason,
        }) + "\n")
