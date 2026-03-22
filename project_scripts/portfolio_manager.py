"""
Portfolio Manager — Capital tracking and profit reinvestment.

Tracks total portfolio value across all trading pairs, captures
realised profits from closed trades, and automatically reinvests
those profits into available capital so the bot scales exponentially
as it compounds gains.

Usage
-----
    pm = PortfolioManager(initial_capital=51.0)
    pm.record_trade_profit("XRP", realized_pnl=3.50)
    new_capital_per_pair = pm.capital_per_pair(num_pairs=5)

Environment variables
---------------------
    TOTAL_TRADING_CAPITAL     Initial total USD capital (default 51.0).
    POSITION_SIZE_PCT         Fraction of capital to deploy per trade (default 0.95).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Immutable record of a completed trade leg."""

    symbol: str
    realized_pnl: float
    entry_price: float
    exit_price: float
    quantity: float
    closed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PortfolioManager:
    """
    Tracks total capital, realised profits, and computes reinvested
    position sizes for exponential portfolio growth.

    Args:
        initial_capital:  Starting USD capital.
        position_size_pct: Fraction of available capital to deploy per trade.
    """

    def __init__(
        self,
        initial_capital: float,
        position_size_pct: float = 0.95,
    ) -> None:
        self._initial_capital: float = initial_capital
        self._available_capital: float = initial_capital
        self._position_size_pct: float = position_size_pct
        self._total_realized_pnl: float = 0.0
        self._trade_records: list[TradeRecord] = []
        self._capital_snapshots: list[tuple[datetime, float]] = [
            (datetime.now(timezone.utc), initial_capital)
        ]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "PortfolioManager":
        """Construct a PortfolioManager from environment variables."""
        initial_capital = float(os.environ.get("TOTAL_TRADING_CAPITAL", "51.0"))
        position_size_pct = float(os.environ.get("POSITION_SIZE_PCT", "0.95"))
        instance = cls(
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
        )
        logger.info(
            "PortfolioManager initialised: capital=$%.2f, position_size=%.0f%%",
            initial_capital,
            position_size_pct * 100,
        )
        return instance

    # ------------------------------------------------------------------
    # Capital management
    # ------------------------------------------------------------------

    def record_trade_profit(
        self,
        symbol: str,
        realized_pnl: float,
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        quantity: float = 0.0,
    ) -> None:
        """
        Record the result of a completed trade and reinvest profits.

        Positive PnL is added to available capital (compounding).
        Losses are deducted from available capital.

        Args:
            symbol:       Trading symbol (e.g. "XRP").
            realized_pnl: Profit/loss from the closed trade (USD).
            entry_price:  Average entry price (for record-keeping).
            exit_price:   Exit price (for record-keeping).
            quantity:     Units traded (for record-keeping).
        """
        self._total_realized_pnl += realized_pnl
        self._available_capital += realized_pnl

        record = TradeRecord(
            symbol=symbol,
            realized_pnl=realized_pnl,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
        )
        self._trade_records.append(record)
        self._capital_snapshots.append(
            (datetime.now(timezone.utc), self._available_capital)
        )

        action = "REINVESTED" if realized_pnl > 0 else "DEDUCTED"
        logger.info(
            "[%s] Trade closed: PnL=$%.2f → %s | Available capital: $%.2f",
            symbol,
            realized_pnl,
            action,
            self._available_capital,
        )

    def set_available_capital(self, capital: float) -> None:
        """
        Manually set available capital (e.g. after querying live balance).

        Args:
            capital: Updated USD cash balance.
        """
        old = self._available_capital
        self._available_capital = max(0.0, capital)
        self._capital_snapshots.append(
            (datetime.now(timezone.utc), self._available_capital)
        )
        logger.info(
            "Capital updated from exchange balance: $%.2f → $%.2f",
            old,
            self._available_capital,
        )

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def capital_per_pair(self, num_pairs: int) -> float:
        """
        Return USD capital to allocate per trading pair.

        Divides deployable capital evenly across active pairs, ensuring
        the bot scales automatically as profits are reinvested.

        Args:
            num_pairs: Number of active trading pairs.

        Returns:
            Capital per pair (USD), or 0.0 if no pairs or capital.
        """
        if num_pairs <= 0 or self._available_capital <= 0:
            return 0.0
        deployable = self._available_capital * self._position_size_pct
        return deployable / num_pairs

    def max_order_notional(self, num_pairs: int) -> float:
        """
        Return the maximum notional USD value of a single order.

        Equal to capital_per_pair — the bot never risks more than its
        per-pair allocation in a single trade.
        """
        return self.capital_per_pair(num_pairs)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    @property
    def total_realized_pnl(self) -> float:
        """Cumulative realised PnL across all symbols (USD)."""
        return self._total_realized_pnl

    @property
    def available_capital(self) -> float:
        """Current available USD capital (initial + reinvested profits)."""
        return self._available_capital

    @property
    def initial_capital(self) -> float:
        """Starting capital (USD)."""
        return self._initial_capital

    @property
    def growth_pct(self) -> float:
        """Portfolio growth percentage since inception."""
        if self._initial_capital <= 0:
            return 0.0
        return (self._available_capital - self._initial_capital) / self._initial_capital * 100.0

    def trade_count(self) -> int:
        """Total number of completed trade records."""
        return len(self._trade_records)

    def winning_trades(self) -> list[TradeRecord]:
        """Return all trade records with positive PnL."""
        return [r for r in self._trade_records if r.realized_pnl > 0]

    def losing_trades(self) -> list[TradeRecord]:
        """Return all trade records with negative or zero PnL."""
        return [r for r in self._trade_records if r.realized_pnl <= 0]

    def win_rate(self) -> float:
        """Win rate as a fraction [0, 1]."""
        total = self.trade_count()
        if total == 0:
            return 0.0
        return len(self.winning_trades()) / total

    def summary(self) -> dict:
        """Return a summary dict for logging and display."""
        return {
            "initial_capital": self._initial_capital,
            "available_capital": self._available_capital,
            "total_realized_pnl": self._total_realized_pnl,
            "growth_pct": round(self.growth_pct, 2),
            "trade_count": self.trade_count(),
            "win_rate_pct": round(self.win_rate() * 100, 1),
            "position_size_pct": self._position_size_pct,
        }
