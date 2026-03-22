"""
Position Manager for Mean Reversion Swing Trading Bot.

Tracks individual trade positions with entry/exit prices and PnL,
supporting both manual/existing holdings and algorithm-generated trades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PositionSource(str, Enum):
    MANUAL = "MANUAL"
    BOT = "BOT"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"


@dataclass
class Position:
    """
    Represents a single position in an asset.

    Attributes:
        symbol:       Trading symbol (e.g. "XRP").
        quantity:     Number of units held.
        entry_price:  Average entry price per unit (USD).
        source:       Whether this position was opened manually or by the bot.
        opened_at:    UTC timestamp when the position was opened/initialised.
        status:       Current lifecycle status of the position.
        exit_price:   Average exit price per unit (USD) after closing.
        closed_at:    UTC timestamp when the position was fully closed.
        realized_pnl: Profit/loss realised on closed portions (USD).
    """

    symbol: str
    quantity: float
    entry_price: float
    source: PositionSource = PositionSource.BOT
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    closed_at: Optional[datetime] = None
    realized_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Read-only helpers
    # ------------------------------------------------------------------

    def unrealized_pnl(self, current_price: float) -> float:
        """Return the mark-to-market PnL for open portions (USD)."""
        if self.status == PositionStatus.CLOSED:
            return 0.0
        return (current_price - self.entry_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Return the unrealized PnL as a percentage of the cost basis."""
        if self.entry_price == 0.0:
            return 0.0
        return ((current_price - self.entry_price) / self.entry_price) * 100.0

    def cost_basis(self) -> float:
        """Return the total cost basis of the open position (USD)."""
        return self.entry_price * self.quantity

    def market_value(self, current_price: float) -> float:
        """Return the current market value of the position (USD)."""
        return current_price * self.quantity

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_to_position(self, quantity: float, price: float) -> None:
        """
        Pyramid into the position (average down/up).

        Recalculates the average entry price using the weighted mean.
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        total_cost = self.entry_price * self.quantity + price * quantity
        self.quantity += quantity
        self.entry_price = total_cost / self.quantity
        self.status = PositionStatus.OPEN
        logger.debug(
            "[%s] Pyramided: +%.4f units @ $%.4f → avg entry $%.4f",
            self.symbol,
            quantity,
            price,
            self.entry_price,
        )

    def close_partial(self, quantity: float, exit_price: float) -> float:
        """
        Close a portion of the position and return the realised PnL.

        Args:
            quantity:   Number of units to close.
            exit_price: The price at which units are sold.

        Returns:
            Realised PnL for the closed portion (USD).

        Raises:
            ValueError: If ``quantity`` exceeds remaining position size.
        """
        if quantity > self.quantity:
            raise ValueError(
                f"Cannot close {quantity} units; only {self.quantity} held"
            )
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        pnl = (exit_price - self.entry_price) * quantity
        self.realized_pnl += pnl
        self.quantity -= quantity
        self.exit_price = exit_price

        if self.quantity == 0.0:
            self.status = PositionStatus.CLOSED
            self.closed_at = datetime.now(timezone.utc)
        else:
            self.status = PositionStatus.PARTIAL

        logger.debug(
            "[%s] Partial close: %.4f units @ $%.4f → PnL $%.2f",
            self.symbol,
            quantity,
            exit_price,
            pnl,
        )
        return pnl

    def close_full(self, exit_price: float) -> float:
        """
        Close the entire position and return the realised PnL.

        Args:
            exit_price: The price at which all units are sold.

        Returns:
            Realised PnL (USD).
        """
        return self.close_partial(self.quantity, exit_price)

    def __repr__(self) -> str:
        return (
            f"Position(symbol={self.symbol!r}, qty={self.quantity:.4f}, "
            f"entry=${self.entry_price:.4f}, source={self.source.value}, "
            f"status={self.status.value})"
        )


class PositionManager:
    """
    Manages all positions for a single asset symbol.

    Distinguishes between existing manual holdings (initialised at bot
    startup from real Kraken balances) and new bot-generated trades.
    Supports partial exits and pyramiding.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._positions: list[Position] = []
        self._trade_count: int = 0
        self._total_realized_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Position creation
    # ------------------------------------------------------------------

    def initialise_existing(self, quantity: float, current_price: float) -> Position:
        """
        Register an existing manual holding as a tracked position.

        The current market price is used as the entry price because the
        original purchase price is not available from the API.

        Args:
            quantity:      Number of units held.
            current_price: Current market price (used as entry price proxy).

        Returns:
            The newly created Position.
        """
        pos = Position(
            symbol=self.symbol,
            quantity=quantity,
            entry_price=current_price,
            source=PositionSource.MANUAL,
        )
        self._positions.append(pos)
        logger.info(
            "[%s] Initialised existing holding: %.4f units @ $%.4f",
            self.symbol,
            quantity,
            current_price,
        )
        return pos

    def open_position(self, quantity: float, entry_price: float) -> Position:
        """
        Open a new bot-generated position.

        If an open position already exists for this symbol, the bot will
        pyramid into it rather than create a duplicate.

        Args:
            quantity:    Number of units to buy.
            entry_price: Purchase price per unit.

        Returns:
            The Position (new or updated).
        """
        open_pos = self.get_open_position()
        if open_pos is not None:
            open_pos.add_to_position(quantity, entry_price)
            self._trade_count += 1
            return open_pos

        pos = Position(
            symbol=self.symbol,
            quantity=quantity,
            entry_price=entry_price,
            source=PositionSource.BOT,
        )
        self._positions.append(pos)
        self._trade_count += 1
        logger.info(
            "[%s] Opened position: %.4f units @ $%.4f",
            self.symbol,
            quantity,
            entry_price,
        )
        return pos

    # ------------------------------------------------------------------
    # Position closing
    # ------------------------------------------------------------------

    def close_position(self, exit_price: float, quantity: Optional[float] = None) -> float:
        """
        Close the current open position (fully or partially).

        Args:
            exit_price: Price at which to close.
            quantity:   Units to close; closes everything if omitted.

        Returns:
            Realised PnL (USD), or 0.0 if no open position exists.
        """
        open_pos = self.get_open_position()
        if open_pos is None:
            logger.warning("[%s] No open position to close", self.symbol)
            return 0.0

        qty_to_close = quantity if quantity is not None else open_pos.quantity
        pnl = open_pos.close_partial(qty_to_close, exit_price)
        self._total_realized_pnl += pnl
        self._trade_count += 1

        logger.info(
            "[%s] Closed %.4f units @ $%.4f → PnL $%.2f (total $%.2f)",
            self.symbol,
            qty_to_close,
            exit_price,
            pnl,
            self._total_realized_pnl,
        )
        return pnl

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_open_position(self) -> Optional[Position]:
        """Return the first open/partial position, or None."""
        for pos in self._positions:
            if pos.status in (PositionStatus.OPEN, PositionStatus.PARTIAL):
                return pos
        return None

    def has_open_position(self) -> bool:
        """Return True if there is an open or partial position."""
        return self.get_open_position() is not None

    def total_quantity(self) -> float:
        """Return total units currently held (open + partial positions)."""
        return sum(
            p.quantity
            for p in self._positions
            if p.status in (PositionStatus.OPEN, PositionStatus.PARTIAL)
        )

    def unrealized_pnl(self, current_price: float) -> float:
        """Return the total unrealized PnL across all open positions (USD)."""
        return sum(
            p.unrealized_pnl(current_price)
            for p in self._positions
            if p.status in (PositionStatus.OPEN, PositionStatus.PARTIAL)
        )

    def realized_pnl(self) -> float:
        """Return the cumulative realised PnL from all closed trades (USD)."""
        return self._total_realized_pnl

    def total_pnl(self, current_price: float) -> float:
        """Return the combined realised + unrealized PnL (USD)."""
        return self.realized_pnl() + self.unrealized_pnl(current_price)

    def trade_count(self) -> int:
        """Return the number of completed trade legs (buys + sells)."""
        return self._trade_count

    def average_entry_price(self) -> float:
        """Return the average entry price across open positions."""
        open_pos = self.get_open_position()
        return open_pos.entry_price if open_pos else 0.0

    def summary(self, current_price: float) -> dict:
        """Return a summary dict suitable for logging and display."""
        open_pos = self.get_open_position()
        return {
            "symbol": self.symbol,
            "quantity": self.total_quantity(),
            "entry_price": self.average_entry_price(),
            "current_price": current_price,
            "market_value": self.total_quantity() * current_price,
            "unrealized_pnl": self.unrealized_pnl(current_price),
            "realized_pnl": self.realized_pnl(),
            "total_pnl": self.total_pnl(current_price),
            "trade_count": self.trade_count(),
            "source": open_pos.source.value if open_pos else "N/A",
            "status": open_pos.status.value if open_pos else "CLOSED",
        }
