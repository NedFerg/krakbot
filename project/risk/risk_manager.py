"""
Risk Manager — Order approval and portfolio protection.

Enforces per-symbol position limits, notional order limits, per-agent
drawdown limits, and a global kill-switch.  Every order intent is passed
through ``RiskManager.approve_order()`` before execution; rejected orders
are logged but never sent to the exchange.

Configuration is driven entirely by environment variables so the same
code runs safely with $51 or $5,000 of capital without code changes.

Environment variables
---------------------
    RISK_MAX_POSITION_PCT       Max fraction of total capital per symbol (default 0.20)
    RISK_MAX_NOTIONAL_PER_ORDER Max USD value of a single order (default 1000.0)
    RISK_MAX_DRAWDOWN_PCT       Per-symbol drawdown that triggers a symbol halt (default 0.15)
    RISK_GLOBAL_MAX_DRAWDOWN_PCT Portfolio-level drawdown that kills all trading (default 0.25)
    TOTAL_TRADING_CAPITAL       Total USD capital under management (default 51.0)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order intent
# ---------------------------------------------------------------------------

@dataclass
class OrderIntent:
    """Represents a proposed order before it is approved by the risk manager."""

    symbol: str
    side: str          # "buy" or "sell"
    quantity: float    # units in base currency
    price: float       # expected execution price (USD)

    @property
    def notional(self) -> float:
        """Return the USD notional value of the order."""
        return self.quantity * self.price


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Stateful risk manager that approves or rejects order intents.

    Args:
        total_capital:           Total USD capital under management.
        max_position_pct:        Maximum fraction of capital per symbol [0, 1].
        max_notional_per_order:  Maximum USD notional for a single order.
        max_drawdown_pct:        Per-symbol drawdown threshold [0, 1].
        global_max_drawdown_pct: Portfolio-level drawdown kill-switch [0, 1].
    """

    def __init__(
        self,
        total_capital: float,
        max_position_pct: float,
        max_notional_per_order: float,
        max_drawdown_pct: float,
        global_max_drawdown_pct: float,
    ) -> None:
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.max_notional_per_order = max_notional_per_order
        self.max_drawdown_pct = max_drawdown_pct
        self.global_max_drawdown_pct = global_max_drawdown_pct

        # Track peak capital for drawdown calculation
        self._peak_capital: float = total_capital
        # Per-symbol peak notional exposure for drawdown tracking
        self._symbol_peak: dict[str, float] = {}
        # Symbols that have been halted due to drawdown
        self._halted_symbols: set[str] = set()
        # Global kill-switch flag
        self._global_halt: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "RiskManager":
        """
        Construct a RiskManager from environment variables.

        Falls back to conservative defaults if variables are not set.
        """
        total_capital = float(os.environ.get("TOTAL_TRADING_CAPITAL", "51.0"))
        max_position_pct = float(os.environ.get("RISK_MAX_POSITION_PCT", "0.20"))
        max_notional_per_order = float(
            os.environ.get("RISK_MAX_NOTIONAL_PER_ORDER", str(total_capital))
        )
        max_drawdown_pct = float(os.environ.get("RISK_MAX_DRAWDOWN_PCT", "0.15"))
        global_max_drawdown_pct = float(
            os.environ.get("RISK_GLOBAL_MAX_DRAWDOWN_PCT", "0.25")
        )

        instance = cls(
            total_capital=total_capital,
            max_position_pct=max_position_pct,
            max_notional_per_order=max_notional_per_order,
            max_drawdown_pct=max_drawdown_pct,
            global_max_drawdown_pct=global_max_drawdown_pct,
        )
        logger.info(
            "RiskManager initialised: capital=$%.2f, max_pos=%.0f%%, "
            "max_notional=$%.2f, drawdown=%.0f%%, global_dd=%.0f%%",
            total_capital,
            max_position_pct * 100,
            max_notional_per_order,
            max_drawdown_pct * 100,
            global_max_drawdown_pct * 100,
        )
        return instance

    def approve_order(
        self,
        intent: OrderIntent,
        current_symbol_exposure: float = 0.0,
        current_portfolio_value: Optional[float] = None,
    ) -> tuple[bool, str]:
        """
        Approve or reject an order intent.

        Args:
            intent:                    The proposed order.
            current_symbol_exposure:   Current USD exposure to this symbol.
            current_portfolio_value:   Current total portfolio value (optional).

        Returns:
            (approved: bool, reason: str)
        """
        # 1. Global kill-switch
        if self._global_halt:
            return False, "Global trading halt — portfolio drawdown limit exceeded"

        # 2. Symbol halt
        if intent.symbol in self._halted_symbols:
            return False, f"Symbol {intent.symbol} halted — per-symbol drawdown limit exceeded"

        # 3. Notional order limit
        if intent.notional > self.max_notional_per_order:
            return (
                False,
                f"Order notional ${intent.notional:.2f} exceeds limit ${self.max_notional_per_order:.2f}",
            )

        # 4. Position concentration limit (buy orders only)
        if intent.side == "buy":
            new_exposure = current_symbol_exposure + intent.notional
            max_exposure = self.total_capital * self.max_position_pct
            if new_exposure > max_exposure:
                return (
                    False,
                    f"Position concentration ${new_exposure:.2f} would exceed "
                    f"{self.max_position_pct*100:.0f}% limit (${max_exposure:.2f})",
                )

        # 5. Check portfolio-level drawdown if value provided
        if current_portfolio_value is not None:
            self._update_peak(current_portfolio_value)
            drawdown = self._portfolio_drawdown(current_portfolio_value)
            if drawdown >= self.global_max_drawdown_pct:
                self._global_halt = True
                logger.critical(
                    "GLOBAL HALT: portfolio drawdown %.1f%% exceeds %.1f%% limit",
                    drawdown * 100,
                    self.global_max_drawdown_pct * 100,
                )
                return False, f"Global trading halt — portfolio drawdown {drawdown*100:.1f}%"

        logger.debug(
            "Order APPROVED: %s %s %.4f @ $%.4f (notional $%.2f)",
            intent.side.upper(),
            intent.symbol,
            intent.quantity,
            intent.price,
            intent.notional,
        )
        return True, "Approved"

    def record_symbol_pnl(self, symbol: str, realized_pnl: float, entry_notional: float) -> None:
        """
        Record a realised trade result and check per-symbol drawdown.

        Args:
            symbol:          Trading symbol.
            realized_pnl:    Realised profit/loss (USD; negative = loss).
            entry_notional:  Original entry notional for this trade.
        """
        if entry_notional <= 0:
            return

        # Convert negative PnL to a positive loss rate for comparison
        loss_pct = (-realized_pnl / entry_notional) if realized_pnl < 0 else 0.0
        if loss_pct >= self.max_drawdown_pct:
            self._halted_symbols.add(symbol)
            logger.warning(
                "[%s] HALTED: realised loss %.1f%% exceeds %.1f%% per-symbol limit",
                symbol,
                loss_pct * 100,
                self.max_drawdown_pct * 100,
            )

    def update_total_capital(self, new_capital: float) -> None:
        """
        Update the total capital figure (e.g. after profit reinvestment).

        This recalibrates position size limits to the new capital level.
        """
        old = self.total_capital
        self.total_capital = new_capital
        self._update_peak(new_capital)
        logger.info(
            "Capital updated: $%.2f → $%.2f (%.1f%% change)",
            old,
            new_capital,
            (new_capital - old) / old * 100 if old > 0 else 0,
        )

    def is_symbol_halted(self, symbol: str) -> bool:
        """Return True if the symbol has been halted due to drawdown."""
        return symbol in self._halted_symbols

    def is_global_halt(self) -> bool:
        """Return True if the global kill-switch is active."""
        return self._global_halt

    def reset_symbol_halt(self, symbol: str) -> None:
        """Manually clear a per-symbol halt (use with caution)."""
        self._halted_symbols.discard(symbol)
        logger.info("[%s] Symbol halt cleared manually", symbol)

    def reset_global_halt(self) -> None:
        """Manually clear the global halt (use with caution)."""
        self._global_halt = False
        logger.info("Global halt cleared manually")

    def status(self) -> dict:
        """Return a summary of the current risk state."""
        return {
            "total_capital": self.total_capital,
            "peak_capital": self._peak_capital,
            "global_halt": self._global_halt,
            "halted_symbols": sorted(self._halted_symbols),
            "max_position_pct": self.max_position_pct,
            "max_notional_per_order": self.max_notional_per_order,
            "max_drawdown_pct": self.max_drawdown_pct,
            "global_max_drawdown_pct": self.global_max_drawdown_pct,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_peak(self, current_value: float) -> None:
        if current_value > self._peak_capital:
            self._peak_capital = current_value

    def _portfolio_drawdown(self, current_value: float) -> float:
        if self._peak_capital <= 0:
            return 0.0
        return max(0.0, (self._peak_capital - current_value) / self._peak_capital)
