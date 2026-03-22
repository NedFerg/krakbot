"""
Enhanced Trading Bot v2 — Mean Reversion Swing Trader.

EnhancedTradeBot combines RSI, MACD, Moving Averages, and
Support/Resistance levels to generate high-quality BUY and SELL signals.

Signal Logic
------------
BUY  (Oversold + Uptrend):
    RSI < 30  AND  fast_MA > slow_MA  AND  MACD histogram > 0

SELL (Overbought + Downtrend):
    RSI > 70  AND  fast_MA < slow_MA  AND  MACD histogram < 0

HOLD:
    Waiting for all conditions to align.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from project_scripts.position_manager import PositionManager
from project_scripts.technical_indicators import (
    MACDResult,
    SupportResistance,
    calculate_atr,
    calculate_macd,
    calculate_moving_average,
    calculate_rsi,
    find_support_resistance,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults (can be overridden via strategy config)
# ---------------------------------------------------------------------------

DEFAULT_FAST_MA_PERIOD = 10
DEFAULT_SLOW_MA_PERIOD = 30
DEFAULT_RSI_PERIOD = 14
DEFAULT_RSI_OVERSOLD = 30.0
DEFAULT_RSI_OVERBOUGHT = 70.0
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_ATR_PERIOD = 14
DEFAULT_SR_LOOKBACK = 20

# Signal string constants
SIGNAL_BUY = "BUY"
SIGNAL_SELL = "SELL"
SIGNAL_HOLD = "HOLD"


@dataclass
class SignalResult:
    """
    Rich signal metadata returned by ``EnhancedTradeBot.generate_signal()``.

    Attributes:
        signal:     "BUY", "SELL", or "HOLD".
        price:      Current close price at signal time.
        rsi:        RSI value (0-100).
        macd:       Full MACD result (line, signal_line, histogram).
        fast_ma:    Fast moving average value.
        slow_ma:    Slow moving average value.
        sr:         Support and resistance levels.
        atr:        ATR value (volatility measure).
        confidence: Fraction of conditions satisfied [0.0, 1.0].
        reason:     Human-readable explanation of the signal.
        timestamp:  UTC time the signal was generated.
    """

    signal: str
    price: float
    rsi: float
    macd: MACDResult
    fast_ma: float
    slow_ma: float
    sr: SupportResistance
    atr: float
    confidence: float
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "price": self.price,
            "rsi": round(self.rsi, 2),
            "macd_line": round(self.macd.macd_line, 6),
            "macd_signal": round(self.macd.signal_line, 6),
            "macd_hist": round(self.macd.histogram, 6),
            "fast_ma": round(self.fast_ma, 4),
            "slow_ma": round(self.slow_ma, 4),
            "support": round(self.sr.support, 4),
            "resistance": round(self.sr.resistance, 4),
            "atr": round(self.atr, 6),
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


class EnhancedTradeBot:
    """
    Mean reversion swing trading bot with composite signal generation.

    The bot is initialised with the current Kraken holdings for its symbol
    so it can manage existing manual positions alongside its own trades.

    Args:
        symbol:            Short asset symbol (e.g. "XRP").
        kraken_pair:       Kraken trading pair (e.g. "XXRPZUSD").
        capital_per_trade: USD allocated for each new bot trade entry.
        existing_qty:      Existing units held (from Kraken balance query).
        existing_price:    Current market price at initialisation time,
                           used as the entry price proxy for existing holdings.
        fast_ma_period:    Fast SMA period (default 10).
        slow_ma_period:    Slow SMA period (default 30).
        rsi_period:        RSI lookback period (default 14).
        rsi_oversold:      RSI threshold for oversold condition (default 30).
        rsi_overbought:    RSI threshold for overbought condition (default 70).
    """

    def __init__(
        self,
        symbol: str,
        kraken_pair: str,
        capital_per_trade: float,
        existing_qty: float = 0.0,
        existing_price: float = 0.0,
        fast_ma_period: int = DEFAULT_FAST_MA_PERIOD,
        slow_ma_period: int = DEFAULT_SLOW_MA_PERIOD,
        rsi_period: int = DEFAULT_RSI_PERIOD,
        rsi_oversold: float = DEFAULT_RSI_OVERSOLD,
        rsi_overbought: float = DEFAULT_RSI_OVERBOUGHT,
    ) -> None:
        self.symbol = symbol
        self.kraken_pair = kraken_pair
        self.capital_per_trade = capital_per_trade
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        self.position_manager = PositionManager(symbol)
        self._signal_history: list[SignalResult] = []

        # Register existing holdings so the bot can manage them
        if existing_qty > 0 and existing_price > 0:
            self.position_manager.initialise_existing(existing_qty, existing_price)
            logger.info(
                "[%s] Registered existing holding: %.4f units @ $%.4f",
                symbol,
                existing_qty,
                existing_price,
            )

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> SignalResult:
        """
        Generate a composite BUY / SELL / HOLD signal from OHLC data.

        Args:
            highs:  Array of high prices (oldest → newest, ≥ 300 candles).
            lows:   Array of low prices (oldest → newest).
            closes: Array of closing prices (oldest → newest).

        Returns:
            SignalResult with full indicator metadata.
        """
        current_price = float(closes[-1])

        rsi = calculate_rsi(closes, self.rsi_period)
        macd = calculate_macd(
            closes,
            fast=DEFAULT_MACD_FAST,
            slow=DEFAULT_MACD_SLOW,
            signal=DEFAULT_MACD_SIGNAL,
        )
        fast_ma = calculate_moving_average(closes, self.fast_ma_period)
        slow_ma = calculate_moving_average(closes, self.slow_ma_period)
        sr = find_support_resistance(highs, lows, closes, lookback=DEFAULT_SR_LOOKBACK)
        atr = calculate_atr(highs, lows, closes, DEFAULT_ATR_PERIOD)

        # Evaluate individual conditions
        is_oversold = rsi < self.rsi_oversold
        is_overbought = rsi > self.rsi_overbought
        is_uptrend = fast_ma > slow_ma
        is_downtrend = fast_ma < slow_ma
        macd_positive = macd.histogram > 0
        macd_negative = macd.histogram < 0
        near_support = current_price <= sr.support * 1.02
        near_resistance = current_price >= sr.resistance * 0.98

        # --- BUY conditions ---
        buy_conditions = [is_oversold, is_uptrend, macd_positive]
        buy_score = sum(buy_conditions)

        # --- SELL conditions ---
        sell_conditions = [is_overbought, is_downtrend, macd_negative]
        sell_score = sum(sell_conditions)

        # Determine signal — require all 3 core conditions for a firm signal
        if buy_score == 3:
            signal = SIGNAL_BUY
            confidence = (buy_score + (1 if near_support else 0)) / 4.0
            parts = [
                f"RSI={rsi:.1f} (oversold<{self.rsi_oversold})",
                f"FastMA({self.fast_ma_period})=${fast_ma:.4f} > SlowMA({self.slow_ma_period})=${slow_ma:.4f}",
                f"MACD hist={macd.histogram:.6f}>0",
            ]
            if near_support:
                parts.append(f"Price near support ${sr.support:.4f}")
            reason = " | ".join(parts)
        elif sell_score == 3:
            signal = SIGNAL_SELL
            confidence = (sell_score + (1 if near_resistance else 0)) / 4.0
            parts = [
                f"RSI={rsi:.1f} (overbought>{self.rsi_overbought})",
                f"FastMA({self.fast_ma_period})=${fast_ma:.4f} < SlowMA({self.slow_ma_period})=${slow_ma:.4f}",
                f"MACD hist={macd.histogram:.6f}<0",
            ]
            if near_resistance:
                parts.append(f"Price near resistance ${sr.resistance:.4f}")
            reason = " | ".join(parts)
        else:
            signal = SIGNAL_HOLD
            confidence = 0.0
            buy_str = f"buy={buy_score}/3"
            sell_str = f"sell={sell_score}/3"
            reason = (
                f"Waiting for signal alignment ({buy_str}, {sell_str}) | "
                f"RSI={rsi:.1f} | MACD hist={macd.histogram:.6f} | "
                f"FastMA={fast_ma:.4f} vs SlowMA={slow_ma:.4f}"
            )

        result = SignalResult(
            signal=signal,
            price=current_price,
            rsi=rsi,
            macd=macd,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            sr=sr,
            atr=atr,
            confidence=confidence,
            reason=reason,
        )

        self._signal_history.append(result)
        logger.info(
            "[%s] %s @ $%.4f | RSI=%.1f | MACD hist=%.6f | confidence=%.0f%%",
            self.symbol,
            signal,
            current_price,
            rsi,
            macd.histogram,
            confidence * 100,
        )
        return result

    # ------------------------------------------------------------------
    # Trade execution helpers
    # ------------------------------------------------------------------

    def execute_buy(self, price: float) -> Optional[float]:
        """
        Open or add to a position using the allocated capital.

        Args:
            price: Execution price per unit.

        Returns:
            Quantity purchased, or None if capital is zero/invalid.
        """
        if price <= 0 or self.capital_per_trade <= 0:
            logger.warning("[%s] Cannot buy: invalid price or capital", self.symbol)
            return None

        qty = self.capital_per_trade / price
        self.position_manager.open_position(qty, price)
        logger.info(
            "[%s] BUY executed: %.4f units @ $%.4f (cost $%.2f)",
            self.symbol,
            qty,
            price,
            self.capital_per_trade,
        )
        return qty

    def execute_sell(self, price: float, quantity: Optional[float] = None) -> float:
        """
        Close the current open position (fully or partially).

        Args:
            price:    Execution price per unit.
            quantity: Units to sell; sells entire position if omitted.

        Returns:
            Realised PnL (USD).
        """
        pnl = self.position_manager.close_position(price, quantity)
        logger.info(
            "[%s] SELL executed @ $%.4f → PnL $%.2f",
            self.symbol,
            price,
            pnl,
        )
        return pnl

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def has_position(self) -> bool:
        return self.position_manager.has_open_position()

    def last_signal(self) -> Optional[SignalResult]:
        return self._signal_history[-1] if self._signal_history else None

    def signal_history(self) -> list[SignalResult]:
        return list(self._signal_history)

    def portfolio_summary(self, current_price: float) -> dict[str, Any]:
        """Return a display-ready summary dict for this bot."""
        pm_summary = self.position_manager.summary(current_price)
        last = self.last_signal()
        return {
            **pm_summary,
            "capital_per_trade": self.capital_per_trade,
            "last_signal": last.signal if last else "N/A",
            "last_rsi": round(last.rsi, 2) if last else None,
            "last_macd_hist": round(last.macd.histogram, 6) if last else None,
            "last_reason": last.reason if last else "No signal yet",
        }
