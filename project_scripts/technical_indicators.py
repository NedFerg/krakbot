"""
Technical Indicators for Mean Reversion Swing Trading Bot.

Provides RSI, MACD, ATR, and Support/Resistance calculations
used by EnhancedTradeBot for signal generation.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


class MACDResult(NamedTuple):
    macd_line: float
    signal_line: float
    histogram: float


class SupportResistance(NamedTuple):
    support: float
    resistance: float


def calculate_rsi(closes: np.ndarray, period: int = 14) -> float:
    """
    Calculate the Relative Strength Index (RSI) for the most recent candle.

    Args:
        closes: Array of closing prices (oldest → newest).
        period:  Lookback period (default 14).

    Returns:
        RSI value in [0, 100], or 50.0 if there is insufficient data.
    """
    if len(closes) < period + 1:
        logger.debug("Insufficient data for RSI calculation (%d candles)", len(closes))
        return 50.0

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple average over first `period` bars
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    # Wilder's smoothing for remaining bars
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0.0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDResult:
    """
    Calculate MACD line, signal line, and histogram for the most recent candle.

    Args:
        closes: Array of closing prices (oldest → newest).
        fast:   Fast EMA period (default 12).
        slow:   Slow EMA period (default 26).
        signal: Signal EMA period (default 9).

    Returns:
        MACDResult(macd_line, signal_line, histogram), all zeros if insufficient data.
    """
    min_len = slow + signal
    if len(closes) < min_len:
        logger.debug("Insufficient data for MACD calculation (%d candles)", len(closes))
        return MACDResult(0.0, 0.0, 0.0)

    fast_ema = _ema_series(closes, fast)
    slow_ema = _ema_series(closes, slow)

    # Align arrays so they share the same length as slow_ema
    if len(fast_ema) > len(slow_ema):
        fast_ema = fast_ema[len(fast_ema) - len(slow_ema):]

    macd_series = fast_ema - slow_ema
    signal_series = _ema_series(macd_series, signal)

    if len(signal_series) == 0:
        return MACDResult(0.0, 0.0, 0.0)

    macd_val = float(macd_series[-1])
    signal_val = float(signal_series[-1])
    hist_val = macd_val - signal_val

    return MACDResult(macd_val, signal_val, hist_val)


def calculate_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> float:
    """
    Calculate Average True Range (ATR) for the most recent candle.

    Args:
        highs:  Array of high prices (oldest → newest).
        lows:   Array of low prices (oldest → newest).
        closes: Array of closing prices (oldest → newest).
        period: Lookback period (default 14).

    Returns:
        ATR value, or 0.0 if there is insufficient data.
    """
    if len(closes) < period + 1:
        logger.debug("Insufficient data for ATR calculation (%d candles)", len(closes))
        return 0.0

    true_ranges: list[float] = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        true_ranges.append(max(hl, hc, lc))

    true_ranges_arr = np.array(true_ranges)

    # Seed
    atr = float(np.mean(true_ranges_arr[:period]))

    # Wilder's smoothing
    for tr in true_ranges_arr[period:]:
        atr = (atr * (period - 1) + tr) / period

    return atr


def find_support_resistance(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    lookback: int = 20,
) -> SupportResistance:
    """
    Identify the nearest support and resistance levels from recent price history.

    The support level is the lowest low and the resistance level is the highest
    high over the last ``lookback`` candles, giving a simple but effective proxy
    for key price levels.

    Args:
        highs:    Array of high prices (oldest → newest).
        lows:     Array of low prices (oldest → newest).
        closes:   Array of closing prices (oldest → newest).
        lookback: Number of recent candles to examine (default 20).

    Returns:
        SupportResistance(support, resistance).
        Falls back to the current close ±1 % if there is insufficient data.
    """
    current_price = float(closes[-1]) if len(closes) > 0 else 1.0

    if len(highs) < lookback or len(lows) < lookback:
        return SupportResistance(
            support=current_price * 0.99,
            resistance=current_price * 1.01,
        )

    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]

    resistance = float(np.max(recent_highs))
    support = float(np.min(recent_lows))

    return SupportResistance(support=support, resistance=resistance)


def calculate_moving_average(closes: np.ndarray, period: int) -> float:
    """
    Calculate the simple moving average (SMA) for the most recent candle.

    Args:
        closes: Array of closing prices (oldest → newest).
        period: SMA period.

    Returns:
        SMA value, or the last close price if there is insufficient data.
    """
    if len(closes) < period:
        return float(closes[-1]) if len(closes) > 0 else 0.0

    return float(np.mean(closes[-period:]))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ema_series(data: np.ndarray, period: int) -> np.ndarray:
    """Return the full EMA series for *data* using the given *period*."""
    if len(data) < period:
        return np.array([], dtype=float)

    k = 2.0 / (period + 1.0)
    ema = np.empty(len(data) - period + 1, dtype=float)
    ema[0] = float(np.mean(data[:period]))

    for i, price in enumerate(data[period:], start=1):
        ema[i] = price * k + ema[i - 1] * (1.0 - k)

    return ema
