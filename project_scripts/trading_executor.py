"""
Original trade executor (kept for compatibility with existing scripts).

Provides thin wrappers around KrakenAPI order placement with paper mode
support. The v2 system (trading_bot_runner_v2.py) embeds equivalent
logic directly; this module is retained for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import Optional

from project_scripts.trading_bot_live import KrakenAPI

logger = logging.getLogger(__name__)


class TradeExecutor:
    """
    Wraps KrakenAPI order placement with paper/live mode and basic validation.

    Args:
        kraken:      Initialised KrakenAPI instance.
        paper_mode:  When True, log trades but do not submit real orders.
    """

    def __init__(self, kraken: KrakenAPI, paper_mode: bool = True) -> None:
        self.kraken = kraken
        self.paper_mode = paper_mode

    def buy(
        self,
        pair: str,
        volume: float,
        price: Optional[float] = None,
    ) -> Optional[str]:
        """
        Execute a buy order.

        Args:
            pair:   Kraken trading pair.
            volume: Units to buy.
            price:  Limit price (market order if None).

        Returns:
            Transaction ID, or None in paper mode or on error.
        """
        if volume <= 0:
            logger.warning("BUY skipped — invalid volume: %.4f", volume)
            return None

        if self.paper_mode:
            price_str = f"${price:.4f}" if price else "market"
            logger.info("[PAPER] BUY  %s: %.4f units @ %s", pair, volume, price_str)
            return None

        order_type = "limit" if price is not None else "market"
        result = self.kraken.add_order(pair, "buy", volume, price, order_type)
        if result:
            txid = result.get("txid", ["unknown"])[0]
            logger.info("BUY  %s: %.4f units → txid=%s", pair, volume, txid)
            return txid
        return None

    def sell(
        self,
        pair: str,
        volume: float,
        price: Optional[float] = None,
    ) -> Optional[str]:
        """
        Execute a sell order.

        Args:
            pair:   Kraken trading pair.
            volume: Units to sell.
            price:  Limit price (market order if None).

        Returns:
            Transaction ID, or None in paper mode or on error.
        """
        if volume <= 0:
            logger.warning("SELL skipped — invalid volume: %.4f", volume)
            return None

        if self.paper_mode:
            price_str = f"${price:.4f}" if price else "market"
            logger.info("[PAPER] SELL %s: %.4f units @ %s", pair, volume, price_str)
            return None

        order_type = "limit" if price is not None else "market"
        result = self.kraken.add_order(pair, "sell", volume, price, order_type)
        if result:
            txid = result.get("txid", ["unknown"])[0]
            logger.info("SELL %s: %.4f units → txid=%s", pair, volume, txid)
            return txid
        return None
