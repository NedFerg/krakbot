"""
Original Kraken API client (kept for compatibility with existing scripts).

The v2 trading system (trading_bot_live_v2.py / trading_bot_runner_v2.py)
calls the Kraken REST API directly; this module provides a thin class
wrapper that mirrors the original interface used on Replit.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import urllib.parse
from base64 import b64decode, b64encode
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

KRAKEN_API_URL = "https://api.kraken.com"


class KrakenAPI:
    """Thin wrapper around the Kraken REST API."""

    def __init__(self, api_key: str = "", api_secret: str = "") -> None:
        self.api_key = api_key
        self.api_secret = api_secret

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def get_ticker(self, pair: str) -> Optional[dict[str, Any]]:
        """Return ticker data for a trading pair."""
        try:
            url = f"{KRAKEN_API_URL}/0/public/Ticker"
            resp = requests.get(url, params={"pair": pair}, timeout=10)
            resp.raise_for_status()
            body = resp.json()
            if body.get("error"):
                logger.error("Kraken ticker error for %s: %s", pair, body["error"])
                return None
            return body.get("result")
        except Exception as exc:
            logger.error("Ticker fetch failed for %s: %s", pair, exc)
            return None

    def get_ohlc(
        self,
        pair: str,
        interval: int = 60,
        since: Optional[int] = None,
    ) -> Optional[list]:
        """
        Fetch OHLC candles for a trading pair.

        Args:
            pair:     Kraken trading pair string.
            interval: Candle interval in minutes (default 60 = hourly).
            since:    Unix timestamp; fetch candles since this time.

        Returns:
            List of candle lists [time, open, high, low, close, vwap, volume, count],
            or None on failure.
        """
        try:
            params: dict[str, Any] = {"pair": pair, "interval": interval}
            if since:
                params["since"] = since
            url = f"{KRAKEN_API_URL}/0/public/OHLC"
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            body = resp.json()
            if body.get("error"):
                logger.error("Kraken OHLC error for %s: %s", pair, body["error"])
                return None
            result = body.get("result", {})
            candle_key = next((k for k in result if k != "last"), None)
            return result.get(candle_key) if candle_key else None
        except Exception as exc:
            logger.error("OHLC fetch failed for %s: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Private endpoints
    # ------------------------------------------------------------------

    def get_balance(self) -> Optional[dict[str, str]]:
        """Return account balances (requires API key/secret)."""
        return self._private_request("/0/private/Balance")

    def add_order(
        self,
        pair: str,
        side: str,
        volume: float,
        price: Optional[float] = None,
        order_type: str = "market",
    ) -> Optional[dict[str, Any]]:
        """
        Place an order on Kraken.

        Args:
            pair:       Trading pair (e.g. "XXRPZUSD").
            side:       "buy" or "sell".
            volume:     Order size in base currency units.
            price:      Limit price (ignored for market orders).
            order_type: "market" or "limit".

        Returns:
            API result dict, or None on failure.
        """
        data: dict[str, Any] = {
            "pair": pair,
            "type": side,
            "ordertype": order_type,
            "volume": str(round(volume, 8)),
        }
        if order_type == "limit" and price is not None:
            data["price"] = str(round(price, 6))

        return self._private_request("/0/private/AddOrder", data)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _private_request(
        self, path: str, data: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        if not self.api_key or not self.api_secret:
            logger.error("API key/secret not configured for private request")
            return None

        if data is None:
            data = {}
        nonce = str(int(time.time() * 1000))
        data["nonce"] = nonce
        post_data = urllib.parse.urlencode(data)
        encoded = (nonce + post_data).encode()
        message = path.encode() + hashlib.sha256(encoded).digest()
        secret = b64decode(self.api_secret)
        signature = b64encode(hmac.new(secret, message, hashlib.sha512).digest()).decode()

        headers = {"API-Key": self.api_key, "API-Sign": signature}
        try:
            resp = requests.post(
                KRAKEN_API_URL + path,
                data=data,
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            body = resp.json()
            if body.get("error"):
                logger.error("Kraken private API error at %s: %s", path, body["error"])
                return None
            return body.get("result")
        except Exception as exc:
            logger.error("Private request failed at %s: %s", path, exc)
            return None
