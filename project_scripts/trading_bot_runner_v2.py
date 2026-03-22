"""
Enhanced Bot Runner v2 — Mean Reversion Swing Trader.

Orchestrates the enhanced trading bots:
  1. Fetches current Kraken holdings to initialise existing positions.
  2. Downloads 300-candle hourly OHLC history for each pair.
  3. Calculates RSI, MACD, MA, and Support/Resistance every 300 s.
  4. Executes BUY/SELL orders (bullish and bearish setups) in paper or live mode.
  5. Enforces risk limits via RiskManager before every order.
  6. Reinvests realised profits via PortfolioManager for exponential growth.
  7. Logs a full portfolio status table every iteration.

Usage
-----
    python -m project_scripts.trading_bot_runner_v2

Environment variables
---------------------
    KRAKEN_API_KEY              Kraken REST API key (required for live/balance queries).
    KRAKEN_API_SECRET           Kraken REST API secret.
    ENABLE_LIVE_TRADING         Set to "true" to enable real order placement (default: paper).
    TOTAL_TRADING_CAPITAL       Total USD capital to deploy (default: 51.0).
    POSITION_SIZE_PCT           Fraction of capital deployed per trade (default: 0.95).
    RISK_MAX_POSITION_PCT       Max capital fraction per symbol (default: 0.20).
    RISK_MAX_NOTIONAL_PER_ORDER Max USD per single order (default: TOTAL_TRADING_CAPITAL).
    RISK_MAX_DRAWDOWN_PCT       Per-symbol drawdown halt threshold (default: 0.15).
    RISK_GLOBAL_MAX_DRAWDOWN_PCT Portfolio drawdown kill-switch (default: 0.25).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
import urllib.parse
from base64 import b64decode, b64encode
from typing import Any, Optional

import numpy as np
import requests

from project_scripts.trading_bot_live_v2 import EnhancedTradeBot, SIGNAL_BUY, SIGNAL_SELL
from project_scripts.portfolio_manager import PortfolioManager
from project.risk.risk_manager import RiskManager, OrderIntent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all values come from environment variables
# ---------------------------------------------------------------------------

KRAKEN_API_URL = "https://api.kraken.com"
KRAKEN_API_KEY = os.environ.get("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.environ.get("KRAKEN_API_SECRET", "")
ENABLE_LIVE_TRADING = os.environ.get("ENABLE_LIVE_TRADING", "false").lower() == "true"

UPDATE_INTERVAL_SECONDS = int(os.environ.get("UPDATE_INTERVAL_SECONDS", "300"))
OHLC_CANDLES = 300
OHLC_INTERVAL_MINUTES = 60  # hourly

# Total capital: read from env so the same code scales from $51 to $5,000+
TOTAL_TRADING_CAPITAL = float(os.environ.get("TOTAL_TRADING_CAPITAL", "51.0"))

# Trading pairs: (short symbol, Kraken pair name)
TRADING_PAIRS: list[tuple[str, str]] = [
    ("XRP", "XXRPZUSD"),
    ("XLM", "XXLMZUSD"),
    ("SOL", "SOLUSD"),
    ("AVAX", "AVAXUSD"),
    ("HBAR", "HBARUSD"),
]

# Symbols to monitor as existing holdings (queried from Kraken balance)
EXISTING_HOLDING_SYMBOLS = {"XRP", "XLM", "SOL", "AVAX", "HBAR"}

# Kraken asset name → short symbol mapping for balance parsing
KRAKEN_ASSET_MAP: dict[str, str] = {
    "XXRP": "XRP",
    "XRP": "XRP",
    "XXLM": "XLM",
    "XLM": "XLM",
    "SOL": "SOL",
    "XSOL": "SOL",
    "AVAX": "AVAX",
    "HBAR": "HBAR",
    "ZUSD": "USD",
    "USD": "USD",
}

# Short symbol → Kraken pair name for price lookups
PRICE_PAIR_MAP: dict[str, str] = {symbol: pair for symbol, pair in TRADING_PAIRS}

# ---------------------------------------------------------------------------
# Kraken REST helpers
# ---------------------------------------------------------------------------


def _kraken_public(path: str, data: Optional[dict] = None) -> Any:
    """Call a Kraken public REST endpoint and return the result payload."""
    url = KRAKEN_API_URL + path
    resp = requests.get(url, params=data, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    if body.get("error"):
        raise RuntimeError(f"Kraken API error: {body['error']}")
    return body["result"]


def _kraken_private(path: str, data: Optional[dict] = None) -> Any:
    """Call a Kraken private REST endpoint (requires API key/secret)."""
    if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
        raise RuntimeError(
            "KRAKEN_API_KEY and KRAKEN_API_SECRET must be set for private API calls"
        )

    if data is None:
        data = {}

    nonce = str(int(time.time() * 1000))
    data["nonce"] = nonce

    post_data = urllib.parse.urlencode(data)
    encoded = (nonce + post_data).encode()
    message = path.encode() + hashlib.sha256(encoded).digest()
    secret = b64decode(KRAKEN_API_SECRET)
    signature = b64encode(hmac.new(secret, message, hashlib.sha512).digest()).decode()

    headers = {
        "API-Key": KRAKEN_API_KEY,
        "API-Sign": signature,
    }
    url = KRAKEN_API_URL + path
    resp = requests.post(url, data=data, headers=headers, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    if body.get("error"):
        raise RuntimeError(f"Kraken API error: {body['error']}")
    return body["result"]


def fetch_account_balances() -> dict[str, float]:
    """
    Fetch all non-zero account balances from Kraken.

    Returns a mapping of short symbol → quantity, e.g. {"XRP": 47.27, "USD": 51.01}.
    Falls back to an empty dict if credentials are not configured.
    """
    if not KRAKEN_API_KEY:
        logger.warning("No API key configured — skipping balance fetch")
        return {}

    try:
        raw = _kraken_private("/0/private/Balance")
        balances: dict[str, float] = {}
        for asset, qty_str in raw.items():
            qty = float(qty_str)
            if qty == 0.0:
                continue
            symbol = KRAKEN_ASSET_MAP.get(asset, asset)
            balances[symbol] = balances.get(symbol, 0.0) + qty
        return balances
    except Exception as exc:
        logger.error("Failed to fetch balances: %s", exc)
        return {}


def fetch_current_price(pair: str) -> Optional[float]:
    """Return the last traded price for a Kraken pair, or None on failure."""
    try:
        result = _kraken_public("/0/public/Ticker", {"pair": pair})
        pair_data = next(iter(result.values()))
        return float(pair_data["c"][0])
    except Exception as exc:
        logger.error("Failed to fetch price for %s: %s", pair, exc)
        return None


def fetch_ohlc(pair: str, interval: int = OHLC_INTERVAL_MINUTES) -> Optional[dict]:
    """
    Fetch OHLC candle data for a Kraken pair.

    Returns a dict with numpy arrays: "highs", "lows", "closes", "volumes".
    Returns None on failure.
    """
    try:
        result = _kraken_public(
            "/0/public/OHLC", {"pair": pair, "interval": interval}
        )
        # Result contains the pair key and a 'last' key
        candle_key = [k for k in result if k != "last"][0]
        candles = result[candle_key]
        # Each candle: [time, open, high, low, close, vwap, volume, count]
        candles = candles[-OHLC_CANDLES:]
        highs = np.array([float(c[2]) for c in candles])
        lows = np.array([float(c[3]) for c in candles])
        closes = np.array([float(c[4]) for c in candles])
        volumes = np.array([float(c[6]) for c in candles])
        logger.info("✓ Fetched %d hourly candles for %s", len(closes), pair)
        return {"highs": highs, "lows": lows, "closes": closes, "volumes": volumes}
    except Exception as exc:
        logger.error("Failed to fetch OHLC for %s: %s", pair, exc)
        return None


def place_order(
    pair: str,
    side: str,
    volume: float,
    price: Optional[float] = None,
) -> Optional[str]:
    """
    Place a market or limit order on Kraken.

    Args:
        pair:   Kraken trading pair (e.g. "XXRPZUSD").
        side:   "buy" or "sell".
        volume: Order volume in base currency units.
        price:  Limit price (optional; omit for market order).

    Returns:
        Transaction ID string, or None on failure / paper mode.
    """
    if not ENABLE_LIVE_TRADING:
        logger.info(
            "[PAPER] %s %.4f units on %s", side.upper(), volume, pair
        )
        return None

    order_data: dict[str, Any] = {
        "pair": pair,
        "type": side,
        "ordertype": "market" if price is None else "limit",
        "volume": str(round(volume, 8)),
    }
    if price is not None:
        order_data["price"] = str(round(price, 6))

    try:
        result = _kraken_private("/0/private/AddOrder", order_data)
        txid = result.get("txid", ["unknown"])[0]
        logger.info("Order placed: %s %s → txid=%s", side.upper(), pair, txid)
        return txid
    except Exception as exc:
        logger.error("Order placement failed for %s: %s", pair, exc)
        return None


# ---------------------------------------------------------------------------
# Portfolio display helpers
# ---------------------------------------------------------------------------

SEP = "=" * 80


def _log_portfolio(
    bots: list[EnhancedTradeBot],
    balances: dict[str, float],
    prices: dict[str, float],
    iteration: int,
    portfolio_mgr: PortfolioManager,
    risk_mgr: RiskManager,
) -> None:
    usd_cash = balances.get("USD", portfolio_mgr.available_capital)

    logger.info("")
    logger.info(SEP)
    logger.info("     [TRADING BOT ALLOCATION]")
    logger.info("     Trading Pairs: %d", len(bots))
    logger.info("     Capital per Pair: $%.2f", portfolio_mgr.capital_per_pair(len(bots)))
    logger.info("     Available Capital: $%.2f", portfolio_mgr.available_capital)
    logger.info("     Portfolio Growth: %+.2f%%", portfolio_mgr.growth_pct)
    logger.info(SEP)
    logger.info("")
    logger.info(SEP)
    logger.info(
        "     >> ITERATION %d - %s UTC",
        iteration,
        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    )
    logger.info(SEP)

    logger.info("")
    logger.info(SEP)
    logger.info("     [PORTFOLIO STATUS]")
    logger.info(SEP)
    logger.info("     Available Cash (USD): $%10.2f", usd_cash)
    logger.info("     Total Realised PnL:   $%10.2f", portfolio_mgr.total_realized_pnl)
    logger.info(
        "     Win Rate:             %10.1f%% (%d trades)",
        portfolio_mgr.win_rate() * 100,
        portfolio_mgr.trade_count(),
    )
    logger.info("")

    # Risk manager status
    risk_status = risk_mgr.status()
    if risk_status["global_halt"]:
        logger.warning("     ⚠️  GLOBAL TRADING HALT ACTIVE")
    if risk_status["halted_symbols"]:
        logger.warning("     ⚠️  Halted symbols: %s", risk_status["halted_symbols"])

    # Existing holdings section
    holdings = {
        sym: qty
        for sym, qty in balances.items()
        if sym in EXISTING_HOLDING_SYMBOLS and qty > 0
    }
    if holdings:
        logger.info("     [EXISTING HOLDINGS] (managed by bot — will sell on overbought signal)")
        total_holding_value = 0.0
        for sym, qty in sorted(holdings.items()):
            pair = PRICE_PAIR_MAP.get(sym, "")
            price = prices.get(pair, 0.0)
            value = qty * price
            total_holding_value += value
            logger.info(
                "       %-6s: %14.8f @ $%10.4f = $%10.2f",
                sym, qty, price, value,
            )
        logger.info(
            "       %-6s: %14s   %10s   $%10.2f",
            "TOTAL", "", "", total_holding_value,
        )
        logger.info("")

    # Active bots section
    logger.info("     [ACTIVE TRADING BOTS] (capital_per_pair=$%.2f)", portfolio_mgr.capital_per_pair(len(bots)))
    for bot in bots:
        pair = PRICE_PAIR_MAP.get(bot.symbol, "")
        price = prices.get(pair, 0.0)
        summary = bot.portfolio_summary(price)
        realized = summary["realized_pnl"]
        unrealized = summary["unrealized_pnl"]
        halted = "HALTED" if risk_mgr.is_symbol_halted(bot.symbol) else ""
        logger.info(
            "       %-6s: Trades=%2d | Realized=$%8.2f | Unrealized=$%8.2f | Signal=%-4s %s",
            bot.symbol,
            summary["trade_count"],
            realized,
            unrealized,
            summary["last_signal"],
            halted,
        )
        if summary["last_reason"] != "No signal yet":
            logger.info("              Reason: %s", summary["last_reason"])

    logger.info(SEP)
    logger.info("     Next update in %ds...", UPDATE_INTERVAL_SECONDS)
    logger.info("")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run() -> None:
    """Main entry point — runs the enhanced trading bot loop."""
    mode = "LIVE" if ENABLE_LIVE_TRADING else "PAPER"
    logger.info("Starting Enhanced Trading Bot v2 (%s mode)", mode)
    logger.info("Total capital configured: $%.2f", TOTAL_TRADING_CAPITAL)

    # ------------------------------------------------------------------
    # Initialise managers
    # ------------------------------------------------------------------
    portfolio_mgr = PortfolioManager.from_env()
    risk_mgr = RiskManager.from_env()

    # ------------------------------------------------------------------
    # Fetch existing balances and initialise bots
    # ------------------------------------------------------------------
    balances = fetch_account_balances()
    if not balances:
        logger.warning("No balances fetched — running with default cash allocation")
        balances = {"USD": TOTAL_TRADING_CAPITAL}
    else:
        # Sync portfolio manager with live USD balance
        usd_balance = balances.get("USD", TOTAL_TRADING_CAPITAL)
        portfolio_mgr.set_available_capital(usd_balance)
        risk_mgr.update_total_capital(usd_balance)

    capital_per_pair = portfolio_mgr.capital_per_pair(len(TRADING_PAIRS))

    # Fetch initial prices to seed existing position entry prices
    initial_prices: dict[str, float] = {}
    for _symbol, pair in TRADING_PAIRS:
        price = fetch_current_price(pair)
        if price:
            initial_prices[pair] = price
        time.sleep(0.3)  # rate limit courtesy pause

    bots: list[EnhancedTradeBot] = []
    for symbol, pair in TRADING_PAIRS:
        existing_qty = balances.get(symbol, 0.0)
        existing_price = initial_prices.get(pair, 0.0)
        bot = EnhancedTradeBot(
            symbol=symbol,
            kraken_pair=pair,
            capital_per_trade=capital_per_pair,
            existing_qty=existing_qty,
            existing_price=existing_price,
        )
        bots.append(bot)
        logger.info(
            "Initialised bot for %s (pair=%s, existing=%.4f, price=$%.4f, capital=$%.2f)",
            symbol,
            pair,
            existing_qty,
            existing_price,
            capital_per_pair,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    iteration = 0
    while True:
        iteration += 1

        # Check global halt before doing any work
        if risk_mgr.is_global_halt():
            logger.critical("GLOBAL TRADING HALT — no orders will be placed this cycle")

        # 2. Fetch OHLC data for all pairs
        logger.info("Fetching hourly OHLC data for %d pairs...", len(TRADING_PAIRS))
        ohlc_data: dict[str, dict] = {}
        current_prices: dict[str, float] = {}

        for _symbol, pair in TRADING_PAIRS:
            candles = fetch_ohlc(pair)
            if candles is not None:
                ohlc_data[pair] = candles
                current_prices[pair] = float(candles["closes"][-1])
            time.sleep(0.5)  # polite rate limiting

        fetched = len(ohlc_data)
        logger.info("Successfully fetched %d/%d pairs", fetched, len(TRADING_PAIRS))

        # Compute current portfolio value for risk monitoring
        portfolio_value = portfolio_mgr.available_capital + sum(
            bot.position_manager.unrealized_pnl(current_prices.get(PRICE_PAIR_MAP.get(bot.symbol, ""), 0.0))
            for bot in bots
        )

        # Refresh capital per pair from portfolio manager (captures reinvested profits)
        capital_per_pair = portfolio_mgr.capital_per_pair(len(TRADING_PAIRS))
        for bot in bots:
            bot.capital_per_trade = capital_per_pair

        # 3. Generate signals and execute trades
        for bot in bots:
            pair = PRICE_PAIR_MAP[bot.symbol]
            candles = ohlc_data.get(pair)
            if candles is None:
                logger.warning("[%s] Skipping — no OHLC data", bot.symbol)
                continue

            price = current_prices.get(pair, 0.0)
            result = bot.generate_signal(
                candles["highs"],
                candles["lows"],
                candles["closes"],
            )

            status = "WAITING"

            if result.signal == SIGNAL_BUY and not risk_mgr.is_global_halt():
                qty = bot.capital_per_trade / price if price > 0 else 0.0
                intent = OrderIntent(
                    symbol=bot.symbol,
                    side="buy",
                    quantity=qty,
                    price=price,
                )
                current_exposure = (
                    bot.position_manager.total_quantity() * price
                )
                approved, reason = risk_mgr.approve_order(
                    intent,
                    current_symbol_exposure=current_exposure,
                    current_portfolio_value=portfolio_value,
                )
                if approved:
                    place_order(pair, "buy", qty)
                    bot.execute_buy(price)
                    status = "BOUGHT"
                else:
                    logger.warning("[%s] BUY rejected by risk manager: %s", bot.symbol, reason)
                    status = "RISK_REJECTED"

            elif result.signal == SIGNAL_SELL and bot.has_position():
                if risk_mgr.is_global_halt():
                    logger.warning("[%s] SELL skipped — global halt active", bot.symbol)
                    status = "HALT"
                else:
                    qty = bot.position_manager.total_quantity()
                    intent = OrderIntent(
                        symbol=bot.symbol,
                        side="sell",
                        quantity=qty,
                        price=price,
                    )
                    approved, reason = risk_mgr.approve_order(
                        intent,
                        current_portfolio_value=portfolio_value,
                    )
                    if approved:
                        entry_price = bot.position_manager.average_entry_price()
                        entry_notional = qty * entry_price
                        place_order(pair, "sell", qty)
                        pnl = bot.execute_sell(price)

                        # Reinvest profits via portfolio manager
                        portfolio_mgr.record_trade_profit(
                            symbol=bot.symbol,
                            realized_pnl=pnl,
                            entry_price=entry_price,
                            exit_price=price,
                            quantity=qty,
                        )
                        # Notify risk manager of the trade outcome
                        risk_mgr.record_symbol_pnl(
                            bot.symbol, pnl, entry_notional
                        )
                        # Update risk manager's capital view
                        risk_mgr.update_total_capital(portfolio_mgr.available_capital)
                        status = "SOLD"
                    else:
                        logger.warning("[%s] SELL rejected by risk manager: %s", bot.symbol, reason)
                        status = "RISK_REJECTED"

            logger.info(
                "[%s] %-4s  | Price: $%10.4f | Status: %s",
                bot.symbol,
                result.signal,
                price,
                status,
            )

        # 4. Display portfolio status
        _log_portfolio(bots, balances, current_prices, iteration, portfolio_mgr, risk_mgr)

        # 5. Wait for next cycle
        time.sleep(UPDATE_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()