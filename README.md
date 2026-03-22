# KrakBot — Automated Mean Reversion Swing Trading Bot

KrakBot is an automated cryptocurrency trading bot for Kraken that:

- Recognises **high-quality bullish and bearish setups** using RSI, MACD, and moving averages
- Enforces **risk limits** (position size, drawdown, notional) via an integrated Risk Manager
- Scales automatically from **$51 to $5,000+** by reading capital from an environment variable
- **Reinvests profits** automatically so the portfolio grows exponentially

---

## Architecture

```
project_scripts/
  trading_bot_runner_v2.py   # Main loop: fetches data, signals, executes orders
  trading_bot_live_v2.py     # EnhancedTradeBot: RSI+MACD+MA signal generation
  technical_indicators.py    # RSI, MACD, ATR, Support/Resistance calculations
  position_manager.py        # Per-symbol position tracking with PnL
  portfolio_manager.py       # Capital tracking, profit reinvestment, win-rate
  trading_bot_live.py        # KrakenAPI REST client (compatibility layer)
  trading_executor.py        # Paper/live order executor (compatibility layer)

project/risk/
  risk_manager.py            # Order approval, drawdown protection, kill-switch
```

---

## Signal Logic

| Signal | Conditions |
|--------|-----------|
| **BUY** | RSI < 30 AND FastMA > SlowMA AND MACD histogram > 0 |
| **SELL** | RSI > 70 AND FastMA < SlowMA AND MACD histogram < 0 |
| **HOLD** | Waiting for all conditions to align |

Both bullish (RSI oversold + uptrend) and bearish (RSI overbought + downtrend) setups
are traded. The bot sells existing holdings when overbought conditions align, then
re-deploys the capital on the next dip — rinse and repeat.

---

## Scaling from $51 to $1,000+ (No Code Changes)

The bot reads capital entirely from environment variables. To scale up, simply
change one line in your `.env` file:

```bash
# Start with $51 for testing
TOTAL_TRADING_CAPITAL=51.0

# Scale to $1,000 when the strategy is proven
TOTAL_TRADING_CAPITAL=1000.0

# Scale to $5,000+ for larger positions
TOTAL_TRADING_CAPITAL=5000.0
```

The bot divides `TOTAL_TRADING_CAPITAL` evenly across the 5 trading pairs and
uses `POSITION_SIZE_PCT` (default 95%) to keep a small buffer for fees.

| Capital | Pairs | Capital per pair |
|---------|-------|-----------------|
| $51     | 5     | ~$9.70          |
| $500    | 5     | ~$95.00         |
| $1,000  | 5     | ~$190.00        |
| $5,000  | 5     | ~$950.00        |

Profits are automatically reinvested, so `capital_per_pair` grows after every
winning trade — creating exponential compounding without any code changes.

---

## Setup

### 1. Install dependencies

```bash
pip install numpy requests
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Kraken API key and desired capital
```

### 3. Run in paper mode (default — no real orders)

```bash
python -m project_scripts.trading_bot_runner_v2
```

### 4. Switch to live trading

```bash
# In .env:
ENABLE_LIVE_TRADING=true
```

---

## Risk Manager

Every order intent passes through the Risk Manager before execution:

| Check | Config variable | Default |
|-------|----------------|---------|
| Max position per symbol | `RISK_MAX_POSITION_PCT` | 20% of capital |
| Max single order notional | `RISK_MAX_NOTIONAL_PER_ORDER` | total capital |
| Per-symbol drawdown halt | `RISK_MAX_DRAWDOWN_PCT` | 15% loss |
| Global portfolio kill-switch | `RISK_GLOBAL_MAX_DRAWDOWN_PCT` | 25% drawdown |

If an order is rejected, it is logged with the rejection reason and skipped —
the bot never sends an unapproved order to the exchange.

---

## Trading Pairs

- XRP/USD (`XXRPZUSD`)
- XLM/USD (`XXLMZUSD`)
- SOL/USD (`SOLUSD`)
- AVAX/USD (`AVAXUSD`)
- HBAR/USD (`HBARUSD`)

---

## Profit Reinvestment (Exponential Growth)

After every SELL trade, the realised PnL is captured by `PortfolioManager`:

```
Trade 1: Start $51 → win $3.50 → capital becomes $54.50
Trade 2: Capital per pair recalculates to $54.50 / 5 = $10.90
Trade 3: Larger positions → larger profits → faster compounding
```

The portfolio grows exponentially as profits are compounded back into each
successive trade. No manual intervention required.
