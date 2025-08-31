# 🚀 Advanced Algorithmic Trading Strategies Collection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![MetaTrader5](https://img.shields.io/badge/MetaTrader5-Live%20Trading-green.svg)](https://www.metatrader5.com)
[![Trading](https://img.shields.io/badge/Trading-XAUUSD-gold.svg)](https://github.com)
[![Strategies](https://img.shields.io/badge/Strategies-11%20Live-red.svg)](https://github.com)

> **A comprehensive collection of 11 sophisticated algorithmic trading strategies designed for live trading on MetaTrader5, primarily focused on XAUUSD (Gold) trading with advanced technical analysis and risk management.**

---

## 📊 Strategy Overview

| Strategy | Type | Accuracy Target | Key Features | Risk Level |
|----------|------|----------------|--------------|------------|
| [Strategy 1](#-strategy-1-smart-market-structure) | Smart Market Structure | 75%+ | Order Blocks, FVG, Multi-timeframe | Medium |
| [Strategy 2](#-strategy-2-fair-value-gap-momentum) | FVG Momentum | 70%+ | Fair Value Gaps, RSI, EMA Trend | Medium |
| [Strategy 3](#-strategy-3-liquidity-zone-hunter) | Liquidity Zones | 80%+ | Support/Resistance, ADX Filter | Low-Medium |
| [Strategy 4](#-strategy-4-advanced-pattern-recognition) | Pattern Recognition | 85%+ | Chart Patterns, Multi-indicator | Medium-High |
| [Strategy 5](#-strategy-5-simple-ema-breakout) | EMA Breakout | 65%+ | Simple EMA crossover | Low |
| [Strategy 6](#-strategy-6-high-frequency-ai-trading) | AI High-Frequency | 90%+ | Machine Learning, Bayesian Models | High |
| [Strategy 7](#-strategy-7-dual-symbol-hedge) | Hedge Trading | 70%+ | Multi-symbol hedging | Medium |
| [Strategy 8](#-strategy-8-pending-order-breakout) | Pending Orders | 75%+ | Breakout capture | Medium |
| [Strategy 9](#-strategy-9-scalping-reversal) | Scalping | 80%+ | Quick reversals, 2.6 pip targets | High |
| [Strategy 10](#-strategy-10-reverse-scalping) | Reverse Scalping | 80%+ | Opposite direction scalping | High |
| [Strategy 11](#-strategy-11-cascade-trading) | Cascade System | 75%+ | Progressive lot sizing | Medium-High |

---

## 🎯 Strategy Details

### 🔥 Strategy 1: Smart Market Structure
**Advanced Order Block & Fair Value Gap Strategy**

- **Core Logic**: Combines Order Blocks detection with Fair Value Gaps for high-probability entries
- **Key Indicators**: EMA50, EMA200, ATR14, RSI
- **Entry Conditions**: 
  - Bullish: Price retraces to order block low + trend confirmation
  - Bearish: Price retraces to order block high + trend confirmation
- **Risk Management**: Dynamic ATR-based stop loss (1.2x ATR), Target: 3.4x ATR
- **Unique Features**: Multi-timeframe trend analysis, volatility filtering
- **Trading Hours**: Indian market hours with time-based filtering

### 📈 Strategy 2: Fair Value Gap Momentum
**FVG-Based Momentum Trading with RSI Confirmation**

- **Core Logic**: Trades Fair Value Gap breakouts with momentum confirmation
- **Key Indicators**: EMA20, EMA50, RSI, ATR14
- **Entry Conditions**: 
  - FVG breakout + RSI > 55 for bullish
  - EMA20 > EMA50 trend confirmation
- **Risk Management**: 2.2x ATR stop loss, 6.2x ATR target
- **Unique Features**: Volatility market filtering, precise FVG detection
- **Success Rate**: Targets 70%+ win rate with favorable R:R

### 🎯 Strategy 3: Liquidity Zone Hunter
**Support/Resistance Liquidity Zone Strategy**

- **Core Logic**: Identifies and trades liquidity zones with multi-timeframe confirmation
- **Key Indicators**: ADX14, ATR14, Support/Resistance levels
- **Entry Conditions**: Price rejection from liquidity zones with trend alignment
- **Risk Management**: Zone-adjusted stop loss (2.3-2.5x ATR), Target: 5.1-5.4x ATR
- **Unique Features**: ADX < 55 filter to avoid choppy markets
- **Special Feature**: Zone bonus system for tighter entries

### 🔬 Strategy 4: Advanced Pattern Recognition
**Multi-Pattern Technical Analysis System**

- **Core Logic**: Combines multiple chart patterns with advanced technical indicators
- **Key Features**: 
  - Head & Shoulders, Double Tops/Bottoms
  - Triangle patterns, Wedges, Channels
  - Order blocks, BOS/CHOCH detection
  - Liquidity sweeps identification
- **Entry Conditions**: Pattern confirmation + multi-indicator alignment
- **Risk Management**: Pattern-specific stop loss and target levels
- **Unique Features**: Comprehensive pattern library integration

### ⚡ Strategy 5: Simple EMA Breakout
**Straightforward EMA-Based Trading**

- **Core Logic**: Simple price action above/below EMA20 for entries
- **Key Indicators**: EMA20
- **Entry Conditions**: 
  - Buy: Price > EMA20
  - Sell: Price < EMA20
- **Risk Management**: Fixed 1000 point stop loss, 1200 point target
- **Trading Hours**: 12:30 PM and 17:30-20:30 (Indian time)
- **Unique Features**: Profit threshold exit at ₹1000

### 🤖 Strategy 6: High-Frequency AI Trading
**Advanced Machine Learning Strategy (90%+ Accuracy Target)**

- **Core Logic**: AI-powered high-frequency trading with multiple prediction models
- **Advanced Features**:
  - Bayesian probability models
  - Kalman filtering for price prediction
  - Hidden Markov Models (HMM)
  - Kelly criterion position sizing
  - Market regime detection
- **Entry Conditions**: Multi-model consensus with 90%+ confidence
- **Risk Management**: Adaptive position sizing, rapid profit taking
- **Unique Features**: 1-second timeframe analysis, hedge trading system

### 🔄 Strategy 7: Dual Symbol Hedge
**Multi-Symbol Hedging Strategy**

- **Core Logic**: Hedges positions across multiple currency pairs
- **Key Features**: 
  - Dual symbol trading (buy one, sell another)
  - Profit target: $0.50+ per trade
  - Trailing stop: $0.01 from peak
- **Risk Management**: Reversal detection and counter-hedging
- **Unique Features**: Cross-pair correlation trading
- **Target Symbols**: Multiple forex pairs

### 📋 Strategy 8: Pending Order Breakout
**Breakout Capture with Pending Orders**

- **Core Logic**: Places pending orders 150 points above/below current price
- **Entry Method**: 
  - Buy Stop: 150 points above current price
  - Sell Stop: 150 points below current price
- **Risk Management**: 100 point stop loss, 200 point target
- **Unique Features**: Automatic pending order management
- **Cycle**: Recreates orders after each completion

### ⚡ Strategy 9: Scalping Reversal
**High-Speed Scalping with Reversal Logic**

- **Core Logic**: Quick scalping with reversal hedging
- **Entry System**:
  - Initial Buy: 0.01 lot, 2.6 pip target
  - Reversal Sell: 0.02 lot when price drops 1 pip
  - Progressive lot sizing: 1.33x multiplier
- **Risk Management**: Tight 2.6 pip targets
- **Unique Features**: Automatic reversal detection, cycle-based trading

### 🔄 Strategy 10: Reverse Scalping
**Opposite Direction Scalping System**

- **Core Logic**: Starts with Sell orders, reverses to Buy
- **Entry System**:
  - Initial Sell: 0.05 lot, 4.6 pip stop loss
  - Reversal Buy: When conditions trigger
  - Progressive sizing: 1.33x multiplier
- **Risk Management**: 4.6 pip stop loss system
- **Unique Features**: Reverse hedging approach

### 🎢 Strategy 11: Cascade Trading
**Progressive Lot Sizing System**

- **Core Logic**: Cascade system with progressive position building
- **Entry System**:
  - Initial Buy: 0.1 lot
  - Reverse Sell: 0.16 lot when price drops 1 pip
  - Target: 4.5 pip profit on each leg
- **Risk Management**: Cycle-based profit targets
- **Unique Features**: Dynamic cascade with termination conditions

---

## 🛠️ Technical Requirements

### Prerequisites
- **Python 3.8+**
- **MetaTrader5** platform
- **Required Libraries**: 
  ```bash
  pip install MetaTrader5 pandas numpy talib scikit-learn
  ```

### Key Dependencies
- `MetaTrader5` - Live trading interface
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `talib` - Technical analysis indicators
- Custom modules: `FVG`, `OrderBlock`, `SupportResistance`, `Chart_pattern`

---

## 🚀 Getting Started

1. **Setup MetaTrader5**
   ```python
   import MetaTrader5 as mt5
   if not mt5.initialize():
       print("MetaTrader5 initialization failed")
   ```

2. **Run a Strategy**
   ```python
   # Example: Run Strategy 1
   python liveStrategy1.py
   ```

3. **Monitor Performance**
   - Each strategy includes built-in logging
   - Real-time trade monitoring
   - Performance metrics tracking

---

## ⚠️ Risk Disclaimer

**IMPORTANT**: These strategies are for educational purposes. Live trading involves significant financial risk. Always:
- Test strategies on demo accounts first
- Use proper risk management
- Never risk more than you can afford to lose
- Consider market conditions and volatility

---

## 📈 Performance Notes

- **Backtesting**: All strategies include historical performance analysis
- **Live Performance**: Results may vary based on market conditions
- **Risk Management**: Each strategy includes built-in risk controls
- **Optimization**: Strategies are optimized for XAUUSD but can be adapted

---

## 🤝 Contributing

Feel free to contribute improvements, bug fixes, or new strategies. Please ensure all contributions include proper testing and documentation.

---

## 📞 Support

For questions, issues, or strategy discussions, please create an issue in this repository.

---

*Happy Trading! 📊💰*
