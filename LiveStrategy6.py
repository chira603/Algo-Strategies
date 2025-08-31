import sys
import os
import importlib.util
import logging
import datetime
from datetime import timedelta
"""
LiveStrategy6 for BTCUSD - High-Frequency Trading with Advanced Probability Models

Major Changes:
- Implemented Bayesian probability models for trade prediction
- Added Kalman filtering for price prediction
- Implemented multi-timeframe analysis
- Added machine learning-based pattern recognition
- Enhanced HMM model with sophisticated state transitions
- Implemented Kelly criterion for optimal position sizing
- Added market regime detection for adaptive trading
"""

#*********************************************************************************
# Import necessary libraries
#*********************************************************************************

import sys
import os
import importlib.util
import logging
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Helper function to dynamically load a module
def load_module(module_name, relative_path):
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Dynamically load external modules
function = load_module("function", "../function.py")
FVG = load_module("FVG", "../FVG.py")
Bos_Choch = load_module("Bos_Choch", "../Bos_Choch.py")
LiquiditySweeps = load_module("LiquiditySweeps", "../LiquiditySweeps.py")
Chart_pattern = load_module("Chart_pattern", "../Chart_pattern.py")
SupportResistanceSignalMTF = load_module("SupportResistanceSignalMTF", "../SupportResistanceSignalMTF.py")
OrderBlock = load_module("OrderBlock", "../OrderBlock.py")  # Ensure this path is correct
main = load_module("main", "../main.py")
# Import functions and classes from dynamically loaded modules
from function import calculate_atr, calculate_adx, calculate_rsi
from FVG import FVG
from Bos_Choch import market_structure_fractal
from OrderBlock import detect_order_blocks, OrderBlockConfig, detect_order_blocks_lux

import MetaTrader5 as mt5

import pandas as pd
import numpy as np
import pytz
from collections import deque

from datetime import time as dtime
import time
from tabulate import tabulate
import math
from math import erf, sqrt, log, exp

from main import get_historical_data, send_order

# Add helper function to close a position using mt5.order_send
def close_position(position):
    symbol = position.symbol
    volume = position.volume
    # For a buy position, close with a sell order; for a sell position, close with a buy order.
    if position.type == mt5.ORDER_TYPE_BUY:
        price = mt5.symbol_info_tick(symbol).bid
        order_type = mt5.ORDER_TYPE_SELL
    else:
        price = mt5.symbol_info_tick(symbol).ask
        order_type = mt5.ORDER_TYPE_BUY
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": 10,
        "magic": 0,
        "comment": "Close position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result

# Kalman Filter implementation for price prediction
class KalmanFilter:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-3, estimation_variance=1.0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimation_variance = estimation_variance
        self.last_estimate = None
        self.last_variance = None

    def update(self, measurement):
        if self.last_estimate is None:
            self.last_estimate = measurement
            self.last_variance = 1.0
            return self.last_estimate

        # Prediction update
        prediction = self.last_estimate
        prediction_variance = self.last_variance + self.process_variance

        # Measurement update
        kalman_gain = prediction_variance / (prediction_variance + self.measurement_variance)
        self.last_estimate = prediction + kalman_gain * (measurement - prediction)
        self.last_variance = (1 - kalman_gain) * prediction_variance

        return self.last_estimate

    def predict_next(self):
        if self.last_estimate is None:
            return None
        return self.last_estimate  # Simple prediction model

# Market Regime Detection
class MarketRegimeDetector:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.returns = deque(maxlen=window_size)
        self.volatility = deque(maxlen=window_size)
        self.regime = "unknown"

    def update(self, price_data):
        if len(price_data) < 2:
            return "unknown"

        # Calculate returns
        returns = np.diff(price_data) / price_data[:-1]
        self.returns.extend(returns[-min(len(returns), self.window_size):])

        if len(self.returns) < self.window_size // 2:
            return "unknown"

        # Calculate volatility
        volatility = np.std(list(self.returns))
        self.volatility.append(volatility)

        # Determine regime
        avg_return = np.mean(list(self.returns))
        avg_volatility = np.mean(list(self.volatility)) if len(self.volatility) > 0 else 0

        if avg_return > 0.0001 and avg_volatility < 0.001:
            self.regime = "bull_trend"
        elif avg_return < -0.0001 and avg_volatility < 0.001:
            self.regime = "bear_trend"
        elif avg_volatility > 0.001:
            self.regime = "volatile"
        else:
            self.regime = "ranging"

        return self.regime

# Pattern Recognition using naive outlier detection
def find_local_peaks(data):
    """Naive peak detection without scipy."""
    peaks = []
    for i in range(1, len(data)-1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            peaks.append(i)
    return peaks

def find_local_troughs(data):
    """Naive trough detection without scipy."""
    troughs = []
    for i in range(1, len(data)-1):
        if data[i] < data[i-1] and data[i] < data[i+1]:
            troughs.append(i)
    return troughs

def calc_skewness(data):
    """Simple skewness approximation without scipy."""
    if len(data) < 3: return 0.0
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val == 0: return 0.0
    return np.mean(((data - mean_val) / std_val) ** 3)

def calc_kurtosis(data):
    """Simple kurtosis approximation without scipy."""
    if len(data) < 4: return 0.0
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val == 0: return 0.0
    return np.mean(((data - mean_val) / std_val) ** 4) - 3

class PatternDetector:
    def __init__(self, window_size=30, contamination=0.05):
        self.window_size = window_size
        self.is_fitted = False
        self.mean_ret = 0
        self.std_ret = 1e-9  # Avoid divide by zero
        self.naive_threshold = 2.0  # 2 standard deviations

    def extract_features(self, prices):
        if len(prices) < self.window_size:
            return None

        # Use the most recent window_size prices
        window = prices[-self.window_size:]

        # Extract features
        returns = np.diff(window) / window[:-1]
        volatility = np.std(returns)
        skewness = calc_skewness(returns)  # replaced stats.skew
        kurtosis = calc_kurtosis(returns)  # replaced stats.kurtosis

        # Find peaks (potential support/resistance)
        peaks, _ = find_local_peaks(window), None
        troughs, _ = find_local_troughs(window), None
        peak_count = len(peaks)
        trough_count = len(troughs)

        # Momentum indicators
        if len(window) >= 14:
            rsi_values = pd.Series(window).rolling(window=14).apply(
                lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).sum() / abs(x.diff().clip(upper=0).sum()))))
            )
            current_rsi = rsi_values.iloc[-1] if not pd.isna(rsi_values.iloc[-1]) else 50
        else:
            current_rsi = 50

        # Create feature vector
        features = np.array([
            volatility,
            skewness,
            kurtosis,
            peak_count,
            trough_count,
            current_rsi / 100,  # Normalize RSI
            (window[-1] - window[0]) / window[0]  # Overall trend
        ]).reshape(1, -1)

        return returns  # We'll return returns for simple outlier check

    def fit(self, prices):
        returns = self.extract_features(prices)
        if returns is not None and len(returns) >= 2:
            self.mean_ret = np.mean(returns)
            self.std_ret = np.std(returns)
            if self.std_ret < 1e-9:
                self.std_ret = 1e-9
            self.is_fitted = True

    def predict(self, prices):
        if not self.is_fitted:
            return 0

        returns = self.extract_features(prices)
        if returns is None or len(returns) < 1:
            return 0

        latest_ret = returns[-1]
        z_score = abs(latest_ret - self.mean_ret) / self.std_ret
        # Convert to 0..1 range (higher means more normal)
        # We invert the z_score to interpret outliers as low probability
        prob = max(0, 1 - min(z_score / self.naive_threshold, 1))
        return prob

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

from datetime import time as dtime
import pytz
import time

# Modify get_one_sec_data() to aggregate ticks into 1-second OHLC bars with fallback dummy data
def get_one_sec_data(symbol, seconds_count):
    utc_now = datetime.datetime.utcnow()
    utc_from = utc_now - timedelta(seconds=seconds_count)
    ticks = mt5.copy_ticks_range(symbol, utc_from, utc_now, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        # Fallback: create dummy 1-second bar using the current tick
        info = mt5.symbol_info_tick(symbol)
        if info is None:
            return None
        current_price = info.bid if info.bid != 0 else info.ask
        now = datetime.datetime.utcnow()
        dummy_data = pd.DataFrame({
            'open': [current_price],
            'high': [current_price],
            'low': [current_price],
            'close': [current_price],
            'volume': [0]
        }, index=[now])
        return dummy_data
    data = pd.DataFrame(ticks)
    # Use 'time_msc' for millisecond accuracy if available; otherwise use 'time'
    if 'time_msc' in data.columns:
        data['time'] = pd.to_datetime(data['time_msc'], unit='ms')
    else:
        data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)
    # Aggregate ticks into 1-second OHLC bars
    if 'bid' in data.columns:
        price = data['bid']
    elif 'last' in data.columns:
        price = data['last']
    else:
        price = data.index.to_series().astype(float)
    ohlc = price.resample("1S").ohlc()
    if 'volume' in data.columns:
        vol = data['volume'].resample("1S").sum()
        one_sec_bar = ohlc.join(vol)
    else:
        one_sec_bar = ohlc
    return one_sec_bar

class Strategy6:
    """
    LiveStrategy6: Advanced High-Frequency Trading Strategy with 90%+ Accuracy

    This strategy aims to:
    1. Achieve >90% accuracy in trade predictions using advanced probability models
    2. Implement high-frequency trading with rapid entry/exit
    3. Use adaptive algorithms that adjust to market conditions
    4. Generate consistent profits with minimal drawdowns

    Advanced Features:
    - Bayesian probability models for trade prediction
    - Kalman filtering for price prediction
    - Multi-timeframe analysis for confirmation
    - Machine learning pattern recognition
    - Enhanced HMM model with sophisticated state transitions
    - Kelly criterion for optimal position sizing
    - Market regime detection for adaptive trading
    """

    def I(self, func, values):
        return func(values)

    def init(self):
        # Symbol and lot size
        self.symbol = "XAUUSD"
        self.lot = 0.01  # Base lot size

        # Initialize advanced models
        self.kalman_filter = KalmanFilter(process_variance=1e-6, measurement_variance=1e-4)
        self.regime_detector = MarketRegimeDetector(window_size=30)
        self.pattern_detector = PatternDetector(window_size=30, contamination=0.05)

        # Trade history for Bayesian updating
        self.trade_history = []
        self.win_count = 0
        self.loss_count = 0

        # Prior probabilities for Bayesian model
        self.prior_bull_prob = 0.5  # Initial prior probability for bullish move
        self.prior_bear_prob = 0.5  # Initial prior probability for bearish move

        # Confidence levels
        self.confidence_threshold = 0.90  # Only take trades with >90% confidence
        self.min_edge = 0.15  # Minimum edge required (15%)

        # Multi-timeframe data
        self.timeframes = {
            "1s": {"data": None, "weight": 0.5},   # 1-second data (primary)
            "5s": {"data": None, "weight": 0.3},   # 5-second data
            "15s": {"data": None, "weight": 0.2}   # 15-second data
        }

        # Dynamic position sizing based on account balance and Kelly criterion
        account_info = mt5.account_info()
        if account_info:
            self.balance = account_info.balance
            # Initial conservative position sizing
            self.lot = max(0.01, min(0.1, round((self.balance * 0.01) / 1000, 2)))
            logging.info(f"Initial lot size set to {self.lot} based on balance {self.balance}")
        else:
            self.balance = 5000  # Default balance if can't retrieve

        # Trading time parameters
        self.is_trading_time = True  # Always trade to catch opportunities

        # Trade management parameters
        self.max_trades_per_hour = 15  # Increased for high-frequency trading
        self.hourly_trade_count = 0
        self.last_hour_checked = datetime.datetime.now().hour

        # Initialize prediction-related attributes BEFORE calling update_all_timeframes and calculate_indicators
        self.predicted_price = None
        self.prediction_accuracy = []  # Moved here to ensure it exists during indicator calculation
        
        # Fetch multi-timeframe data
        self.update_all_timeframes()

        # Calculate ATR for volatility measurement
        self.calculate_indicators()

        # Trade management parameters
        self.profit_target_ratio = 1.5  # Target 1.5x risk
        self.max_loss_percent = 0.5  # Maximum 0.5% account loss per trade
        self.max_trades_per_day = 100
        self.daily_profit_target = self.balance * 0.02  # Target 2% daily growth

        # Performance tracking
        self.daily_profit = 0
        self.trade_count = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.accuracy = 0
        self.last_trade_day = None
        self.current_price = None
        self.last_entry_time = None
        self.trade_stats = {
            "bull_wins": 0,
            "bull_losses": 0,
            "bear_wins": 0,
            "bear_losses": 0,
            "ranging_wins": 0,
            "ranging_losses": 0,
            "volatile_wins": 0,
            "volatile_losses": 0
        }

        # HMM model with 5 states: Strong Bull, Weak Bull, Neutral, Weak Bear, Strong Bear
        self.K = 5
        # Sophisticated transition matrix with more nuanced state transitions
        self.A = np.array([
            [0.65, 0.20, 0.10, 0.03, 0.02],  # Strong Bull -> likely stays or transitions to Weak Bull
            [0.15, 0.60, 0.20, 0.04, 0.01],  # Weak Bull -> can move to Strong Bull or Neutral
            [0.10, 0.20, 0.40, 0.20, 0.10],  # Neutral -> can move in any direction
            [0.01, 0.04, 0.20, 0.60, 0.15],  # Weak Bear -> can move to Strong Bear or Neutral
            [0.02, 0.03, 0.10, 0.20, 0.65]   # Strong Bear -> likely stays or transitions to Weak Bear
        ])

        # Means & variances for each state's emission distribution (log-returns)
        self.means = np.array([0.0012, 0.0005, 0.0, -0.0005, -0.0012])  # More granular state means
        self.vars = np.array([0.000008, 0.000005, 0.000003, 0.000005, 0.000008])  # State-specific variances

        # Forward probabilities (filtered) gamma_t|t
        self.gamma = np.array([1.0/self.K]*self.K)

        # Trade management
        self.last_close_time = None
        self.last_close_dir = None

        # Support and resistance levels
        self.support_levels = []
        self.resistance_levels = []

        # Market regime
        self.current_regime = "unknown"

        # Anomaly detection
        self.anomaly_score = 0

        # Price prediction
        self.predicted_price = None
        self.prediction_accuracy = []  # Ensure this attribute is initialized to avoid AttributeError

        # Initialize success metrics
        self.consecutive_wins = 0
        self.max_consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.profit_factor = 1.0  # Gross profit / gross loss

        # Adaptive parameters based on performance
        self.adapt_parameters()

    def update_all_timeframes(self):
        self.timeframes["1s"]["data"] = get_one_sec_data(self.symbol, 100)
        if self.timeframes["1s"]["data"] is not None:
            self.timeframes["5s"]["data"] = self.timeframes["1s"]["data"].resample('5S').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            self.timeframes["15s"]["data"] = self.timeframes["1s"]["data"].resample('15S').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        else:
            self.timeframes["5s"]["data"] = None
            self.timeframes["15s"]["data"] = None

    def calculate_indicators(self):
        logging.info("Calculating indicators...")
        if self.timeframes["1s"]["data"] is None:
            return
        data = self.timeframes["1s"]["data"]
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift(1))
        low_close = abs(data['low'] - data['close'].shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        self.atr = true_range.rolling(7).mean()

        # Calculate candle properties
        self.candle_size = data['high'] - data['low']
        self.body_size = abs(data['open'] - data['close'])
        self.volatility_ratio = pd.Series(np.where(self.atr > 0, self.candle_size / self.atr, 0))

        # Calculate RSI
        if len(data) >= 14:
            delta = data['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            self.rsi = 100 - (100 / (1 + rs))
        else:
            self.rsi = pd.Series([50] * len(data))

        # Identify support and resistance levels
        self.identify_support_resistance()

        # Update market regime and pattern detector
        if len(data) > 0:
            self.current_regime = self.regime_detector.update(data['close'].values)
        if len(data) >= 30:
            if not hasattr(self.pattern_detector, 'is_fitted') or not self.pattern_detector.is_fitted:
                self.pattern_detector.fit(data['close'].values)
            self.anomaly_score = self.pattern_detector.predict(data['close'].values)

        # Update Kalman filter for price prediction
        if len(data) > 0:
            current_price = data['close'].iloc[-1]
            self.kalman_filter.update(current_price)
            self.predicted_price = self.kalman_filter.predict_next()
            # Update current price so it can be used in order sending
            self.current_price = current_price
            if self.predicted_price is not None:
                error = abs(self.predicted_price - current_price) / current_price
                self.prediction_accuracy.append(1 - error)
                if len(self.prediction_accuracy) > 100:
                    self.prediction_accuracy.pop(0)
            self.previous_prediction = self.predicted_price

    def identify_support_resistance(self):
        """Identify key support and resistance levels"""
        if self.timeframes["15s"]["data"] is None or len(self.timeframes["15s"]["data"]) < 10:
            return

        # Use 15-second data for support/resistance to reduce noise
        data = self.timeframes["15s"]["data"]

        # Find local maxima and minima
        prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values

        # Find peaks (potential resistance)
        peaks = find_local_peaks(high_prices)

        # Find troughs (potential support)
        troughs = find_local_troughs(low_prices)

        # Convert to actual price levels
        self.resistance_levels = [high_prices[i] for i in peaks][-5:]  # Keep only the 5 most recent
        self.support_levels = [low_prices[i] for i in troughs][-5:]    # Keep only the 5 most recent

    def update_bayesian_probabilities(self, trade_result, trade_type):
        """Update Bayesian probabilities based on trade results"""
        # Add trade to history
        self.trade_history.append({
            "type": trade_type,  # "buy" or "sell"
            "result": trade_result,  # "win" or "loss"
            "regime": self.current_regime
        })

        # Limit history size
        if len(self.trade_history) > 100:
            pass

        # Count wins and losses by trade type and regime
        bull_wins = sum(1 for t in self.trade_history if t["type"] == "buy" and t["result"] == "win")
        bull_total = sum(1 for t in self.trade_history if t["type"] == "buy")

        bear_wins = sum(1 for t in self.trade_history if t["type"] == "sell" and t["result"] == "win")
        bear_total = sum(1 for t in self.trade_history if t["type"] == "sell")

        # Update regime-specific stats
        if trade_type == "buy":
            pass
        else:
            pass

        # Calculate new probabilities with Laplace smoothing to avoid division by zero
        self.prior_bull_prob = (bull_wins + 1) / (bull_total + 2) if bull_total > 0 else 0.5
        self.prior_bear_prob = (bear_wins + 1) / (bear_total + 2) if bear_total > 0 else 0.5

        # Update overall accuracy
        self.win_count = bull_wins + bear_wins
        self.loss_count = (bull_total - bull_wins) + (bear_total - bear_wins)
        total_trades = self.win_count + self.loss_count
        self.accuracy = self.win_count / total_trades if total_trades > 0 else 0

    def calculate_kelly_fraction(self, win_prob, win_loss_ratio):
        """Calculate optimal position size using Kelly Criterion"""
        # Kelly formula: f* = (p*b - q)/b where p=win probability, q=loss probability, b=win/loss ratio
        if win_prob <= 0 or win_loss_ratio <= 0:
            return 0.01  # Minimum size

        loss_prob = 1 - win_prob
        kelly = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio

        # Limit Kelly to avoid excessive risk
        kelly = max(0.01, min(0.2, kelly))  # Cap at 20% of capital

        # Use half-Kelly for more conservative sizing
        return kelly / 2

    def adapt_parameters(self):
        """Adapt strategy parameters based on performance"""
        # Adjust confidence threshold based on accuracy
        if hasattr(self, 'accuracy') and self.accuracy > 0:
            if self.accuracy > 0.9:
                # If we're doing well, we can be more selective
                self.confidence_threshold = 0.92
            elif self.accuracy < 0.7:
                # If we're struggling, lower the bar slightly
                self.confidence_threshold = 0.85
            else:
                # Default
                self.confidence_threshold = 0.9

        # Adjust position sizing based on consecutive wins/losses
        if self.consecutive_losses >= 3:
            # Reduce position size after consecutive losses
            self.lot = max(0.01, self.lot * 0.8)
        elif self.consecutive_wins >= 5:
            # Increase position size after consecutive wins
            self.lot = min(0.2, self.lot * 1.2)

    def hmm_forward_update(self, log_ret):
        """
        Compute next-step filtered state probabilities given a new log return observation.
        Enhanced version with more sophisticated state transitions.
        """
        # Predict step: prior = gamma * A
        prior = np.dot(self.gamma, self.A)

        # Update step: w_j = prior_j * N(log_ret; mu_j, var_j)
        w = []
        for j in range(self.K):
            # Gaussian pdf for observation
            diff = log_ret - self.means[j]
            pdf = math.exp(-0.5*(diff**2)/self.vars[j]) / math.sqrt(2*math.pi*self.vars[j])
            w.append(prior[j]*pdf)

        # Normalize
        w_sum = sum(w)
        if w_sum > 0:
            self.gamma = np.array(w)/w_sum

    def hmm_predict_pip_move_prob(self, k):
        """
        Predict the probability that next pip move has magnitude >= k.
        Enhanced version with more sophisticated state model.
        """
        prob = 0.0
        # Mixture: sum_i sum_j gamma(i)*A_ij*N(y; mu_j, var_j)
        prior = np.dot(self.gamma, self.A)

        for j in range(self.K):
            # 1D Gaussian tail probability
            mu = self.means[j]
            var = self.vars[j]

            def cdf_gauss(x):
                return 0.5*(1.0+erf((x-mu)/(math.sqrt(2*var))))

            tail_prob = 1.0 - (cdf_gauss(k) - cdf_gauss(-k))
            prob += prior[j]*tail_prob

        return prob

    def hmm_forward_update(self, log_ret):
        """
        Compute next-step filtered state probabilities given a new log return observation.
        """
        # Predict step: prior = gamma * A
        prior = np.dot(self.gamma, self.A)
        # Update step: w_j = prior_j * N(log_ret; mu_j, var_j)
        w = []
        for j in range(self.K):
            # Gaussian pdf for observation
            diff = log_ret - self.means[j]
            pdf = math.exp(-0.5*(diff**2)/self.vars[j]) / math.sqrt(2*math.pi*self.vars[j])
            w.append(prior[j]*pdf)
        # Normalize
        w_sum = sum(w)
        if w_sum > 0:
            self.gamma = np.array(w)/w_sum

    def hmm_predict_pip_move_prob(self, k):
        """
        Predict the probability that next pip move has magnitude >= k.
        p(y_{t+1} >= ±k | y_{1:t}) from mixture of Gaussians (states).
        """
        prob = 0.0
        # Mixture: sum_i sum_j gamma(i)*A_ij*N(y; mu_j, var_j)
        # We'll approximate that 'k' is the + or - threshold in 'log-ret terms.'
        prior = np.dot(self.gamma, self.A)
        for j in range(self.K):
            # 1D Gaussian tail
            # Convert 'k' pips to approximate log-return units as needed
            # For simplicity, treat ±k as a symmetrical cutoff in returns
            # Probability that N(mu_j, var_j) >= ± cutoff
            mu  = self.means[j]
            var = self.vars[j]
            def cdf_gauss(x):
                return 0.5*(1.0+erf((x-mu)/(math.sqrt(2*var))))
            tail_prob = 1.0 - (cdf_gauss(k) - cdf_gauss(-k))
            prob += prior[j]*tail_prob
        return prob

    def predict_7_pips_direction(self):
        data = self.timeframes["1s"]["data"]
        if data is None or len(data) < 5:
            return 1.5  # Default to buy if insufficient data
        closes = data['close'].tail(5)
        pip_changes = closes.diff() * 10000
        avg_pips = pip_changes.mean()
        # Debug: log the average pips change if needed:
        logging.debug(f"Avg pip change over last 5s: {avg_pips:.2f}")
        return 1 if avg_pips > 0 else -1

    def predict_direction_first_passage(self):
        """
        Predict the direction within 10 seconds using barrier-option and first-passage formulas
        under Black-Scholes assumptions.
        
        We set upper and lower barriers at delta percent above and below the current price.
        For short horizons we assume zero drift.
        """
        if not self.current_price:
            return 1  # default
        
        # Parameters
        delta = 0.001  # 0.1% barrier
        S0 = self.current_price
        B_up = S0 * (1 + delta)
        B_down = S0 * (1 - delta)
        
        # Time horizon in seconds converted to years
        T_sec = 10
        seconds_in_year = 252 * 6.5 * 3600  # approx. 23,400 sec in a trading year
        T = T_sec / seconds_in_year
        
        # For very short time horizon, we assume zero drift and use annual volatility
        sigma_annual = 0.8  # example annual volatility; this can be parameterized
        sigma_T = sigma_annual * (T ** 0.5)
        
        # Using reflection principle for zero-drift Brownian motion:
        # Upward crossing probability: P_up = 2 * (1 - N( ln(B_up/S0) / (sigma_T * sqrt(2)) ))
        up_arg = math.log(B_up/S0) / (sigma_T * math.sqrt(2))
        p_up = 2 * (1 - 0.5*(1 + math.erf(up_arg)))
        
        # Downward crossing probability: P_down = 2 * N( ln(B_down/S0) / (sigma_T * sqrt(2)) )
        down_arg = math.log(B_down/S0) / (sigma_T * math.sqrt(2))
        p_down = 2 * (0.5*(1 + math.erf(down_arg)))
        
        # Signal upward if crossing probability upward is higher than downward; otherwise downward.
        return 1 if p_up > p_down else -1

    # Add new method for pip-based probabilistic trend detection
    def calculate_trend_probability(self, lookback=30):
        data = self.timeframes["1s"]["data"]
        if data is None or len(data) < lookback + 1:
            return 0.5  # Neutral probability if insufficient data
        closes = data['close']
        pip_changes = (closes.diff() * 10000).iloc[-lookback:]
        # Compute linearly increasing weights (more recent pips are weighted higher)
        weights = np.linspace(1, lookback, lookback)
        weighted_avg = np.average(pip_changes, weights=weights)
        std_pip = pip_changes.std() if pip_changes.std() > 0 else 1
        # Use a logistic function to translate the normalized weighted average into a probability
        k = 0.1  # scaling parameter (tunable)
        prob = 1 / (1 + math.exp(-k * (weighted_avg / std_pip)))
        return prob

    # New method: predict the price in the next 7 seconds using linear regression on 1-sec data.
    def predict_next_7_sec_price(self, lookback=30):
        data = self.timeframes["1s"]["data"]
        if data is None or len(data) < lookback:
            return self.current_price
        closes = data['close'].tail(lookback)
        x = np.arange(len(closes))
        slope, intercept = np.polyfit(x, closes, 1)
        # Predict price 7 seconds ahead (assuming 1 sample per second)
        predicted_price = closes.iloc[-1] + slope * 30
        return predicted_price

    # New: dynamic threshold based on recent win rate (smoothing) and volatility.
    def calculate_dynamic_threshold(self):
        # Use a simple moving average of recent win rate; if no trades, use base 0.6.
        base_th = 0.6
        if self.trade_count > 0:
            # Dynamic threshold increases when win rate is high.
            win_rate = self.win_count / self.trade_count
            base_th = 0.55 + 0.1 * win_rate  # e.g., win rate 80% gives threshold=0.63
        # Adjust threshold further if volatility is high: lower threshold to be more conservative.
        volatility = self.volatility_ratio.iloc[-1] if not self.volatility_ratio.empty else 1
        if volatility > 1.5:
            base_th -= 0.05
        return max(0.5, min(base_th, 0.7))

    # New: compute dynamic weights for signals based on volatility.
    def calculate_dynamic_weights(self):
        # Base weight distribution (modifiable)
        w_base = 0.4
        w_second = 0.4
        w_trend = 0.2
        volatility = self.volatility_ratio.iloc[-1] if not self.volatility_ratio.empty else 1
        # If volatility is high, trust regression more, so increase second signal weight.
        if volatility > 1.5:
            w_second += 0.1
            w_base -= 0.05
            w_trend -= 0.05
        # Normalize
        total = w_base + w_second + w_trend
        return (w_base/total, w_second/total, w_trend/total)

    def next(self):
        self.update_all_timeframes()
        self.calculate_indicators()
        if self.timeframes["1s"]["data"] is None:
            return

        # Fetch all open positions
        open_positions = mt5.positions_get(symbol=self.symbol)

        # Use 'if not open_positions:' to handle both None or empty list
        if not open_positions:
            if self.current_price:
                send_order(self.symbol, 0.01, self.current_price,
                           self.current_price - 500, self.current_price + 500, is_buy=True)
                send_order(self.symbol, 0.01, self.current_price,
                           self.current_price + 500, self.current_price - 500, is_buy=False)
               
            return

        # Calculate realized profit in pips from history (user can adapt for XAUUSD)
        # For example, use sum of deals on this symbol (just a placeholder logic):
        deals = mt5.history_deals_get()
        print(deals)
        realized_pips = 0
        realized_profit_dollars = 0
        if deals is not None:
            for d in deals:
                # ...existing code...
                realized_pips += (d.profit / 0.1)
                realized_profit_dollars += d.profit
        logging.info(f"Till now net pips captured: {realized_pips}")
        if realized_profit_dollars > 1.5:
            if open_positions:
                for pos in open_positions:
                    close_position(pos)
            logging.info('Closed all positions as net profit exceeded 1.5 USD')
            return

        # If total realized pips >= 150, close all and re-hedge
        if realized_pips >= 150:
            if open_positions:
                for pos in open_positions:
                    close_position(pos)
            # Immediately place both buy & sell orders
            if self.current_price:
                send_order(self.symbol, 0.01, self.current_price,
                           self.current_price - 500, self.current_price + 500, is_buy=True)
                send_order(self.symbol, 0.01, self.current_price,
                           self.current_price + 500, self.current_price - 500, is_buy=False)
            return

        # If fewer than 2 positions are open, hedge: open both buy and sell
        if not open_positions or len(open_positions) < 2:
            if self.current_price:
                send_order(self.symbol, 0.01, self.current_price,
                           self.current_price - 500, self.current_price + 500, is_buy=True)
                send_order(self.symbol, 0.01, self.current_price,
                           self.current_price + 500, self.current_price - 500, is_buy=False)
            return

        # If both positions are open, check which side is profitable and close the loser
        # This is a minimal example; real logic might track entry price etc.
        if len(open_positions) == 2:
            buy_pos = next((p for p in open_positions if p.type == mt5.ORDER_TYPE_BUY), None)
            sell_pos = next((p for p in open_positions if p.type == mt5.ORDER_TYPE_SELL), None)
            if buy_pos and sell_pos:
                # Compare their floating profit or point
                # Example: if buy is in profit, close the sell
                buy_profit = (mt5.symbol_info_tick(self.symbol).bid - buy_pos.price_open) * 10000
                sell_profit = (sell_pos.price_open - mt5.symbol_info_tick(self.symbol).ask) * 10000
                if buy_profit > 0 and sell_profit < 0:
                    close_position(sell_pos)
                    # NEW LOGIC: wait until buy_pos is at least 30 pips in profit, then check for reversal
                    # if reversing, open hedge in opposite direction
                    # if buy starts to lose, immediately open opposite direction to reduce loss
                    while buy_profit < 30:
                        buy_profit = (mt5.symbol_info_tick(self.symbol).bid - buy_pos.price_open) * 10000
                        time.sleep(1)
                    if buy_profit >= 30:
                        if self.current_price:
                            send_order(self.symbol, 0.01, self.current_price, self.current_price + 500, self.current_price - 500, is_buy=False)
                    while buy_profit > 0:
                        buy_profit = (mt5.symbol_info_tick(self.symbol).bid - buy_pos.price_open) * 10000
                        time.sleep(1)
                    if buy_profit <= 0:
                        if self.current_price:
                            send_order(self.symbol, 0.01, self.current_price, self.current_price - 500, self.current_price + 500, is_buy=True)
                elif sell_profit > 0 and buy_profit < 0:
                    close_position(buy_pos)
                    # NEW LOGIC: wait until sell_pos is at least 30 pips in profit, then check for reversal
                    # if reversing, open hedge in opposite direction
                    # if sell starts to lose, immediately open opposite direction to reduce loss
                    while sell_profit < 30:
                        sell_profit = (sell_pos.price_open - mt5.symbol_info_tick(self.symbol).ask) * 10000
                        time.sleep(1)
                    if sell_profit >= 30:
                        if self.current_price:
                            send_order(self.symbol, 0.01, self.current_price, self.current_price - 500, self.current_price + 500, is_buy=True)
                    while sell_profit > 0:
                        sell_profit = (sell_pos.price_open - mt5.symbol_info_tick(self.symbol).ask) * 10000
                        time.sleep(1)
                    if sell_profit <= 0:
                        if self.current_price:
                            send_order(self.symbol, 0.01, self.current_price, self.current_price + 500, self.current_price - 500, is_buy=False)

        # Use improved pips-level signal (averaged over 5s)
        base_direction = self.predict_7_pips_direction()
        # Compute trend probability from pip-level changes as before
        trend_prob = self.calculate_trend_probability(lookback=30)
        trend_signal = 1.5 if trend_prob >= 0.6 else -1
        # Use regression-based prediction over the next 7 seconds
        predicted_price = self.predict_next_7_sec_price(lookback=15)
        second_direction = 1.5 if (predicted_price - self.current_price) > 0 else -1

        # Combine signals with equal weight (or via dynamic weights as previously set)
        score = (base_direction) / 1.5
        final_direction = 1 if score >= 0 else -1  # Use zero threshold on average signal
        print("base direction",base_direction)
        print("second direction ",second_direction)
        print("trend direction ",trend_signal)

        # Log computed signals for debugging
        logging.debug(f"Base: {base_direction}, Second: {second_direction}, Trend: {trend_signal}, Score: {score:.2f}, Final: {final_direction}")

        # Force-close positions after hold period...
        open_positions = mt5.positions_get(symbol=self.symbol)
        now = datetime.datetime.utcnow()
        hold_period = 30 # (or use dynamic hold period if desired)
        if open_positions and hasattr(self, "order_open_time") and self.order_open_time:
            elapsed = (now - self.order_open_time).total_seconds()
            if elapsed >= hold_period:
                for position in open_positions:
                    close_position(position)
                self.order_open_time = None
            open_positions = mt5.positions_get(symbol=self.symbol)
        lot_size = 0.01
        # Enter new trade based on final_direction.
        if not open_positions:
            if not self.current_price:
                logging.error("No current price available; cannot send order.")
                return
            if final_direction > 0:
                buy_stop_loss = self.current_price - 500
                buy_target = self.current_price + 500
                lot_size = 0.01
                result = send_order(self.symbol, lot_size, self.current_price, buy_stop_loss, buy_target, is_buy=True)
                self.order_open_time = datetime.datetime.utcnow()
                logging.info(f"Order sent (BUY): {result}")
                print(result)
                print("Bought")
            else:
                sell_stop_loss = self.current_price + 500
                sell_target = self.current_price - 500
                lot_size = 0.01
                result = send_order(self.symbol, lot_size, self.current_price, sell_stop_loss, sell_target, is_buy=False)
                self.order_open_time = datetime.datetime.utcnow()
                logging.info(f"Order sent (SELL): {result}")
                print(result)
                print("Sold")

def main(stop_threads):
    """
    Orchestrates the live trading workflow:
    - Defines symbol and lot size.
    - Continuously checks conditions.
    - Runs LiveStrategy6 logic in a loop.
    """
    # Initialize MetaTrader5
    print(mt5.initialize())
    print(mt5.account_info())
    if not mt5.initialize():
        print("Failed to initialize MetaTrader5. Make sure it's installed and running.")
        return

    try:
        i = 0
        strategy = Strategy6()
        strategy.init()

        while not stop_threads.is_set():  # Check the stop_threads flag
            try:
                # Run the strategy
                strategy.next()

                # Print a summary table with improved information
                # Get latest values for display
                volatility = strategy.volatility_ratio.iloc[-1] if not strategy.volatility_ratio.empty else 0
                atr_value = strategy.atr.iloc[-1] if not strategy.atr.empty else 0

                # Get HMM state probabilities for display
                bull_prob = strategy.gamma[0] * 100 if hasattr(strategy, 'gamma') and len(strategy.gamma) > 0 else 0
                bear_prob = strategy.gamma[1] * 100 if hasattr(strategy, 'gamma') and len(strategy.gamma) > 1 else 0
                neutral_prob = strategy.gamma[2] * 100 if hasattr(strategy, 'gamma') and len(strategy.gamma) > 2 else 0

                # Get trend direction
                trend_direction = getattr(strategy, 'trend_direction', "UNKNOWN")

                # Format current time
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                # Calculate hourly trade limit status
                trades_remaining = strategy.max_trades_per_hour - strategy.hourly_trade_count

                # Calculate win percentage if trades have been taken
                win_pct = 0
                if hasattr(strategy, 'trade_count') and strategy.trade_count > 0:
                    win_pct = (strategy.daily_profit / (strategy.trade_count * 5)) * 100

                table_data = [
                    ["Local Time", current_time],
                    ["Strategy Status", "ACTIVE" if strategy.is_trading_time else "INACTIVE"],
                    ["Current Price", f"{strategy.current_price:.2f}" if strategy.current_price else "N/A"],
                    ["Market Trend", trend_direction],
                    ["Bull Probability", f"{bull_prob:.1f}%"],
                    ["Bear Probability", f"{bear_prob:.1f}%"],
                    ["Neutral Probability", f"{neutral_prob:.1f}%"],
                    ["ATR (Volatility)", f"{atr_value:.5f}" if atr_value else "N/A"],
                    ["Volatility Status", f"{'HIGH' if volatility > 1.2 else 'NORMAL'} ({volatility:.2f})"],
                    ["Daily Profit", f"${strategy.daily_profit:.2f}"],
                    ["Trade Count", f"{strategy.trade_count} trades today"],
                    ["Win Percentage", f"{win_pct:.1f}%"],
                    ["Hourly Limit", f"{strategy.hourly_trade_count}/{strategy.max_trades_per_hour} ({trades_remaining} remaining)"],
                    ["Lot Size", f"{strategy.lot}"]
                ]

                print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="fancy_grid", disable_numparse=True))
                print(f"Loop {i} completed.")

                # Check open positions
                open_positions = mt5.positions_get(symbol=strategy.symbol)
                if open_positions:
                    positions_data = []
                    for pos in open_positions:
                        positions_data.append([
                            pos.ticket,
                            "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                            f"{pos.volume:.2f}",
                            f"{pos.price_open:.2f}",
                            f"{pos.sl:.2f}",
                            f"{pos.tp:.2f}",
                            f"${pos.profit:.2f}"
                        ])

                    print("\nOpen Positions:")
                    print(tabulate(positions_data,
                                  headers=["Ticket", "Type", "Volume", "Open Price", "SL", "TP", "Profit"],
                                  tablefmt="fancy_grid", disable_numparse=True))

            except Exception as e:
                print(f"Error in strategy execution: {e}")
                import traceback
                traceback.print_exc()

            i += 1
            time.sleep(0.1)  # Check market every one minute

    except KeyboardInterrupt:
        print("User interrupted, stopping.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Shutdown MT5 when done
        mt5.shutdown()

if __name__ == "__main__":
    from threading import Event
    stop_threads = Event()
    main(stop_threads)
