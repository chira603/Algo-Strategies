"""
Live trading strategy for XAUUSD (M5) using Strategy1 logic.

Enhancements:
- Organized imports
- Added concise docstrings
- Provided inline comments where relevant
- Kept functionality intact
"""








#********************************************************************************

# Import necessary libraries

#********************************************************************************

import sys
import os
import importlib.util

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
FVG = load_module("FVG", "../FVG.py")  # Updated path to parent directory
OrderBlock = load_module("OrderBlock", "../OrderBlock.py")
Bos_Choch = load_module("Bos_Choch", "../Bos_Choch.py")
LiquiditySweeps = load_module("LiquiditySweeps", "../LiquiditySweeps.py")
Chart_pattern = load_module("Chart_pattern", "../Chart_pattern.py")
InternalOrderBlock = load_module("InternalOrderBlock", "../InternalOrderBlock.py")
SupportResistanceSignalMTF = load_module("SupportResistanceSignalMTF", "../SupportResistanceSignalMTF.py")  # Add this line
main = load_module("main", "../main.py")
# Import functions and classes from dynamically loaded modules
from function import (
    calculate_rsi, calculate_ema, calculate_atr, calculate_adx
)
from FVG import FVG
from SupportResistanceSignalMTF import detect_support_resistance_mtf  # Ensure this import works
from OrderBlock import detect_order_blocks, OrderBlockConfig, detect_order_blocks_lux


from InternalOrderBlock import store_order_block

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from tqdm import tqdm
from tabulate import tabulate  # Add this import for tabular formatting
from main import send_order, get_historical_data
from datetime import time as dtime
import pytz
import time

















#********************************************************************************

# Define constants

#********************************************************************************
class LiveStrategy1():
    """
    Main class for live strategy execution.
    - Collects and processes market data.
    - Calculates technical indicators.
    - Monitors conditions for trades.
    - Places orders via MetaTrader 5.
    """

    def I(self, func, *args):
        """
        Proxy method to wrap indicator functions.
        """
        return func(*args)

    def init(self):
        """
        Initializes indicators and key parameters:
        - Data retrieval from M5 timeframe.
        - EMA50, EMA200 for trend detection.ection.
        - ATR14 for volatility-based SL calculations.
        - is_trading_time to control trade windows.
        - internalOrderBlocks and fairValueGaps for advanced signals.
        """

        # *********************************************************************************

        # Getting Historical Data,ema50,ema200,atr14

        # *********************************************************************************
        # *********************************************************************************
        self.data = type('DataContainer', (), {})()
        self.data.df = get_historical_data("XAUUSD", mt5.TIMEFRAME_M5, bars=17280)
        
        # Convert to lowercase column names
        self.data.df.columns = [c.lower() for c in self.data.df.columns]
        self.data.Close = self.data.df['close']
        self.data.High = self.data.df['high']
        self.data.Low = self.data.df['low']
        self.data.Open = self.data.df['open']
        self.ema20 = self.I(lambda c: pd.Series(c).ewm(span=20).mean(), self.data.Close)
        self.ema50 = self.I(lambda c: pd.Series(c).ewm(span=50).mean(), self.data.Close)
        self.ema200 = self.I(lambda c: pd.Series(c).ewm(span=200).mean(), self.data.Close)
        self.atr14 = self.I(lambda h, l, c: calculate_atr(pd.DataFrame({'high': h, 'low': l, 'close': c}), 14),
                            self.data.High, self.data.Low, self.data.Close)
        
        # Initialize defaults
        self.internalOrderBlocks = []
        self.fairValueGaps = []
        self.is_volatile_market = False
        self.is_trading_time = False
        self.bullishChochCount = 0
        self.bearishChochCount = 0
        self.stopLossATRMultiplier = 2.0
        self.spreadPips = 0.05
        self.lot = 0.03
        self.leverage = 50
        self.minADX = 20
        self.trades = []























    #**********************************************************************************

    # Main function to execute strategy logic

    #**********************************************************************************
    #**********************************************************************************
   
   
   
   
   
    def next(self):
        """
        Replaces placeholder logic with Strategy1-like checks:
        - ADX threshold
        - RSI-based bias
        - Trend direction from EMA cross
        - Order placement aligned with strong signals
        """

        # Convert M5 data, prepare for resampling
        data_m5 = self.data.df.copy()
        data_m5.columns = [c.lower() for c in data_m5.columns]
       
        # Ensure data_m5 has a proper DatetimeIndex for resampling
        if not isinstance(data_m5.index, pd.DatetimeIndex):
            try:
                # If index contains datetime-like strings, convert to
                # DatetimeIndex
                data_m5.index = pd.to_datetime(data_m5.index)
            except:
                print("Warning: Could not convert index to DatetimeIndex")
                return  # Exit if conversion fails
       
        # Resample to 1H & 4H for retest
        data_1h = data_m5.resample('1H').agg({'high': 'max', 'low': 'min', 'close': 'last'})
        data_4h = data_m5.resample('4H').agg({'high': 'max', 'low': 'min', 'close': 'last'})
       
        if not data_1h.empty and not data_4h.empty:
            last_10_1h = data_1h.tail(10)
            last_10_4h = data_4h.tail(10)
            high1h = last_10_1h['high'].max()
            low1h = last_10_1h['low'].min()
            high4h = last_10_4h['high'].max()
            low4h = last_10_4h['low'].min()
            current_close = self.data.Close.iloc[-1]
            fast_ht_retest = ((current_close < high1h) and (current_close > low1h)) or \
                             ((current_close < high4h) and (current_close > low4h))
        else:
            fast_ht_retest = False












        # Check volatility on a higher timeframe (H1)
        data_h1 = get_historical_data("XAUUSD", mt5.TIMEFRAME_H1, bars=1000)
        # Ensure proper column naming for consistency
        if not data_h1.empty:
            data_h1.rename(columns={'high': 'high', 'low': 'low', 'close': 'close'}, inplace=True)
       
        if not data_h1.empty:
            atr_h1 = calculate_atr(data_h1, period=14)
            atr_ema_h1 = atr_h1.ewm(span=20).mean()
            
            self.is_volatile_market = atr_h1.iloc[-1] > atr_ema_h1.iloc[-1]
            print(atr_h1.iloc[-1], atr_ema_h1.iloc[-1])
        else:
            self.is_volatile_market = False




















        # Fair Value Gap check (last 70 bars)
        last_70 = data_m5.tail(70)
        fast_fvg = False
        fvg_data = pd.DataFrame()
        if not last_70.empty:
            fvg_result = FVG.fvg(last_70)
            fvg_data = pd.DataFrame({
                "is_bull": fvg_result["is_bull"],
                "is_bear": fvg_result["is_bear"],
                "gap_top": fvg_result["gap_top"],
                "gap_bottom": fvg_result["gap_bottom"]
            })
            if not fvg_data.empty:
                current_close = self.data.Close.iloc[-1]
                latest_gaps = fvg_data.tail(5)
                for _, gap in latest_gaps.iterrows():
                    if gap["is_bull"] and gap["gap_bottom"] <= current_close <= gap["gap_top"]:
                        fast_fvg = True
                    if gap["is_bear"] and gap["gap_bottom"] <= current_close <= gap["gap_top"]:
                        fast_fvg = True











        # Daily trend check with SMA & RSI
        data_d1 = data_m5.resample('1D').agg({'close': 'last'}).dropna()
        if len(data_d1) >= 20:
            sma50 = data_d1['close'].rolling(50).mean()
            sma200 = data_d1['close'].rolling(200).mean()
            rsi_d1 = calculate_rsi(data_d1['close'], 14)
            if sma50.iloc[-1] > sma200.iloc[-1] and rsi_d1.iloc[-1] > 50:
                fast_trend = "BULLISH"
            elif sma50.iloc[-1] < sma200.iloc[-1] and rsi_d1.iloc[-1] < 50:
                fast_trend = "BEARISH"
            else:
                fast_trend = "NEUTRAL"
        else:
            fast_trend = "NEUTRAL"










        # Determine trend direction (same as Strategy1)
        trendDirection = "BULLISH" if self.ema20.iloc[-1] > self.ema200.iloc[-1] else "BEARISH"

        # ADX filtering
        current_adx = calculate_adx(pd.DataFrame({
            'high': self.data.High[-100:],
            'low': self.data.Low[-100:],
            'close': self.data.Close[-100:]
        }), 14).iloc[-1]
        


        fast_strong = self.ema50.iloc[-1] > self.ema200.iloc[-1]
        fast_low_vol = False if self.atr14.iloc[-1] <= self.atr14.iloc[-1] else True






        # Trading time (Asia/Kolkata)
        if (fast_ht_retest or fast_fvg):
            india_tz = pytz.timezone("Asia/Kolkata")
            current_bar_time = pd.Timestamp.now(tz=pytz.utc).tz_convert(india_tz).time()
            morning_start, morning_end = dtime(8, 30), dtime(12, 30)
            evening_start, evening_end = dtime(17, 30), dtime(20, 30)
            if (morning_start <= current_bar_time <= morning_end) or (evening_start <= current_bar_time <= evening_end):
                self.is_trading_time = True
            else:
                self.is_trading_time = False
            
        if current_adx < self.minADX:
            return






        # Ensure at least one internal order block
        if not self.internalOrderBlocks:
            class DummyPivot:
                def __init__(self, barIndex):
                    self.barIndex = barIndex
            pivot = DummyPivot(0)
            parsedHighs = data_m5['high'].tolist()
            parsedLows = data_m5['low'].tolist()
            times = data_m5.index.tolist()
            bias = 1 if fast_trend == "BULLISH" else -1
            store_order_block(pivot, len(parsedHighs), bias, parsedHighs, parsedLows, times, self.internalOrderBlocks)







        # Check for open trades using MT5 directly
        open_positions = mt5.positions_get(symbol="XAUUSD")
        open_trades = len(open_positions) if open_positions else 0
        print(open_trades)










        # Order Block Logic
        if self.internalOrderBlocks and self.is_trading_time and fast_strong and (not fast_low_vol) and (open_trades == 0) and self.is_volatile_market:
            orderBlock = self.internalOrderBlocks[0]
           
            rawATR = self.atr14.iloc[-1] or 0
            dynamicStopLoss = max(abs(rawATR) * self.stopLossATRMultiplier, 1e-6)
            if len(self.data.Close) == 0:
                return
            if trendDirection == "BULLISH" and self.data.Close.iloc[-1] < orderBlock.barLow:
                entryPrice = self.data.Close.iloc[-1]
                stopLossPrice = entryPrice - dynamicStopLoss * 1.2
                targetPrice = entryPrice + dynamicStopLoss * 3.4
                send_order("XAUUSD", self.lot, entryPrice, stopLossPrice, targetPrice, is_buy=True)
            elif trendDirection == "BEARISH" and self.data.Close.iloc[-1] > orderBlock.barHigh:
                entryPrice = self.data.Close.iloc[-1]
                stopLossPrice = entryPrice + dynamicStopLoss * 1.2
                targetPrice = entryPrice - dynamicStopLoss * 3.4
                send_order("XAUUSD", self.lot, entryPrice, stopLossPrice, targetPrice, is_buy=False)







        if not fvg_data.empty:
                self.fairValueGaps = []
                for _, gap in fvg_data.iterrows():
                    self.fairValueGaps.append({
                        'is_bull': gap['is_bull'],
                        'is_bear': gap['is_bear'],
                        'top': gap['gap_top'],
                        'bottom': gap['gap_bottom']
                    })




        # FVG Logic
        if fast_fvg and self.is_trading_time and fast_strong and (open_trades == 0) and self.is_volatile_market:
           
            rawATR = self.atr14.iloc[-1] or 0
            dynamicStopLoss = rawATR * self.stopLossATRMultiplier if rawATR > 0 else 1e-6
            dynamicStopLoss = max(abs(rawATR) * self.stopLossATRMultiplier, 1e-6)
            if trendDirection == "BULLISH":
                for gap in fvg_data.to_dict('records'):
                    if self.data.Close.iloc[-1] > gap['gap_bottom'] and self.data.Close.iloc[-1] < gap['gap_top'] and open_trades == 0:
                        entryPrice = gap['gap_bottom'] + self.spreadPips
                        stopLossPrice = entryPrice - dynamicStopLoss * 1.2
                        targetPrice = entryPrice + dynamicStopLoss * 3.4
                        send_order("XAUUSD", self.lot, entryPrice, stopLossPrice, targetPrice, is_buy=True)
                        break
            else:
                for gap in fvg_data.to_dict('records'):
                    if self.data.Close.iloc[-1] < gap['gap_top'] and self.data.Close.iloc[-1] > gap['gap_bottom'] and open_trades == 0:
                        entryPrice = gap['gap_top'] - self.spreadPips
                        stopLossPrice = entryPrice + dynamicStopLoss * 1.2
                        targetPrice = entryPrice - dynamicStopLoss * 3.4
                        send_order("XAUUSD", self.lot, entryPrice, stopLossPrice, targetPrice, is_buy=False)
                        break

# **********************************************************************************

# here we are checking does it take retest from higher time frame

# **********************************************************************************
# **********************************************************************************
def checkHigherTimeframeRetest() -> bool:
    """
    Verifies if current price is within recent high-low ranges
    on higher timeframes (H1, H4).
    """
    data_1h = get_historical_data("XAUUSD", mt5.TIMEFRAME_H1, bars=10)
    data_4h = get_historical_data("XAUUSD", mt5.TIMEFRAME_H4, bars=10)
    if data_1h.empty or data_4h.empty:
        return False
    high1h = data_1h['high'].max()
    low1h = data_1h['low'].min()
    high4h = data_4h['high'].max()
    low4h = data_4h['low'].min()
    current_close = get_historical_data("XAUUSD", mt5.TIMEFRAME_M5, bars=1)['close'].iloc[-1]
    retest1h = (current_close < high1h) and (current_close > low1h)
    retest4h = (current_close < high4h) and (current_close > low4h)
    return retest1h or retest4h

# **********************************************************************************

# here we are checking does it take retest from Fair Value Gap

# **********************************************************************************
# **********************************************************************************
def checkFairValueGapRetest() -> bool:
    """
    Checks M5 for Fair Value Gaps and verifies if
    current price is within any recent gap ranges.
    """
    data = get_historical_data("XAUUSD", mt5.TIMEFRAME_M5, bars=70)
    if data.empty:
        return False
    fvg_result = FVG.fvg(data)
    fvg_data = pd.DataFrame({
        "is_bull": fvg_result["is_bull"],
        "is_bear": fvg_result["is_bear"],
        "gap_top": fvg_result["gap_top"],
        "gap_bottom": fvg_result["gap_bottom"]
    })
    if fvg_data.empty:
        return False
    current_close = get_historical_data("XAUUSD", mt5.TIMEFRAME_M5, bars=1)['close'].iloc[-1]
    for _, gap in fvg_data.tail(5).iterrows():
        if gap["gap_bottom"] <= current_close <= gap["gap_top"]:
            return True
    return False

# **********************************************************************************

# here we are checking does it strong trend

# **********************************************************************************
# **********************************************************************************
def isStrongTrend() -> bool:
    """
    Checks if the current trend is strong based on EMA50 and EMA200.
    """
    df = get_historical_data("XAUUSD", mt5.TIMEFRAME_M5, bars=300)
    if df.empty:
        return False
    ema50 = df['close'].ewm(span=50).mean().iloc[-1]
    ema200 = df['close'].ewm(span=200).mean().iloc[-1]
    return ema50 > ema200

# **********************************************************************************

# here we are checking does it low volatility

# **********************************************************************************
# **********************************************************************************
def isLowVolatility() -> bool:
    """
    Checks if the current market condition is of low volatility
    based on ATR14 and its 50-period EMA.
    """
    df = get_historical_data("XAUUSD", mt5.TIMEFRAME_M5, bars=300)
    if df.empty:
        return False
    atr14 = calculate_atr(df, period=14)
    atr_ema50 = atr14.ewm(span=50).mean()
    return atr14.iloc[-1] < atr_ema50.iloc[-1]

# **********************************************************************************

# here we are checking does it higher time frame trend

# **********************************************************************************
# **********************************************************************************
def getHigherTimeframeTrend() -> str:
    """
    Determines the higher timeframe trend based on SMA50 and SMA200.
    """
    df = get_historical_data("XAUUSD", mt5.TIMEFRAME_D1, bars=300)
    if df.empty:
        return "BULLISH"
    sma50 = df['close'].rolling(50).mean().iloc[-1]
    sma200 = df['close'].rolling(200).mean().iloc[-1]
    return "BULLISH" if sma50 > sma200 else "BEARISH"

# **********************************************************************************

# Main function to run the live strategy

# **********************************************************************************
# **********************************************************************************
def monitor_and_trade():
    """
    Creates a LiveStrategy1 instance, initializes it,
    and executes 'next' for scanning trade opportunities.
    Uses the complete Strategy1 logic for trade decisions.
    """
    strategy = LiveStrategy1()
    strategy.init()
    strategy.next()

def main(stop_threads):
    """
    Orchestrates the live trading workflow:
    - Defines symbol and lot size.
    - Continuously checks conditions.
    - Runs LiveStrategy1 logic in a loop.
    """
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M5
    lot_size = 0.03

    # Initialize MetaTrader5
    if not mt5.initialize():
        print("Failed to initialize MetaTrader5. Make sure it's installed and running.")
        return

    try:
        i = 0
        while not stop_threads.is_set():  # Check the stop_threads flag
            try:
                # Run the strategy directly instead of checking conditions separately
                strategy = LiveStrategy1()
                strategy.init()
                strategy.next()

                # Status reporting for monitoring
                table_data = [
                    ["Condition", "Status"],
                    ["Higher Timeframe Retest", "Checked in strategy logic"],
                    ["Fair Value Gap Retest", "Checked in strategy logic"],
                    ["Is 5m Timeframe", "Yes"],
                    ["Is Strong Trend", str(strategy.ema50.iloc[-1] > strategy.ema200.iloc[-1])],
                    ["Is Volatile Market", str(strategy.is_volatile_market)],
                    ["Trend Direction", "BULLISH" if strategy.ema50.iloc[-1] > strategy.ema200.iloc[-1] else "BEARISH"],
                    ["ADX Value", str(calculate_adx(pd.DataFrame({'high': strategy.data.High.iloc[-100:],
                                                                'low': strategy.data.Low.iloc[-100:],
                                                                'close': strategy.data.Close.iloc[-100:]}),
                                                  period=14).iloc[-1])],
                    ["Is Trading Time", str(strategy.is_trading_time)],
                    ["Order Blocks", str(len(strategy.internalOrderBlocks))],
                    ["Fair Value Gaps", str(len(strategy.fairValueGaps))]
                ]

                current_local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                table_data.insert(0, ["Local Time", current_local_time])
                print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))
                print(f"Loop {i} completed.")
            except Exception as e:
                print(f"Error in strategy execution: {e}")
                import traceback
                traceback.print_exc()

            i += 1
            time.sleep(60)  # Check market every minute
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