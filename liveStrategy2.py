"""
LiveStrategy2 for XAUUSD (M5), adapted from Strategy2 logic.

Major Changes:
- Refactored variable names (trend_bias).
- Shortened docstrings and consolidated FVG logic.
- Maintained real order calls placeholders.
"""

#*********************************************************************************
# Import necessary libraries
#*********************************************************************************

import sys
import os
import importlib.util
import logging

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
import pytz

from datetime import time as dtime
import time
from tabulate import tabulate

from main import get_historical_data, send_order
























#*********************************************************************************
# Implementation of LiveStrategy2 for live trading
#*********************************************************************************




class LiveStrategy2:
    """
    LiveStrategy2 adapts the backtest logic of Strategy2 for live trading.
    It combines Fair Value Gap (FVG) detection, support/resistance checks,
    and trend filters, and then places live orders using send_order.
    """
















    def init(self, symbol="XAUUSD", data=None):
        """
        Initialization with crucial variables for live trading.
        """
        self.symbol = symbol
        self.is_trading_time = False
        # Retrieve live historical M5 data
        self.data = type('DataContainer', (), {})()
        self.data.df = get_historical_data(symbol, mt5.TIMEFRAME_M5, bars=17280)  # Use sufficient bars for indicators
        self.data.Close = self.data.df['close']
        self.data.High = self.data.df['high']
        self.data.Low = self.data.df['low']
        self.data.Open = self.data.df['open']
        # Initialize indicators (EMA, ATR, ADX, RSI)
        self.ema20 = self.data.df['close'].ewm(span=20, adjust=False).mean()
        self.ema50 = self.data.df['close'].ewm(span=50, adjust=False).mean()
        self.ema200 = self.data.df['close'].ewm(span=200, adjust=False).mean()
        self.atr14 = calculate_atr(self.data.df, period=14)
        self.adx14 = calculate_adx(self.data.df, period=14)
        # Compute RSI using the imported calculate_rsi function
        self.rsi14 = calculate_rsi(self.data.df['close'], period=14)
        # Initialize lists for FVG and support/resistance data
        self.fvg_data = []
        self.support_resistance_levels = []
        # Set lot size and stop loss multiplier
        self.lot = 0.03
        self.stopLossATRMultiplier = 2.0
        self.is_volatile_market = False
        self.trades = []  # Initialize empty trades list for tracking open positions













    def next(self):
        """
        Executes live trading logic based on Strategy2.
        Steps:
         1. Check trading time.
         2. Determine trend bias (from EMA comparisons).
         3. Detect FVG markers and support/resistance levels.
         4. Place live orders via send_order based on signals.
        """





        #*********************** Check trading time ************************************
        local_time = time.localtime()
        current_dt = dtime(local_time.tm_hour, local_time.tm_min)
        if not ((dtime(8, 30) <= current_dt <= dtime(12, 30)) or (dtime(17, 30) <= current_dt <= dtime(20, 30))):
            self.is_trading_time = False
        else : 
            self.is_trading_time = True












        # --- Trend Bias ---
        ema20_last = self.ema20.iloc[-1]
        ema50_last = self.ema50.iloc[-1]
        ema200_last = self.ema200.iloc[-1]
        trend_bias = "BULLISH" if ema20_last > ema200_last else "BEARISH"
        current_price = self.data.Close.iloc[-1]
        atr14_last = self.atr14.iloc[-1]
        rsi_last = self.rsi14.iloc[-1]









        # --- Process Data for FVG and S/R Detection ---
        # Prepare a processed copy with standard column names
        if not hasattr(self, '_processed_data'):
            self._processed_data = self.data.df.copy()
            self._processed_data.rename(columns={'open': 'open', 'high': 'high',
                                                  'low': 'low', 'close': 'close'}, inplace=True)
            self._processed_data['time'] = self._processed_data.index













        # Detect FVG markers using the FVG module
        fvg_markers = FVG.fvg(self._processed_data)
        if fvg_markers:
            self.fvg_data = [
                {
                    "gap_bottom": gap["gap_bottom"],
                    "gap_top": gap["gap_top"],
                    "is_bull": gap["is_bull"],
                    "is_bear": gap["is_bear"]
                }
                for gap in fvg_markers.get("gaps", [])
            ]
        else:
            self.fvg_data = []
















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
       
      








        # ADX filter
        adx_value = self.adx14.iloc[-1]
        if adx_value < 20:
            return












       
        # Check for open trades using MT5 directly
        open_positions = mt5.positions_get(symbol=self.symbol)
        open_trades = len(open_positions) if open_positions else 0
        print(open_trades)








        # --- Live Order Execution Based on Signals ---
        table_data = []
        # Example: FVG-based signals (add RSI checks)
        for gap in self.fvg_data:
            # For live orders, we mimic Strategy2's conditions exactly:
            if  gap["is_bull"] and trend_bias == "BULLISH" and rsi_last > 55 and current_price > gap["gap_top"] and self.is_trading_time and open_trades==0 and self.ema20.iloc[-1]>self.ema50.iloc[-1] and self.is_volatile_market:
                entry_price = current_price   # adding spread
                stop_loss = entry_price - 2.2 * atr14_last
                target_price = entry_price + 6.2 * atr14_last
                send_order(self.symbol, self.lot, entry_price, stop_loss, target_price, is_buy=True, comment="LiveStrategy2 Buy (FVG)")
                table_data.append(["Buy Order (FVG)", entry_price, stop_loss, target_price])
            elif gap["is_bear"] and trend_bias == "BEARISH" and rsi_last < 45 and current_price < gap["gap_bottom"] and self.is_trading_time and open_trades==0 and self.ema20.iloc[-1]<self.ema50.iloc[-1] and self.is_volatile_market:
                entry_price = current_price
                stop_loss = entry_price + 2.2 * atr14_last
                target_price = entry_price - 6.2 * atr14_last
                send_order(self.symbol, self.lot, entry_price, stop_loss, target_price, is_buy=False, comment="LiveStrategy2 Sell (FVG)")
                table_data.append(["Sell Order (FVG)", entry_price, stop_loss, target_price])

        # Print trade details for debugging - only print once
        if table_data:
            print(tabulate(table_data, headers=["Signal", "Entry", "SL", "TP"], tablefmt="grid"))
            # Clear table_data after printing to avoid duplicate prints
            table_data = []


















    def run_live(self):
        """
        Summarizes the process for your live trading environment. Invokes next() method.
        """
        self.next()










# **********************************************************************************

# Main function to run the live strategy

# **********************************************************************************




def monitor_and_trade():
    """
    Creates a LiveStrategy1 instance, initializes it,
    and executes 'next' for scanning trade opportunities.
    """
    strategy = LiveStrategy2()
    strategy.init()
    strategy.next()

def main(stop_threads):
    """
    Orchestrates the live trading workflow:
    - Defines symbol and lot size.
    - Continuously checks conditions.
    - Runs LiveStrategy2 logic in a loop.
    """
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M5

    # Initialize MetaTrader5
    if not mt5.initialize():
        logging.error("Failed to initialize MetaTrader5. Make sure it's installed and running.")
        return

    try:
        i = 0
        while not stop_threads.is_set():  # Check the stop_threads flag
            try:
                logging.info("Fetching historical data...")
                data = get_historical_data(symbol, timeframe, bars=300)
                if data.empty:
                    logging.warning("No data fetched. Retrying...")
                    time.sleep(60)
                    continue

                # Initialize and run the strategy
                strategy = LiveStrategy2()
                strategy.init(symbol)
                strategy.next()

                # Print function statuses in a table
                status_table = [
                    ["Condition", "Status"],
                    ["Trading Time", strategy.is_trading_time],
                    ["FVG Data Count", len(strategy.fvg_data)],
                    ["Support/Resistance Levels", len(strategy.support_resistance_levels)],
                    ["Is Volatile Market", str(strategy.is_volatile_market)],
                    ["Trend Direction (Bullish)" , str(strategy.ema20.iloc[-1]>strategy.ema200.iloc[-1])],
                   
                ]
                print(tabulate(status_table, headers="firstrow", tablefmt="fancy_grid"))

                logging.info(f"Loop {i} completed.")
            except Exception as e:
                logging.error(f"Error in strategy execution: {e}")
                import traceback
                traceback.print_exc()

            i += 1
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("User interrupted, stopping.")
        stop_threads.set()
    finally:
        # Shutdown MT5 when done
        mt5.shutdown()

if __name__ == "__main__":
    import threading
    stop_threads = threading.Event()
    try:
        main(stop_threads)
    except KeyboardInterrupt:
        stop_threads.set()
        print("Stopping threads...")
