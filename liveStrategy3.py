#*********************************************************************************
"""

LiveStrategy3 for XAUUSD (M5), adapted from Strategy3 logic.

Major Changes:
- Adapted Strategy3 logic for live trading.
- Replaced backtesting-specific methods with live trading placeholders.
- Followed the format and structure of LiveStrategy2.
"""
#*********************************************************************************










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
FVG = load_module("FVG", "../FVG.py")  # Add this line to dynamically load FVG
OrderBlock = load_module("OrderBlock", "../OrderBlock.py")
Bos_Choch = load_module("Bos_Choch", "../Bos_Choch.py")
LiquiditySweeps = load_module("LiquiditySweeps", "../LiquiditySweeps.py")
Chart_pattern = load_module("Chart_pattern", "../Chart_pattern.py")
InternalOrderBlock = load_module("InternalOrderBlock", "../InternalOrderBlock.py")
SupportResistanceSignalMTF = load_module("SupportResistanceSignalMTF", "../SupportResistanceSignalMTF.py")
main = load_module("main", "../main.py")

# Import functions and classes from dynamically loaded modules
from function import (
    calculate_rsi, calculate_ema, calculate_atr, calculate_adx
)
from FVG import FVG  # Ensure this import works
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pytz
from datetime import time as dtime
from function import calculate_atr, calculate_adx
from LiquiditySweeps import detect_liquidity_sweeps
from Bos_Choch import market_structure_fractal
from main import get_historical_data, send_order
from tabulate import tabulate
import time
















#*********************************************************************************

# Implementation of LiveStrategy3 for live trading

#*********************************************************************************


class LiveStrategy3:
    """
    LiveStrategy3 extends the real-time execution of Strategy3 logic in a
    live market environment. Combines liquidity zone detection, support/resistance checks,
    ADX/RSI filters, and EMA trend confirmations for live order placement.
    """









    #**********************************************************************************

    # Initialization method

    #**********************************************************************************


    def init(self, symbol="XAUUSD", data=None):
        """
        Initialization with crucial variables for live trading.
        """
        self.symbol = symbol
        self.is_trading_time = False
        self.liquidity_zones = []
        self.support_resistance_levels = []
        self.trades = []  # Initialize empty trades list
        self.is_volatile_market = False  # Initialize volatility flag

        # Fetch historical data
        self.data = type('DataContainer', (), {})()
        self.data.df = get_historical_data(symbol, mt5.TIMEFRAME_M5, bars=17280)
        self.data.Close = self.data.df['close']
        self.data.High = self.data.df['high']
        self.data.Low = self.data.df['low']
        self.data.Open = self.data.df['open']

        # Initialize indicators
        self.ema20 = self.data.df['close'].ewm(span=20, adjust=False).mean()
        self.ema50 = self.data.df['close'].ewm(span=50, adjust=False).mean()
        self.ema200 = self.data.df['close'].ewm(span=200, adjust=False).mean()
        self.atr14 = calculate_atr(self.data.df, period=14)
        self.adx14 = calculate_adx(self.data.df, period=14)
        self.rsi14 = self.compute_rsi(self.data.df['close'], period=14)
        self.is_trading_time = False
        self.lot = 0.03
        











    def compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Computes RSI for the given series and period (without talib).
        """
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ema_up = up.ewm(span=period, adjust=False).mean()
        ema_down = down.ewm(span=period, adjust=False).mean()
        rs = ema_up / ema_down
        return 100 - (100 / (1 + rs))












    #**********************************************************************************

    # Main method for each new bar/step in live trading

    #**********************************************************************************

    def next(self):
        """
        Implements live trading logic based on Strategy3:
        - Applies ADX, RSI filters, and EMA trend confirmations.
        - Detects liquidity zones and support/resistance levels.
        - Places live orders accordingly.
        """



        #***********************************************************************************

        # Check if current time is within trading hours (Indian market hours)

        #***********************************************************************************

        #*********************** Check trading time ************************************
        local_time = time.localtime()
        current_dt = dtime(local_time.tm_hour, local_time.tm_min)
        if not ((dtime(8, 30) <= current_dt <= dtime(12, 30)) or (dtime(17, 30) <= current_dt <= dtime(20, 30))):
            self.is_trading_time = False
        else :
            self.is_trading_time = True










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









        # Check for open trades using MT5 directly
        open_positions = mt5.positions_get(symbol=self.symbol)
        open_trades = len(open_positions) if open_positions else 0
        print(open_trades)








        #***********************************************************************************

        # Trend and signal confirmation

        #***********************************************************************************

        ema20_last = self.ema20.iloc[-1]
        ema50_last = self.ema50.iloc[-1]
        ema200_last = self.ema200.iloc[-1]
        atr14_last = self.atr14.iloc[-1]
        current_price = self.data.df['close'].iloc[-1]
        last_5_closes = self.data.Close[-5:]
        any_close_below_lower = False
        any_close_above_upper = False
        zone_lower = None
        zone_upper = None


        # Determine trend based on EMA crossover
        trend_direction = "BULLISH" if ema20_last > ema50_last else "BEARISH"

        # # Additional trend filter: for bullish signals, EMA20 must be above EMA200; for bearish signals, EMA20 below EMA200.
        if trend_direction == "BULLISH" and ema20_last < self.ema200.iloc[-1]:
            return
        if trend_direction == "BEARISH" and ema20_last > self.ema200.iloc[-1]:
            return
        

        # Prepare processed data
        if not hasattr(self, '_processed_data'):
            self._processed_data = self.data.df.copy()
            self._processed_data.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}, inplace=True)
            self._processed_data['time'] = self._processed_data.index

        # Detect liquidity zones (only once per bar)
        if not hasattr(self, '_liquidity_data') or self._liquidity_data.index.iloc[-1] != self.data.index.iloc[-1]:
            self._liquidity_data = detect_liquidity_sweeps(self._processed_data)
            if not self._liquidity_data.empty:
                self.liquidity_zones = self._liquidity_data.to_dict('records')
            else:
                self.liquidity_zones = []


        # Detect support/resistance levels (only once per bar)
        if not hasattr(self, '_support_resistance_data') or self._support_resistance_data.index[-1] != self.data.index[-1]:
            self._support_resistance_data = market_structure_fractal(
                self._processed_data, length=5, show_support=True, show_resistance=True
            )
            if not self._support_resistance_data.empty:
                self.support_resistance_levels = self._support_resistance_data.tail(5).to_dict('records')
            else:
                self.support_resistance_levels = []




        if self.liquidity_zones:
            sample_zone = self.liquidity_zones[0]
            zone_lower = sample_zone.get('low', None)
            print(zone_lower)
            zone_upper = sample_zone.get('high', None)
            print(zone_upper)
            if zone_lower is not None:
                any_close_below_lower = any(close < zone_lower for close in last_5_closes)
            if zone_upper is not None:
                any_close_above_upper = any(close > zone_upper for close in last_5_closes)

        
        

        
        if not self.liquidity_zones and not self.support_resistance_levels:
            return
        if len(self.atr14) == 0 or self.atr14.iloc[-1] is None:
            return

        
        

        table_data = []

        # Exactly match Strategy3 logic
        if trend_direction == "BULLISH" and zone_lower is not None and current_price < zone_lower and any_close_below_lower and self.is_trading_time and (open_trades == 0) and self.adx14.iloc[-1] < 55 and self.is_volatile_market:
            entry_price = current_price
            # Zone-Bonus: adjust SL/TP for BUY to allow a higher profit target (exactly as in Strategy3)
            if (zone_lower - current_price) < (0.2 * atr14_last):
                stop_loss = entry_price - 2.3 * atr14_last
                target_price = entry_price + 5.1 * atr14_last
            else:
                stop_loss = entry_price - 2.5 * atr14_last
                target_price = entry_price + 5.4 * atr14_last
            send_order(self.symbol, self.lot, entry_price, stop_loss, target_price, is_buy=True)
            table_data.append(["Buy Signal (Liquidity Zone)", entry_price, stop_loss, target_price])
        elif trend_direction == "BEARISH" and zone_upper is not None and current_price > zone_upper and any_close_above_upper and self.is_trading_time and (open_trades == 0) and self.adx14.iloc[-1] < 55 and self.is_volatile_market:
            entry_price = current_price
            # Zone-Bonus: adjust SL/TP for SELL to allow a higher profit target (exactly as in Strategy3)
            if (current_price - zone_upper) < (0.2 * atr14_last):
                stop_loss = entry_price + 2.3 * atr14_last
                target_price = entry_price - 5.1 * atr14_last
            else:
                stop_loss = entry_price + 2.5 * atr14_last
                target_price = entry_price - 5.4 * atr14_last
            send_order(self.symbol, self.lot, entry_price, stop_loss, target_price, is_buy=False)
            table_data.append(["Sell Signal (Liquidity Zone)", entry_price, stop_loss, target_price])

      
        # Print trade details in a table format
        if table_data:
            print(tabulate(table_data, headers=["Signal", "Entry", "SL", "TP"], tablefmt="grid"))


# **********************************************************************************

# Main function to run the live strategy

# **********************************************************************************
def main(stop_threads):
    """
    Orchestrates the live trading workflow:
    - Defines symbol and lot size.
    - Continuously checks conditions.
    - Runs LiveStrategy3 logic in a loop.
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
                strategy = LiveStrategy3()
                strategy.init(symbol, data)
                strategy.next()

                # Status reporting for monitoring
                table_data = [
                    ["Condition", "Status"],
                    ["Trading Time", str(strategy.is_trading_time)],
                    ["Liquidity Zones Count", str(len(strategy.liquidity_zones))],
                    ["Support/Resistance Levels", str(len(strategy.support_resistance_levels))],
                    ["Is Volatile Market", str(strategy.is_volatile_market)]
                ]

                # Safely add ADX value
                adx_value = "N/A"
                if isinstance(strategy.adx14, pd.Series) and not strategy.adx14.empty:
                    adx_value = f"{strategy.adx14.iloc[-1]:.2f}"
                elif isinstance(strategy.adx14, np.ndarray) and len(strategy.adx14) > 0:
                    adx_value = f"{strategy.adx14[-1]:.2f}"
                table_data.append(["ADX", adx_value])

                current_local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                table_data.insert(0, ["Local Time", current_local_time])
                print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))

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
