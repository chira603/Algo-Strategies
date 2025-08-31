import sys
import os
import importlib.util
import numpy as np
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
from Bos_Choch import market_structure_fractal
from LiquiditySweeps import detect_liquidity_sweeps
from Chart_pattern import (
    detect_head_shoulder,
    detect_multiple_tops_bottoms,
    calculate_support_resistance,
    detect_triangle_pattern,
    detect_wedge,
    detect_channel,
    detect_double_top_bottom,
    detect_trendline,
    find_pivots
)
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


class LiveStrategy4():
    """
    LiveStrategy4 extends the backtesting 'Strategy' base class to implement
    live trading logic for Strategy4. It includes real-time data handling,
    dynamic stop-loss adjustments, and trading time checks.
    """
    def I(self, func, *args):
            """
            Proxy method to wrap indicator functions.
            """
            return func(*args)
    








    def init(self):
        """
        Initializes technical indicators and key parameters for live trading:
        - EMA20, EMA50, EMA200: Exponential moving averages for trend detection.
        - ATR14: Average True Range for volatility-based stop loss.
        - internalOrderBlocks, fairValueGaps: Lists to store discovered support/resistance data.
        - is_trading_time: Boolean to allow or forbid trades based on time blocks.
        - stopLossATRMultiplier: Multiplier for ATR-based stop losses.
        - spreadPips: Spread in pips to adjust entry prices.
        - minADX: Minimum ADX threshold for filtering weak-trend markets.
        """
        self.ema20 = self.I(lambda c: pd.Series(c).ewm(span=20).mean(), self.data.Close)
        self.ema50 = self.I(lambda c: pd.Series(c).ewm(span=50).mean(), self.data.Close)
        self.ema200 = self.I(lambda c: pd.Series(c).ewm(span=200).mean(), self.data.Close)
        self.atr14 = self.I(lambda h, l, c: calculate_atr(pd.DataFrame({'high': h, 'low': l, 'close': c}), period=14),
                            self.data.High, self.data.Low, self.data.Close)
        self.adx14 = self.I(lambda h, l, c: calculate_adx(pd.DataFrame({'high': h, 'low': l, 'close': c}), period=14),
                            self.data.High, self.data.Low, self.data.Close)
        self.internalOrderBlocks = []
        self.fairValueGaps = []
        self.is_trading_time = False
        self.is_volatile_market = False
        self.stopLossATRMultiplier = 2.0
        self.spreadPips = 0
        self.minADX = 20
        self.lot = 0.03
        self.trades = []  # Initialize empty trades list

    def next(self):
        """
        Core method executed on each new bar during live trading.
        - Checks for Fair Value Gaps (FVG) and internal order blocks.
        - Confirms trading time windows and trend strength.
        - Places trades based on detected signals.
        """


        # Check trading time
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





        ema20_last = self.ema20.iloc[-1]
        ema50_last = self.ema50.iloc[-1]
        ema200_last = self.ema200.iloc[-1]
        atr14_last = self.atr14.iloc[-1]
        current_price = self.data.Close.iloc[-1]
        
       




        # Determine trend direction
        trend_direction = "BULLISH" if ema20_last > ema200_last else "BEARISH"

        




        # Ensure the DataFrame has the required columns
        if not hasattr(self, '_processed_data'):
            self._processed_data = self.data.df.copy()
            self._processed_data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)
            self._processed_data['time'] = self._processed_data.index  # Ensure 'time' column exists

        

        





        # Check for open trades using MT5 directly
        open_positions = mt5.positions_get(symbol="XAUUSD")
        open_trades = len(open_positions) if open_positions else 0
        print(open_trades)
        table_data = []

       









        # Exactly match Strategy4 logic
        if (open_trades == 0) and self.adx14.iloc[-1] < 40 and self.is_volatile_market:
            if trend_direction == "BULLISH" and current_price > ema20_last and self.rsi14.iloc[-1] > 60:
                entry_price = current_price
                stop_loss = entry_price - 3 * atr14_last
                target_price = entry_price + 7.4 * atr14_last
                send_order("XAUUSD", self.lot, entry_price, stop_loss, target_price, is_buy=True)
                table_data.append(["Buy Signal (EMA Fallback)", entry_price, stop_loss, target_price])
                # print(f"Buy Signal (EMA Fallback): Entry={entry_price}, SL={stop_loss}, TP={target_price}")
            elif trend_direction == "BEARISH" and current_price < ema20_last and self.rsi14.iloc[-1] < 40:
                entry_price = current_price
                stop_loss = entry_price + 3 * atr14_last
                target_price = entry_price - 7.4 * atr14_last
                send_order("XAUUSD", self.lot, entry_price, stop_loss, target_price, is_buy=False)  # Fixed is_buy=False for sell
                table_data.append(["Sell Signal (EMA Fallback)", entry_price, stop_loss, target_price])
                # print(f"Sell Signal (EMA Fallback): Entry={entry_price}, SL={stop_loss}, TP={target_price}")
            if table_data:
                print(tabulate(table_data, headers=["Signal", "Entry", "SL", "TP"], tablefmt="grid"))
                # Clear table_data after printing to avoid duplicate prints
                table_data = []
        # ----- End of Fallback Entry -----









        
def main(stop_threads):
    """
    Orchestrates the live trading workflow for Strategy4:
    - Defines symbol and timeframe parameters
    - Continuously fetches market data and runs strategy
    - Displays current market conditions and strategy status
    """
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M5

    # Initialize MetaTrader5
    if not mt5.initialize():
        print("Failed to initialize MetaTrader5. Make sure it's installed and running.")
        return

    try:
        i = 0
        while not stop_threads.is_set():
            try:
                # Fetch recent market data
                df = get_historical_data(symbol, timeframe, 300)
                if df is None or df.empty:
                    print("Failed to get market data, retrying in 60 seconds...")
                    time.sleep(60)
                    continue

                # Initialize strategy with current data
                strategy = LiveStrategy4()

                # Create data object structure that strategy expects
                class DataWrapper:
                    def __init__(self, dataframe):
                        self.df = dataframe
                        # Make sure the index is DatetimeIndex
                        if not isinstance(dataframe.index, pd.DatetimeIndex):
                            try:
                                dataframe.index = pd.to_datetime(dataframe.index)
                            except:
                                # If conversion fails, create a new index
                                dataframe.index = pd.date_range(start=pd.Timestamp.now(), periods=len(dataframe), freq='5min')

                        # Store both as Series and numpy arrays for flexibility
                        self.Close = dataframe['close']
                        self.Open = dataframe['open']
                        self.High = dataframe['high']
                        self.Low = dataframe['low']

                strategy.data = DataWrapper(df)
                strategy.init()

                # Calculate missing indicators referenced in next()
                strategy.rsi14 = calculate_rsi(df['close'], 14)
                strategy.adx14 = calculate_adx(df, 14)

                # Run strategy logic
                strategy.next()

                # Status reporting for monitoring
                table_data = [
                    ["Condition", "Status"],
                ]

                # Safe access to indicators for display
                if isinstance(strategy.ema20, pd.Series):
                    table_data.append(["Check current_price > ema20 (C1)", str(strategy.data.Close.iloc[-1] > strategy.ema20.iloc[-1])])
                elif isinstance(strategy.ema20, np.ndarray):
                    table_data.append(["Check current_price > ema20 (C1)", str(strategy.data.Close.iloc[-1] > strategy.ema20[-1])])

                if isinstance(strategy.ema50, pd.Series):
                    table_data.append(["EMA50", f"{strategy.ema50.iloc[-1]:.2f}"])
                elif isinstance(strategy.ema50, np.ndarray):
                    table_data.append(["EMA50", f"{strategy.ema50[-1]:.2f}"])

                if isinstance(strategy.ema200, pd.Series):
                    table_data.append(["EMA200", f"{strategy.ema200.iloc[-1]:.2f}"])
                elif isinstance(strategy.ema200, np.ndarray):
                    table_data.append(["EMA200", f"{strategy.ema200[-1]:.2f}"])


                
                if isinstance(strategy.rsi14, pd.Series):
                    table_data.append(["Check rsi14 > 60 (C3)", str(strategy.rsi14.iloc[-1] > 60)])
                
                if isinstance(strategy.rsi14, pd.Series):
                    table_data.append(["Check rsi14 < 40 (C3)", str(strategy.rsi14.iloc[-1] < 40)])

                if isinstance(strategy.adx14, pd.Series):
                    table_data.append(["Check adx14 < 40 (C2)", str(strategy.adx14.iloc[-1] < 40)])
                elif isinstance(strategy.adx14, np.ndarray):
                    table_data.append(["Check adx14 < 40 (C2)", str(strategy.adx14[-1] < 40)])

                if isinstance(strategy.atr14, pd.Series):
                    table_data.append(["ATR", f"{strategy.atr14.iloc[-1]:.2f}"])
                elif isinstance(strategy.atr14, np.ndarray):
                    table_data.append(["ATR", f"{strategy.atr14[-1]:.2f}"])

                # Determine trend direction safely
                if isinstance(strategy.ema20, pd.Series) and isinstance(strategy.ema200, pd.Series):
                    trend = "BULLISH" if strategy.ema20.iloc[-1] > strategy.ema200.iloc[-1] else "BEARISH"
                elif isinstance(strategy.ema20, np.ndarray) and isinstance(strategy.ema200, np.ndarray):
                    trend = "BULLISH" if strategy.ema20[-1] > strategy.ema200[-1] else "BEARISH"
                else:
                    trend = "UNKNOWN"

                table_data.append(["Trend Direction", trend])
                table_data.append(["Is Volatile Market", str(strategy.is_volatile_market)])
                table_data.append(["Is Trading Time", str(strategy.is_trading_time)])

                if isinstance(df['close'], pd.Series):
                    table_data.append(["Current Price", f"{df['close'].iloc[-1]:.2f}"])
                else:
                    table_data.append(["Current Price", f"{df['close'][-1]:.2f}"])

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