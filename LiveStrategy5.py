import sys
import os
import importlib.util
import logging

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

class Strategy5:
    def I(self, func, values):
        return func(values)

    def init(self):
        # Fetch minimal bars for quick checks
        self.symbol = "XAUUSD"
        self.is_trading_time = False
        self.lot = 0.01
        self.data = get_historical_data(self.symbol, mt5.TIMEFRAME_M5, 100)
        self.ema20 = self.I(lambda c: pd.Series(c).ewm(span=9).mean(), self.data['close'])
        

    def next(self):
        # Run each minute, place trade if price is above/below short EMA
        current_dt = dtime(time.localtime().tm_hour, time.localtime().tm_min)
        if not ((dtime(12, 30)) or (dtime(17, 30) <= current_dt <= dtime(20, 30))):
            self.is_trading_time = False
            return
        self.is_trading_time = True

        self.current_price = self.data['close'].iloc[-1]
        open_positions = mt5.positions_get(symbol=self.symbol)
        print(open_positions)
        if open_positions:
            # Check profit/loss, exit if in profit
            for pos in open_positions:
                if pos.profit > 1000:  # Example profit threshold
                    request_result = mt5.Close(pos.ticket)  # Hypothetical exit call
                    logging.info(f"Closed position with profit: {request_result}")
            return

        # Place trade logic
        if self.current_price > self.ema20.iloc[-1]:
            stop_loss = self.current_price - 1000 * mt5.symbol_info(self.symbol).point
            target_price = self.current_price + 1200 * mt5.symbol_info(self.symbol).point
            send_order(self.symbol, self.lot, self.current_price, stop_loss, target_price, is_buy=True)
        else:
            stop_loss = self.current_price + 1000 * mt5.symbol_info(self.symbol).point
            target_price = self.current_price - 1200 * mt5.symbol_info(self.symbol).point
            send_order(self.symbol, self.lot, self.current_price, stop_loss, target_price, is_buy=False)

        
def main(stop_threads):
    """
    Orchestrates the live trading workflow:
    - Defines symbol and lot size.
    - Continuously checks conditions.
    - Runs LiveStrategy1 logic in a loop.
    """
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M5
    lot_size = 0.01

    # Initialize MetaTrader5
    if not mt5.initialize():
        print("Failed to initialize MetaTrader5. Make sure it's installed and running.")
        return

    try:
        i = 0
        while not stop_threads.is_set():  # Check the stop_threads flag
            try:
                # Run the strategy directly instead of checking conditions separately
                strategy = Strategy5()
                strategy.init()
                strategy.next()

                # Print a quick summary table
                table_data = [
                    ["Trading Time", str(strategy.is_trading_time)],
                    ["Current Price", f"{strategy.current_price:.2f}"],
                    ["EMA(9)", f"{strategy.ema20.iloc[-1]:.2f}"]
                ]
                print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="fancy_grid", disable_numparse=True))

                current_local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                table_data.insert(0, ["Local Time", current_local_time])
                print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid", disable_numparse=True))
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


