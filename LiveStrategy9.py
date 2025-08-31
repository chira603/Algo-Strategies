import MetaTrader5 as mt5
import time
import math

def close_all_positions():
    # Closes all open positions.
    positions = mt5.positions_get()
    if positions:
        for pos in positions:
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
                "deviation": 10,
                "magic": 234000,
                "comment": "Close All",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            mt5.order_send(req)
            time.sleep(1)

def new_scalping_strategy(symbol="XAUUSD"):
    pip = 1  # standard pip value for most pairs
    while True:
        # Start each cycle by closing all positions.
        close_all_positions()
        time.sleep(2)
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            print("Tick not available for", symbol)
            time.sleep(1)
            continue

        # Initial Buy
        entry_price = tick.ask
        initial_lot = 0.01
        buy_tp = entry_price + 2.6 * pip
        req_buy = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": initial_lot,
            "type": mt5.ORDER_TYPE_BUY,
            "price": entry_price,
            "deviation": 10,
            "magic": 234000,
            "comment": "Initial Buy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res_buy = mt5.order_send(req_buy)
        if res_buy.retcode != mt5.TRADE_RETCODE_DONE:
            print("Initial Buy order error:", res_buy)
            close_all_positions()
            continue
        last_buy = {"ticket": res_buy.order, "entry": entry_price, "lot": initial_lot, "tp": buy_tp}
        first_buy_entry = entry_price          # store initial buy price
        last_sell = None
        first_sell_entry = None
        # initially no sell order
        last_side = "buy"                      # track last executed order side
        print(f"Placed Initial Buy at {entry_price} with TP at {buy_tp}")

        # Monitoring loop for the cycle
        while True:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                time.sleep(1)
                continue
            current_bid = tick.bid
            current_ask = tick.ask

            # Check TP conditions for the most recent orders.
            if last_buy and current_bid >= last_buy["tp"]:
                print("TP reached for Buy order.")
                break
            if last_sell and current_ask <= last_sell["tp"]:
                print("TP reached for Sell order.")
                break

            # New Reversal Sell condition inside the monitoring loop:
            # If no reversal sell has been executed (first_sell_entry is None) and the price drops below last_buy entry by 1 pip,
            # then execute the reversal sell.
            if last_side == "buy" and first_sell_entry is None and current_bid <= last_buy["entry"] - 1 * pip:
                sell_entry = current_bid
                sell_lot = 0.02
                sell_tp = sell_entry - 2.6 * pip
                req_sell = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": sell_lot,
                    "type": mt5.ORDER_TYPE_SELL,
                    "price": sell_entry,
                    "deviation": 10,
                    "magic": 234000,
                    "comment": "Reversal Sell",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                res_sell = mt5.order_send(req_sell)
                if res_sell.retcode != mt5.TRADE_RETCODE_DONE:
                    print("Reversal Sell order error:", res_sell)
                    close_all_positions()
                    break
                last_sell = {"ticket": res_sell.order, "entry": sell_entry, "lot": sell_lot, "tp": sell_tp}
                first_sell_entry = sell_entry      # store first sell price
                last_side = "sell"                # update last order side
                print(f"Placed Reversal Sell at {sell_entry} with TP at {sell_tp}")

            # Example: Subsequent Buy trigger
            # When the last order was a sell and the current ask price is at or above the initial buy entry,
            # a subsequent buy order is created.
            # For example, if first_buy_entry is 1800.0 and the ask price reaches 1800.0 or higher, a buy is taken.
            if last_sell and last_side == "sell" and mt5.symbol_info_tick(symbol).ask >= first_buy_entry:
                new_buy_entry = mt5.symbol_info_tick(symbol).ask
                new_buy_lot =  0.02 
                new_buy_tp = new_buy_entry + 2.6 * pip
                req_buy = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": new_buy_lot,
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": new_buy_entry,
                    "deviation": 10,
                    "magic": 234000,
                    "comment": "Subsequent Buy",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                res_buy = mt5.order_send(req_buy)
                if res_buy.retcode != mt5.TRADE_RETCODE_DONE:
                    print("Subsequent Buy order error:", res_buy)
                    close_all_positions()
                    break
                last_buy = {"ticket": res_buy.order, "entry": new_buy_entry, "lot": new_buy_lot, "tp": new_buy_tp}
                last_side = "buy"                 # update last order side
                print(f"Placed Subsequent Buy at {new_buy_entry} with TP at {new_buy_tp}")

            # Example: Subsequent Sell trigger
            # When the last order was a buy and the current bid price falls to or below the first sell entry,
            # a subsequent sell order is issued.
            # For instance, if first_sell_entry is 1795.0 and the bid price drops to 1795.0 or lower, a sell is executed.
            if last_buy and first_sell_entry and last_side == "buy" and mt5.symbol_info_tick(symbol).bid <= current_bid <= first_buy_entry - 1 * pip:
                new_sell_entry = mt5.symbol_info_tick(symbol).bid
                new_sell_lot = 0.02
                new_sell_tp = new_sell_entry - 2.6 * pip
                req_sell = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": new_sell_lot,
                    "type": mt5.ORDER_TYPE_SELL,
                    "price": new_sell_entry,
                    "deviation": 10,
                    "magic": 234000,
                    "comment": "Subsequent Sell",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                res_sell = mt5.order_send(req_sell)
                if res_sell.retcode != mt5.TRADE_RETCODE_DONE:
                    print("Subsequent Sell order error:", res_sell)
                last_sell = {"ticket": res_sell.order, "entry": new_sell_entry, "lot": new_sell_lot, "tp": new_sell_tp}
                last_side = "sell"                # update last order side
                print(f"Placed Subsequent Sell at {new_sell_entry} with TP at {new_sell_tp}")

            time.sleep(1)
        # On TP or error, close positions and restart cycle.
        close_all_positions()
        print("Cycle complete. Restarting strategy...")
        time.sleep(2)

if __name__ == "__main__":
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
    try:
        new_scalping_strategy()
    finally:
        mt5.shutdown()
