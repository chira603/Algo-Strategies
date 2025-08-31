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

        # Initial Sell: set SL above entry
        entry_price = tick.bid  # use bid for sell order
        initial_lot = 0.05
        sell_sl = entry_price + 4.6 * pip
        req_sell = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": initial_lot,
            "type": mt5.ORDER_TYPE_SELL,
            "price": entry_price,
            "deviation": 10,
            "magic": 234000,
            "comment": "Initial Sell",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res_sell = mt5.order_send(req_sell)
        if res_sell.retcode != mt5.TRADE_RETCODE_DONE:
            print("Initial Sell order error:", res_sell)
            close_all_positions()
            continue
        last_sell = {"ticket": res_sell.order, "entry": entry_price, "lot": initial_lot, "sl": sell_sl}
        first_sell_entry = entry_price          # store initial sell price
        last_buy = None                         # initially no reversal buy order
        first_buy_entry = None
        last_side = "sell"                      # track last executed order side
        print(f"Placed Initial Sell at {entry_price} with SL at {sell_sl}")

        # Monitoring loop for the cycle
        while True:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                time.sleep(1)
                continue
            current_bid = tick.bid
            current_ask = tick.ask

            # SL conditions (adjusted for reverse hedging)
            if last_sell and current_ask >= last_sell["sl"]:
                print("SL reached for Sell order.")
                break
            if last_buy and current_bid <= last_buy["sl"]:
                print("SL reached for Buy order.")
                break

            # Reversal Buy condition (mapped from reversal sell): set SL below entry.
            if last_side == "sell" and first_buy_entry is None and current_ask <= last_sell["entry"] - 1 * pip:
                buy_entry = current_ask
                buy_lot = 0.08
                buy_sl = buy_entry - 4.6 * pip
                req_buy = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": buy_lot,
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": buy_entry,
                    "deviation": 10,
                    "magic": 234000,
                    "comment": "Reversal Buy",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                res_buy = mt5.order_send(req_buy)
                if res_buy.retcode != mt5.TRADE_RETCODE_DONE:
                    print("Reversal Buy order error:", res_buy)
                    close_all_positions()
                    break
                last_buy = {"ticket": res_buy.order, "entry": buy_entry, "lot": buy_lot, "sl": buy_sl}
                first_buy_entry = buy_entry    # store first reversal buy price
                last_side = "buy"              # update last order side
                print(f"Placed Reversal Buy at {buy_entry} with SL at {buy_sl}")

            # Subsequent Sell trigger (mapped from original Subsequent Buy): SL above entry.
            if last_buy and last_side == "buy" and mt5.symbol_info_tick(symbol).bid >= first_sell_entry:
                new_sell_entry = mt5.symbol_info_tick(symbol).bid
                new_sell_lot = math.floor(last_buy["lot"] * 1.33 * 100) / 100.0
                new_sell_sl = new_sell_entry + 4.6 * pip
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
                    close_all_positions()
                    break
                last_sell = {"ticket": res_sell.order, "entry": new_sell_entry, "lot": new_sell_lot, "sl": new_sell_sl}
                last_side = "sell"
                print(f"Placed Subsequent Sell at {new_sell_entry} with SL at {new_sell_sl}")

            # Subsequent Buy trigger (mapped from original Subsequent Sell): SL below entry.
            if last_sell and first_buy_entry and last_side == "sell" and current_ask <= first_sell_entry - 1 * pip:
                new_buy_entry = current_ask
                new_buy_lot = math.floor(last_sell["lot"] * 1.33 * 100) / 100.0
                new_buy_sl = new_buy_entry - 4.6 * pip
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
                last_buy = {"ticket": res_buy.order, "entry": new_buy_entry, "lot": new_buy_lot, "sl": new_buy_sl}
                last_side = "buy"
                print(f"Placed Subsequent Buy at {new_buy_entry} with SL at {new_buy_sl}")

            time.sleep(1)
        # On SL or error, close positions and restart cycle.
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
