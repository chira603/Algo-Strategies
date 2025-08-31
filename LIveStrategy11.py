import MetaTrader5 as mt5
import time

def new_cascade_strategy(symbol="XAUUSD"):
    pip = 1  # standard pip value
    while True:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            time.sleep(1)
            continue
        first_buy_price = tick.ask
        
        # Place Initial Buy (0.1 lot at first price)
        req_initial_buy = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.1,
            "type": mt5.ORDER_TYPE_BUY,
            "price": first_buy_price,
            "deviation": 10,
            "magic": 234000,
            "comment": "Initial Buy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(req_initial_buy)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            print("Initial Buy error:", res)
            time.sleep(1)
            continue
        print(f"Placed Initial Buy at {first_buy_price} (0.1 lot)")
        
        # Wait until price drops 1 pip below first buy price
        while True:
            tick = mt5.symbol_info_tick(symbol)
            if tick.bid <= first_buy_price - 1 * pip:
                break
            time.sleep(0.5)
        
        # Place Reverse Sell (0.16 lot at first_buy_price - 1) with TP 4.5 pips below price
        reverse_sell_price = first_buy_price - 1 * pip
        req_reverse_sell = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.16,
            "type": mt5.ORDER_TYPE_SELL,
            "price": reverse_sell_price,
            "deviation": 10,
            "magic": 234000,
            "comment": "Reverse Sell",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "tp": reverse_sell_price - 4.5 * pip,  # added TP for reverse sell
        }
        res = mt5.order_send(req_reverse_sell)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            print("Reverse Sell error:", res)
            time.sleep(1)
            continue
        print(f"Placed Reverse Sell at {reverse_sell_price} (0.16 lot)")
        
        # Set overall TP targets for cycle completion:
        initial_buy_tp = first_buy_price + 4.5 * pip   # initial buy TP condition
        reverse_sell_tp = reverse_sell_price - 4.5 * pip  # reverse sell TP condition

        # --- Begin dynamic cascade cycle ---
        current_direction = "sell"  # last trade was sell (Reverse Sell)
        last_order_price = reverse_sell_price
        while True:
            tick = mt5.symbol_info_tick(symbol)
            # Overall termination conditions:
            if tick.ask >= initial_buy_tp:
                print("Initial buy TP hit, cycle complete. Closing open positions.")
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    for pos in positions:
                        close_price = mt5.symbol_info_tick(symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
                        close_req = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "position": pos.ticket,
                            "symbol": symbol,
                            "volume": pos.volume,
                            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                            "price": close_price,
                            "deviation": 10,
                            "magic": 234000,
                            "comment": "Close position",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        close_res = mt5.order_send(close_req)
                        if close_res.retcode != mt5.TRADE_RETCODE_DONE:
                            print("Close position error:", close_res)
                break
            if tick.bid <= reverse_sell_tp:
                print("Reverse sell TP hit, cycle complete.")
                break

            if current_direction == "sell":
                # Check if current sell order's TP is hit
                if tick.bid <= first_buy_price - 5.5 * pip:
                    print("Sell order TP hit, cycle complete.")
                    break
                # When price reverses upward from the sell price, place a buy order
                elif tick.ask >= last_order_price:
                    req_buy = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": 0.08,
                        "type": mt5.ORDER_TYPE_BUY,
                        "price": last_order_price,
                        "deviation": 10,
                        "magic": 234000,
                        "comment": "Buy after sell",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                        "tp": last_order_price + 5.5 * pip,
                    }
                    res = mt5.order_send(req_buy)
                    if res.retcode != mt5.TRADE_RETCODE_DONE:
                        print("Buy after sell error:", res)
                        time.sleep(1)
                        continue
                    print(f"Placed Buy order after sell at {last_order_price} (0.08 lot)")
                    current_direction = "buy"
                    # For the next reversal, we use the same order price
                else:
                    time.sleep(0.5)
            elif current_direction == "buy":
                # Check if current buy order's TP is hit
                if tick.ask >= first_buy_price + 4.5 * pip:
                    print("Buy order TP hit, cycle complete.")
                    break
                # When price reverses downward from the buy price, place a sell order
                elif tick.bid <= last_order_price:
                    req_sell = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": 0.08,
                        "type": mt5.ORDER_TYPE_SELL,
                        "price": last_order_price,
                        "deviation": 10,
                        "magic": 234000,
                        "comment": "Sell after buy",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                        "tp": last_order_price - 4.5 * pip,
                    }
                    res = mt5.order_send(req_sell)
                    if res.retcode != mt5.TRADE_RETCODE_DONE:
                        print("Sell after buy error:", res)
                        time.sleep(1)
                        continue
                    print(f"Placed Sell order after buy at {last_order_price} (0.08 lot)")
                    current_direction = "sell"
                    # For next reversal, maintain the same reference price
                else:
                    time.sleep(0.5)
        # --- End dynamic cascade cycle ---
        
        print("Cycle complete. Restarting cascade strategy...")
        time.sleep(2)

if __name__ == "__main__":
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
    try:
        new_cascade_strategy()
    finally:
        mt5.shutdown()
