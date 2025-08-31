import MetaTrader5 as mt5
import time
import math

# NEW: Clamp lot volume between 0.01 and 0.06
def clamp_lot(lot):
    return max(0.01, min(lot, 0.06))

# Constants for pending order offset and SL/TP
PENDING_OFFSET = 1   # points above/below current price for pending orders
SL_DISTANCE = 1   # Stop loss 100 points from the pending order price
TP_DISTANCE = 2   # Take profit 200 points from the pending order price

def open_pending_orders(symbol, lot):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("Failed to get tick for", symbol)
        return None, None
    # NEW: Adjust lot using symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print("Failed to get symbol info for", symbol)
        return None, None
    if lot < symbol_info.volume_min:
        lot = symbol_info.volume_min
    else:
        lot = math.floor(lot / symbol_info.volume_step) * symbol_info.volume_step
    lot = clamp_lot(lot)   # NEW: clamp lot between 0.01 and 0.06
    # Calculate pending prices
    buy_pending_price  = tick.ask + PENDING_OFFSET
    sell_pending_price = tick.bid - PENDING_OFFSET
    # For a BUY pending order, set SL 100 points below and TP 200 points above the pending price
    buy_sl = buy_pending_price - SL_DISTANCE
    buy_tp = buy_pending_price + TP_DISTANCE
    # For a SELL pending order, set SL 100 points above and TP 200 points below the pending price
    sell_sl = sell_pending_price + SL_DISTANCE
    sell_tp = sell_pending_price - TP_DISTANCE
    # Create Buy Stop pending order
    result_buy = mt5.order_send({
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY_STOP,
        "price": buy_pending_price,
        "sl": buy_sl,
        "tp": buy_tp,
        "deviation": 10,
        "magic": 234000,
        "comment": "Pending Buy Stop",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    })
    if result_buy.retcode != mt5.TRADE_RETCODE_DONE:
        print("Buy pending order failed:", result_buy)
        pending_buy = None
    else:
        pending_buy = {"ticket": result_buy.order, "price": buy_pending_price, "type": "buy", "symbol": symbol}
    # Create Sell Stop pending order
    result_sell = mt5.order_send({
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL_STOP,
        "price": sell_pending_price,
        "sl": sell_sl,
        "tp": sell_tp,
        "deviation": 10,
        "magic": 234000,
        "comment": "Pending Sell Stop",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    })
    if result_sell.retcode != mt5.TRADE_RETCODE_DONE:
        print("Sell pending order failed:", result_sell)
        pending_sell = None
    else:
        pending_sell = {"ticket": result_sell.order, "price": sell_pending_price, "type": "sell", "symbol": symbol}
    return pending_buy, pending_sell

def monitor_pending_orders(pending_buy, pending_sell, symbol):
    # Monitor pending orders until one is activated and then closed (TP or SL hit)
    # A simple approach: check session net profit continuously. When a pending order is executed,
    # its profit will eventually reflect TP or SL. For simplicity, assume that once either order is activated,
    # we wait until the active position's profit is non-zero and then it's closed.
    while True:
        time.sleep(5)
        net_profit = 0
        orders = [pending_buy, pending_sell]
        for order in orders:
            if order is not None:
                profit = mt5.history_deals_total()  # or query get_trade_profit(order["ticket"])
                # For simulation, we check if get_trade_profit(order["ticket"]) != 0
                profit = get_trade_profit(order["ticket"])
                net_profit += profit
        # If any pending order resulted in a non-zero profit then assume one has been triggered and closed
        if net_profit != 0:
            print("A pending order was activated and closed with profit:", net_profit)
            break

def get_trade_profit(ticket):
    positions = mt5.positions_get()
    if positions:
        for pos in positions:
            if pos.ticket == ticket:
                return pos.profit
    now = time.time()
    history = mt5.history_deals_get(now - 86400, now)
    if history:
        for deal in history:
            if deal.ticket == ticket:
                return deal.profit
    return 0.0

def pending_strategy(symbol="XAUUSD", lot=0.01):
    while True:
        # Place pending orders 150 points above and below current market
        pending_buy, pending_sell = open_pending_orders(symbol, lot)
        if pending_buy is None and pending_sell is None:
            print("Failed to place pending orders. Retrying...")
            time.sleep(5)
            continue
        print("Pending orders placed. Buy ticket:", pending_buy["ticket"] if pending_buy else "None", 
              "Sell ticket:", pending_sell["ticket"] if pending_sell else "None")
        # Monitor pending orders until one gets activated and closed at TP or SL
        monitor_pending_orders(pending_buy, pending_sell, symbol)
        # After TP or SL hit, delete any remaining pending orders
        # (In actual MT5, you may need to cancel the untriggered order)
        for order in [pending_buy, pending_sell]:
            if order is not None:
                ret = mt5.order_delete(order["ticket"])
                print("Cancelled pending order", order["ticket"], ":", ret)
        # New cycle: create new pending orders
        print("Recreating pending orders...")
        time.sleep(5)  # wait briefly before new cycle

if __name__ == "__main__":
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
    try:
        pending_strategy()
    finally:
        mt5.shutdown()
