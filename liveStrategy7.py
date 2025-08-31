import MetaTrader5 as mt5
import time
import math

def open_hedge_trades(symbol, lot):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("Failed to get tick for", symbol)
        return None
    # Ensure the symbol is selected
    if not mt5.symbol_select(symbol, True):
        print("Failed to select symbol", symbol)
        return None

    # NEW: Adjust the lot to ensure it meets the symbol's minimum and volume step requirements.
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print("Failed to get symbol info for", symbol)
        return None
    # Clamp lot between minimum and maximum allowed volumes, then adjust by volume_step.
    if lot < symbol_info.volume_min:
        lot = symbol_info.volume_min
    elif lot > symbol_info.volume_max:
        lot = symbol_info.volume_max
    else:
        lot = math.floor(lot / symbol_info.volume_step) * symbol_info.volume_step
    # Ensure lot has the proper decimal precision based on volume_step.
    decimals = 0
    if symbol_info.volume_step < 1:
        decimals = abs(int(math.floor(math.log10(symbol_info.volume_step))))
    lot = round(lot, decimals)

    # Create Buy order request using computed lot size.
    request_buy = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,  # use computed lot size
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "deviation": 10,
        "magic": 234000,
        "comment": "Hedged Scalping Buy",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result_buy = mt5.order_send(request_buy)
    
    # Create Sell order request using the same computed lot size.
    request_sell = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,  # use computed lot size
        "type": mt5.ORDER_TYPE_SELL,
        "price": tick.bid,
        "deviation": 10,
        "magic": 234000,
        "comment": "Hedged Scalping Sell",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result_sell = mt5.order_send(request_sell)
    
    if result_buy.retcode != mt5.TRADE_RETCODE_DONE or result_sell.retcode != mt5.TRADE_RETCODE_DONE:
        print("Order send failed:", result_buy, result_sell)
        return None
    return {
        'buy': {"ticket": result_buy.order, "price": tick.ask, "type": "buy", "symbol": symbol},
        'sell': {"ticket": result_sell.order, "price": tick.bid, "type": "sell", "symbol": symbol}
    }

def close_trade(order_info):
    ticket = order_info["ticket"]
    symbol = order_info["symbol"]
    positions = mt5.positions_get(symbol=symbol)
    pos_to_close = None
    for pos in positions:
        if pos.ticket == ticket:
            pos_to_close = pos
            break
    if not pos_to_close:
        print("Position not found for ticket", ticket)
        return
    tick = mt5.symbol_info_tick(symbol)
    if order_info["type"] == "buy":
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
    else:
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": pos_to_close.volume,
        "type": order_type,
        "position": ticket,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "Close position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Failed to close position", ticket, result)

def get_trade_profit(ticket):
    # Check open positions first.
    positions = mt5.positions_get()
    if positions:
        for pos in positions:
            if pos.ticket == ticket:
                return pos.profit
    # If position not open, check deal history (assuming trade completed recently).
    now = time.time()
    history = mt5.history_deals_get(now - 86400, now)
    if history:
        for deal in history:
            if deal.ticket == ticket:
                return deal.profit
    return 0.0

def check_for_reversal_signal(active_order, symbol, threshold=0.0005):
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False
    # For a BUY trade, if current bid has fallen significantly below the entry price, consider it a reversal.
    if active_order["type"] == "buy":
        if tick.bid < active_order["price"] - threshold:
            return True
    # For a SELL trade, if current ask has risen significantly above the entry price, it's a reversal.
    else:
        if tick.ask > active_order["price"] + threshold:
            return True
    return False

def get_session_net_profit(trade_ids):
    net_profit = 0
    for order_info in trade_ids:
        net_profit += get_trade_profit(order_info["ticket"])
    return net_profit

def open_reverse_trade(active_order, symbol, lot):
    # Opens a trade in the reverse direction of the active order.
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print("Failed to get tick for reverse trade", symbol)
        return None
    order_type = mt5.ORDER_TYPE_SELL if active_order["type"] == "buy" else mt5.ORDER_TYPE_BUY
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "Hedged Scalping Reverse Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Failed to open reverse trade for", symbol, result)
        return None
    return {"ticket": result.order, "price": price, "type": "buy" if order_type == mt5.ORDER_TYPE_BUY else "sell", "symbol": symbol}

# New commission optimization: subtract a 30-point commission cost per trade.
COMMISSION_COST = 0.5  # Commission cost (points) per trade

def get_adjusted_profit(ticket):
    # Returns the profit adjusted for commission cost.
    return get_trade_profit(ticket) - COMMISSION_COST

def open_same_side_trade(order, symbol, lot):
    # Opens a new trade on the same side as ‘order’
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print("Failed to get tick for same side trade", symbol)
        return None
    if order["type"] == "buy":
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
        comment = "Same side Buy"
    else:
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
        comment = "Same side Sell"
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Failed to open same side trade", symbol, result)
        return None
    return {"ticket": result.order, "price": price, "type": order["type"], "symbol": symbol}

#############################################
# New: Position Sizing and Risk Management Functions

def calculate_half_kelly(win_rate, win_loss_ratio):
    """
    Returns the fraction (half-Kelly) of capital to risk.
    f* = (W - (1-W)/R)
    Use half of f* to reduce volatility.
    """
    if win_loss_ratio <= 0:
        return 0.0
    f_star = win_rate - (1.0 - win_rate) / win_loss_ratio
    return max(0.0, f_star * 0.5)

def calculate_position_size(capital, risk_fraction, price, symbol, contract_size=1):
    """
    Given the capital, fraction to risk and price, determine the number of lots.
    Clamps the lot size between symbol's volume_min and volume_max and rounds 
    down according to the symbol's volume_step.
    """
    risk_money = capital * risk_fraction
    raw_lot = risk_money / price / contract_size
    si = mt5.symbol_info(symbol)
    if si:
        # Clamp raw_lot between symbol's volume_min and volume_max.
        raw_lot = max(si.volume_min, min(raw_lot, si.volume_max))
        # Round down to the nearest multiple of the volume_step.
        raw_lot = math.floor(raw_lot / si.volume_step) * si.volume_step
    # Limit the lot size between 0.01 and 0.09.
    raw_lot = max(0.01, min(raw_lot, 0.09))
    return raw_lot

def compute_sharpe_ratio(returns, risk_free_rate=0):
    """
    Placeholder function to compute Sharpe ratio.
    returns: list of trade returns.
    """
    if not returns:
        return 0.0
    avg_return = sum(returns) / len(returns)
    vol = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns))
    if vol == 0:
        return 0
    return (avg_return - risk_free_rate) / vol

def adjust_lot_size_martingale(last_trade_won, current_lot):
    """
    Implements simple Martingale and Anti-Martingale rules:
    - If last trade lost, double the lot (Martingale).
    - If last trade won, increase lot by 50% (Anti-Martingale) or keep same.
    """
    if last_trade_won:
        return current_lot * 1.5  # Increase exposure modestly.
    else:
        return current_lot * 2.0  # Double the lot after a loss.

# NEW: New function to open hedge orders on different currency pairs.
def open_hedge_trades_dual(buy_symbol, sell_symbol, lot_buy, lot_sell):
    # Get ticks and select symbols
    tick_buy = mt5.symbol_info_tick(buy_symbol)
    tick_sell = mt5.symbol_info_tick(sell_symbol)
    if tick_buy is None or tick_sell is None:
        print("Failed to get tick for one or both symbols:", buy_symbol, sell_symbol)
        return None
    if not mt5.symbol_select(buy_symbol, True) or not mt5.symbol_select(sell_symbol, True):
        print("Failed to select one or both symbols:", buy_symbol, sell_symbol)
        return None
    # Adjust lot for each symbol using same clamping logic.
    for sym, lot in [(buy_symbol, lot_buy), (sell_symbol, lot_sell)]:
        si = mt5.symbol_info(sym)
        if si is None:
            print("Failed to get symbol info for", sym)
            return None
    # Adjust buy lot.
    buy_si = mt5.symbol_info(buy_symbol)
    if lot_buy < buy_si.volume_min:
        lot_buy = buy_si.volume_min
    elif lot_buy > buy_si.volume_max:
        lot_buy = buy_si.volume_max
    else:
        lot_buy = math.floor(lot_buy / buy_si.volume_step) * buy_si.volume_step
    decimals = 0
    if buy_si.volume_step < 1:
        decimals = abs(int(math.floor(math.log10(buy_si.volume_step))))
    lot_buy = round(lot_buy, decimals)
    # Adjust sell lot.
    sell_si = mt5.symbol_info(sell_symbol)
    if lot_sell < sell_si.volume_min:
        lot_sell = sell_si.volume_min
    elif lot_sell > sell_si.volume_max:
        lot_sell = sell_si.volume_max
    else:
        lot_sell = math.floor(lot_sell / sell_si.volume_step) * sell_si.volume_step
    decimals = 0
    if sell_si.volume_step < 1:
        decimals = abs(int(math.floor(math.log10(sell_si.volume_step))))
    lot_sell = round(lot_sell, decimals)
    
    # Create Buy order for buy_symbol.
    request_buy = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": buy_symbol,
        "volume": lot_buy,
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick_buy.ask,
        "deviation": 10,
        "magic": 234000,
        "comment": "Hedged Scalping Buy",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result_buy = mt5.order_send(request_buy)
    
    # Create Sell order for sell_symbol.
    request_sell = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": sell_symbol,
        "volume": lot_sell,
        "type": mt5.ORDER_TYPE_SELL,
        "price": tick_sell.bid,
        "deviation": 10,
        "magic": 234000,
        "comment": "Hedged Scalping Sell",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result_sell = mt5.order_send(request_sell)
    
    if result_buy.retcode != mt5.TRADE_RETCODE_DONE or result_sell.retcode != mt5.TRADE_RETCODE_DONE:
        print("Order send failed:", result_buy, result_sell)
        return None
    return {
        'buy': {"ticket": result_buy.order, "price": tick_buy.ask, "type": "buy", "symbol": buy_symbol},
        'sell': {"ticket": result_sell.order, "price": tick_sell.bid, "type": "sell", "symbol": sell_symbol}
    }

# NEW: Predictive function for market direction based on strong trend.
def predict_direction(symbol):
    # Placeholder prediction logic based on strong trend.
    # Replace with your own trend analysis algorithm.
    trend_strength = get_trend_strength(symbol)  # Assume this function returns a value indicating trend strength.
    if trend_strength > 0:  # Positive trend indicates a buy signal.
        return "buy"
    elif trend_strength < 0:  # Negative trend indicates a sell signal.
        return "sell"
    else:  # Neutral trend defaults to buy.
        return "buy"

# NEW: Function to determine trend strength for a symbol.
def get_trend_strength(symbol):
    # Placeholder logic for trend strength calculation.
    # Replace with your own implementation (e.g., based on moving averages, RSI, etc.).
    # Positive values indicate upward trend, negative values indicate downward trend.
    if symbol == "EURUSD":
        return 1.5  # Strong upward trend.
    elif symbol == "GBPUSD":
        return -1.2  # Strong downward trend.
    else:
        return 0  # Neutral trend.

# Modify hedged_scalping_strategy to decide logically which currency to buy and which to sell.
def hedged_scalping_strategy(symbols=["EURUSD", "GBPUSD"], initial_lot=0.01, capital=10000):
    # Require at least two symbols for dual hedge trading.
    if len(symbols) < 2:
        print("At least two symbols required for dual hedge trading.")
        return
    buy_symbol = symbols[0]
    sell_symbol = symbols[1]
    
    # Determine predicted directions.
    direction_buy = predict_direction(buy_symbol)
    direction_sell = predict_direction(sell_symbol)
    print(f"Prediction: {buy_symbol} -> {direction_buy}, {sell_symbol} -> {direction_sell}")
    
    # If predictions are reversed, swap the roles.
    if direction_buy != "buy" or direction_sell != "sell":
        print("Predictions not as expected. Swapping symbol roles.")
        buy_symbol, sell_symbol = sell_symbol, buy_symbol
    
    win_rate = 0.55
    win_loss_ratio = 1.5
    risk_fraction = calculate_half_kelly(win_rate, win_loss_ratio)
    lot_buy = calculate_position_size(capital, risk_fraction, mt5.symbol_info_tick(buy_symbol).ask, buy_symbol)
    lot_sell = calculate_position_size(capital, risk_fraction, mt5.symbol_info_tick(sell_symbol).ask, sell_symbol)
    print(f"{buy_symbol}: Lot size: {lot_buy}")
    print(f"{sell_symbol}: Lot size: {lot_sell}")
    
    # Define dictionaries for later use.
    lots = {buy_symbol: lot_buy, sell_symbol: lot_sell}
    last_trade_won = {buy_symbol: True, sell_symbol: True}
    active_trades = {buy_symbol: [], sell_symbol: []}
    
    while True:
        # Only execute dual hedge if there are no open positions on both symbols.
        pos_buy = mt5.positions_get(symbol=buy_symbol)
        pos_sell = mt5.positions_get(symbol=sell_symbol)
        if (not pos_buy or len(pos_buy)==0) and (not pos_sell or len(pos_sell)==0):
            trade_info = open_hedge_trades_dual(buy_symbol, sell_symbol, lot_buy, lot_sell)
            if trade_info is None:
                time.sleep(1)
                continue
            time.sleep(5)
            # ...insert management for dual hedge orders here...
        # Process each symbol in the pair.
        for sym in [buy_symbol, sell_symbol]:
            positions = mt5.positions_get(symbol=sym)
            if positions is None or len(positions) == 0:
                print(f"{sym}: No open positions detected. Executing hedge trades.")
                trade_info = open_hedge_trades(sym, lots[sym])
                if trade_info is None:
                    continue
                time.sleep(5)  # monitoring phase wait period

                # Evaluate trades for immediate win management.
                buy_profit = get_adjusted_profit(trade_info['buy']["ticket"])
                sell_profit = get_adjusted_profit(trade_info['sell']["ticket"])
                if buy_profit < sell_profit:
                    print(f"{sym}: Buy trade is winning (adjusted); closing sell trade.")
                    close_trade(trade_info['sell'])
                    active_order = trade_info['buy']
                    last_trade_won[sym] = True
                else:
                    print(f"{sym}: Sell trade is winning (adjusted); closing buy trade.")
                    close_trade(trade_info['buy'])
                    active_order = trade_info['sell']
                    last_trade_won[sym] = True

                active_trades[sym] = [active_order]
                reversal_taken = False
                max_profit = get_adjusted_profit(active_order["ticket"])
                PROFIT_TARGET = 0.5  # Book small profits

                # Modified monitoring loop for each symbol:
                while True:
                    time.sleep(2)
                    current_profit = get_adjusted_profit(active_order["ticket"])
                    if current_profit > max_profit:
                        max_profit = current_profit
                    # When profit target is met, wait 3 seconds then close the position.
                    if current_profit >= PROFIT_TARGET:
                        print(f"{sym}: Profit target reached with profit {current_profit}. Waiting 3 sec before closing trade.")
                        time.sleep(3)
                        close_trade(active_order)
                        break
                    # Trailing stop: if profit drops by 0.01 from the peak, close trade.
                    if (max_profit - current_profit) >= 0.01:
                        print(f"{sym}: Trailing stop triggered. Peak profit was {max_profit} and current profit is {current_profit}. Closing trade.")
                        close_trade(active_order)
                        break
                    # Check for reversal signal if not already hedged.
                    if not reversal_taken and check_for_reversal_signal(active_order, sym):
                        print(f"{sym}: Reversal detected for ticket {active_order['ticket']}")
                        reverse_order = open_reverse_trade(active_order, sym, lots[sym]*2)
                        if reverse_order:
                            active_trades[sym].append(reverse_order)
                            reversal_taken = True
                            print(f"{sym}: Opened reverse trade for hedging. Active trades: {[trade['ticket'] for trade in active_trades[sym]]}")
                            time.sleep(3)
                    combined_profit = sum(get_adjusted_profit(trade["ticket"]) for trade in active_trades[sym])
                    if combined_profit >= PROFIT_TARGET:
                        print(f"{sym}: Profit target reached (adjusted). Closing trades: {[trade['ticket'] for trade in active_trades[sym]]}")
                        for order in active_trades[sym]:
                            close_trade(order)
                        active_trades[sym] = []
                        break
                # After monitoring, adjust lot size based on trade outcome.
                trade_profit = get_trade_profit(active_order["ticket"])
                last_trade_won[sym] = True if trade_profit > 0 else False
                lots[sym] = adjust_lot_size_martingale(last_trade_won[sym], lots[sym])
                print(f"{sym}: Adjusted lot size for next trade: {lots[sym]}")
        time.sleep(1)

if __name__ == "__main__":
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
    try:
        hedged_scalping_strategy()
    finally:
        mt5.shutdown()
