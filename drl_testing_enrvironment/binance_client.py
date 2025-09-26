from binance.client import Client
from binance.enums import *
import os
import sys

# ====== Fill in your Testnet API Key and Secret ======
API_KEY = "LyKVJBT0ddo4rUtyqdsQ8GLdE1AyIJ2N8qiPpRtuvxBrEkBNo3A6QN1dQPOj7DK9"
API_SECRET = "dA0PDEQyjTc730HnkR3GTAM7ypuXIUxLbmh4Fiyc2qVeKenVRbN70vg0cMtxhBbi"


# Create client for Binance Testnet
client = Client(API_KEY, API_SECRET, testnet=True)
client.API_URL = 'https://testnet.binance.vision/api'
client.get_server_time()


def get_symbol_ticker(symbol):
    """Get latest price ticker for a symbol"""
    return client.get_symbol_ticker(symbol=symbol)

def get_symbol_info(symbol):
    """Get trading rules for a specific symbol"""
    return client.get_symbol_info(symbol)

def get_balance(asset=None):
    """Check account balance"""
    account_info = client.get_account()
    balances = account_info["balances"]
    if asset:
        for b in balances:
            if b["asset"] == asset:
                return float(b["free"]), float(b["locked"])
        return 0.0, 0.0
    else:
        # Return all non-zero balances
        return [b for b in balances if float(b["free"]) > 0 or float(b["locked"]) > 0]

def place_limit_order(symbol, side, quantity, price):
    """Place a limit order"""
    return client.create_order(
        symbol=symbol,
        side=side,
        type=ORDER_TYPE_LIMIT,
        timeInForce=TIME_IN_FORCE_GTC,
        quantity=quantity,
        price=str(price)
    )

def place_market_order(symbol, side, quantity):
    """Place a market order"""
    return client.create_order(
        symbol=symbol,
        side=side,
        type=ORDER_TYPE_MARKET,
        quantity=quantity
    )

def list_open_orders(symbol="ETHUSDT"):
    """List all open orders"""
    return client.get_open_orders(symbol=symbol)

def cancel_order(symbol, order_id):
    """Cancel a specific order by ID"""
    return client.cancel_order(symbol=symbol, orderId=order_id)

if __name__ == "__main__":
    print("====== Binance Testnet Trading Terminal ======")

    while True:
        cmd = input("\nEnter command (BALANCE / ORDERS / BUY / SELL / CANCEL / EXIT): ").strip().upper()

        if cmd == "EXIT":
            print("Exiting...")
            sys.exit(0)

        elif cmd == "BALANCE":
            balances = get_balance()
            print("Balances:")
            for b in balances:
                print(f"{b['asset']}: free={b['free']}, locked={b['locked']}")

        elif cmd == "ORDERS":
            orders = list_open_orders()
            if not orders:
                print("No open orders.")
            else:
                for o in orders:
                    print(f"ID={o['orderId']} {o['side']} {o['origQty']} {o['symbol']} @ {o['price']} Status={o['status']}")

        elif cmd in ["BUY", "SELL"]:
            qty = float(input("Enter quantity (e.g. 0.01): "))
            price_input = input("Enter price (leave blank for market order): ").strip()

            if price_input == "":
                order = place_market_order("ETHUSDT", cmd, qty)
                print("Market order submitted:", order)
            else:
                price = float(price_input)
                order = place_limit_order("ETHUSDT", cmd, qty, price)
                print("Limit order submitted:", order)

        elif cmd == "CANCEL":
            order_id = int(input("Enter the order ID to cancel: "))
            result = cancel_order("ETHUSDT", order_id)
            print("Order canceled:", result)

        else:
            print("Invalid command. Please enter BALANCE / ORDERS / BUY / SELL / CANCEL / EXIT")