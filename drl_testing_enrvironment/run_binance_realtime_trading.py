import asyncio
import logging
import signal
import sys
import json
import time
import os
import websockets
import requests
import ssl
import math
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt

# Import our components
from model_loader import ModelLoader
from online_feature_engine import OnlineFeatureEngine
import binance_client as binance_api

# === CONFIGURATION & CONSTANTS ===
SYMBOL = "ETHUSDT"
INTERVAL = "1m"
RAW_DIR = f"data/price/raw/{SYMBOL}"
FEAT_DIR = f"data/price/features/raw/{SYMBOL}"
STATE_PATH = f"state/indicators_{SYMBOL}.json"
SENT_1MIN_PATH = "data/reddit/weighted/sentiment_1min_vader_s1_s5.csv"

# Binance Endpoints
REST_KLINES = "https://api.binance.com/api/v3/klines"
WS_STREAM = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@kline_1m"

# === HELPER FUNCTIONS ===
def to_utc_ms(dt_: datetime) -> int:
    return int(dt_.replace(tzinfo=timezone.utc).timestamp() * 1000)

def kline_list_to_df(data):
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["ts"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["ts", "open", "high", "low", "close", "volume"]].dropna()

def append_df_to_parquet(df: pd.DataFrame, out_path: str):
    if df.empty: return
    if os.path.exists(out_path):
        old = pd.read_parquet(out_path)
        df = pd.concat([old, df]).drop_duplicates(subset=["ts"]).sort_values("ts")
    df.to_parquet(out_path, index=False)

@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(5))
def rest_fetch_klines(start_ms: int, end_ms: int, limit: int = 1000):
    params = {"symbol": SYMBOL, "interval": INTERVAL, "startTime": start_ms, "endTime": end_ms, "limit": limit}
    r = requests.get(REST_KLINES, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def backfill_klines(days: int = 3):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    start_ms, end_ms = to_utc_ms(start), to_utc_ms(end)
    logging.info(f"[REST] Backfilling {SYMBOL} {INTERVAL}: {start} -> {end}")
    cur = start_ms
    while cur < end_ms:
        batch_end = min(cur + 1000 * 60 * 1000, end_ms)
        data = rest_fetch_klines(cur, batch_end)
        if not data:
            cur += 1000 * 60 * 1000
            continue
        df = kline_list_to_df(data)
        if not df.empty:
            for day, g in df.groupby(df["ts"].dt.strftime("%Y-%m-%d")):
                day_path = os.path.join(RAW_DIR, f"{SYMBOL}_{day}.parquet")
                append_df_to_parquet(g, day_path)
        last_close = data[-1][6]
        cur = last_close + 1
        time.sleep(0.2)
    logging.info("[REST] Backfill done.")

def adjust_to_lot_size(quantity, step_size):
    return math.floor(quantity / step_size) * step_size

class TradingSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.is_running = False
        self.main_task = None
        
        self.model_loader: Optional[ModelLoader] = None
        self.feature_engine: Optional[OnlineFeatureEngine] = None
        
        self.initial_portfolio_value = 0.0
        self.current_position = 0.0
        self.position_change = 0.0
        self.cash_balance = 0.0
        self.portfolio_value = 0.0
        self.trades_count = 0
        
        # Exchange Rules
        self.lot_step_size = None
        self.min_notional = None
        self.quantity_precision = None

        self.actions_history = []
        self.rewards_history = []
        self.portfolio_history = []
        self.start_time = None
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('trading_system.log'), logging.StreamHandler(sys.stdout)])
        self.logger = logging.getLogger(__name__)
        
        self._initialize_components()
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _initialize_components(self):
        try:
            model_path = self.config.get('model_path')
            if not model_path: raise ValueError("model_path required in config")
            self.model_loader = ModelLoader(model_path, self.config.get('scaler_path'))
            self.logger.info(f"Model loaded: {self.model_loader.get_model_info()}")

            self.logger.info("Using binance_client.py for exchange communication (hardcoded to Testnet).")
            self._fetch_exchange_rules()
            self.sync_portfolio_state()

            try:
                ticker = binance_api.get_symbol_ticker(symbol=SYMBOL)
                initial_price = float(ticker['price'])
                self.portfolio_value = self.cash_balance + (self.current_position * initial_price)
                self.initial_portfolio_value = self.portfolio_value
                self.logger.info(f"Initial portfolio value calculated at ${self.initial_portfolio_value:.2f} with price ${initial_price}")
            except Exception as e:
                self.logger.error(f"Could not fetch initial price to set portfolio value: {e}")
                self.initial_portfolio_value = 0.0

            os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
            self.feature_engine = OnlineFeatureEngine(symbol=SYMBOL, state_path=STATE_PATH, sentiment_1min_path=self.config.get('sentiment_path', SENT_1MIN_PATH))
            self.logger.info("OnlineFeatureEngine initialized.")

            os.makedirs(RAW_DIR, exist_ok=True)
            os.makedirs(FEAT_DIR, exist_ok=True)

            self.logger.info("All components initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    def _fetch_exchange_rules(self):
        self.logger.info(f"Fetching trading rules for {SYMBOL}...")
        info = binance_api.get_symbol_info(SYMBOL)
        
        lot_size_filter = next((f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        notional_filter = next((f for f in info['filters'] if f['filterType'] == 'NOTIONAL'), None)

        if lot_size_filter:
            self.lot_step_size = float(lot_size_filter['stepSize'])
            self.quantity_precision = int(-math.log10(self.lot_step_size))
            self.logger.info(f"LOT_SIZE stepSize: {self.lot_step_size}, Precision: {self.quantity_precision} decimals")
        else:
            raise ValueError("Could not fetch LOT_SIZE filter for symbol.")

        if notional_filter:
            self.min_notional = float(notional_filter['minNotional'])
            self.logger.info(f"MIN_NOTIONAL: {self.min_notional}")
        else:
            raise ValueError("Could not fetch NOTIONAL filter for symbol.")

    def sync_portfolio_state(self, current_price: float = None):
        try:
            base_asset, quote_asset = SYMBOL.replace("USDT", ""), "USDT"
            eth_balance, _ = binance_api.get_balance(asset=base_asset)
            usdt_balance, _ = binance_api.get_balance(asset=quote_asset)
            self.current_position = eth_balance
            self.cash_balance = usdt_balance
            if current_price:
                self.portfolio_value = self.cash_balance + (self.current_position * current_price)
        except Exception as e:
            self.logger.error(f"Failed to sync portfolio state: {e}")

    def _signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.is_running = False
        if self.main_task:
            self.main_task.cancel()
        
    async def start(self):
        self.logger.info("--- STARTING NEW SESSION ---")
        await self._flatten_position(reason="Starting session with a flat position.")
        self.sync_portfolio_state()
        self.initial_portfolio_value = self.cash_balance # After flattening, value is just cash
        self.logger.info(f"Initial portfolio value set to ${self.initial_portfolio_value:.2f}")

        self.is_running = True
        self.start_time = datetime.now()
        try:
            self.logger.info("Backfilling historical data for feature engine warmup...")
            backfill_klines(days=3)
            self.logger.info("Backfill complete.")
            self.main_task = asyncio.create_task(self._stream_klines(on_bar_callback=self._on_new_bar))
            await self.main_task
        except asyncio.CancelledError:
            self.logger.info("Main task cancelled. Shutting down.")
        finally:
            await self.stop()
            
    async def stop(self):
        self.logger.info("--- ENDING SESSION ---")
        self._generate_report()
        await self._flatten_position(reason="Ending session with a flat position.")
        if self.feature_engine: self.feature_engine.save_state()
        self.logger.info("Trading system stopped")

    async def _flatten_position(self, reason: str):
        self.logger.info(reason)
        self.sync_portfolio_state()
        min_trade_size = float(self.lot_step_size) * 2
        if self.current_position > min_trade_size:
            self.logger.info(f"Selling remaining {self.current_position} ETH to flatten position.")
            quantity_to_sell = adjust_to_lot_size(self.current_position, self.lot_step_size)
            try:
                formatted_qty = "{:.{prec}f}".format(quantity_to_sell, prec=self.quantity_precision)
                binance_api.place_market_order(symbol=SYMBOL, side='SELL', quantity=formatted_qty)
                self.logger.info("Position flattened successfully.")
            except Exception as e:
                self.logger.error(f"Failed to flatten position: {e}")
        else:
            self.logger.info("No significant position to flatten.")

    async def _stream_klines(self, on_bar_callback=None):
        self.logger.info(f"[WS] Connecting to {WS_STREAM}")
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        while self.is_running:
            try:
                async with websockets.connect(WS_STREAM, ping_interval=20, ping_timeout=20, ssl=ssl_context) as ws:
                    self.logger.info("[WS] Connection successful.")
                    while self.is_running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            m = json.loads(msg)
                            if m.get("e") != "kline" or not m.get("k", {}).get("x"):
                                continue
                            k = m["k"]
                            bar = pd.Series({"ts": pd.to_datetime(k["T"], unit="ms", utc=True), "open": float(k["o"]), "high": float(k["h"]), "low": float(k["l"]), "close": float(k["c"]), "volume": float(k["v"]),
                            })
                            if on_bar_callback:
                                await on_bar_callback(bar)
                        except asyncio.TimeoutError:
                            continue
                        except websockets.exceptions.ConnectionClosed:
                            self.logger.warning("[WS] Connection closed, will reconnect...")
                            break
            except asyncio.CancelledError:
                self.logger.info("[WS] Stream task cancelled.")
                break
            except Exception as e:
                if not self.is_running: break
                self.logger.error(f"[WS] Error: {e}; reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _on_new_bar(self, bar: pd.Series):
        try:
            day = bar["ts"].strftime("%Y-%m-%d")
            day_path = os.path.join(RAW_DIR, f"{SYMBOL}_{day}.parquet")
            append_df_to_parquet(pd.DataFrame([bar]), day_path)
            features = self.feature_engine.update(o=bar["open"], h=bar["high"], l=bar["low"], c=bar["close"], v=bar["volume"], ts_utc=bar["ts"])
            state_vector = np.zeros(15, dtype=np.float32)
            state_vector[0] = self.current_position
            state_vector[1:5] = [features.get(k, 0.0) for k in ['z_score', 'zone_norm', 'price_momentum', 'z_score_momentum']]
            state_vector[5] = self.position_change
            state_vector[6:14] = [features.get(k, 0.0) for k in ['macd_norm', 'macd_signal_norm', 'macd_diff_norm', 'rsi_norm', 'bb_mid_norm', 'bb_high_norm', 'bb_low_norm', 'obv_norm']]
            state_vector[14] = features.get('sentiment_score', 0.0)
            action = self.model_loader.predict(state_vector, deterministic=True)
            reward = self._execute_action(action, bar['close'])
            self.actions_history.append(action)
            self.rewards_history.append(reward)
            self.portfolio_history.append(self.portfolio_value)
            self._log_status(state_vector, action, bar['close'], reward)
        except Exception as e:
            self.logger.error(f"Error processing new bar: {e}", exc_info=True)
            
    def _execute_action(self, action: float, current_price: float) -> float:
        self.sync_portfolio_state(current_price=current_price)
        old_portfolio_value = self.portfolio_value
        target_position_eth = action * (self.portfolio_value / current_price)
        self.position_change = target_position_eth - self.current_position
        
        # Check against exchange rules before placing order
        quantity = adjust_to_lot_size(abs(self.position_change), self.lot_step_size)
        notional_value = quantity * current_price

        if notional_value < self.min_notional:
            self.logger.info(f"Skipping trade. Notional value ${notional_value:.2f} is below minimum of ${self.min_notional:.2f}")
            self.position_change = 0
            return 0.0

        side = 'BUY' if self.position_change > 0 else 'SELL'
        formatted_qty = "{:.{prec}f}".format(quantity, prec=self.quantity_precision)

        self.logger.info(f"Agent action: {action:.4f} -> Target: {target_position_eth:.6f} ETH. Executing {side} {formatted_qty} ETH.")
        try:
            order_result = binance_api.place_market_order(symbol=SYMBOL, side=side, quantity=formatted_qty)
            self.logger.info(f"Order successful: {order_result}")
            self.trades_count += 1
        except Exception as e:
            self.logger.error(f"ORDER FAILED: {e}")
            self.position_change = 0
            return 0.0
        
        self.sync_portfolio_state(current_price=current_price)
        reward = (self.portfolio_value - old_portfolio_value) / old_portfolio_value if old_portfolio_value != 0 else 0.0
        return reward
        
    def _log_status(self, state: np.ndarray, action: float, price: float, reward: float):
        state_str = np.array2string(state, precision=3, separator=',', suppress_small=True)
        log_message = (
            f"\n--- Timestep Report ---\n"
            f"  State      : {state_str}\n"
            f"  Action     : {action:+.4f}\n"
            f"  Reward     : {reward:+.6f}\n"
            f"  Price      : ${price:.2f}\n"
            f"  Position   : {self.current_position:.4f} ETH\n"
            f"  Portfolio  : ${self.portfolio_value:.2f}\n"
            f"  ---------------------"
        )
        self.logger.info(log_message)
        
    def _generate_report(self):
        self.logger.info("--- SESSION REPORT ---")
        # Sync one last time to get final values for report
        self.sync_portfolio_state()
        try:
            ticker = binance_api.get_symbol_ticker(symbol=SYMBOL)
            final_price = float(ticker['price'])
            final_value = self.cash_balance + (self.current_position * final_price)
        except Exception as e:
            self.logger.error(f"Could not fetch final price for report: {e}")
            final_value = self.portfolio_value # Fallback to last known value

        if not self.portfolio_history:
            self.logger.info("No trades were made during this session.")
            return

        pnl = final_value - self.initial_portfolio_value
        total_return_pct = (pnl / self.initial_portfolio_value) * 100 if self.initial_portfolio_value else 0

        report = {
            'session_duration': str(datetime.now() - self.start_time),
            'initial_portfolio_value': self.initial_portfolio_value,
            'final_portfolio_value': final_value,
            'pnl': pnl,
            'total_return_pct': total_return_pct,
            'total_trades': self.trades_count,
            'avg_reward': np.mean(self.rewards_history) if self.rewards_history else 0,
        }

        self.logger.info(f"Duration: {report['session_duration']}")
        self.logger.info(f"Total Trades: {report['total_trades']}")
        self.logger.info(f"Initial Portfolio: ${report['initial_portfolio_value']:.2f}")
        self.logger.info(f"Final Portfolio: ${report['final_portfolio_value']:.2f}")
        self.logger.info(f"Profit/Loss: ${report['pnl']:.2f} ({report['total_return_pct']:.2f}%) ")
        
        report_dir = Path(__file__).parent / "trading_reports"
        report_dir.mkdir(exist_ok=True) # Ensure directory exists
        report_file = report_dir / f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        self.logger.info(f"Full report saved to {report_file}")

def load_config() -> Dict:
    config_file = Path("trading_config.json")
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        default_config = {
            "model_path": "drl_training/models/a2c_0001_final.zip",
            "scaler_path": None,
            "sentiment_path": "/Users/choemanseung/4th year/760/760-ethereum-trading/data/reddit/weighted/sentiment_1min_vader_s1_s5.csv"
        }
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        return default_config

async def main():
    try:
        config = load_config()
        system = TradingSystem(config)
        await system.start()
    except Exception as e:
        logging.critical(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    print("DRL Trading System Starting...")
    print("This script will execute REAL trades.")
    print("Press Ctrl+C to stop")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("System stopped by user.")
