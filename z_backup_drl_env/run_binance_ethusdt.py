# run_binance_ethusdt.py
# Binance · ETHUSDT · 1min K线：REST 回补 + WebSocket 实时
# 分钟收盘触发：落盘原始K线 + 产出一行15D特征（含情绪）到 features/raw/ETHUSDT/YYYY-MM-DD.parquet

import os, time, json, math, asyncio, websockets, requests
from datetime import datetime, timezone, timedelta
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt

# === 目录结构 ===
SYMBOL      = "ETHUSDT"
INTERVAL    = "1m"
RAW_DIR     = f"data/price/raw/{SYMBOL}"
MERGED_PQ   = f"data/price/merged/{SYMBOL}.parquet"
FEAT_DIR    = f"data/price/features/raw/{SYMBOL}"   # 按天产出特征
STATE_PATH  = f"state/indicators_{SYMBOL}.json"     # 在线指标状态
SENT_1MIN   = "data/reddit/weighted/sentiment_1min_vader_s1_s5.csv"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MERGED_PQ), exist_ok=True)
os.makedirs(FEAT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)

# === Binance 端点 ===
REST_KLINES = "https://api.binance.com/api/v3/klines"  # params: symbol, interval, startTime, endTime, limit<=1000
WS_STREAM   = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@kline_1m"

# === 引入在线特征引擎 ===
from online_feature_engine import OnlineFeatureEngine

# ---------- 工具 ----------
def to_utc_ms(dt_: datetime) -> int:
    return int(dt_.replace(tzinfo=timezone.utc).timestamp() * 1000)

def utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def kline_list_to_df(data):
    # Binance kline schema:
    # [open_time, open, high, low, close, volume, close_time, quote_asset_vol, trades, taker_base_vol, taker_quote_vol, ignore]
    cols = ["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["ts"]     = pd.to_datetime(df["close_time"], unit="ms", utc=True)  # 以分钟收盘时间作为时间戳
    df["open"]   = pd.to_numeric(df["open"], errors="coerce")
    df["high"]   = pd.to_numeric(df["high"], errors="coerce")
    df["low"]    = pd.to_numeric(df["low"], errors="coerce")
    df["close"]  = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    return df[["ts","open","high","low","close","volume"]].dropna()

def append_raw(df: pd.DataFrame):
    if df.empty: return
    day = df["ts"].dt.strftime("%Y-%m-%d").iloc[0]
    out = os.path.join(RAW_DIR, f"{SYMBOL}_{day}.parquet")
    if os.path.exists(out):
        old = pd.read_parquet(out)
        df = pd.concat([old, df]).drop_duplicates(subset=["ts"]).sort_values("ts")
    df.to_parquet(out, index=False)

def merge_all_raw_to_one():
    files = sorted([os.path.join(RAW_DIR, x) for x in os.listdir(RAW_DIR) if x.endswith(".parquet")])
    if not files: return
    dfs = [pd.read_parquet(f) for f in files]
    all_df = pd.concat(dfs).drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    all_df.to_parquet(MERGED_PQ, index=False)
    return all_df

def append_feature_row(ts, feats: dict, o,h,l,c,v):
    """
    每分钟收盘产出一行 15D 特征，按天写入 features/raw/ETHUSDT/YYYY-MM-DD.parquet（幂等去重）
    """
    day = pd.to_datetime(ts).tz_convert("UTC").strftime("%Y-%m-%d")
    out = os.path.join(FEAT_DIR, f"{SYMBOL}_features_{day}.parquet")
    row = {
        "ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v,
        # 15D（其中 position/position_change 由环境补）
        "z_score": feats["z_score"],
        "zone_norm": feats["zone_norm"],
        "price_momentum": feats["price_momentum"],
        "z_score_momentum": feats["z_score_momentum"],
        "macd_norm": feats["macd_norm"],
        "macd_signal_norm": feats["macd_signal_norm"],
        "macd_diff_norm": feats["macd_diff_norm"],
        "rsi_norm": feats["rsi_norm"],
        "bb_mid_norm": feats["bb_mid_norm"],
        "bb_high_norm": feats["bb_high_norm"],
        "bb_low_norm": feats["bb_low_norm"],
        "obv_norm": feats["obv_norm"],
        "sentiment_score": feats["sentiment_score"],
    }
    new = pd.DataFrame([row])
    if os.path.exists(out):
        old = pd.read_parquet(out)
        new = pd.concat([old, new]).drop_duplicates(subset=["ts"]).sort_values("ts")
    new.to_parquet(out, index=False)

# ---------- 1) REST 回补 ----------
@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(5))
def rest_fetch_klines(start_ms: int, end_ms: int, limit: int = 1000):
    params = {
        "symbol": SYMBOL, "interval": INTERVAL,
        "startTime": start_ms, "endTime": end_ms, "limit": limit
    }
    r = requests.get(REST_KLINES, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def backfill_klines(days: int = 3):
    """
    回补最近 days 天的 1m K 线（UTC 分钟收盘）
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    start_ms = to_utc_ms(start)
    end_ms   = to_utc_ms(end)

    print(f"[REST] backfill {SYMBOL} {INTERVAL}: {start} -> {end}")
    cur = start_ms
    while cur < end_ms:
        batch_end = min(cur + 1000*60*1000, end_ms)  # 每批最多 1000 根
        data = rest_fetch_klines(cur, batch_end)
        if not data:
            cur += 1000*60*1000
            continue
        df = kline_list_to_df(data)
        if not df.empty:
            # 按天落 raw
            for _, g in df.groupby(df["ts"].dt.strftime("%Y-%m-%d")):
                append_raw(g)
        last_close = data[-1][6]
        cur = last_close + 1  # 推进到下一毫秒，避免重叠
        time.sleep(0.2)
    merge_all_raw_to_one()
    print("[REST] backfill done.")

# ---------- 2) WebSocket 实时 ----------
async def stream_klines(on_bar_callback=None):
    """
    订阅 Binance 1m K 线；仅当 kline['x']==True（该分钟收盘）时触发回调/落盘
    """
    print(f"[WS] connect {WS_STREAM}")
    async for ws in websockets.connect(WS_STREAM, ping_interval=20, ping_timeout=20, max_size=1_000_000):
        try:
            async for msg in ws:
                m = json.loads(msg)
                if m.get("e") != "kline":
                    continue
                k = m["k"]
                if not k.get("x"):  # 该分钟尚未收盘
                    continue
                close_time_ms = k["T"]
                bar = {
                    "ts":     pd.to_datetime(close_time_ms, unit="ms", utc=True),
                    "open":   float(k["o"]),
                    "high":   float(k["h"]),
                    "low":    float(k["l"]),
                    "close":  float(k["c"]),
                    "volume": float(k["v"]),
                }
                df = pd.DataFrame([bar])
                append_raw(df)
                merge_all_raw_to_one()
                if on_bar_callback:
                    on_bar_callback(df.iloc[0])
        except Exception as e:
            print(f"[WS] error: {e}; reconnect in 3s...")
            await asyncio.sleep(3)
            continue

# ---------- 3) 将 K 线 → 15D 特征（含情绪） ----------
def make_on_bar_callback():
    ofe = OnlineFeatureEngine(
        symbol=SYMBOL,
        state_path=STATE_PATH,
        sentiment_1min_path=SENT_1MIN
    )
    def _cb(row):
        feats = ofe.update(
            o=row["open"], h=row["high"], l=row["low"], c=row["close"], v=row["volume"], ts_utc=row["ts"]
        )
        ofe.save_state()
        append_feature_row(
            ts=row["ts"], feats=feats,
            o=row["open"], h=row["high"], l=row["low"], c=row["close"], v=row["volume"]
        )
        # 你也可以在此把 feats 送入策略推断/纸交易执行网关
        print(f"[feats] {row['ts']} z={feats['z_score']:.3f} sent={feats['sentiment_score']:.3f}")
    return _cb

# ---------- 入口 ----------
if __name__ == "__main__":
    # 1) 盘前/启动先回补最近 N 天（确保指标预热 & 无缺口）
    backfill_klines(days=3)

    # 2) 实时订阅（分钟收盘产出）
    cb = make_on_bar_callback()
    asyncio.run(stream_klines(on_bar_callback=cb))
