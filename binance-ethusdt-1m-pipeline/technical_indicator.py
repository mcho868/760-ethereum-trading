import pandas as pd
import ta  # 技术指标库 / technical analysis library
from sklearn.preprocessing import MinMaxScaler  # 数据归一化工具 / normalization tool

# === 读取 parquet 文件 / Read parquet file ===
df = pd.read_parquet("out/ETHUSDT_1m.parquet")

# === 确认 ts 是 datetime 并设为索引 / Ensure 'ts' is datetime and set as index ===
df["ts"] = pd.to_datetime(df["ts"], utc=True)
df = df.set_index("ts").sort_index()

# === 技术指标计算 / Technical indicators calculation ===

# 1. RSI (相对强弱指数 / Relative Strength Index)
df["RSI"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

# 2. 布林带 / Bollinger Bands
boll = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
df["BB_mid"] = boll.bollinger_mavg()    # 中轨 / Middle band
df["BB_high"] = boll.bollinger_hband()  # 上轨 / Upper band
df["BB_low"]  = boll.bollinger_lband()  # 下轨 / Lower band

# 3. EMA (指数移动平均线 / Exponential Moving Average)
df["EMA_12"] = ta.trend.EMAIndicator(close=df["close"], window=12).ema_indicator()
df["EMA_26"] = ta.trend.EMAIndicator(close=df["close"], window=26).ema_indicator()

# 4. MACD (指数平滑异同移动平均线 / Moving Average Convergence Divergence)
macd = ta.trend.MACD(close=df["close"])
df["MACD"] = macd.macd()                 # MACD 主线 / MACD main line
df["MACD_signal"] = macd.macd_signal()   # 信号线 / Signal line
df["MACD_diff"] = macd.macd_diff()       # 柱状图差值 / Histogram difference

# 5. ATR (平均真实波幅 / Average True Range)
atr = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
df["ATR"] = atr.average_true_range()

# === 归一化到 [0,1] 区间 / Normalize all indicators to [0,1] ===
scaler = MinMaxScaler()

# 要归一化的列 / Columns to normalize
cols_to_scale = ["RSI", "BB_mid", "BB_high", "BB_low",
                 "EMA_12", "EMA_26", "MACD", "MACD_signal", "MACD_diff", "ATR"]

# 创建归一化版本副本 / Create a scaled copy
df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# === 保存完整 parquet 文件 / Save full parquet file ===
df_scaled.to_parquet("out/ETHUSDT_1m_with_indicators_scaled.parquet")

# === 保存前500行 CSV / Save first 500 rows to CSV ===
df_scaled.head(500).to_csv("out/ethusdt_indicators_scaled_head500.csv", index=True)

print("✅ 已保存归一化后的 parquet 和前500行 CSV / Normalized parquet and first 500 rows CSV saved")
