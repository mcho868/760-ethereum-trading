# -*- coding: utf-8 -*-
"""
merge_eth_1m.py
合并 Binance ETHUSDT 1 分钟K线月包为一个 Parquet；自动识别时间戳单位；统一 UTC；去重。
Merge monthly Binance ETHUSDT 1-minute kline ZIPs into a single Parquet; auto-detect timestamp unit;
normalize to UTC; drop duplicates.

Usage / 用法:
    python merge_eth_1m.py <zip_dir> <out_dir>

It will produce / 将生成:
    <out_dir>/ETHUSDT_1m.parquet
"""

import os
import sys
import glob
import zipfile
import pandas as pd

# 原始 CSV 列名（Binance 公共数据的标准顺序）
# Column names in the raw Binance CSV (standard order)
COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]


def to_ts_auto(x):
    """
    将 open_time 自动识别为 ns/us/ms/s 或字符串时间；统一转换为 UTC 的 pandas.Timestamp 序列。
    Auto-detect whether `open_time` is in ns/us/ms/s (or string datetime) and convert to UTC Timestamps.

    返回 / return:
        ts: pd.Series of UTC timestamps
        unit: detected unit string (e.g., 'ms', 'us', 'ns', 's', or None)
    """
    num = pd.to_numeric(x, errors="coerce")           # 尝试转为数字 / try numeric
    med = num.dropna().median()                       # 用中位数判断量级 / use median to infer magnitude
    unit = None

    if pd.isna(med):
        # 本身就是可解析的字符串时间 / already a parseable datetime-like string
        ts = pd.to_datetime(x, utc=True, errors="coerce")
    else:
        # 依据数量级推断单位（经验阈值）/ infer unit by magnitude (heuristics)
        if med > 1e18:
            unit = "ns"
        elif med > 1e15:
            unit = "us"
        elif med > 1e12:
            unit = "ms"
        elif med > 1e9:
            unit = "s"

        if unit:
            ts = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(x, utc=True, errors="coerce")

    # 夹取一个合理区间，过滤异常时间戳 / clamp to a reasonable range to drop outliers
    lo = pd.Timestamp("2015-01-01", tz="UTC")
    hi = pd.Timestamp("2035-01-01", tz="UTC")
    ts = ts.where((ts >= lo) & (ts < hi))
    return ts, unit


def read_zip(zpath):
    """
    读取单个 ZIP 月包并返回清洗后的 DataFrame（只保留建模需要的列）
    Read one monthly ZIP and return a cleaned DataFrame (only modeling-needed columns).
    """
    with zipfile.ZipFile(zpath) as zf:
        # 找出其中的 CSV 文件（通常只有一个）
        # Locate the CSV file inside (usually one)
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            print(f"[WARN] no CSV inside {os.path.basename(zpath)}")
            return None

        with zf.open(csvs[0]) as f:
            # Binance 月包无表头，因此 header=None，names=COLS
            # No header in monthly CSV; provide names explicitly
            df = pd.read_csv(f, header=None, names=COLS)

    # 自动识别并转换 open_time → ts（UTC）
    # Auto-detect unit and convert open_time → ts (UTC)
    ts, unit = to_ts_auto(df["open_time"])
    df["ts"] = ts

    # 强制数值类型（避免字符串混入）
    # Enforce numeric types to avoid string contamination
    for c in ["open", "high", "low", "close", "volume", "number_of_trades"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    # 丢弃缺失关键字段的行；记录该月时间范围
    # Drop rows with missing key fields; compute the visible time range
    df = df.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
    rng = f"{df['ts'].min()} -> {df['ts'].max()}" if len(df) else "EMPTY"
    print(f"[READ] {os.path.basename(zpath):<24} rows={before}->{len(df)}  unit={unit}  range={rng}")

    # 仅保留建模必要列 / keep only necessary columns for modeling
    return df[["ts", "open", "high", "low", "close", "volume", "number_of_trades"]]


def main():
    # 参数检查 / argument check
    if len(sys.argv) < 3:
        print("Usage: python merge_eth_1m.py <zip_dir> <out_dir>")
        sys.exit(1)

    zip_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    # 收集该目录下所有 ETHUSDT 1m 的月度 ZIP 文件
    # Collect all monthly ZIPs for ETHUSDT 1m
    files = sorted(glob.glob(os.path.join(zip_dir, "ETHUSDT-1m-*.zip")))
    print(f"[INFO] found {len(files)} zips")
    print(f"[INFO] last 8:", [os.path.basename(f) for f in files[-8:]])

    dfs = []
    for z in files:
        d = read_zip(z)
        if d is not None and len(d):
            dfs.append(d)

    if not dfs:
        print("[ERROR] nothing read"); sys.exit(1)

    # 合并所有月份 → 去重（按 ts 保留最后出现）→ 按时间排序
    # Concatenate months → drop duplicates on ts (keep last) → sort by ts
    df = (pd.concat(dfs, ignore_index=True)
            .drop_duplicates(subset=["ts"], keep="last")
            .sort_values("ts"))

    # 类型修正与标记交易对 / fix dtype and set symbol
    df["number_of_trades"] = df["number_of_trades"].astype("Int64")
    df["symbol"] = "ETHUSDT"

    # 简要汇总日志 / summary log
    print("[SUM] rows=", len(df),
          "\nyears=", sorted(df["ts"].dt.year.unique().tolist()),
          "\n" + "tail=", df["ts"].iloc[-3:].tolist())

    # 输出 Parquet / write Parquet
    out = os.path.join(out_dir, "ETHUSDT_1m.parquet")
    df.to_parquet(out, index=False)
    print("[DONE] ->", out)


if __name__ == "__main__":
    main()
