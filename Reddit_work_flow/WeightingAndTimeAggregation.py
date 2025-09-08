# -*- coding: utf-8 -*-
"""
Project: Reddit submissions (with VADER sentiment) -> weighting -> daily aggregation -> 1min forward fill
Usage:
  python sentiment_weight_aggregate_to_minute.py --in INPUT_CSV --outdir OUTDIR
Input CSV must contain at least: created_utc, sentiment, score (optional num_comments)
"""

import os
import math
import argparse
import pandas as pd

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def read_csv_with_fallback(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, encoding="latin-1")

def compute_weight(df: pd.DataFrame,
                   score_col="score",
                   comments_col="num_comments") -> pd.Series:
    """
    你可以在这里调整权重公式：
      w = log1p(score) + 0.5 * log1p(num_comments)
    如没有评论列，就用 w = log1p(score)
    """
    score = pd.to_numeric(df.get(score_col, 0), errors="coerce").fillna(0).clip(lower=0)
    if comments_col in df.columns:
        numc = pd.to_numeric(df.get(comments_col, 0), errors="coerce").fillna(0).clip(lower=0)
        w = score.apply(math.log1p) + 0.5 * numc.apply(math.log1p)
    else:
        w = score.apply(math.log1p)
    # 避免全 0 导致后续除以 0，可按需加一个很小的下限
    return w

def main(args):
    in_path = args.input
    out_dir = args.outdir
    ensure_dir(out_dir)

    df = read_csv_with_fallback(in_path)

    # 基础字段检查
    if "sentiment" not in df.columns:
        raise ValueError("Input CSV must contain a 'sentiment' column (from previous VADER step).")
    if "created_utc" not in df.columns and "created_time_utc" not in df.columns:
        raise ValueError("Input CSV must contain 'created_utc' (unix ts) or 'created_time_utc' (ISO time).")

    # 统一生成时间列（UTC）
    if "created_time_utc" in df.columns:
        t = pd.to_datetime(df["created_time_utc"], errors="coerce", utc=False)
    else:
        t = pd.to_datetime(df["created_utc"], unit="s", errors="coerce", utc=False)

    df["created_time_utc"] = t
    # 过滤异常时间（Reddit 2005 年上线，规避 1970/空值等）
    df = df[df["created_time_utc"] >= "2005-01-01"]
    df = df.dropna(subset=["created_time_utc"])

    # 去重（如有重复 id）
    if "id" in df.columns:
        df = df.sort_values("created_time_utc").drop_duplicates("id", keep="last")

    # 权重
    df["weight"] = compute_weight(df, score_col="score", comments_col="num_comments")

    # 可选：裁剪 sentiment 异常值（VADER 理论范围[-1,1]，但稳妥起见）
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").clip(-1, 1)

    # === 日级聚合 ===
    df["date_utc"] = df["created_time_utc"].dt.date
    # 加权平均： sum(sent*w) / sum(w)；若某日全 w=0，则退化为简单均值
    def _daily_agg(g: pd.DataFrame) -> float:
        w = g["weight"].fillna(0)
        s = g["sentiment"].fillna(0)
        num = (s * w).sum()
        den = w.sum()
        if den > 0:
            return num / den
        # backup: 简单均值
        return s.mean()

    daily = df.groupby("date_utc").apply(_daily_agg).rename("sentiment").reset_index()

    # 保存日级
    daily_out = os.path.join(out_dir, "sentiment_daily.csv")
    daily.to_csv(daily_out, index=False, encoding="utf-8")

    # === 扩展到 1 分钟级 ===
    start = pd.to_datetime(str(daily["date_utc"].min()))
    end   = pd.to_datetime(str(daily["date_utc"].max())) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    idx_min = pd.date_range(start=start, end=end, freq="T")  # 每分钟

    # 把日级映射到分钟：先构造 DataFrame，再前向填充
    daily_ts = pd.Series(daily["sentiment"].values,
                         index=pd.to_datetime(daily["date_utc"]))
    # 先生成每天 00:00 的值，再 ffill
    minute_df = pd.DataFrame(index=idx_min)
    minute_df["sentiment"] = daily_ts.reindex(minute_df.index.normalize(), method="ffill").values

    # 保存分钟级（只有时间和分数）
    minute_out = os.path.join(out_dir, "sentiment_1min.csv")
    minute_df = minute_df.reset_index().rename(columns={"index": "time_utc"})
    minute_df.to_csv(minute_out, index=False, encoding="utf-8")

    # 控制台摘要
    print("Done.")
    print(f"Input              : {in_path}")
    print(f"Days covered       : {daily['date_utc'].min()}  ~  {daily['date_utc'].max()}")
    print(f"Daily CSV          : {daily_out}")
    print(f"1-min CSV          : {minute_out}")
    print("\nDaily preview:")
    print(daily.head(3))
    print("\n1-min preview:")
    print(minute_df.head(3))

if __name__ == "__main__":
    # 输入文件（你的情绪打分结果）
    in_path = r"C:\Users\Jimmy\Desktop\760\cleaned\submissions_2025-07_eth_with_sentiment.csv"

    # 输出目录
    out_dir = r"C:\Users\Jimmy\Desktop\760\weighted"
    os.makedirs(out_dir, exist_ok=True)

    class Args:  # 模拟 argparse 的 args
        input = in_path
        outdir = out_dir

    args = Args()
    main(args)
