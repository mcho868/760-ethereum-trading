# -*- coding: utf-8 -*-
"""
Project: Reddit submissions -> Clean + VADER sentiment + Visualizations
Usage:
  python sentiment_clean_and_score.py --in submissions_2025-07_eth.csv --outdir ./out
"""

import os
import re
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- utils ----------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def clean_text(text: str) -> str:
    """lowercase + remove URLs + remove punctuation + squeeze spaces"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)       # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)           # keep [a-z0-9] and space
    text = re.sub(r"\s+", " ", text).strip()
    return text

def read_csv_with_fallback(path: str) -> pd.DataFrame:
    """Try utf-8 first, fall back to latin-1 to avoid decode errors."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, encoding="latin-1")

# ---------- main ----------
def main(args):
    in_path  = args.input
    out_dir  = args.outdir
    ensure_dir(out_dir)

    # 1) load
    df = read_csv_with_fallback(in_path)

    # 2) cleaning
    # 2.1 drop unnamed columns
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
    # 2.2 drop rows with empty title
    if "title" in df.columns:
        df = df.dropna(subset=["title"])
    else:
        raise ValueError("Input CSV must contain a 'title' column.")

    # 2.3 merge and clean text
    df["selftext"] = df.get("selftext", "").fillna("")
    df["text_clean"] = (df["title"].astype(str) + " " + df["selftext"].astype(str)).apply(clean_text)

    # 2.4 timestamp -> readable
    if "created_utc" in df.columns:
        df["created_time_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")

    # 2.5 reorder columns (optional)
    cols_order = [c for c in ["id","subreddit","author","created_utc","created_time_utc",
                              "score","num_comments","title","selftext","text_clean"] if c in df.columns]
    df = df[cols_order]

    # 3) save cleaned
    base = os.path.splitext(os.path.basename(in_path))[0]
    cleaned_csv = os.path.join(out_dir, f"{base}_clean.csv")
    df.to_csv(cleaned_csv, index=False, encoding="utf-8")

    # 4) VADER sentiment
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df["text_clean"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

    # 5) save with sentiment
    with_sent_csv = os.path.join(out_dir, f"{base}_with_sentiment.csv")
    df.to_csv(with_sent_csv, index=False, encoding="utf-8")c

    # 6) visualizations
    # 6.1 sentiment histogram
    plt.figure(figsize=(8,5))
    plt.hist(df["sentiment"].dropna(), bins=40)
    plt.title("Sentiment (VADER compound) distribution")
    plt.xlabel("compound score [-1, 1]")
    plt.ylabel("count")
    plt.tight_layout()
    fig1 = os.path.join(out_dir, f"{base}_sentiment_hist.png")
    plt.savefig(fig1, dpi=150)
    plt.close()

    # 6.2 score histogram (log1p to reduce tail)
    if "score" in df.columns:
        score = pd.to_numeric(df["score"], errors="coerce").fillna(0).clip(lower=0)
        score_log1p = score.apply(math.log1p)
        plt.figure(figsize=(8,5))
        plt.hist(score_log1p, bins=40)
        plt.title("log1p(score) distribution")
        plt.xlabel("log1p(score)")
        plt.ylabel("count")
        plt.tight_layout()
        fig2 = os.path.join(out_dir, f"{base}_score_log1p_hist.png")
        plt.savefig(fig2, dpi=150)
        plt.close()

    # 6.3 sentiment vs. score scatter (log1p(score))
    if "score" in df.columns:
        plt.figure(figsize=(8,5))
        plt.scatter(score_log1p, df["sentiment"], s=6, alpha=0.5)
        plt.title("Sentiment vs. log1p(score)")
        plt.xlabel("log1p(score)")
        plt.ylabel("sentiment")
        plt.tight_layout()
        fig3 = os.path.join(out_dir, f"{base}_sentiment_vs_score.png")
        plt.savefig(fig3, dpi=150)
        plt.close()

    # 7) quick summary in console
    print(f"\nDone.")
    print(f"Cleaned CSV      -> {cleaned_csv}")
    print(f"With sentiment   -> {with_sent_csv}")
    print(f"Figures saved to -> {out_dir}")
    if "created_time_utc" in df.columns:
        print(f"Time range       -> {df['created_time_utc'].min()}  ~  {df['created_time_utc'].max()}")
    print(df[["id","subreddit","sentiment"]].head(5))

if __name__ == "__main__":
    # 这里写上你自己的默认输入/输出路径（不传参数时就用这里）
    DEFAULT_INPUT  = r"C:\Users\Jimmy\Desktop\760\submissions_2025-07_eth.csv"
    DEFAULT_OUTDIR = r"C:\Users\Jimmy\Desktop\760\cleaned"

    parser = argparse.ArgumentParser()
    parser.add_argument("--in",  dest="input",  required=False, help="path to input CSV")
    parser.add_argument("--outdir", required=False, help="output directory")
    args = parser.parse_args()

    # 如果没传参数，就用默认值
    if not args.input:
        args.input = DEFAULT_INPUT
    if not args.outdir:
        args.outdir = DEFAULT_OUTDIR

    main(args)
