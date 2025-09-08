# extract_rs_month_eth.py
# -*- coding: utf-8 -*-
import os, json, re
import zstandard as zstd

# === 改这里：你的RS文件路径 ===
IN_PATH  = r"submissions\RS_2025-06.zst"   # 例：C:\Users\Jimmy\Desktop\760\dumps\RS_2025-07.zst
OUT_DIR  = r"out"
OUT_DATA = os.path.join(OUT_DIR, "submissions_2025-06_eth.jsonl")
OUT_IDS  = os.path.join(OUT_DIR, "submission_ids_2025-06.txt")

# 关键词&子版（按需改）
PAT = re.compile(r"\b(ethereum|eth)\b", re.IGNORECASE)
KEEP_SUBS = {"ethereum", "ethtrader", "CryptoCurrency"}  # 不想限制就设为 None

os.makedirs(OUT_DIR, exist_ok=True)

def keep(obj):
    if KEEP_SUBS and str(obj.get("subreddit","")).lower() not in KEEP_SUBS:
        return False
    title = obj.get("title") or ""
    selftext = obj.get("selftext") or ""
    return PAT.search(title + "\n" + selftext) is not None

def project(obj):
    return {
        "id": obj.get("id"),
        "title": obj.get("title"),
        "selftext": obj.get("selftext"),
        "score": obj.get("score"),
        "num_comments": obj.get("num_comments"),
        "created_utc": obj.get("created_utc"),
        "subreddit": obj.get("subreddit"),
        "author": obj.get("author"),
    }

def main():
    kept = 0; total = 0
    ids = []
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    with open(IN_PATH, "rb") as fh, \
         dctx.stream_reader(fh) as reader, \
         open(OUT_DATA, "w", encoding="utf-8") as f_out:
        buf = b""
        while True:
            chunk = reader.read(2**20)
            if not chunk: break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line: continue
                total += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if keep(obj):
                    rec = project(obj)
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if rec["id"]:
                        ids.append(rec["id"])
                    kept += 1

    with open(OUT_IDS, "w", encoding="utf-8") as f_ids:
        for i in ids:
            f_ids.write(i + "\n")

    print(f"RS done. scanned={total:,}, kept={kept:,}")
    print(f" -> {OUT_DATA}")
    print(f" -> {OUT_IDS}  (给评论解析用)")

if __name__ == "__main__":
    main()
