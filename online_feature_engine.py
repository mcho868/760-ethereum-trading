# online_feature_engine.py
# 在线 15D 特征引擎（与线下口径一致；分钟级 O(1) 更新；T+1 日度情绪分钟化并入）
# 依赖：pandas, numpy

import json
from pathlib import Path
import pandas as pd
import numpy as np

class OnlineFeatureEngine:
    """
    每分钟收盘时，调用 update(o,h,l,c,v,ts_utc) 即可返回一行与线下 15D 对齐的特征：
      - core(4D): z_score, zone_norm, price_momentum, z_score_momentum
      - tech(8D): macd_norm, macd_signal_norm, macd_diff_norm, rsi_norm,
                  bb_mid_norm, bb_high_norm, bb_low_norm, obv_norm
      - senti(1D): sentiment_score（来源：data/reddit/weighted/sentiment_1min_vader_s1_s5.csv）
    注意：position / position_change 由你的交易环境补齐，不在此引擎输出。
    """

    def __init__(
        self,
        symbol: str = "ETHUSDT",
        state_path: str = "state/indicators_ETHUSDT.json",
        ma_period: int = 20,
        z_win: int = 60,
        open_th: float = 1.5,
        close_th: float = 0.5,
        sentiment_1min_path: str = "data/reddit/weighted/sentiment_1min_vader_s1_s5.csv",
    ):
        self.symbol = symbol
        self.state_path = Path(state_path)
        self.ma_period = ma_period
        self.z_win = z_win
        self.open_th = open_th
        self.close_th = close_th

        self.state = self._load_state()

        # 预加载分钟情绪（UTC 时间戳为 index；今天使用的应是 T+1 日度定值分钟化表）
        self.sent = None
        sp = Path(sentiment_1min_path)
        if sp.exists():
            s = pd.read_csv(sp, parse_dates=["ts"])
            s = s.set_index("ts").sort_index()
            # 你的分钟情绪表里通常有 vader, s1..s5：取列的均值作为单一 sentiment_score
            sent_cols = [c for c in s.columns if c.lower() in ("vader","s1","s2","s3","s4","s5")]
            if sent_cols:
                s["sentiment_score"] = s[sent_cols].mean(axis=1)
                self.sent = s["sentiment_score"]

    # ---------- 状态 ----------
    def _default_state(self):
        return {
            "prev_close": None,
            # SMA for pseudo-spread baseline
            "ma_sum": 0.0, "ma_q": [],
            # rolling spread window for z-score
            "spread_q": [],
            # MACD(12,26,9)
            "ema12": None, "ema26": None, "macd_signal": None,
            # RSI(14) Wilder 平滑
            "rsi_avg_gain": None, "rsi_avg_loss": None,
            # BOLL(20,2) 维护滚动均值/方差
            "boll_q": [], "boll_sum": 0.0, "boll_sumsq": 0.0, "boll_n": 0,
            # OBV
            "obv": 0.0,
            # 缓存
            "prev_z": None,
        }

    def _load_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except Exception:
                pass
        return self._default_state()

    def save_state(self):
        self.state_path.write_text(json.dumps(self.state))

    # ---------- helpers ----------
    @staticmethod
    def _ema_step(prev, x, span):
        alpha = 2/(span+1)
        return x if prev is None else (prev + alpha*(x - prev))

    @staticmethod
    def _wilder(prev, x, n):
        return x if prev is None else ((prev*(n-1) + x)/n)

    def _zone_from_z(self, z):
        if z > self.open_th:
            return 1.0
        elif z > self.close_th:
            return 0.5
        elif z >= -self.close_th:
            return 0.0
        elif z >= -self.open_th:
            return -0.5
        else:
            return -1.0

    def _get_sentiment(self, ts_utc: pd.Timestamp) -> float:
        if self.sent is None:
            return 0.0
        try:
            v = self.sent.asof(ts_utc)  # 取 <= ts 最近值（你的分钟表每日定值，asof 合理）
            return float(v) if pd.notna(v) else 0.0
        except Exception:
            return 0.0

    # ---------- 核心：每分钟收盘更新 ----------
    def update(self, o, h, l, c, v, ts_utc: pd.Timestamp) -> dict:
        st = self.state

        # OBV
        if st["prev_close"] is not None:
            if c > st["prev_close"]:
                st["obv"] += float(v)
            elif c < st["prev_close"]:
                st["obv"] -= float(v)
        obv = st["obv"]

        # pseudo-spread baseline: SMA(ma_period)
        q = st["ma_q"]
        if len(q) == self.ma_period:
            st["ma_sum"] -= q.pop(0)
        q.append(c)
        st["ma_sum"] += c
        ma_baseline = st["ma_sum"] / len(q)

        spread = c - ma_baseline
        sq = st["spread_q"]
        sq.append(spread)
        if len(sq) > self.z_win:
            sq.pop(0)
        arr = np.array(sq, dtype=float)
        if arr.size >= 2:
            z = (spread - arr.mean()) / (arr.std(ddof=0) + 1e-8)
        else:
            z = 0.0
        zone = self._zone_from_z(z)

        price_mom = 0.0 if st["prev_close"] is None else np.clip((c - st["prev_close"]) / st["prev_close"], -0.1, 0.1)
        prev_z = st["prev_z"]
        z_mom = 0.0 if prev_z is None else np.clip(z - prev_z, -2.0, 2.0)
        st["prev_z"] = z

        # MACD(12,26,9)
        st["ema12"] = self._ema_step(st["ema12"], c, 12)
        st["ema26"] = self._ema_step(st["ema26"], c, 26)
        macd = None if (st["ema12"] is None or st["ema26"] is None) else (st["ema12"] - st["ema26"])
        st["macd_signal"] = None if macd is None else self._ema_step(st["macd_signal"], macd, 9)
        macd_sig = st["macd_signal"]
        macd_hist = None if (macd is None or macd_sig is None) else (macd - macd_sig)

        # RSI(14) Wilder
        if st["prev_close"] is None:
            gain = loss = None
        else:
            ch = c - st["prev_close"]
            gain = max(ch, 0.0); loss = max(-ch, 0.0)
        st["rsi_avg_gain"] = None if gain is None else self._wilder(st["rsi_avg_gain"], gain, 14)
        st["rsi_avg_loss"] = None if loss is None else self._wilder(st["rsi_avg_loss"], loss, 14)
        rsi = None
        if st["rsi_avg_gain"] is not None and st["rsi_avg_loss"] is not None and st["rsi_avg_loss"] != 0:
            rs = st["rsi_avg_gain"] / st["rsi_avg_loss"]
            rsi = 100 - 100/(1+rs)
        rsi_norm = 0.0 if rsi is None else (rsi / 100.0)

        # BOLL(20,2)
        bq = st["boll_q"]; n_old = st["boll_n"]
        if n_old == 20:
            old = bq.pop(0)
            st["boll_sum"]   -= old
            st["boll_sumsq"] -= old*old
            n_old -= 1
        bq.append(c)
        st["boll_sum"]   += c
        st["boll_sumsq"] += c*c
        n_old += 1
        st["boll_n"] = n_old

        if n_old >= 1:
            mean = st["boll_sum"]/n_old
            var  = max(st["boll_sumsq"]/n_old - mean*mean, 0.0)
            std  = var**0.5
        else:
            mean = c; std = 0.0

        bb_mid = mean
        bb_up  = bb_mid + 2*std
        bb_low = bb_mid - 2*std

        # 情绪（分钟定值）
        sent_score = self._get_sentiment(ts_utc)

        # update prev_close
        st["prev_close"] = c

        feats = {
            # core
            "z_score": z,
            "zone_norm": zone,
            "price_momentum": price_mom,
            "z_score_momentum": z_mom,
            # tech
            "macd_norm": 0.0 if macd is None else macd,
            "macd_signal_norm": 0.0 if macd_sig is None else macd_sig,
            "macd_diff_norm": 0.0 if (macd is None or macd_sig is None) else (macd - macd_sig),
            "rsi_norm": rsi_norm,
            "bb_mid_norm": bb_mid,
            "bb_high_norm": bb_up,
            "bb_low_norm": bb_low,
            "obv_norm": obv,
            # sentiment
            "sentiment_score": sent_score
        }
        return feats
