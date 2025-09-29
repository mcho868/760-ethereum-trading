import asyncio
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import praw
import requests
from dotenv import load_dotenv
from nltk import download as nltk_download
from nltk import data as nltk_data
from nltk.sentiment import SentimentIntensityAnalyzer
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    target_date: date
    fetched_posts: int
    scored_rows: int
    master_path: Optional[Path]
    daily_path: Optional[Path]
    minute_path: Optional[Path]
    skipped: bool
    message: str
    runtime_seconds: float


class LMStudioModelManager:
    def __init__(self, base_url: str, api_key: Optional[str], enabled: bool, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.enabled = enabled
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def load(self, model: str):
        if not self.enabled:
            return
        try:
            requests.post(
                f"{self.base_url}/models/load",
                json={"model": model},
                headers=self._headers(),
                timeout=self.timeout,
            )
        except Exception as exc:
            logger.debug("LM Studio load failed for %s: %s", model, exc)

    def unload(self, model: str):
        if not self.enabled:
            return
        try:
            requests.post(
                f"{self.base_url}/models/unload",
                json={"model": model},
                headers=self._headers(),
                timeout=self.timeout,
            )
        except Exception as exc:
            logger.debug("LM Studio unload failed for %s: %s", model, exc)


class RealTimeSentimentPipeline:
    DEFAULT_SUBREDDITS = ["ethereum", "ethfinance", "ethtrader", "CryptoCurrency", "defi", "ethdev"]
    DEFAULT_KEYWORDS = [
        r"\beth\b",
        r"\bethereum\b",
        r"\beth/usdt\b",
        r"\bethusd\b",
        r"\bethusdt\b",
        r"\bspot\b",
        r"\bfutures?\b",
        r"\bperps?\b",
        r"\btrade|trading|trader\b",
        r"\bposition\b",
        r"\blong\b",
        r"\bshort\b",
        r"\bentry\b",
        r"\bexit\b",
        r"\bmarket\b",
        r"\border\b",
        r"\bliquidation\b",
        r"\bhedge\b",
        r"\bleverage\b",
    ]

    DEFAULT_MODEL_SEQUENCE = [
        {"model": "meta-llama-3.1-8b-instruct", "column": "s1"},
        {"model": "google/gemma-2-9b", "column": "s2", "fallback_to_vader_sign": True},
        {"model": "qwen2.5-7b-instruct-1m", "column": "s3"},
        {"model": "mistralai/mistral-7b-instruct-v0.3", "column": "s4"},
        {"model": "nous-hermes-2-mistral-7b-dpo", "column": "s5"},
    ]

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        subreddits: Optional[Iterable[str]] = None,
        keyword_patterns: Optional[Iterable[str]] = None,
        lmstudio_config: Optional[Dict] = None,
        model_sequence: Optional[List[Dict]] = None,
        reddit_credentials: Optional[Dict[str, str]] = None,
    ):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
        self.subreddits = list(subreddits) if subreddits else self.DEFAULT_SUBREDDITS
        patterns = list(keyword_patterns) if keyword_patterns else self.DEFAULT_KEYWORDS
        self.keyword_regex = re.compile("|".join(patterns), flags=re.IGNORECASE)
        self.max_per_subreddit = 5000
        self.request_sleep = 0.5
        self.model_sequence = model_sequence or self.DEFAULT_MODEL_SEQUENCE

        self.yesterday_dir = self.base_dir / "data" / "reddit" / "yesterday"
        self.processed_dir = self.base_dir / "data" / "reddit" / "processed"
        self.cleaned_dir = self.base_dir / "cleaned"
        self.scored_dir = self.base_dir / "data" / "reddit" / "scored"
        self.weighted_dir = self.base_dir / "data" / "reddit" / "weighted"
        for path in [self.yesterday_dir, self.processed_dir, self.cleaned_dir, self.scored_dir, self.weighted_dir]:
            path.mkdir(parents=True, exist_ok=True)

        load_dotenv(self.base_dir / ".env")
        reddit_credentials = reddit_credentials or {}
        self.reddit_client_id = reddit_credentials.get("client_id") or os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = reddit_credentials.get("client_secret") or os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = reddit_credentials.get("user_agent") or os.getenv("REDDIT_USER_AGENT", "eth-sentiment-bot/0.1")
        if not self.reddit_client_id or not self.reddit_client_secret:
            raise RuntimeError("Reddit credentials missing: set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env or pass via config")

        lmstudio_config = lmstudio_config or {}
        base_url = lmstudio_config.get("base_url", "http://127.0.0.1:1234/v1")
        api_key = lmstudio_config.get("api_key", "lm-studio")
        concurrency = lmstudio_config.get("concurrency", 2)
        self.model_concurrency = max(1, int(concurrency))
        self.model_retry = lmstudio_config.get("retry", 3)
        max_len = lmstudio_config.get("max_text_len", 256)
        self.max_text_len = max(32, int(max_len))
        manage_models = lmstudio_config.get("manage_models", True)
        self.model_manager = LMStudioModelManager(base_url, api_key, manage_models)
        self.lmstudio_base_url = base_url
        self.lmstudio_api_key = api_key

        self.sentiment_prompt = (
            "Classify the sentiment as exactly one token from {{POS, NEU, NEG}}.\n"
            "Text:\n{text}\n"
            "Answer:"
        )
        self.sentiment_grammar = 'root ::= "POS" | "NEU" | "NEG"'

    async def run(self, target_date: Optional[date] = None, force: bool = False) -> PipelineResult:
        start_ts = time.time()
        target_date = target_date or (datetime.now(timezone.utc).date() - timedelta(days=1))
        skipped = False
        message = ""
        fetched_posts = 0
        scored_rows = 0
        master_path = None
        daily_path = None
        minute_path = None

        if not force and self._is_already_processed(target_date):
            skipped = True
            message = f"Sentiment already processed for {target_date}"
            master_path = self.scored_dir / f"posts_scores_{target_date}.csv"
            daily_path, minute_path = self._latest_aggregated_paths()
            runtime = time.time() - start_ts
            return PipelineResult(target_date, 0, 0, master_path, daily_path, minute_path, skipped, message, runtime)

        loop = asyncio.get_running_loop()
        rows_df, raw_path = await loop.run_in_executor(None, self._fetch_reddit_posts, target_date)
        fetched_posts = len(rows_df)
        if fetched_posts == 0:
            message = f"No Reddit posts found for {target_date}"
        self._save_dataframe(rows_df, raw_path)

        normalized_df, processed_path = self._normalize(rows_df, target_date)
        cleaned_df, cleaned_path = self._clean(processed_path)
        master_df, master_path = self._score_vader(cleaned_df, target_date)

        scored_rows += len(master_df)

        for cfg in self.model_sequence:
            model_name = cfg.get("model")
            logger.info(f"Scoring with model: {model_name}...")
            try:
                scored_rows += await self._score_single_model(master_path, cfg)
                logger.info(f"Finished scoring with model: {model_name}.")
            except Exception as exc:
                logger.warning("Model %s failed:", model_name, exc_info=True)

        try:
            daily_path, minute_path = self._aggregate_weighted()
        except Exception as exc:
            logger.warning("Aggregation failed: %s", exc)

        runtime = time.time() - start_ts
        if not message:
            message = f"Sentiment pipeline completed for {target_date}"
        return PipelineResult(target_date, fetched_posts, scored_rows, master_path, daily_path, minute_path, skipped, message, runtime)

    def _is_already_processed(self, target_date: date) -> bool:
        target_str = str(target_date)
        daily_path = self.weighted_dir / "sentiment_daily_vader_s1_s5.csv"
        if daily_path.exists():
            try:
                df = pd.read_csv(daily_path, parse_dates=["ts"])
                if not df.empty and target_str in df["ts"].dt.date.astype(str).values:
                    return True
            except Exception:
                return False
        return False

    def _latest_aggregated_paths(self) -> Tuple[Optional[Path], Optional[Path]]:
        daily = self.weighted_dir / "sentiment_daily_vader_s1_s5.csv"
        minute = self.weighted_dir / "sentiment_1min_vader_s1_s5.csv"
        return (daily if daily.exists() else None, minute if minute.exists() else None)

    def get_last_processed_date(self) -> Optional[date]:
        """
        Finds the most recent date in the aggregated daily sentiment file.
        Returns the date object or None if not found.
        """
        daily_path = self.weighted_dir / "sentiment_daily_vader_s1_s5.csv"
        if not daily_path.exists():
            return None
        try:
            df = pd.read_csv(daily_path, parse_dates=["ts"])
            if df.empty or "ts" not in df.columns:
                return None
            # Ensure 'ts' is datetime before calling .dt
            if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
                 df["ts"] = pd.to_datetime(df["ts"], errors='coerce', utc=True)
            
            # Drop NaT values and find max
            valid_dates = df["ts"].dropna()
            if valid_dates.empty:
                return None
            
            return valid_dates.max().date()
        except Exception as e:
            logger.warning(f"Could not read last processed date from {daily_path}: {e}")
            return None

    def _fetch_reddit_posts(self, target_date: date) -> Tuple[pd.DataFrame, Path]:
        start = datetime.combine(target_date, dt_time.min, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        reddit = praw.Reddit(
            client_id=self.reddit_client_id,
            client_secret=self.reddit_client_secret,
            user_agent=self.reddit_user_agent,
            ratelimit_seconds=5,
        )
        reddit.read_only = True

        rows: List[Dict] = []
        for sub in self.subreddits:
            logger.info("Collecting from r/%s...", sub)
            count = 0
            try:
                for submission in reddit.subreddit(sub).new(limit=None):
                    created_ts = getattr(submission, "created_utc", None)
                    if created_ts is None:
                        continue
                    created = datetime.fromtimestamp(created_ts, tz=timezone.utc)
                    if created >= end:
                        continue
                    if created < start:
                        break
                    title = submission.title or ""
                    selftext = submission.selftext or ""
                    if not self.keyword_regex.search(f"{title}\n{selftext}"):
                        continue
                    rows.append(
                        {
                            "id": submission.id,
                            "subreddit": submission.subreddit.display_name,
                            "created_utc": created.isoformat(),
                            "title": title,
                            "selftext": selftext,
                            "url": submission.url or "",
                            "is_self": bool(submission.is_self),
                            "author": str(submission.author) if submission.author else None,
                            "score": int(submission.score or 0),
                            "upvote_ratio": float(submission.upvote_ratio or 0.0),
                            "num_comments": int(submission.num_comments or 0),
                            "over_18": bool(getattr(submission, "over_18", False)),
                            "stickied": bool(getattr(submission, "stickied", False)),
                            "crosspost_parent": getattr(submission, "crosspost_parent", None),
                            "permalink": f"https://www.reddit.com{submission.permalink}" if getattr(submission, "permalink", None) else "",
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                            "fetch_window_start": start.isoformat(),
                            "fetch_window_end": end.isoformat(),
                            "fetch_version": "submissions_only_v1",
                        }
                    )
                    count += 1
                    if count > 0 and count % 100 == 0:
                        logger.info("... matched %d posts from r/%s", count, sub)
                    if count >= self.max_per_subreddit:
                        break
                    time.sleep(self.request_sleep)
            except Exception as exc:
                logger.warning("Failed to collect r/%s: %s", sub, exc)

        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values(["id", "score", "fetched_at"], ascending=[True, False, False], inplace=True)
            df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
            df["is_crosspost"] = df["crosspost_parent"].notna()
        filename = f"reddit_eth_submissions_{target_date}.csv"
        return df, self.yesterday_dir / filename

    def _save_dataframe(self, df: pd.DataFrame, path: Path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, path)

    def _normalize(self, df: pd.DataFrame, target_date: date) -> Tuple[pd.DataFrame, Path]:
        keep_fields = [
            "id",
            "author",
            "subreddit",
            "created_utc",
            "created",
            "created_time_utc",
            "title",
            "selftext",
            "body",
            "url",
            "permalink",
            "score",
            "upvote_ratio",
            "num_comments",
            "num_crossposts",
            "over_18",
            "is_self",
        ]
        out = pd.DataFrame(columns=keep_fields)
        if not df.empty:
            out = pd.DataFrame(index=df.index, columns=keep_fields)
            out["id"] = df.get("id")
            out["author"] = df.get("author")
            out["subreddit"] = df.get("subreddit")
            out["created_utc"] = df.get("created_utc")
            ts = pd.to_datetime(out["created_utc"], errors="coerce", utc=True)
            out["created"] = ts.dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M:%S")
            out["created_time_utc"] = out["created_utc"]
            out["title"] = df.get("title", "").fillna("")
            out["selftext"] = df.get("selftext", "").fillna("")
            out["body"] = ""
            out["url"] = df.get("url", "").fillna("")
            out["permalink"] = df.get("permalink", "").fillna("")
            out["score"] = df.get("score", 0).fillna(0).astype("Int64")
            out["upvote_ratio"] = df.get("upvote_ratio", 0.0).fillna(0.0)
            out["num_comments"] = df.get("num_comments", 0).fillna(0).astype("Int64")
            if "num_crossposts" in df.columns:
                out["num_crossposts"] = df["num_crossposts"].fillna(0).astype("Int64")
            else:
                out["num_crossposts"] = df.get("crosspost_parent").notna().astype(int)
            out["over_18"] = df.get("over_18", False).fillna(False).astype(bool)
            out["is_self"] = df.get("is_self", False).fillna(False).astype(bool)
            out = out.sort_values(["id", "score"], ascending=[True, False]).drop_duplicates("id").reset_index(drop=True)
        out_path = self.processed_dir / f"reddit_eth_standard_{target_date}.csv"
        self._save_dataframe(out, out_path)
        return out, out_path

    def _clean(self, processed_path: Path) -> Tuple[pd.DataFrame, Path]:
        try:
            df = pd.read_csv(processed_path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(processed_path, encoding="latin-1", low_memory=False)
        if df.empty:
            out_path = self.cleaned_dir / f"{processed_path.stem}_clean.csv"
            self._save_dataframe(df, out_path)
            return df, out_path

        if "title" not in df.columns:
            raise ValueError("Processed CSV missing 'title' column")
        if "selftext" not in df.columns:
            df["selftext"] = ""
        if "body" not in df.columns:
            df["body"] = ""

        df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
        df = df.dropna(subset=["title"])
        df["text_raw"] = df["title"].astype(str).fillna("") + " " + df["selftext"].astype(str).fillna("")
        mask_short = df["text_raw"].str.len().fillna(0) < 3
        df.loc[mask_short, "text_raw"] = df.loc[mask_short, "text_raw"] + " " + df["body"].astype(str)
        df["text_raw"] = df["text_raw"].astype(str).str.slice(0, 20000)
        df["text_clean"] = df["text_raw"].map(self._clean_text)
        df = df[df["text_clean"].str.len() >= 5].copy()

        if "created_time_utc" not in df.columns:
            if "created_utc" in df.columns:
                df["created_time_utc"] = self._parse_datetime(df["created_utc"])
            elif "created" in df.columns:
                df["created_time_utc"] = self._parse_datetime(df["created"])
            else:
                df["created_time_utc"] = pd.NaT
        else:
            df["created_time_utc"] = self._parse_datetime(df["created_time_utc"])

        if df["created_time_utc"].notna().any():
            df["year_month"] = pd.to_datetime(df["created_time_utc"]).dt.strftime("%Y-%m")
        elif "source_file" in df.columns:
            df["year_month"] = df["source_file"].str.extract(r"((20\d{2})[-_](\d{2}))")[0]
        else:
            df["year_month"] = None

        before = len(df)
        if "id" in df.columns:
            df = df.sort_values(["id", "score"] if "score" in df.columns else ["id"]).drop_duplicates("id")
        else:
            keys = [k for k in ["title", "created_time_utc", "subreddit"] if k in df.columns]
            df = df.drop_duplicates(subset=keys) if keys else df.drop_duplicates()
        logger.info("Cleaned posts: %s -> %s", before, len(df))
        out_path = self.cleaned_dir / f"{processed_path.stem}_clean.csv"
        self._save_dataframe(df, out_path)
        return df, out_path

    def _clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        s = str(text)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
        s = re.sub(r"http\S+|www\.\S+", " ", s, flags=re.IGNORECASE)
        s = re.sub(r"<[^>]+>", " ", s)
        s = re.sub(r"[^A-Za-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    def _parse_datetime(self, series: pd.Series) -> pd.Series:
        ts = pd.to_datetime(series, errors="coerce", utc=True)
        if ts.notna().sum() >= len(series) * 0.5:
            return ts
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            unit = "ms" if numeric.dropna().median() > 10 ** 12 else "s"
            return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
        return ts

    def _score_vader(self, cleaned_df: pd.DataFrame, target_date: date) -> Tuple[pd.DataFrame, Path]:
        if cleaned_df.empty:
            master = pd.DataFrame()
            master_path = self.scored_dir / f"posts_scores_{target_date}.csv"
            self._save_dataframe(master, master_path)
            return master, master_path

        base_cols = [
            "id",
            "subreddit",
            "created_time_utc",
            "title",
            "selftext",
            "body",
            "text_clean",
            "score",
            "num_comments",
            "upvote_ratio",
            "permalink",
            "url",
        ]
        df = cleaned_df[[c for c in base_cols if c in cleaned_df.columns]].copy()
        if "text_clean" in df.columns:
            text_col = "text_clean"
        else:
            text_col = "__text_tmp__"
            df[text_col] = [
                " ".join([str(x) for x in xs if pd.notna(x)])
                for xs in zip(df.get("title", ""), df.get("selftext", ""), df.get("body", ""))
            ]

        try:
            nltk_data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk_download("vader_lexicon")

        sia = SentimentIntensityAnalyzer()
        df["vader"] = df[text_col].map(lambda t: sia.polarity_scores(str(t))["compound"])
        master_path = self.scored_dir / f"posts_scores_{target_date}.csv"
        self._save_dataframe(df, master_path)
        return df, master_path

    async def _score_single_model(self, master_path: Path, cfg: Dict) -> int:
        model_name = cfg.get("model")
        out_col = cfg.get("column")
        if not model_name or not out_col:
            return 0
        try:
            df = pd.read_csv(master_path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(master_path, encoding="latin-1", low_memory=False)
        if df.empty:
            return 0
        text_col = "text_clean" if "text_clean" in df.columns else None
        if text_col is None:
            df["__text_tmp__"] = [
                " ".join([str(x) for x in xs if pd.notna(x)])
                for xs in zip(df.get("title", ""), df.get("selftext", ""), df.get("body", ""))
            ]
            text_col = "__text_tmp__"
        if out_col not in df.columns:
            df[out_col] = np.nan
        mask = df[out_col].isna()
        if cfg.get("vader_edge") is not None and "vader" in df.columns:
            mask &= df["vader"].abs() < cfg["vader_edge"]
        todo = df.loc[mask].copy()
        if todo.empty:
            return 0
        todo["__txt"] = todo[text_col].astype(str).str.slice(0, self.max_text_len)
        groups = todo.groupby("__txt").indices
        unique_texts = list(groups.keys())

        self.model_manager.load(model_name)
        try:
            scores_map = await self._run_model_requests(unique_texts, model_name)
        finally:
            self.model_manager.unload(model_name)

        for text, indices in groups.items():
            df.loc[indices, out_col] = scores_map.get(text, np.nan)

        fallback_to_vader = cfg.get("fallback_to_vader_sign", False)
        if fallback_to_vader and df[out_col].isna().any() and "vader" in df.columns:
            sign = np.sign(df["vader"].fillna(0.0))
            df.loc[df[out_col].isna(), out_col] = sign.replace(0, 0.0)

        before_missing = mask.sum()
        remaining_missing = df[out_col].isna().sum()
        tmp = master_path.with_suffix(master_path.suffix + ".tmp")
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, master_path)
        return max(0, before_missing - remaining_missing)

    async def _run_model_requests(self, texts: List[str], model_name: str) -> Dict[str, float]:
        if not texts:
            return {}
        
        semaphore = asyncio.Semaphore(self.model_concurrency)
        results: Dict[str, float] = {}
        url = f"{self.lmstudio_base_url.rstrip('/')}/chat/completions"

        async def classify(session: aiohttp.ClientSession, text: str) -> Tuple[str, float]:
            prompt = self.sentiment_prompt.format(text=text)
            
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 3,  # As requested
                "grammar": self.sentiment_grammar
            }

            for attempt in range(self.model_retry):
                try:
                    async with session.post(url, json=payload, timeout=30) as response:
                        response.raise_for_status()
                        json_response = await response.json()
                        label = json_response["choices"][0]["message"]["content"]
                        return text, self._map_label_to_score(label)
                except Exception as exc:
                    await asyncio.sleep(0.6 * (attempt + 1))
                    if attempt == self.model_retry - 1:
                        logger.warning("Model %s failed on text: %s", model_name, exc)
                        return text, float("nan")
            return text, float("nan")

        async def worker(session: aiohttp.ClientSession, text: str):
            async with semaphore:
                result = await classify(session, text)
                results[result[0]] = result[1]

        async with aiohttp.ClientSession() as session:
            tasks = [asyncio.create_task(worker(session, text)) for text in texts]
            await asyncio.gather(*tasks)
            
        return results

    def _map_label_to_score(self, label: Optional[str]) -> float:
        token = (label or "").strip().upper()
        if not token:
            return 0.0
        token = re.split(r"\s+", token)[0]
        if token == "POS":
            return 1.0
        if token == "NEG":
            return -1.0
        if token == "NEU":
            return 0.0
        if "POS" in token:
            return 1.0
        if "NEG" in token:
            return -1.0
        if "NEU" in token:
            return 0.0
        return 0.0

    def _aggregate_weighted(self) -> Tuple[Optional[Path], Optional[Path]]:
        paths = sorted(self.scored_dir.glob("posts_scores_*.csv"))
        if not paths:
            return None, None
        frames = []
        for path in paths:
            try:
                df = pd.read_csv(path, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="latin-1", low_memory=False)
            df["__source"] = path.name
            frames.append(df)
        if not frames:
            return None, None
        df = pd.concat(frames, ignore_index=True)
        sent_cols: List[str] = []
        if "vader" in df.columns:
            sent_cols.append("vader")
        for col in ["s1", "s2", "s3", "s4", "s5"]:
            if col in df.columns:
                sent_cols.append(col)
            elif f"sent_{col}" in df.columns:
                df[col] = pd.to_numeric(df[f"sent_{col}"], errors="coerce")
                sent_cols.append(col)
        if not sent_cols:
            return None, None
        if "created_time_utc" in df.columns:
            t = pd.to_datetime(df["created_time_utc"], errors="coerce", utc=True)
        elif "created_utc" in df.columns:
            t = pd.to_datetime(df["created_utc"], unit="s", errors="coerce", utc=True)
        else:
            raise ValueError("posts_scores CSV missing timestamp columns")
        df["created_time_utc"] = t
        df = df.dropna(subset=["created_time_utc"])
        df = df[df["created_time_utc"] >= "2005-01-01"]
        if "id" in df.columns:
            df = df.sort_values("created_time_utc").drop_duplicates("id", keep="last")
        score = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0).clip(lower=0)
        if "num_comments" in df.columns:
            num_comments = pd.to_numeric(df.get("num_comments", 0), errors="coerce").fillna(0).clip(lower=0)
            weight = score.apply(math.log1p) + 0.5 * num_comments.apply(math.log1p)
        else:
            weight = score.apply(math.log1p)
        df["__w"] = weight
        for col in sent_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(-1, 1)
        df["date_utc"] = df["created_time_utc"].dt.date
        daily_rows = []
        for d, group in df.groupby("date_utc"):
            row = {"date_utc": d}
            for col in sent_cols:
                row[col] = self._weighted_average(group[col], group["__w"])
            daily_rows.append(row)
        daily = pd.DataFrame(daily_rows).sort_values("date_utc").reset_index(drop=True)
        if daily.empty:
            return None, None

        daily_ff = daily.copy()
        daily_ff["ts_day"] = pd.to_datetime(daily_ff["date_utc"]).dt.tz_localize("UTC")
        start = daily_ff["ts_day"].min()
        end = daily_ff["ts_day"].max() + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        minute_df = pd.DataFrame({"ts": pd.date_range(start=start, end=end, freq="min", tz="UTC")})
        joined = pd.merge_asof(
            minute_df.sort_values("ts"),
            daily_ff[["ts_day"] + sent_cols].sort_values("ts_day"),
            left_on="ts",
            right_on="ts_day",
            direction="backward",
        )
        minute_df = joined.drop(columns=["ts_day"])
        minute_df = minute_df[["ts"] + sent_cols]

        daily_out = self.weighted_dir / "sentiment_daily_vader_s1_s5.csv"
        minute_out = self.weighted_dir / "sentiment_1min_vader_s1_s5.csv"
        daily_export = daily.rename(columns={"date_utc": "ts"})
        daily_export["ts"] = pd.to_datetime(daily_export["ts"]).dt.tz_localize("UTC")
        daily_export.to_csv(daily_out, index=False, encoding="utf-8")
        minute_df.to_csv(minute_out, index=False, encoding="utf-8")
        logger.info(
            "Sentiment aggregation updated: days %s -> %s",
            daily_export["ts"].min().date(),
            daily_export["ts"].max().date(),
        )
        return daily_out, minute_out

    def _weighted_average(self, series: pd.Series, weights: pd.Series) -> float:
        s = series.astype(float)
        w = weights.astype(float)
        denominator = np.nansum(w)
        if denominator > 0:
            return float(np.nansum(s * w) / denominator)
        return float(np.nanmean(s))
