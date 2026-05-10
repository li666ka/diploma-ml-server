"""Persistent embedding cache для MiniLM.

Структура:
  /path/to/cache_root/
  └── dataset_{dataset_id}/
      ├── articles_minilm.pt      # dict[id_str, tensor[384]]
      ├── tweets_minilm.pt
      ├── retweets_minilm.pt
      ├── replies_minilm.pt
      └── metadata.json           # model_name, max_chars, file hashes

Invalidation:
  - Якщо CSV file hash змінився → cache miss → re-encode
  - Якщо model_name або max_chars відрізняються → cache miss
  - Manual: видалити dataset_{id} директорію
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from ml_server.config import MODELS_ROOT
from ml_server.encoder import encode_dataframe_column

log = logging.getLogger(__name__)

CACHE_ROOT = Path(MODELS_ROOT).parent / "embeddings_cache"

ENCODER_VERSION = "minilm-v1"


def _compute_file_hash(filepath: Path) -> str:
    """SHA256 hash першого 1MB файлу + size (швидко для великих CSV)."""
    if not filepath.exists():
        return "missing"

    size = filepath.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())

    with open(filepath, "rb") as f:
        h.update(f.read(1024 * 1024))

    return h.hexdigest()[:16]


def _get_cache_dir(dataset_id) -> Path:
    cache_dir = CACHE_ROOT / f"dataset_{dataset_id}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_metadata(cache_dir: Path) -> Optional[dict]:
    meta_path = cache_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to load cache metadata: {e}")
        return None


def _save_metadata(cache_dir: Path, metadata: dict) -> None:
    meta_path = cache_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _is_cache_valid(
    cache_dir: Path,
    expected_model: str,
    expected_max_chars: dict,
    csv_hashes: dict,
) -> bool:
    metadata = _load_metadata(cache_dir)
    if metadata is None:
        return False

    if metadata.get("encoder_version") != ENCODER_VERSION:
        log.info("Cache invalid: encoder_version mismatch")
        return False

    if metadata.get("model_name") != expected_model:
        log.info("Cache invalid: model_name changed")
        return False

    cached_max_chars = metadata.get("max_chars", {})
    for category, expected in expected_max_chars.items():
        if cached_max_chars.get(category) != expected:
            log.info(f"Cache invalid: max_chars[{category}] changed")
            return False

    cached_hashes = metadata.get("data_files_hash", {})
    for filename, expected_hash in csv_hashes.items():
        if cached_hashes.get(filename) != expected_hash:
            log.info(f"Cache invalid: {filename} changed")
            return False

    return True


def _save_embeddings(embeddings: dict, filepath: Path) -> None:
    if not embeddings:
        log.info(f"Skipping save (empty embeddings): {filepath.name}")
        return

    log.info(f"Saving {len(embeddings):,} embeddings → {filepath.name}")
    torch.save(embeddings, filepath)


def _load_embeddings(filepath: Path) -> dict:
    if not filepath.exists():
        log.warning(f"Cache file missing: {filepath.name}")
        return {}

    log.info(f"Loading embeddings ← {filepath.name}")
    try:
        embeddings = torch.load(filepath, weights_only=False)
        log.info(f"  Loaded {len(embeddings):,} embeddings")
        return embeddings
    except Exception as e:
        log.error(f"Failed to load {filepath.name}: {e}")
        return {}


def encode_with_cache(
    dataset_id,
    dataset_folder: str,
    articles_df: pd.DataFrame,
    tweets_df: pd.DataFrame,
    retweets_df: pd.DataFrame,
    replies_df: pd.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_chars_articles: int = 2000,
    max_chars_tweets: int = 500,
    force_recompute: bool = False,
    progress_callback: Optional[callable] = None,
) -> dict:
    """Encode all texts з кешуванням. Повертає dict з embeddings + cache stats."""
    t0 = time.time()

    cache_dir = _get_cache_dir(dataset_id)
    log.info(f"Embedding cache dir: {cache_dir}")

    dataset_path = Path(dataset_folder) if dataset_folder else None
    csv_files = ["news.csv", "tweets.csv", "retweets.csv", "replies.csv"]
    if dataset_path is not None and dataset_path.exists():
        csv_hashes = {
            f: _compute_file_hash(dataset_path / f)
            for f in csv_files
            if (dataset_path / f).exists()
        }
    else:
        log.warning(f"dataset_folder not found: {dataset_folder} — cache may be unsafe")
        csv_hashes = {}

    expected_max_chars = {
        "articles": max_chars_articles,
        "tweets": max_chars_tweets,
        "retweets": max_chars_tweets,
        "replies": max_chars_tweets,
    }

    cache_valid = (not force_recompute) and _is_cache_valid(
        cache_dir, model_name, expected_max_chars, csv_hashes
    )

    cache_hits = {
        "articles": False,
        "tweets": False,
        "retweets": False,
        "replies": False,
    }

    # ── Articles ──
    articles_path = cache_dir / "articles_minilm.pt"
    if cache_valid and articles_path.exists():
        article_emb = _load_embeddings(articles_path)
        cache_hits["articles"] = True
    else:
        if progress_callback:
            progress_callback("encoding_articles")
        log.info(f"Encoding {len(articles_df):,} articles...")
        article_emb = encode_dataframe_column(
            articles_df, "article_id", "combined_text",
            max_chars=max_chars_articles, batch_size=64,
        )
        _save_embeddings(article_emb, articles_path)

    # ── Tweets ──
    tweets_path = cache_dir / "tweets_minilm.pt"
    if cache_valid and tweets_path.exists():
        tweet_emb = _load_embeddings(tweets_path)
        cache_hits["tweets"] = True
    else:
        if progress_callback:
            progress_callback("encoding_tweets")
        tweet_emb = {}
        if len(tweets_df) > 0 and "tweet_text" in tweets_df.columns:
            log.info(f"Encoding {len(tweets_df):,} tweets...")
            tweet_emb = encode_dataframe_column(
                tweets_df, "tweet_id", "tweet_text",
                max_chars=max_chars_tweets, batch_size=64,
            )
            _save_embeddings(tweet_emb, tweets_path)

    # ── Retweets ──
    retweets_path = cache_dir / "retweets_minilm.pt"
    if cache_valid and retweets_path.exists():
        retweet_emb = _load_embeddings(retweets_path)
        cache_hits["retweets"] = True
    else:
        if progress_callback:
            progress_callback("encoding_retweets")
        retweet_emb = {}
        if len(retweets_df) > 0 and "retweet_text" in retweets_df.columns:
            log.info(f"Encoding {len(retweets_df):,} retweets...")
            retweet_emb = encode_dataframe_column(
                retweets_df, "retweet_id", "retweet_text",
                max_chars=max_chars_tweets, batch_size=64,
            )
            _save_embeddings(retweet_emb, retweets_path)

    # ── Replies ──
    replies_path = cache_dir / "replies_minilm.pt"
    if cache_valid and replies_path.exists():
        reply_emb = _load_embeddings(replies_path)
        cache_hits["replies"] = True
    else:
        if progress_callback:
            progress_callback("encoding_replies")
        reply_emb = {}
        if len(replies_df) > 0 and "reply_text" in replies_df.columns:
            log.info(f"Encoding {len(replies_df):,} replies...")
            reply_emb = encode_dataframe_column(
                replies_df, "reply_id", "reply_text",
                max_chars=max_chars_tweets, batch_size=64,
            )
            _save_embeddings(reply_emb, replies_path)

    all_from_cache = all(cache_hits.values())
    if not all_from_cache:
        metadata = {
            "encoder_version": ENCODER_VERSION,
            "model_name": model_name,
            "max_chars": expected_max_chars,
            "n_articles": len(article_emb),
            "n_tweets": len(tweet_emb),
            "n_retweets": len(retweet_emb),
            "n_replies": len(reply_emb),
            "embedding_dim": 384,
            "data_files_hash": csv_hashes,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _save_metadata(cache_dir, metadata)

    encoding_time = time.time() - t0

    n_cached = sum(cache_hits.values())
    log.info(
        f"Encoding complete: {n_cached}/4 from cache, "
        f"time={encoding_time:.1f}s"
    )

    return {
        "article_emb": article_emb,
        "tweet_emb": tweet_emb,
        "retweet_emb": retweet_emb,
        "reply_emb": reply_emb,
        "from_cache": all_from_cache,
        "cache_hits": cache_hits,
        "encoding_time": encoding_time,
    }


def invalidate_cache(dataset_id) -> bool:
    """Видалити cache для конкретного dataset. Returns True якщо успішно."""
    import shutil

    cache_dir = CACHE_ROOT / f"dataset_{dataset_id}"
    if not cache_dir.exists():
        return False

    log.info(f"Invalidating cache: {cache_dir}")
    shutil.rmtree(cache_dir)
    return True


def get_cache_info(dataset_id) -> dict:
    """Повернути info про cache: чи існує, розмір, метадата."""
    cache_dir = CACHE_ROOT / f"dataset_{dataset_id}"

    if not cache_dir.exists():
        return {"exists": False}

    metadata = _load_metadata(cache_dir)

    total_size_bytes = sum(
        f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
    )

    return {
        "exists": True,
        "cache_dir": str(cache_dir),
        "metadata": metadata,
        "total_size_mb": round(total_size_bytes / 1024 / 1024, 2),
        "files": [f.name for f in cache_dir.iterdir() if f.is_file()],
    }
