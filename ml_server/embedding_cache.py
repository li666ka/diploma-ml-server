"""Incremental embedding cache: зберігаємо кожну категорію ОДРАЗУ після encoding.
Дозволяє resume після OOM crash — повертаємось до точки зупинки.

Структура:
  /content/drive/MyDrive/diploma/embeddings_cache/
  └── dataset_{id}/
      ├── articles.pt          # dict[id, tensor[384]] — зберігається першим
      ├── tweets.pt            # — другим
      ├── retweets.pt          # — третім
      ├── replies.pt           # — четвертим
      └── metadata.json        # encoder_version, max_chars, file_hashes
"""
import gc
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
import torch

from ml_server.config import MODELS_ROOT
from ml_server.encoder import encode_dataframe_column

log = logging.getLogger(__name__)

CACHE_ROOT = Path(MODELS_ROOT).parent / "embeddings_cache"

ENCODER_VERSION = "minilm-v1"


def _cache_dir(dataset_id) -> Path:
    d = CACHE_ROOT / f"dataset_{dataset_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _file_hash(filepath: Path) -> str:
    """Швидкий hash: size + перший 1MB."""
    if not filepath.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(str(filepath.stat().st_size).encode())
    with open(filepath, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()[:16]


def _load_metadata(cache_dir: Path) -> Optional[dict]:
    meta_path = cache_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to load metadata: {e}")
        return None


def _save_metadata(cache_dir: Path, metadata: dict) -> None:
    """Atomic save через temp + rename."""
    meta_path = cache_dir / "metadata.json"
    tmp_path = cache_dir / "metadata.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(metadata, f, indent=2)
    tmp_path.replace(meta_path)


def _ram_snapshot() -> Optional[tuple]:
    """(percent, used_gb, total_gb) або None якщо psutil недоступний."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.percent, mem.used / 1e9, mem.total / 1e9
    except Exception:
        return None


def _log_ram(category: str, when: str) -> None:
    snap = _ram_snapshot()
    if snap is None:
        return
    pct, used, total = snap
    log.info(f"[{category}] RAM {when}: {pct:.0f}% ({used:.1f}/{total:.1f} GB)")


def _is_category_cached(
    cache_dir: Path,
    category: str,
    expected_hash: str,
    expected_max_chars: int,
    expected_model: str,
) -> bool:
    pt_path = cache_dir / f"{category}.pt"
    if not pt_path.exists():
        return False

    metadata = _load_metadata(cache_dir)
    if metadata is None:
        return False

    if metadata.get("encoder_version") != ENCODER_VERSION:
        log.info(f"  [{category}] cache stale: encoder version changed")
        return False

    if metadata.get("model_name") != expected_model:
        log.info(f"  [{category}] cache stale: model_name changed")
        return False

    cat_meta = metadata.get("categories", {}).get(category)
    if cat_meta is None:
        return False

    if cat_meta.get("source_hash") != expected_hash:
        log.info(f"  [{category}] cache stale: source CSV changed")
        return False

    if cat_meta.get("max_chars") != expected_max_chars:
        log.info(f"  [{category}] cache stale: max_chars changed")
        return False

    return True


def _save_category(
    cache_dir: Path,
    category: str,
    embeddings: dict,
    source_hash: str,
    max_chars: int,
    model_name: str,
) -> None:
    """Atomic save: .pt + update metadata. Зберігається ОДРАЗУ після encoding."""
    if not embeddings:
        log.info(f"  [{category}] skip save: empty embeddings")
        return

    pt_path = cache_dir / f"{category}.pt"
    tmp_path = cache_dir / f"{category}.pt.tmp"

    log.info(f"  [{category}] saving {len(embeddings):,} embeddings → {pt_path.name}")
    torch.save(embeddings, tmp_path)
    tmp_path.replace(pt_path)

    metadata = _load_metadata(cache_dir) or {
        "encoder_version": ENCODER_VERSION,
        "model_name": model_name,
        "embedding_dim": 384,
        "categories": {},
    }
    metadata["model_name"] = model_name
    metadata["encoder_version"] = ENCODER_VERSION
    metadata.setdefault("categories", {})[category] = {
        "source_hash": source_hash,
        "max_chars": max_chars,
        "n_items": len(embeddings),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_metadata(cache_dir, metadata)
    log.info(f"  [{category}] ✓ saved")


def _load_category(cache_dir: Path, category: str) -> dict:
    pt_path = cache_dir / f"{category}.pt"
    if not pt_path.exists():
        return {}

    log.info(f"  [{category}] loading from cache: {pt_path.name}")
    try:
        embeddings = torch.load(pt_path, weights_only=False)
        log.info(f"  [{category}] ✓ loaded {len(embeddings):,}")
        return embeddings
    except Exception as e:
        log.error(f"  [{category}] failed to load: {e}")
        return {}


def encode_incrementally(
    dataset_id,
    dataset_folder: str,
    articles_df: pd.DataFrame,
    tweets_df: pd.DataFrame,
    retweets_df: pd.DataFrame,
    replies_df: pd.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_chars_articles: int = 2000,
    max_chars_tweets: int = 500,
    batch_size: int = 64,
    force_recompute: bool = False,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """Encode + SAVE одразу після кожної категорії.

    Якщо одна категорія вже закешована — пропускаємо.
    Якщо crash посередині — наступний run підбере з місця зупинки.
    """
    cache_dir = _cache_dir(dataset_id)
    dataset_path = Path(dataset_folder) if dataset_folder else None

    log.info(f"═══ Incremental encoding for dataset {dataset_id} ═══")
    log.info(f"Cache dir: {cache_dir}")

    def _hash(name: str) -> str:
        if dataset_path is None:
            return "no-folder"
        return _file_hash(dataset_path / name)

    hashes = {
        "articles": _hash("news.csv"),
        "tweets": _hash("tweets.csv"),
        "retweets": _hash("retweets.csv"),
        "replies": _hash("replies.csv"),
    }

    categories = [
        {
            "name": "articles", "df": articles_df,
            "id_col": "article_id", "text_col": "combined_text",
            "max_chars": max_chars_articles, "result_key": "article_emb",
        },
        {
            "name": "tweets", "df": tweets_df,
            "id_col": "tweet_id", "text_col": "tweet_text",
            "max_chars": max_chars_tweets, "result_key": "tweet_emb",
        },
        {
            "name": "retweets", "df": retweets_df,
            "id_col": "retweet_id", "text_col": "retweet_text",
            "max_chars": max_chars_tweets, "result_key": "retweet_emb",
        },
        {
            "name": "replies", "df": replies_df,
            "id_col": "reply_id", "text_col": "reply_text",
            "max_chars": max_chars_tweets, "result_key": "reply_emb",
        },
    ]

    results = {}
    cache_hits = {}
    t0 = time.time()

    for cat in categories:
        name = cat["name"]
        df = cat["df"]

        _log_ram(name, "before")

        if df is None or len(df) == 0 or cat["text_col"] not in df.columns:
            log.info(f"[{name}] skip: empty or missing column '{cat['text_col']}'")
            results[cat["result_key"]] = {}
            cache_hits[name] = "skipped"
            continue

        cached = (not force_recompute) and _is_category_cached(
            cache_dir,
            category=name,
            expected_hash=hashes[name],
            expected_max_chars=cat["max_chars"],
            expected_model=model_name,
        )

        if cached:
            embeddings = _load_category(cache_dir, name)
            if embeddings:
                results[cat["result_key"]] = embeddings
                cache_hits[name] = "hit"
                log.info(f"[{name}] ✓ from cache ({len(embeddings):,} items)")
                continue
            cache_hits[name] = "load_failed"
        else:
            cache_hits[name] = "miss"

        if progress_callback:
            progress_callback(f"encoding_{name}")

        log.info(f"[{name}] encoding {len(df):,} items (batch_size={batch_size})...")

        try:
            embeddings = encode_dataframe_column(
                df,
                cat["id_col"],
                cat["text_col"],
                max_chars=cat["max_chars"],
                batch_size=batch_size,
            )
        except Exception as e:
            log.error(f"[{name}] encoding FAILED: {e}")
            log.error(f"[{name}] partial results NOT saved (encode_dataframe_column атомарний)")
            raise

        _save_category(
            cache_dir=cache_dir,
            category=name,
            embeddings=embeddings,
            source_hash=hashes[name],
            max_chars=cat["max_chars"],
            model_name=model_name,
        )

        results[cat["result_key"]] = embeddings

        _log_ram(name, "after")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    elapsed = time.time() - t0
    n_hits = sum(1 for v in cache_hits.values() if v == "hit")

    log.info(f"═══ Encoding complete: {n_hits}/4 from cache, time={elapsed:.1f}s ═══")

    return {
        **results,
        "cache_hits": cache_hits,
        "encoding_time": elapsed,
    }


def get_cache_status(dataset_id) -> dict:
    """Перевірити який стан кешу для dataset (per-category)."""
    cache_dir = _cache_dir(dataset_id)

    status = {
        "cache_dir": str(cache_dir),
        "exists": cache_dir.exists(),
        "categories": {},
    }

    if not cache_dir.exists():
        return status

    metadata = _load_metadata(cache_dir)

    total_size = 0
    for cat_name in ["articles", "tweets", "retweets", "replies"]:
        pt_path = cache_dir / f"{cat_name}.pt"
        cat_meta = (metadata or {}).get("categories", {}).get(cat_name, {})
        size_mb = round(pt_path.stat().st_size / 1e6, 2) if pt_path.exists() else 0
        total_size += size_mb

        status["categories"][cat_name] = {
            "exists": pt_path.exists(),
            "size_mb": size_mb,
            "n_items": cat_meta.get("n_items", 0),
            "saved_at": cat_meta.get("saved_at"),
        }

    status["total_size_mb"] = round(total_size, 2)
    status["metadata"] = metadata
    return status


def invalidate_category(dataset_id, category: str) -> bool:
    """Видалити одну категорію (наприклад, для re-encode тільки tweets)."""
    cache_dir = _cache_dir(dataset_id)
    pt_path = cache_dir / f"{category}.pt"

    if pt_path.exists():
        pt_path.unlink()

        metadata = _load_metadata(cache_dir)
        if metadata and category in metadata.get("categories", {}):
            del metadata["categories"][category]
            _save_metadata(cache_dir, metadata)

        log.info(f"Invalidated cache: {pt_path.name}")
        return True

    return False


def invalidate_all(dataset_id) -> bool:
    """Видалити весь кеш для dataset."""
    import shutil

    cache_dir = CACHE_ROOT / f"dataset_{dataset_id}"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        log.info(f"Invalidated all cache: {cache_dir}")
        return True
    return False
