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

import numpy as np
import pandas as pd
import torch

from ml_server.config import MODELS_ROOT
from ml_server.encoder import encode_dataframe_column

log = logging.getLogger(__name__)

CACHE_ROOT = Path(MODELS_ROOT).parent / "embeddings_cache"

ENCODER_VERSION = "minilm-v1"


class CorruptMemmapCache(Exception):
    """Cache на диску неконсистентний (ids.json ↔ .npy size mismatch)."""


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


# ── Memmap-backed encoding (для дуже великих datasets) ────────────────────

def encode_incrementally_memmap(
    dataset_id,
    dataset_folder: str,
    articles_df: pd.DataFrame,
    tweets_df: pd.DataFrame,
    retweets_df: pd.DataFrame,
    replies_df: pd.DataFrame,
    max_chars_articles: int = 2000,
    max_chars_tweets: int = 500,
    batch_size: int = 64,
    force_recompute: bool = False,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """Disk-backed encoding для дуже великих datasets.

    Замість dict[id, tensor] зберігаємо:
      - {category}.npy   — numpy memmap[N, 384] (на диску, доступне як array)
      - {category}_ids.json — {id_str: row_idx} mapping
    """
    cache_dir = _cache_dir(dataset_id)
    dataset_path = Path(dataset_folder) if dataset_folder else None

    log.info(f"═══ Memmap encoding: dataset_{dataset_id} ═══")
    log.info(f"Cache dir: {cache_dir}")

    def _hash(filename):
        if dataset_path is None:
            return "no_folder"
        return _file_hash(dataset_path / filename)

    hashes = {
        "articles": _hash("news.csv"),
        "tweets": _hash("tweets.csv"),
        "retweets": _hash("retweets.csv"),
        "replies": _hash("replies.csv"),
    }

    categories = [
        ("articles", articles_df, "article_id", "combined_text", max_chars_articles),
        ("tweets", tweets_df, "tweet_id", "tweet_text", max_chars_tweets),
        ("retweets", retweets_df, "retweet_id", "retweet_text", max_chars_tweets),
        ("replies", replies_df, "reply_id", "reply_text", max_chars_tweets),
    ]

    lookups = {}
    cache_hits = {}
    t0 = time.time()

    for name, df, id_col, text_col, max_chars in categories:
        _log_ram(name, "before")

        npy_path = cache_dir / f"{name}.npy"
        ids_path = cache_dir / f"{name}_ids.json"

        if df is None or len(df) == 0 or text_col not in df.columns:
            log.info(f"[{name}] skip: empty")
            lookups[name] = None
            cache_hits[name] = "skipped"
            continue

        cached = (
            not force_recompute
            and npy_path.exists()
            and ids_path.exists()
            and _is_memmap_cached(cache_dir, name, hashes[name], max_chars)
        )

        if cached:
            log.info(f"[{name}] loading from memmap cache...")
            try:
                lookups[name] = MemmapLookup(npy_path, ids_path)
                cache_hits[name] = "hit"
                _log_ram(name, "after_cache_hit")
                continue
            except CorruptMemmapCache as e:
                log.warning(f"[{name}] {e} — re-encoding from scratch")
                if npy_path.exists():
                    npy_path.unlink()
                if ids_path.exists():
                    ids_path.unlink()
                # fall through to re-encoding below

        cache_hits[name] = "miss"

        if progress_callback:
            progress_callback(f"encoding_{name}")

        log.info(f"[{name}] encoding {len(df):,} items → memmap...")

        try:
            _encode_to_memmap(
                df=df,
                id_col=id_col,
                text_col=text_col,
                max_chars=max_chars,
                batch_size=batch_size,
                npy_path=npy_path,
                ids_path=ids_path,
            )
        except Exception as e:
            log.error(f"[{name}] encoding FAILED: {e}")
            if npy_path.exists():
                npy_path.unlink()
            if ids_path.exists():
                ids_path.unlink()
            raise

        _update_memmap_metadata(cache_dir, name, hashes[name], max_chars)

        lookups[name] = MemmapLookup(npy_path, ids_path)
        log.info(f"  [{name}] ✓ memmap created and opened")

        _log_ram(name, "after_save")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    elapsed = time.time() - t0
    log.info(f"═══ Memmap encoding done: time={elapsed:.1f}s ═══")
    _log_ram("final", "")

    return {
        "article_lookup": lookups.get("articles"),
        "tweet_lookup": lookups.get("tweets"),
        "retweet_lookup": lookups.get("retweets"),
        "reply_lookup": lookups.get("replies"),
        "cache_hits": cache_hits,
        "encoding_time": elapsed,
    }


def _encode_to_memmap(
    df: pd.DataFrame,
    id_col: str,
    text_col: str,
    max_chars: int,
    batch_size: int,
    npy_path: Path,
    ids_path: Path,
):
    """Encode + save directly to disk (numpy memmap)."""
    from ml_server.encoder import get_encoder

    df_clean = df[[id_col, text_col]].copy()
    df_clean[text_col] = df_clean[text_col].fillna("").astype(str).str.slice(0, max_chars)
    df_clean[id_col] = df_clean[id_col].astype(str)
    df_clean = df_clean.drop_duplicates(subset=[id_col]).reset_index(drop=True)

    n_items = len(df_clean)
    if n_items == 0:
        return

    EMBEDDING_DIM = 384
    chunk_size = 50_000

    id_to_row = {df_clean[id_col].iloc[i]: i for i in range(n_items)}

    # ВАЖЛИВО: пишемо .npy ПЕРШИМ і коммітимо ids.json лише після успішного
    # `tmp_npy.replace(npy_path)`. Це дає атомарну консистентність — якщо
    # процес впаде під час encoding, або .npy частково записаний, ids.json
    # залишається старим (або відсутнім), і кеш-перевірка не побачить hit.
    tmp_npy = npy_path.with_suffix(".npy.tmp")
    memmap = np.memmap(
        tmp_npy,
        dtype=np.float32,
        mode="w+",
        shape=(n_items, EMBEDDING_DIM),
    )

    encoder = get_encoder()
    texts = df_clean[text_col].tolist()

    n_chunks = (n_items + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_items)

        log.info(f"    Chunk {chunk_idx + 1}/{n_chunks}: encoding rows {start:,}-{end:,}")

        chunk_texts = texts[start:end]
        chunk_embs = encoder.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        memmap[start:end] = chunk_embs

        del chunk_embs, chunk_texts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    memmap.flush()
    del memmap
    gc.collect()

    tmp_npy.replace(npy_path)

    # Commit ids.json лише після того, як .npy успішно опинився на диску.
    tmp_ids = ids_path.with_suffix(".json.tmp")
    with open(tmp_ids, "w") as f:
        json.dump(id_to_row, f)
    tmp_ids.replace(ids_path)

    log.info(f"    ✓ memmap saved: {npy_path.name} ({n_items:,} × {EMBEDDING_DIM})")


def _is_memmap_cached(
    cache_dir: Path,
    category: str,
    expected_hash: str,
    expected_max_chars: int,
) -> bool:
    meta = _load_metadata(cache_dir)
    if not meta or meta.get("encoder_version") != ENCODER_VERSION:
        return False

    cat = meta.get("memmap_categories", {}).get(category)
    if not cat:
        return False

    if cat.get("source_hash") != expected_hash:
        log.info(f"  [{category}] memmap cache stale: source changed")
        return False

    if cat.get("max_chars") != expected_max_chars:
        log.info(f"  [{category}] memmap cache stale: max_chars changed")
        return False

    return True


def _update_memmap_metadata(
    cache_dir: Path,
    category: str,
    source_hash: str,
    max_chars: int,
):
    meta = _load_metadata(cache_dir) or {
        "encoder_version": ENCODER_VERSION,
        "memmap_categories": {},
    }
    meta["encoder_version"] = ENCODER_VERSION
    meta.setdefault("memmap_categories", {})[category] = {
        "source_hash": source_hash,
        "max_chars": max_chars,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_metadata(cache_dir, meta)


class MemmapLookup:
    """Disk-backed dict[id, tensor[384]] використовуючи numpy memmap.

    Замість тримати GB у RAM, читаємо rows тільки коли потрібно.
    OS сам кешує hot pages — для warm access швидкість майже як у RAM.
    """

    def __init__(self, npy_path: Path, ids_path: Path):
        self.npy_path = Path(npy_path)
        self.ids_path = Path(ids_path)

        with open(self.ids_path) as f:
            self.id_to_row = json.load(f)

        n_items = len(self.id_to_row)
        expected_bytes = n_items * 384 * 4  # float32
        actual_bytes = self.npy_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise CorruptMemmapCache(
                f"Memmap cache inconsistent for {self.npy_path.name}: "
                f"ids.json claims {n_items} items ({expected_bytes} bytes), "
                f"but .npy is {actual_bytes} bytes. Cache will be rebuilt."
            )
        self.memmap = np.memmap(
            self.npy_path,
            dtype=np.float32,
            mode="r",
            shape=(n_items, 384),
        )

        log.info(f"  MemmapLookup opened: {self.npy_path.name} ({n_items:,} items)")

    def __contains__(self, key) -> bool:
        return str(key) in self.id_to_row

    def __len__(self) -> int:
        return len(self.id_to_row)

    def get(self, key, default=None):
        key_str = str(key)
        if key_str not in self.id_to_row:
            return default
        row_idx = self.id_to_row[key_str]
        arr = np.array(self.memmap[row_idx], dtype=np.float32)
        return torch.from_numpy(arr)

    def __getitem__(self, key):
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def keys(self):
        return self.id_to_row.keys()
