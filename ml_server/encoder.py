"""MiniLM encoder з chunked encoding для безпеки на Colab."""
import gc
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ml_server.utils import log


_encoder = None


def get_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Lazy-load MiniLM encoder. Reused across calls."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading encoder {model_name} on {device}")
        _encoder = SentenceTransformer(model_name, device=device)
    return _encoder


def _cleanup_memory():
    """Force GC + GPU cache clear."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def encode_texts_batch(
    texts: list[str],
    batch_size: int = 64,
    max_chars: int = 500,
    show_progress: bool = True,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Encode texts → numpy [N, 384].

    Для великих datasets (>chunk_size) використовуємо chunked encoding:
    - кожен chunk обробляється окремо
    - cleanup memory між chunks
    - peak RAM ~200MB замість ~2GB на повний batch
    """
    encoder = get_encoder()
    texts_clean = [str(t)[:max_chars] if t is not None else "" for t in texts]
    n_total = len(texts_clean)

    if n_total == 0:
        return np.zeros((0, 384), dtype=np.float32)

    if n_total <= chunk_size:
        embs = encoder.encode(
            texts_clean,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        _cleanup_memory()
        return embs

    n_chunks = (n_total + chunk_size - 1) // chunk_size
    log.info(f"  Chunked encoding: {n_total:,} items у {n_chunks} chunks по {chunk_size:,}")

    all_embs = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_total)
        chunk = texts_clean[start:end]

        log.info(f"    Chunk {i + 1}/{n_chunks}: encoding {len(chunk):,} items...")

        chunk_embs = encoder.encode(
            chunk,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        all_embs.append(chunk_embs)
        _cleanup_memory()

    result = np.concatenate(all_embs, axis=0)
    del all_embs
    _cleanup_memory()

    return result


def encode_dataframe_column(
    df: pd.DataFrame,
    id_col: str,
    text_col: str,
    max_chars: int = 500,
    batch_size: int = 64,
) -> dict:
    """Encode column → {id_str: tensor[384]}.

    Memory-safe: tensors створюються лише в кінці, не накопичуються list[Tensor].
    """
    df_clean = df[[id_col, text_col]].copy()
    df_clean[text_col] = (
        df_clean[text_col].fillna("").astype(str).str.slice(0, max_chars)
    )
    df_clean[id_col] = df_clean[id_col].astype(str)
    df_clean = df_clean.drop_duplicates(subset=[id_col]).reset_index(drop=True)

    if len(df_clean) == 0:
        return {}

    log.info(f"Encoding {len(df_clean):,} unique {id_col} values (batch={batch_size})...")

    embs = encode_texts_batch(
        df_clean[text_col].tolist(),
        batch_size=batch_size,
        max_chars=max_chars,
    )

    ids = df_clean[id_col].tolist()
    result = {ids[i]: torch.from_numpy(embs[i]) for i in range(len(ids))}

    del embs, df_clean, ids
    _cleanup_memory()

    return result
