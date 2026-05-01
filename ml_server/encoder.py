"""MiniLM encoder helpers (lazy-load + batch encoding)."""
import gc
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ml_server.utils import log


# Lazy-loaded global instance
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


def encode_texts_batch(
    texts: list[str],
    batch_size: int = 512,
    max_chars: int = 500,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode list of texts → numpy [N, dim] (зазвичай 384)."""
    encoder = get_encoder()
    texts_clean = [str(t)[:max_chars] if t is not None else "" for t in texts]
    embs = encoder.encode(
        texts_clean,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return embs


def encode_dataframe_column(
    df: pd.DataFrame,
    id_col: str,
    text_col: str,
    max_chars: int = 500,
    batch_size: int = 512,
) -> dict[str, torch.Tensor]:
    """
    Encode column from DataFrame. Returns {id_str: tensor[dim]}.
    Дедуплікація по id_col.
    """
    df_clean = df[[id_col, text_col]].copy()
    df_clean[text_col] = (
        df_clean[text_col].fillna("").astype(str).str.slice(0, max_chars)
    )
    df_clean[id_col] = df_clean[id_col].astype(str)
    df_clean = df_clean.drop_duplicates(subset=[id_col]).reset_index(drop=True)

    if len(df_clean) == 0:
        return {}

    log.info(f"Encoding {len(df_clean):,} unique {id_col} values...")
    embs = encode_texts_batch(
        df_clean[text_col].tolist(),
        batch_size=batch_size,
        max_chars=max_chars,
    )

    result = {
        df_clean[id_col].iloc[i]: torch.from_numpy(embs[i])
        for i in range(len(df_clean))
    }

    # Free GPU mem
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return result
