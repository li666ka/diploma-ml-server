"""Завантаження даних з Drive (або локально) у різних форматах:
- aggregated (для NB) → ml_server.aggregated_loader
- article-level (для DistilBERT і GNN) → тут
"""
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_server.config import DATASETS_ROOT
from ml_server.utils import log


def resolve_dataset_path(
    dataset_id: int | str,
    dataset_name: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """
    Знайти папку датасету у Drive/локально.
    Підтримує і папки, і ZIP-архіви.

    Returns: (csv_dir, tmpdir_for_cleanup_or_None)
    """
    candidates = []
    if dataset_id is not None:
        candidates.append((f"{DATASETS_ROOT}/{dataset_id}", False))
        candidates.append((f"{DATASETS_ROOT}/{dataset_id}.zip", True))
    if dataset_name:
        safe = str(dataset_name).strip().replace("/", "_")
        candidates.append((f"{DATASETS_ROOT}/{safe}", False))
        candidates.append((f"{DATASETS_ROOT}/{safe}.zip", True))

    for path, is_zip in candidates:
        if not os.path.exists(path):
            continue

        if is_zip:
            extract_dir = tempfile.mkdtemp(prefix=f"ds_{dataset_id}_")
            log.info(f"Extracting {path} → {extract_dir}")
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(extract_dir)
            for root, _dirs, files in os.walk(extract_dir):
                if "news.csv" in files:
                    log.info(f"✓ Extracted, CSVs at {root}")
                    return root, extract_dir
            shutil.rmtree(extract_dir, ignore_errors=True)
            raise ValueError(f"ZIP {path} не містить news.csv")
        else:
            for root, _dirs, files in os.walk(path):
                if "news.csv" in files:
                    log.info(f"✓ Folder dataset found: {root}")
                    return root, None

    searched = "\n    ".join(p for p, _ in candidates)
    raise FileNotFoundError(f"Dataset не знайдено. Шукав:\n    {searched}")


def _try_load_external_splits(
    csv_dir: str,
    news: pd.DataFrame,
    splits_subdir: Optional[str] = None,
):
    """Try to load pre-computed splits from CSV files.

    Args:
        csv_dir: dataset folder path
        news: filtered DataFrame з article_id (вже після length filter і label)
        splits_subdir: name of splits folder (наприклад "splits_cross_domain").
            If None — fallback на legacy "splits/" якщо існує.
            If string — використовує цю папку конкретно. Якщо її нема —
            log.warning і повертає None (caller робить fallback на auto-split).

    Returns: (train_df, val_df, test_df) або None.
    """
    if splits_subdir:
        splits_dir = os.path.join(csv_dir, splits_subdir)
        if not os.path.isdir(splits_dir):
            log.warning(
                f"Requested splits folder not found: {splits_subdir}. "
                f"Will use auto-split."
            )
            return None
    else:
        splits_dir = os.path.join(csv_dir, "splits")
        if not os.path.isdir(splits_dir):
            return None

    train_csv = os.path.join(splits_dir, "split_train.csv")
    val_csv = os.path.join(splits_dir, "split_val.csv")
    test_csv = os.path.join(splits_dir, "split_test.csv")
    for path in (train_csv, val_csv, test_csv):
        if not os.path.exists(path):
            log.warning(
                f"Missing {os.path.basename(path)} in "
                f"{os.path.basename(splits_dir)}/. Will use auto-split."
            )
            return None

    log.info(f"✓ Loading external splits from {os.path.basename(splits_dir)}/")

    train_ids = set(pd.read_csv(train_csv, dtype={"article_id": str})["article_id"])
    val_ids = set(pd.read_csv(val_csv, dtype={"article_id": str})["article_id"])
    test_ids = set(pd.read_csv(test_csv, dtype={"article_id": str})["article_id"])

    train_df = news[news["article_id"].isin(train_ids)].reset_index(drop=True)
    val_df = news[news["article_id"].isin(val_ids)].reset_index(drop=True)
    test_df = news[news["article_id"].isin(test_ids)].reset_index(drop=True)

    log.info(
        f"  External splits: train={len(train_df):,} "
        f"val={len(val_df):,} test={len(test_df):,}"
    )
    return train_df, val_df, test_df


def build_article_level_data(
    dataset_id: int | str,
    dataset_name: Optional[str] = None,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    min_text_length: int = 30,
    seed: int = 42,
    require_tweets: bool = False,
    require_social: bool = False,
    splits_subdir: Optional[str] = None,
):
    """
    Article-level dataset для DistilBERT і GNN.

    Args:
        splits_subdir: ім'я папки `splits_*` для зовнішніх splits (опціонально).
            None → fallback на legacy `splits/` якщо є, інакше auto-split.

    Returns: (train_df, val_df, test_df, full_data_dict, stats, tmpdir)

    де:
      train/val/test_df — DataFrame з [article_id, label, combined_text, ...]
      full_data_dict    — {tweets, retweets, replies, users} (тільки якщо
                          require_tweets=True)
      stats             — статистика (включно зі `splits_used`)
      tmpdir            — для cleanup після ZIP
    """
    csv_dir, tmpdir = resolve_dataset_path(dataset_id, dataset_name)

    news_path = os.path.join(csv_dir, "news.csv")
    log.info(f"Loading news.csv ({os.path.getsize(news_path) / 1024 / 1024:.1f} MB)")
    news = pd.read_csv(news_path, low_memory=False, dtype={"article_id": str})
    log.info(f"  {len(news):,} articles")

    news["article_text"] = news["article_text"].fillna("").astype(str)
    news["article_title"] = news["article_title"].fillna("").astype(str)
    news["combined_text"] = news["article_title"] + ". " + news["article_text"]

    before = len(news)
    news = news[news["combined_text"].str.len() >= min_text_length].copy()
    log.info(f"  After length filter (>={min_text_length}): {len(news):,} "
             f"(-{before - len(news):,})")

    news["label"] = (
        news["article_label"].astype(str).str.upper() == "FAKE"
    ).astype(int)
    news = news.reset_index(drop=True)

    # ── External splits (optional) → fallback на stratified auto-split ──
    external = _try_load_external_splits(csv_dir, news, splits_subdir=splits_subdir)
    if external is not None:
        train_df, val_df, test_df = external
        if splits_subdir:
            splits_used = splits_subdir.replace("splits_", "", 1)
        else:
            splits_used = "splits"  # legacy
    else:
        trainval, test_df = train_test_split(
            news, test_size=test_ratio,
            stratify=news["label"], random_state=seed,
        )
        val_relative = val_ratio / (1 - test_ratio)
        train_df, val_df = train_test_split(
            trainval, test_size=val_relative,
            stratify=trainval["label"], random_state=seed,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        splits_used = "auto"

    log.info(f"  Train: {len(train_df):,} (FAKE={(train_df.label==1).sum():,}, "
             f"REAL={(train_df.label==0).sum():,})")
    log.info(f"  Val:   {len(val_df):,} (FAKE={(val_df.label==1).sum():,}, "
             f"REAL={(val_df.label==0).sum():,})")
    log.info(f"  Test:  {len(test_df):,} (FAKE={(test_df.label==1).sum():,}, "
             f"REAL={(test_df.label==0).sum():,})")

    # ── Optional: tweets/retweets/replies/users (для GNN та NB+social) ──
    full_data = None
    if require_tweets or require_social:
        full_data = _load_social_data(csv_dir)

    stats = {
        "mode": "article-level",
        "total_articles": len(news),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_fake": int((train_df.label == 1).sum()),
        "train_real": int((train_df.label == 0).sum()),
        "val_fake": int((val_df.label == 1).sum()),
        "val_real": int((val_df.label == 0).sum()),
        "test_fake": int((test_df.label == 1).sum()),
        "test_real": int((test_df.label == 0).sum()),
        "splits_used": splits_used,
        "has_full_data": full_data is not None,
    }

    return train_df, val_df, test_df, full_data, stats, tmpdir


def _load_social_data(csv_dir: str) -> dict:
    """Завантажити tweets/retweets/replies/users (для GNN)."""
    tweets_path = os.path.join(csv_dir, "tweets.csv")
    retweets_path = os.path.join(csv_dir, "retweets.csv")
    replies_path = os.path.join(csv_dir, "replies.csv")
    users_path = os.path.join(csv_dir, "users.csv")

    if not os.path.exists(tweets_path):
        raise FileNotFoundError(f"GNN потребує tweets.csv: {tweets_path}")

    log.info("Loading social data (tweets/retweets/replies/users)...")
    tweets = pd.read_csv(
        tweets_path, low_memory=False,
        dtype={"tweet_id": str, "article_id": str, "user_id": str},
    )
    log.info(f"  tweets:   {len(tweets):,}")

    retweets = pd.DataFrame()
    if os.path.exists(retweets_path):
        retweets = pd.read_csv(
            retweets_path, low_memory=False,
            dtype={"retweet_id": str, "original_tweet_id": str, "user_id": str},
        )
        log.info(f"  retweets: {len(retweets):,}")

    replies = pd.DataFrame()
    if os.path.exists(replies_path):
        replies = pd.read_csv(
            replies_path, low_memory=False,
            dtype={"reply_id": str, "parent_tweet_id": str, "user_id": str},
        )
        log.info(f"  replies:  {len(replies):,}")

    users = pd.DataFrame()
    if os.path.exists(users_path):
        users = pd.read_csv(users_path, low_memory=False, dtype={"user_id": str})
        log.info(f"  users:    {len(users):,}")

    return {
        "tweets": tweets,
        "retweets": retweets,
        "replies": replies,
        "users": users,
    }
