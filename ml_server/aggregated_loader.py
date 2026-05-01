"""Aggregated pipeline для NB (legacy v16).

1 row per article = "[TWEET] top_tweet [ARTICLE] article_text" 
+ 8 social aggregate columns.

Цей модуль — спрощений варіант. Повний (з emotional features) 
треба портувати з v16 ноутбука.
"""
from typing import Optional

import numpy as np
import pandas as pd

from ml_server.config import AGGREGATED_SOCIAL_COLS
from ml_server.data_loader import _try_load_external_splits, resolve_dataset_path
from ml_server.utils import log


def build_aggregated_dataframe(
    news: pd.DataFrame,
    tweets: pd.DataFrame,
    users: Optional[pd.DataFrame] = None,
    top_tweet_strategy: str = "popularity",
) -> pd.DataFrame:
    """Aggregated DataFrame: 1 row per article."""
    log.info(f"  Building aggregates (strategy={top_tweet_strategy})...")
    tw = tweets.copy()

    tw["_is_retweet"] = (
        tw["tweet_retweeted_status_id"].notna()
        if "tweet_retweeted_status_id" in tw.columns
        else False
    )
    tw["_likes"] = pd.to_numeric(
        tw.get("like_count", pd.Series(0, index=tw.index)),
        errors="coerce",
    ).fillna(0)

    # Merge user profile
    if users is not None and len(users) > 0:
        u = users.copy()
        if "user_created_at" in u.columns:
            u["_created"] = pd.to_datetime(
                u["user_created_at"], unit="s", errors="coerce"
            )
            now = pd.Timestamp.utcnow().tz_localize(None)
            u["_account_age_days"] = (now - u["_created"]).dt.total_seconds() / 86400
        else:
            u["_account_age_days"] = 0.0

        if "user_verified" in u.columns:
            u["_verified"] = u["user_verified"].fillna(False).astype(bool)
        else:
            u["_verified"] = False

        cols_to_merge = ["user_id", "_verified", "_account_age_days"]
        for c in ["user_followers_count", "user_friends_count", "user_statuses_count"]:
            if c in u.columns:
                cols_to_merge.append(c)

        tw = tw.merge(u[cols_to_merge], on="user_id", how="left")

    # Top tweet per article
    if top_tweet_strategy == "popularity":
        tw_sorted = tw.sort_values(
            ["article_id", "_likes"], ascending=[True, False]
        )
    elif top_tweet_strategy == "first":
        time_col = (
            "tweet_created_at" if "tweet_created_at" in tw.columns else "tweet_id"
        )
        tw_sorted = tw.sort_values([("article_id"), time_col], ascending=[True, True])
    else:
        tw_sorted = tw.sample(frac=1, random_state=42).sort_values(
            "article_id", kind="stable"
        )

    top_tweets = (
        tw_sorted.groupby("article_id", as_index=False)
        .first()[["article_id", "tweet_text"]]
        .rename(columns={"tweet_text": "top_tweet_text"})
    )

    # Aggregates
    agg_dict = {
        "tweet_count": ("tweet_id", "count"),
        "mean_retweets": ("_is_retweet", "mean"),
        "mean_favorites": ("_likes", "mean"),
    }
    if "user_followers_count" in tw.columns:
        agg_dict["mean_followers"] = ("user_followers_count", "mean")
    if "user_friends_count" in tw.columns:
        agg_dict["mean_friends"] = ("user_friends_count", "mean")
    if "_verified" in tw.columns:
        agg_dict["verified_ratio"] = ("_verified", "mean")
    if "user_statuses_count" in tw.columns:
        agg_dict["mean_statuses"] = ("user_statuses_count", "mean")
    if "_account_age_days" in tw.columns:
        agg_dict["mean_account_age_days"] = ("_account_age_days", "mean")

    agg = tw.groupby("article_id", as_index=False).agg(**agg_dict)

    for col in AGGREGATED_SOCIAL_COLS:
        if col not in agg.columns:
            agg[col] = 0.0
        agg[col] = agg[col].fillna(0)

    log.info(f"  Aggregates computed for {len(agg):,} articles")

    df = news.merge(top_tweets, on="article_id", how="inner")
    df = df.merge(agg, on="article_id", how="left")
    for col in AGGREGATED_SOCIAL_COLS:
        df[col] = df[col].fillna(0)

    def _build_text(row):
        top_tweet = str(row.get("top_tweet_text", "") or "").strip()
        article = str(row.get("article_text", "") or "").strip()
        if not article:
            article = str(row.get("article_title", "") or "").strip()
        return f"[TWEET] {top_tweet} [ARTICLE] {article}"

    df["text"] = df.apply(_build_text, axis=1)
    df["label"] = (df["article_label"].astype(str).str.upper() == "FAKE").astype(int)

    log.info(f"  Final aggregated dataset: {len(df):,} articles")
    lc = df["label"].value_counts()
    log.info(f"    FAKE: {lc.get(1, 0):,}, REAL: {lc.get(0, 0):,}")

    return df


def build_aggregated_data(
    dataset_id: int | str,
    dataset_name: Optional[str] = None,
    top_tweet_strategy: str = "popularity",
    min_text_length: int = 30,
    test_ratio: float = 0.20,
    seed: int = 42,
    splits_subdir: Optional[str] = None,
):
    """Build aggregated train/test для NB. Returns (train, test, stats, tmpdir).

    Args:
        splits_subdir: ім'я папки `splits_*` для зовнішніх splits.
            None → fallback на legacy `splits/`, інакше random split.
            Aggregated режим не використовує val_df: якщо знайдено external
            splits з val — він мерджиться у train.
    """
    import os

    csv_dir, tmpdir = resolve_dataset_path(dataset_id, dataset_name)

    news = pd.read_csv(
        os.path.join(csv_dir, "news.csv"),
        low_memory=False, dtype={"article_id": str},
    )
    tweets = pd.read_csv(
        os.path.join(csv_dir, "tweets.csv"),
        low_memory=False, dtype={"article_id": str},
    )

    users = None
    users_path = os.path.join(csv_dir, "users.csv")
    if os.path.exists(users_path):
        users = pd.read_csv(users_path, low_memory=False)

    df = build_aggregated_dataframe(news, tweets, users, top_tweet_strategy)

    # Length filter
    df = df[df["text"].str.len() >= min_text_length].copy()
    df["article_id"] = df["article_id"].astype(str)

    # ── External splits (optional) → fallback на random split ──
    external = _try_load_external_splits(csv_dir, df, splits_subdir=splits_subdir)
    if external is not None:
        train_part, val_part, test_df = external
        # NB-aggregated не має окремого val — мерджимо у train.
        if len(val_part) > 0:
            train_df = pd.concat([train_part, val_part], ignore_index=True)
        else:
            train_df = train_part.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        if splits_subdir:
            splits_used = splits_subdir.replace("splits_", "", 1)
        else:
            splits_used = "splits"  # legacy
    else:
        rng = np.random.RandomState(seed)
        all_ids = df["article_id"].tolist()
        shuffled = rng.permutation(all_ids).tolist()
        n_test = max(1, int(len(shuffled) * test_ratio))
        test_ids = set(shuffled[:n_test])

        train_df = df[~df["article_id"].isin(test_ids)].reset_index(drop=True)
        test_df = df[df["article_id"].isin(test_ids)].reset_index(drop=True)
        splits_used = "auto"

    stats = {
        "mode": "aggregated",
        "samples_train_val": len(train_df),
        "samples_test": len(test_df),
        "label_train_fake": int((train_df["label"] == 1).sum()),
        "label_train_real": int((train_df["label"] == 0).sum()),
        "label_test_fake": int((test_df["label"] == 1).sum()),
        "label_test_real": int((test_df["label"] == 0).sum()),
        "social_feature_cols": list(AGGREGATED_SOCIAL_COLS),
        "splits_used": splits_used,
    }

    return train_df, test_df, stats, tmpdir
