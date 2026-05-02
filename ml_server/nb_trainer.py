"""Naive Bayes trainer.

Дві окремі функції:
- ``train_nb``            — article-level pipeline (default, apples-to-apples
                            з DistilBERT і GNN). Використовує ``combined_text``,
                            робить auto-tuning alpha по validation set.
- ``train_nb_aggregated`` — legacy aggregated pipeline (1 row per article =
                            top-tweet + article + 8 social aggregates +
                            optional emotional/stylistic/rhetorical features).
                            Залишено для майбутніх ablation-експериментів.
"""
import os
import time
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from ml_server.utils import compute_metrics, create_download_url, log, preprocess_text


# ── Helpers ──────────────────────────────────────────────────────────────

def get_nb_classifier(variant: str, alpha: float):
    if variant == "complement":
        return ComplementNB(alpha=alpha)
    return MultinomialNB(alpha=alpha)


def get_vectorizer(vec_type: str, ngram_range, max_features: int = 50000):
    if isinstance(ngram_range, str):
        a, b = ngram_range.split(",")
        ngram = (int(a), int(b))
    else:
        ngram = tuple(ngram_range)
    if vec_type == "count":
        return CountVectorizer(ngram_range=ngram, max_features=max_features)
    return TfidfVectorizer(
        ngram_range=ngram, max_features=max_features,
        min_df=2, max_df=0.95, sublinear_tf=True,
    )


def _extract_top_words(pipeline: Pipeline, vec_step: str = "vectorizer",
                       clf_step: str = "classifier", top_n: int = 20) -> dict:
    """Витягнути топ-N дискримінативних n-gram для FAKE/REAL."""
    top_words = {"fake": [], "real": []}
    try:
        vec = pipeline.named_steps[vec_step]
        clf = pipeline.named_steps[clf_step]
        vocab = vec.get_feature_names_out()
        log_prob = clf.feature_log_prob_
        n_text = len(vocab)
        if log_prob.shape[1] >= n_text:
            diff = log_prob[1, :n_text] - log_prob[0, :n_text]
            fake_idx = diff.argsort()[-top_n:][::-1]
            real_idx = diff.argsort()[:top_n]
            top_words["fake"] = [
                {"word": vocab[i], "score": float(diff[i])} for i in fake_idx
            ]
            top_words["real"] = [
                {"word": vocab[i], "score": float(-diff[i])} for i in real_idx
            ]
    except Exception as e:
        log.warning(f"top words extraction failed: {e}")
    return top_words


# ── Per-article social/graph aggregation (для article-level NB) ──────────

# Graph features обчислюються окремо (per-article cascade); решта SOCIAL_FEATURES
# усереднюється по всіх твітах статті.
_GRAPH_FEAT_NAMES = {
    "cascade_depth_norm", "cascade_breadth_norm", "lifetime_hours_norm",
    "retweets_per_tweet", "replies_per_tweet", "unique_users_norm",
}


def _compute_social_aggregates_per_article(
    article_ids, tweets_df, retweets_df, replies_df, users_df, social_features,
):
    """Returns DataFrame: article_id × social_feature columns.
    Усереднює social features по всіх твітах статті (Variant A).
    """
    from ml_server.features import SOCIAL_FEATURES, extract_social_features

    # Build user_id → user_row dict
    user_lookup = {}
    if users_df is not None and len(users_df) > 0:
        for _, row in users_df.iterrows():
            uid = str(row.get("user_id", ""))
            if uid:
                user_lookup[uid] = row

    # ── Tweet engagement lookup ──
    # tweets.csv не містить retweet_count/reply_count, тож обчислюємо їх
    # один раз через groupby по retweets.csv / replies.csv.
    tweet_engagement_lookup: dict[str, dict[str, int]] = {}
    if len(retweets_df) > 0 and "original_tweet_id" in retweets_df.columns:
        rt_clean = retweets_df.dropna(subset=["original_tweet_id"])
        rt_counts = rt_clean.groupby(
            rt_clean["original_tweet_id"].astype(str)
        ).size()
        for tid, count in rt_counts.items():
            tweet_engagement_lookup[tid] = {"retweets": int(count), "replies": 0}

    if len(replies_df) > 0 and "parent_tweet_id" in replies_df.columns:
        rep_clean = replies_df.dropna(subset=["parent_tweet_id"])
        rep_counts = rep_clean.groupby(
            rep_clean["parent_tweet_id"].astype(str)
        ).size()
        for tid, count in rep_counts.items():
            entry = tweet_engagement_lookup.get(tid)
            if entry is None:
                tweet_engagement_lookup[tid] = {
                    "retweets": 0, "replies": int(count),
                }
            else:
                entry["replies"] = int(count)

    log.info(
        f"  Built engagement lookup: {len(tweet_engagement_lookup):,} tweets"
    )

    pure_social = [
        f for f in social_features
        if f in SOCIAL_FEATURES and f not in _GRAPH_FEAT_NAMES
    ]

    # ── Pre-compute: groupby один раз (O(N+M) замість O(N*M)) ──
    if len(tweets_df) > 0 and "article_id" in tweets_df.columns:
        aid_keys = tweets_df["article_id"].astype(str)
        tweets_by_article = dict(tuple(tweets_df.groupby(aid_keys)))
    else:
        tweets_by_article = {}

    rows = []
    for aid in article_ids:
        aid_str = str(aid)
        article_tweets = tweets_by_article.get(aid_str, pd.DataFrame())

        feat_accum = {f: [] for f in pure_social}
        for _, twt in article_tweets.iterrows():
            uid = str(twt.get("user_id", ""))
            user_row = user_lookup.get(uid)
            feats = extract_social_features(
                user_row, pure_social, tweet_row=twt,
                tweet_engagement_lookup=tweet_engagement_lookup,
            )
            for f, v in feats.items():
                feat_accum[f].append(v)

        article_feats = {"article_id": aid_str}
        for f, vals in feat_accum.items():
            article_feats[f] = float(np.mean(vals)) if vals else 0.0
        rows.append(article_feats)

    return pd.DataFrame(rows)


def _compute_graph_aggregates_per_article(
    article_ids, tweets_df, retweets_df, replies_df, graph_features,
):
    """Returns DataFrame: article_id × graph_feature columns."""
    from ml_server.features import extract_graph_features

    # ── Pre-compute: groupby один раз (O(1) lookup замість full scan) ──
    if len(tweets_df) > 0 and "article_id" in tweets_df.columns:
        aid_keys = tweets_df["article_id"].astype(str)
        tweets_by_article = dict(tuple(tweets_df.groupby(aid_keys)))
    else:
        tweets_by_article = {}

    if len(retweets_df) > 0 and "original_tweet_id" in retweets_df.columns:
        rt_keys = retweets_df["original_tweet_id"].astype(str)
        retweets_by_orig = dict(tuple(retweets_df.groupby(rt_keys)))
    else:
        retweets_by_orig = {}

    if len(replies_df) > 0 and "parent_tweet_id" in replies_df.columns:
        rep_keys = replies_df["parent_tweet_id"].astype(str)
        replies_by_parent = dict(tuple(replies_df.groupby(rep_keys)))
    else:
        replies_by_parent = {}

    empty = pd.DataFrame()

    rows = []
    for aid in article_ids:
        aid_str = str(aid)
        article_tweets = tweets_by_article.get(aid_str, empty)

        if len(article_tweets) == 0:
            tweet_ids_list: list[str] = []
        else:
            tweet_ids_list = article_tweets["tweet_id"].astype(str).tolist()

        # Retweets для цих tweet_id
        rt_parts = [retweets_by_orig[t] for t in tweet_ids_list if t in retweets_by_orig]
        article_retweets = (
            pd.concat(rt_parts, ignore_index=True) if rt_parts else empty
        )

        # Replies — прямі діти tweets + транзитивне розширення на 1 крок
        # через reply_id (replies можуть посилатися на інші replies).
        rep_parts = [replies_by_parent[t] for t in tweet_ids_list if t in replies_by_parent]
        article_replies = (
            pd.concat(rep_parts, ignore_index=True) if rep_parts else empty
        )
        if len(article_replies) > 0 and "reply_id" in article_replies.columns:
            initial_reply_ids = article_replies["reply_id"].astype(str).tolist()
            extra_parts = [
                replies_by_parent[r]
                for r in initial_reply_ids
                if r in replies_by_parent
            ]
            if extra_parts:
                article_replies = pd.concat(
                    [article_replies] + extra_parts, ignore_index=True,
                )

        feats = extract_graph_features(
            aid_str, article_tweets, article_retweets, article_replies, graph_features,
        )
        feats["article_id"] = aid_str
        rows.append(feats)

    return pd.DataFrame(rows)


# ── Article-level NB (DEFAULT) ───────────────────────────────────────────

ALPHA_GRID = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]


def train_nb(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    user_id,
    experiment_id,
    *,
    nb_variant: str = "complement",
    vectorizer_type: str = "tfidf",
    ngram_range: str = "1,2",
    alpha: Optional[float] = None,
    tfidf_max_features: int = 50000,
    preprocessing: Optional[dict] = None,
    additional_features: Optional[list] = None,
    full_data: Optional[dict] = None,
):
    """Article-level NB на ``combined_text`` (article_title + article_text).

    Якщо ``alpha`` is None — автоматичний тюнінг по ALPHA_GRID за val_f1_macro.
    Якщо передано конкретне значення — auto-tuning пропускається.

    Args:
        train_df, val_df, test_df: DataFrame з колонками ``combined_text``, ``label``.
            ``val_df`` може бути None — тоді auto-tuning пропускається,
            використовується ``alpha`` (або 1.0).
        user_id, experiment_id: для save path.
        nb_variant: "complement" (default) або "multinomial".
        vectorizer_type: "tfidf" (default) або "count".
        ngram_range: рядок "a,b" (default "1,2").
        alpha: float — фіксоване значення (skip auto-tune); None — auto-tune.
        tfidf_max_features: розмір словника.
        preprocessing: опції для preprocess_text.
    """
    if preprocessing is None:
        preprocessing = {}

    text_col = "combined_text"
    if text_col not in train_df.columns:
        raise ValueError(
            f"train_df missing required column '{text_col}'. "
            f"Got: {list(train_df.columns)}"
        )

    df_train = train_df.copy()
    df_test = test_df.copy()
    df_val = val_df.copy() if val_df is not None else None

    # Preprocess text
    df_train["text_processed"] = df_train[text_col].astype(str).apply(
        lambda t: preprocess_text(t, preprocessing)
    )
    df_test["text_processed"] = df_test[text_col].astype(str).apply(
        lambda t: preprocess_text(t, preprocessing)
    )
    if df_val is not None:
        df_val["text_processed"] = df_val[text_col].astype(str).apply(
            lambda t: preprocess_text(t, preprocessing)
        )

    y_train = df_train["label"].astype(int).values
    y_test = df_test["label"].astype(int).values

    log.info(
        f"train_nb (article-level): variant={nb_variant}, vec={vectorizer_type}, "
        f"ngrams={ngram_range}, train={len(df_train)}, "
        f"val={len(df_val) if df_val is not None else 0}, test={len(df_test)}"
    )

    # ── Additional features (опційно) ──
    add_feat_cols: list[str] = []
    if additional_features:
        log.info(f"Processing additional features: {additional_features}")

        from ml_server.features import (
            EMOTIONAL_FEATURES, SOCIAL_FEATURES, STYLISTIC_FEATURES,
            extract_features, nrc_el, nrc_eil,
        )

        text_feats = [f for f in additional_features
                      if f in EMOTIONAL_FEATURES or f in STYLISTIC_FEATURES]
        social_feats_pure = [f for f in additional_features
                             if f in SOCIAL_FEATURES and f not in _GRAPH_FEAT_NAMES]
        graph_feats = [f for f in additional_features if f in _GRAPH_FEAT_NAMES]

        log.info(
            f"  Split: text={len(text_feats)}, social={len(social_feats_pure)}, "
            f"graph={len(graph_feats)}"
        )

        df_parts = [df_train, df_test]
        if df_val is not None:
            df_parts.insert(1, df_val)

        # Text-based features (per-row)
        if text_feats:
            for df_part in df_parts:
                feats_list = df_part["combined_text"].apply(
                    lambda t: extract_features(t, nrc_el, nrc_eil, text_feats)
                )
                feats_df = pd.DataFrame(feats_list.tolist(), index=df_part.index)
                for col in feats_df.columns:
                    df_part[col] = feats_df[col]
                    if col not in add_feat_cols:
                        add_feat_cols.append(col)

        # Social/graph features (потребують full_data)
        if (social_feats_pure or graph_feats) and full_data is not None:
            tweets_df = full_data.get("tweets", pd.DataFrame())
            retweets_df = full_data.get("retweets", pd.DataFrame())
            replies_df = full_data.get("replies", pd.DataFrame())
            users_df = full_data.get("users", pd.DataFrame())

            for i, df_part in enumerate(df_parts):
                article_ids = df_part["article_id"].tolist()

                if social_feats_pure:
                    soc_df = _compute_social_aggregates_per_article(
                        article_ids, tweets_df, retweets_df, replies_df, users_df,
                        social_feats_pure,
                    )
                    df_part = df_part.merge(soc_df, on="article_id", how="left")
                    for f in social_feats_pure:
                        df_part[f] = df_part[f].fillna(0.0)
                        if f not in add_feat_cols:
                            add_feat_cols.append(f)
                    df_parts[i] = df_part

                if graph_feats:
                    gr_df = _compute_graph_aggregates_per_article(
                        article_ids, tweets_df, retweets_df, replies_df,
                        graph_feats,
                    )
                    df_part = df_part.merge(gr_df, on="article_id", how="left")
                    for f in graph_feats:
                        df_part[f] = df_part[f].fillna(0.0)
                        if f not in add_feat_cols:
                            add_feat_cols.append(f)
                    df_parts[i] = df_part

            # Re-bind після merge (merge створює нові DataFrame)
            df_train = df_parts[0]
            if df_val is not None:
                df_val = df_parts[1]
                df_test = df_parts[2]
            else:
                df_test = df_parts[1]
        elif social_feats_pure or graph_feats:
            log.warning(
                "Social/graph features requested але full_data=None. "
                "Пропускаю social/graph (text features залишаються)."
            )

    # ── Pipeline builder ──
    use_features = bool(add_feat_cols)

    def _make_pipeline(a):
        """Build pipeline for given alpha. With or without additional features."""
        if use_features:
            transformers = [
                ("text_vec",
                 get_vectorizer(vectorizer_type, ngram_range, tfidf_max_features),
                 "text_processed"),
                ("num_scaler", MinMaxScaler(), add_feat_cols),
            ]
            return Pipeline([
                ("preprocessor", ColumnTransformer(transformers=transformers)),
                ("classifier", get_nb_classifier(nb_variant, a)),
            ])
        return Pipeline([
            ("vectorizer", get_vectorizer(
                vectorizer_type, ngram_range, tfidf_max_features
            )),
            ("classifier", get_nb_classifier(nb_variant, a)),
        ])

    def _make_X(df_part):
        if use_features:
            return df_part[["text_processed"] + add_feat_cols]
        return df_part["text_processed"].values

    X_train = _make_X(df_train)
    X_test = _make_X(df_test)

    # ── Alpha selection ──
    alpha_search: list[dict] = []
    best_val_f1 = None

    if alpha is not None:
        chosen_alpha = float(alpha)
        log.info(f"  alpha fixed by user: {chosen_alpha}")
    elif df_val is None or len(df_val) == 0:
        chosen_alpha = 1.0
        log.warning("  no val_df provided — defaulting alpha=1.0 (no tuning)")
    else:
        X_val = _make_X(df_val)
        y_val = df_val["label"].astype(int).values
        log.info(f"  Auto-tuning alpha across {ALPHA_GRID} on val set...")
        chosen_alpha = ALPHA_GRID[0]
        best_val_f1 = -1.0
        for a in ALPHA_GRID:
            pipe = _make_pipeline(a)
            pipe.fit(X_train, y_train)
            y_val_pred = pipe.predict(X_val)
            from sklearn.metrics import f1_score
            f1m = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
            alpha_search.append({"alpha": a, "val_f1_macro": round(f1m, 4)})
            log.info(f"    alpha={a:<6} → val_f1_macro={f1m:.4f}")
            if f1m > best_val_f1:
                best_val_f1 = f1m
                chosen_alpha = a
        log.info(f"  ✓ Best alpha={chosen_alpha} (val_f1_macro={best_val_f1:.4f})")

    # ── Final training on train (з обраним alpha) ──
    pipeline = _make_pipeline(chosen_alpha)

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)

    y_pred = pipeline.predict(X_test)

    # ROC-AUC: predict_proba повертає [N, 2], беремо стовпець FAKE (index 1)
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except Exception as e:
        log.warning(f"predict_proba failed: {e}, ROC-AUC буде відсутній")
        y_proba = None

    metrics = compute_metrics(y_test, y_pred, elapsed, y_proba=y_proba)
    metrics["train_size"] = len(df_train)
    metrics["additional_features_used"] = list(add_feat_cols)
    if best_val_f1 is not None:
        metrics["val_f1_macro"] = round(float(best_val_f1), 4)

    log.info(
        f"NB (article-level) done: acc={metrics['accuracy']:.4f}, "
        f"f1={metrics['f1_score']:.4f}, f1_macro={metrics['f1_macro']:.4f}"
        + (f", roc_auc={metrics['roc_auc']:.4f}" if metrics.get("roc_auc") is not None else "")
    )

    # top_words: для ColumnTransformer-based pipeline vectorizer лежить інакше.
    if use_features:
        top_words = {"fake": [], "real": []}
        try:
            preproc = pipeline.named_steps["preprocessor"]
            text_vec = preproc.named_transformers_["text_vec"]
            clf = pipeline.named_steps["classifier"]
            vocab = text_vec.get_feature_names_out()
            log_prob = clf.feature_log_prob_
            n_text = len(vocab)
            if log_prob.shape[1] >= n_text:
                diff = log_prob[1, :n_text] - log_prob[0, :n_text]
                top_n = 20
                fake_idx = diff.argsort()[-top_n:][::-1]
                real_idx = diff.argsort()[:top_n]
                top_words["fake"] = [
                    {"word": vocab[i], "score": float(diff[i])} for i in fake_idx
                ]
                top_words["real"] = [
                    {"word": vocab[i], "score": float(-diff[i])} for i in real_idx
                ]
        except Exception as e:
            log.warning(f"top words extraction (with features) failed: {e}")
    else:
        top_words = _extract_top_words(pipeline)

    save_path = (
        f"/content/drive/MyDrive/diploma_models/user_{user_id}/"
        f"model_nb_{experiment_id}.pkl"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "preprocessing": dict(preprocessing),
        "type": "nb",
        "pipeline_type": "article",
        "nb_variant": nb_variant,
        "vectorizer_type": vectorizer_type,
        "ngram_range": ngram_range,
        "alpha": float(chosen_alpha),
        "alpha_search": alpha_search,
        "tfidf_max_features": tfidf_max_features,
        "additional_features": list(additional_features) if additional_features else [],
    }
    joblib.dump(bundle, save_path)
    log.info(f"  Saved article-level NB → {save_path}")

    return {
        "path": save_path,
        "metrics": metrics,
        "top_words": top_words,
        "alpha_search": alpha_search,
        "best_alpha": float(chosen_alpha),
        "download_url": create_download_url(save_path),
    }


# ── Aggregated NB (legacy, для майбутніх ablation) ───────────────────────

def train_nb_aggregated(
    train_df, test_df, user_id, experiment_id,
    emotional_features=None, stylistic_features=None,
    rhetorical_features=None, social_features=None,
    use_text=True, nb_variant="complement", vectorizer_type="tfidf",
    ngram_range="1,1", alpha=1.0, preprocessing=None,
):
    """Legacy aggregated pipeline (1 row per article = top-tweet+article+social).

    Залишено для ablation. Працює на колонці ``text`` (як її формує
    ``aggregated_loader.build_aggregated_data``).
    """
    from ml_server.features import extract_features, nrc_el, nrc_eil

    if preprocessing is None:
        preprocessing = {}
    if emotional_features is None:
        emotional_features = []
    if stylistic_features is None:
        stylistic_features = []
    if rhetorical_features is None:
        rhetorical_features = []
    if social_features is None:
        social_features = []

    text_based_additional = (
        list(emotional_features) + list(stylistic_features) + list(rhetorical_features)
    )
    all_additional = text_based_additional + list(social_features)

    log.info(
        f"train_nb_aggregated: use_text={use_text}, "
        f"emo={emotional_features}, styl={stylistic_features}, "
        f"rhet={rhetorical_features}, soc={social_features}"
    )

    df_train = train_df.copy()
    df_test = test_df.copy()

    if text_based_additional:
        for part in (df_train, df_test):
            feats = part["text"].apply(
                lambda x: extract_features(x, nrc_el, nrc_eil, text_based_additional)
            )
            feat_df = pd.DataFrame(feats.tolist(), index=part.index)
            for col in feat_df.columns:
                part[col] = feat_df[col]

    if social_features:
        missing = [f for f in social_features if f not in df_train.columns]
        if missing:
            log.warning(f"  Social features missing from df: {missing}. Filling with 0.")
            for f in missing:
                df_train[f] = 0.0
                df_test[f] = 0.0

    df_train["text_processed"] = df_train["text"].apply(
        lambda t: preprocess_text(t, preprocessing)
    )
    df_test["text_processed"] = df_test["text"].apply(
        lambda t: preprocess_text(t, preprocessing)
    )

    cols = (["text_processed"] if use_text else []) + all_additional
    X_train = df_train[cols]
    y_train = df_train["label"].astype(int)
    X_test = df_test[cols]
    y_test = df_test["label"].astype(int)

    transformers = []
    if use_text:
        transformers.append((
            "text_vec",
            get_vectorizer(vectorizer_type, ngram_range),
            "text_processed",
        ))
    if all_additional:
        transformers.append(("num_scaler", MinMaxScaler(), all_additional))

    if not transformers:
        raise ValueError("Need at least one feature (text or additional)")

    preprocessor = ColumnTransformer(transformers=transformers)
    clf = get_nb_classifier(nb_variant, alpha)
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)

    y_pred = pipeline.predict(X_test)

    # ROC-AUC: predict_proba повертає [N, 2], беремо стовпець FAKE (index 1)
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except Exception as e:
        log.warning(f"predict_proba failed: {e}, ROC-AUC буде відсутній")
        y_proba = None

    metrics = compute_metrics(y_test, y_pred, elapsed, y_proba=y_proba)
    metrics["train_size"] = len(X_train)

    log.info(
        f"NB (aggregated) done: acc={metrics['accuracy']:.4f}, "
        f"f1={metrics['f1_score']:.4f}"
        + (f", roc_auc={metrics['roc_auc']:.4f}" if metrics.get("roc_auc") is not None else "")
    )

    top_words = {"fake": [], "real": []}
    try:
        nb_clf = pipeline.named_steps["classifier"]
        if use_text:
            text_vec = pipeline.named_steps["preprocessor"].transformers_[0][1]
            vocab = text_vec.get_feature_names_out()
            log_prob = nb_clf.feature_log_prob_
            n_text = len(vocab)
            if log_prob.shape[1] >= n_text:
                diff = log_prob[1, :n_text] - log_prob[0, :n_text]
                top_n = 20
                fake_idx = diff.argsort()[-top_n:][::-1]
                real_idx = diff.argsort()[:top_n]
                top_words["fake"] = [
                    {"word": vocab[i], "score": float(diff[i])} for i in fake_idx
                ]
                top_words["real"] = [
                    {"word": vocab[i], "score": float(-diff[i])} for i in real_idx
                ]
    except Exception as e:
        log.warning(f"top words failed: {e}")

    social_train_mean = {}
    for f in social_features:
        if f in df_train.columns:
            social_train_mean[f] = float(df_train[f].mean())

    save_path = (
        f"/content/drive/MyDrive/diploma_models/user_{user_id}/"
        f"model_nb_{experiment_id}.pkl"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "preprocessing": dict(preprocessing),
        "type": "nb",
        "pipeline_type": "aggregated",
        "emotional_features": list(emotional_features),
        "stylistic_features": list(stylistic_features),
        "rhetorical_features": list(rhetorical_features),
        "social_features": list(social_features),
        "social_train_mean": social_train_mean,
    }, save_path)
    log.info(f"  Saved aggregated NB → {save_path}")

    return {
        "path": save_path,
        "metrics": metrics,
        "top_words": top_words,
        "download_url": create_download_url(save_path),
    }
