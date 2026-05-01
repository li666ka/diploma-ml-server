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

    X_train = df_train["text_processed"].values
    y_train = df_train["label"].astype(int).values
    X_test = df_test["text_processed"].values
    y_test = df_test["label"].astype(int).values

    log.info(
        f"train_nb (article-level): variant={nb_variant}, vec={vectorizer_type}, "
        f"ngrams={ngram_range}, train={len(X_train)}, "
        f"val={len(df_val) if df_val is not None else 0}, test={len(X_test)}"
    )

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
        X_val = df_val["text_processed"].values
        y_val = df_val["label"].astype(int).values
        log.info(f"  Auto-tuning alpha across {ALPHA_GRID} on val set...")
        chosen_alpha = ALPHA_GRID[0]
        best_val_f1 = -1.0
        for a in ALPHA_GRID:
            pipe = Pipeline([
                ("vectorizer", get_vectorizer(
                    vectorizer_type, ngram_range, tfidf_max_features
                )),
                ("classifier", get_nb_classifier(nb_variant, a)),
            ])
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
    pipeline = Pipeline([
        ("vectorizer", get_vectorizer(
            vectorizer_type, ngram_range, tfidf_max_features
        )),
        ("classifier", get_nb_classifier(nb_variant, chosen_alpha)),
    ])

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)

    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, elapsed)
    metrics["train_size"] = len(X_train)
    if best_val_f1 is not None:
        metrics["val_f1_macro"] = round(float(best_val_f1), 4)

    log.info(
        f"NB (article-level) done: acc={metrics['accuracy']:.4f}, "
        f"f1={metrics['f1_score']:.4f}, f1_macro={metrics['f1_macro']:.4f}"
    )

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
    metrics = compute_metrics(y_test, y_pred, elapsed)
    metrics["train_size"] = len(X_train)

    log.info(
        f"NB (aggregated) done: acc={metrics['accuracy']:.4f}, "
        f"f1={metrics['f1_score']:.4f}"
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
