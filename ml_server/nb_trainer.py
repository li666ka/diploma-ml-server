"""Aggregated NB trainer (existing pipeline from v16 notebook).

ВАЖЛИВО: цей модуль — простий stub з основною логікою.
Повний код (з emotional/stylistic/rhetorical features + NRC lexicon)
треба перенести з v16 ноутбука у цей файл.

Поки що тут простий ComplementNB pipeline з TF-IDF.
Якщо потрібен повний функціонал — портувати з ноутбука.
"""
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

from ml_server.config import MODELS_ROOT
from ml_server.utils import compute_metrics, create_download_url, log, preprocess_text


def get_nb_classifier(variant: str, alpha: float):
    if variant == "complement":
        return ComplementNB(alpha=alpha)
    return MultinomialNB(alpha=alpha)


def get_vectorizer(vec_type: str, ngram_range: str):
    if isinstance(ngram_range, str):
        a, b = ngram_range.split(",")
        ngram = (int(a), int(b))
    else:
        ngram = ngram_range
    if vec_type == "count":
        return CountVectorizer(ngram_range=ngram, max_features=50000)
    return TfidfVectorizer(
        ngram_range=ngram, max_features=50000,
        min_df=2, max_df=0.95, sublinear_tf=True,
    )


def train_nb(train_df, test_df, user_id, experiment_id, emotional_features,
             stylistic_features=None, rhetorical_features=None,
             social_features=None,
             use_text=True, nb_variant="multinomial", vectorizer_type="tfidf",
             ngram_range="1,1", alpha=1.0, preprocessing=None):

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

    # Text-based features (from text)
    text_based_additional = list(emotional_features) + list(stylistic_features) + list(rhetorical_features)
    # Social features (from user_row)
    all_additional = text_based_additional + list(social_features)

    log.info(f"train_nb: use_text={use_text}, emo={emotional_features}, styl={stylistic_features}, rhet={rhetorical_features}, soc={social_features}")

    df_train = train_df.copy()
    df_test = test_df.copy()

    # Compute text-based features (emotional/stylistic/rhetorical)
    if text_based_additional:
        for part in (df_train, df_test):
            feats = part["text"].apply(lambda x: extract_features(x, nrc_el, nrc_eil, text_based_additional))
            feat_df = pd.DataFrame(feats.tolist(), index=part.index)
            for col in feat_df.columns:
                part[col] = feat_df[col]

    # Social features — для aggregated моделі вони ВЖЕ обчислені в Cell 6
    # як числові колонки (tweet_count, mean_followers, ...). Просто перевіряємо
    # що вони існують у df. Якщо ні — це інший тип моделі (не aggregated).
    if social_features:
        missing = [f for f in social_features if f not in df_train.columns]
        if missing:
            log.warning(f"  Social features missing from df: {missing}. Filling with 0.")
            for f in missing:
                df_train[f] = 0.0
                df_test[f] = 0.0
        log.info(f"  Social features (from aggregates): {social_features}")
        # Логуємо stats для перших трьох
        for sf in social_features[:3]:
            if sf in df_train.columns:
                log.info(f"    {sf}: mean={df_train[sf].mean():.3f}, "
                         f"std={df_train[sf].std():.3f}, "
                         f"max={df_train[sf].max():.3f}")

    # Preprocess text (після того як обчислили stylistic з оригіналу)
    df_train["text_processed"] = df_train["text"].apply(lambda t: preprocess_text(t, preprocessing))
    df_test["text_processed"] = df_test["text"].apply(lambda t: preprocess_text(t, preprocessing))

    # ── Collect feature samples for UI (3 FAKE + 3 REAL with breakdown) ──
    feature_samples = []
    try:
        # 3 FAKE + 3 REAL випадкових
        fake_part = df_train[df_train["label"] == 1]
        real_part = df_train[df_train["label"] == 0]
        picks = []
        if len(fake_part) > 0:
            picks.append(fake_part.sample(n=min(3, len(fake_part)), random_state=42))
        if len(real_part) > 0:
            picks.append(real_part.sample(n=min(3, len(real_part)), random_state=42))
        sample = pd.concat(picks) if picks else df_train.sample(n=min(6, len(df_train)), random_state=42)

        print("\n" + "=" * 60)
        print("📋 Feature samples (3 FAKE + 3 REAL)")
        print("=" * 60)

        for _, row in sample.iterrows():
            lbl = "FAKE" if row["label"] == 1 else "REAL"
            src = row.get("article_source", "?")
            text_raw = str(row["text"])[:200]
            text_proc = str(row["text_processed"])[:200]

            # Collect features breakdown (already computed у df)
            emo_dict = {f: float(row[f]) for f in emotional_features if f in row.index}
            styl_dict = {f: float(row[f]) for f in stylistic_features if f in row.index}
            rhet_dict = {f: float(row[f]) for f in rhetorical_features if f in row.index}
            soc_dict = {f: float(row[f]) for f in social_features if f in row.index}

            sample_obj = {
                "label": lbl,
                "source": str(src),
                "text_raw": text_raw,
                "text_processed": text_proc,
                "emotional": emo_dict,
                "stylistic": styl_dict,
                "rhetorical": rhet_dict,
                "social": soc_dict,
            }
            feature_samples.append(sample_obj)

            # Print to Colab log
            print(f"\n[{lbl} / {src}]")
            print(f"  BEFORE: {text_raw[:100]}")
            print(f"  AFTER:  {text_proc[:100]}")
            if emo_dict:
                top_emo = sorted(emo_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                print(f"  Emotional (top 5): {', '.join(f'{k}={v:.3f}' for k, v in top_emo)}")
            if styl_dict:
                print(f"  Stylistic: {', '.join(f'{k}={v:.3f}' for k, v in styl_dict.items())}")
            if rhet_dict:
                print(f"  Rhetorical: {', '.join(f'{k}={v:.3f}' for k, v in rhet_dict.items())}")
            if soc_dict:
                print(f"  Social: {', '.join(f'{k}={v:.3f}' for k, v in soc_dict.items())}")

        print("=" * 60 + "\n")
    except Exception as e:
        log.warning(f"Feature samples collection failed: {e}")

    cols = (["text_processed"] if use_text else []) + all_additional
    X_train = df_train[cols]
    y_train = df_train["label"].astype(int)
    X_test = df_test[cols]
    y_test = df_test["label"].astype(int)

    vectorizer = get_vectorizer(vectorizer_type, ngram_range)
    transformers = []
    if use_text:
        transformers.append(("text_vec", vectorizer, "text_processed"))
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

    log.info(f"NB done: acc={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}")

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
                top_words["fake"] = [{"word": vocab[i], "score": float(diff[i])} for i in fake_idx]
                top_words["real"] = [{"word": vocab[i], "score": float(-diff[i])} for i in real_idx]
    except Exception as e:
        log.warning(f"top words failed: {e}")

    # Зберегти train_mean соц.ознак — для inference fallback коли agg недоступні
    social_train_mean = {}
    for f in social_features:
        if f in df_train.columns:
            social_train_mean[f] = float(df_train[f].mean())

    save_path = f"/content/drive/MyDrive/diploma_models/user_{user_id}/model_nb_{experiment_id}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "preprocessing": preprocessing.copy(),
        "emotional_features": list(emotional_features),
        "stylistic_features": list(stylistic_features),
        "rhetorical_features": list(rhetorical_features),
        "social_features": list(social_features),
        "social_train_mean": social_train_mean,
    }, save_path)
    log.info(f"  Saved social_train_mean for {len(social_train_mean)} features")

    global nb_pipeline, nb_preprocessing
    global nb_emotional_features, nb_stylistic_features, nb_rhetorical_features, nb_social_features
    global nb_social_train_mean
    nb_pipeline = pipeline
    nb_preprocessing = preprocessing.copy()
    nb_emotional_features = list(emotional_features)
    nb_stylistic_features = list(stylistic_features)
    nb_rhetorical_features = list(rhetorical_features)
    nb_social_features = list(social_features)
    nb_social_train_mean = dict(social_train_mean)

    return {
        "path": save_path,
        "metrics": metrics,
        "top_words": top_words,
        "feature_samples": feature_samples,
        "download_url": create_download_url(save_path),
    }

