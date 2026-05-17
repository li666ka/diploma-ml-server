"""Local explanation для Naive Bayes через log-odds per feature.

Працює для всіх трьох режимів `nb_trainer.train_nb`:
  Mode A: Pipeline([("vectorizer", TfidfVec), ("classifier", NB)])
          → лише text-tokens у tokens_attribution
  Mode B: Pipeline([("preprocessor", ColumnTransformer(text+num)), ("classifier", NB)])
          → text-tokens + handcrafted features
  Mode C: Pipeline([("num_scaler", MinMaxScaler), ("classifier", NB)])
          → лише handcrafted features (tokens_attribution=[])

Метод: attribution(f) = value(f) × (log P(f|FAKE) − log P(f|REAL))
Знак → напрямок (>0 FAKE, <0 REAL); модуль → сила.
"""
from __future__ import annotations

import logging
from typing import Optional

import joblib

log = logging.getLogger(__name__)


def explain_nb_prediction(
    model_path: str,
    text: str,
    feature_values: Optional[dict] = None,
    top_k: int = 15,
) -> dict:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        return {"method": "log_odds", "error": "invalid_bundle",
                "message": "Bundle must be dict (legacy raw Pipeline не підтримується)"}

    pipeline = bundle.get("pipeline")
    if pipeline is None:
        return {"method": "log_odds", "error": "no_pipeline_in_bundle",
                "bundle_keys": sorted(bundle.keys())}

    use_text = bool(bundle.get("use_text", True))
    additional_features: list[str] = list(bundle.get("additional_features") or [])
    use_features = bool(additional_features)

    # ── Знайти classifier (має feature_log_prob_) ─────────────────────
    clf = None
    for _name, step in pipeline.steps:
        if hasattr(step, "feature_log_prob_"):
            clf = step
            break
    if clf is None:
        return {"method": "log_odds", "error": "no_classifier_with_log_prob"}

    classes = list(getattr(clf, "classes_", []))
    if 1 not in classes:
        return {"method": "log_odds", "error": "no_fake_class",
                "classes": classes}
    fake_idx = classes.index(1)
    real_idx = 1 - fake_idx
    log_probs = clf.feature_log_prob_  # (2, n_features_total)
    diff = log_probs[fake_idx] - log_probs[real_idx]

    # ── Визначити mode + витягти vectorizer (якщо є) ──────────────────
    named = dict(pipeline.named_steps)
    mode: Optional[str] = None
    vectorizer = None
    text_feature_count = 0

    if "vectorizer" in named:
        mode = "A"
        vectorizer = named["vectorizer"]
        text_feature_count = len(vectorizer.get_feature_names_out())
    elif "preprocessor" in named:
        mode = "B"
        preproc = named["preprocessor"]
        # ColumnTransformer кладе transformers у named_transformers_
        try:
            vectorizer = preproc.named_transformers_.get("text_vec")
        except AttributeError:
            vectorizer = None
        if vectorizer is not None and hasattr(vectorizer, "get_feature_names_out"):
            text_feature_count = len(vectorizer.get_feature_names_out())
    elif "num_scaler" in named:
        mode = "C"
    else:
        return {"method": "log_odds", "error": "unknown_pipeline_structure",
                "steps": [(n, type(s).__name__) for n, s in pipeline.steps]}

    # ── Препроцесинг тексту як у тренуванні ───────────────────────────
    from ml_server.utils import preprocess_text
    processed_text = preprocess_text(text, bundle.get("preprocessing") or {}) or text

    tokens_attribution: list[dict] = []
    feature_attribution: list[dict] = []

    # ── Text contributions (Mode A/B) ─────────────────────────────────
    if vectorizer is not None and use_text:
        X_text = vectorizer.transform([processed_text])
        feature_names = vectorizer.get_feature_names_out()
        text_diff = diff[:text_feature_count]
        coo = X_text.tocoo()
        for col, count in zip(coo.col, coo.data):
            attribution = float(count) * float(text_diff[col])
            tokens_attribution.append({
                "token": str(feature_names[col]),
                "count": float(count),
                "log_odds_diff": round(float(text_diff[col]), 4),
                "attribution": round(attribution, 4),
            })

    # ── Handcrafted features (Mode B/C) ───────────────────────────────
    if use_features:
        # У diff: спочатку text features, потім numeric (порядок, який задано
        # ColumnTransformer transformers list — у нашому trainer:
        #   [("text_vec", vec, "text_processed"), ("num_scaler", scaler, add_feat_cols)]).
        # Для Mode C text_feature_count=0, тож feature_diffs = весь diff.
        feature_diffs = diff[text_feature_count:]
        fv = feature_values or {}
        for i, fname in enumerate(additional_features):
            if i >= len(feature_diffs):
                break
            value = float(fv.get(fname, 0.0))
            attribution = value * float(feature_diffs[i])
            feature_attribution.append({
                "feature": fname,
                "raw_value": value,
                "log_odds_diff": round(float(feature_diffs[i]), 4),
                "attribution": round(attribution, 4),
            })

    tokens_attribution.sort(key=lambda x: abs(x["attribution"]), reverse=True)
    feature_attribution.sort(key=lambda x: abs(x["attribution"]), reverse=True)

    total = (
        sum(c["attribution"] for c in tokens_attribution)
        + sum(c["attribution"] for c in feature_attribution)
    )
    total = round(float(total), 4)

    return {
        "method": "log_odds",
        "mode": mode,
        "method_params": {
            "use_text": use_text,
            "use_features": use_features,
            "n_text_features": text_feature_count,
            "n_handcrafted_features": len(additional_features),
            "classifier": type(clf).__name__,
        },
        "tokens": tokens_attribution[:top_k],
        "all_tokens": tokens_attribution,
        "feature_attributions": feature_attribution,
        "total_log_odds": total,
        "prediction": "FAKE" if total > 0 else "REAL",
    }
