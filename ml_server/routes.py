"""Flask routes — registers all endpoints."""
import os
import shutil
import threading
import uuid
from datetime import datetime

from flask import Blueprint, Flask, jsonify, request

from ml_server.aggregated_loader import build_aggregated_data
from ml_server.data_loader import build_article_level_data
from ml_server.distilbert_trainer import (
    get_distilbert_state,
    set_distilbert_state,
    train_distilbert_article_level,
)
from ml_server.features import (
    EMOTIONAL_FEATURES,
    SOCIAL_FEATURES,
    STYLISTIC_FEATURES,
)
from ml_server.gnn_trainer import train_gnn
from ml_server.nb_trainer import train_nb, train_nb_aggregated
from ml_server.upload_handlers import (
    dataset_status,
    handle_upload_chunk,
    handle_upload_finalize,
    list_datasets,
    list_splits,
)
from ml_server.utils import log, preprocess_text


# ── GNNExplainer in-memory LRU cache ────────────────────────────────────
# GNNExplainer повільний (5-15s GPU, до 60s CPU), а одна стаття часто
# explain'иться кілька разів підряд з UI. Кеш на 16 свіжих результатів,
# TTL 5 хв.
_GNN_EXPLAIN_CACHE: dict = {}
_GNN_EXPLAIN_TTL = 300.0  # секунд

# ── NB pipeline LRU cache ───────────────────────────────────────────────
# joblib.load з Drive ~1-2s — небажано на кожен /predict_nb. Сам pipeline
# 1-2 МБ, легко кешуємо 16 свіжих.
_NB_PIPELINE_CACHE: dict = {}


def _detect_nb_mode(pipeline) -> str:
    """Authoritative детектор Mode за фактичним типом першого pipeline-кроку.

    Метадані bundle (`use_text`, `additional_features`) могли збреху́ти у
    legacy-моделях, тому покладаємось на структуру самого pipeline:

      Mode A: перший step — text-vectorizer (TfidfVectorizer, CountVectorizer)
              → приймає list[str].
      Mode B: перший step — ColumnTransformer → потрібен DataFrame з
              текстовою + numeric колонками.
      Mode C: усе інше (MinMaxScaler, StandardScaler, etc.) → numeric only.
    """
    from sklearn.compose import ColumnTransformer

    if not getattr(pipeline, "steps", None):
        return "unknown"
    first = pipeline.steps[0][1]
    if isinstance(first, ColumnTransformer):
        return "B"
    # vectorizer-like: fitted має vocabulary_; також ловимо за class name
    cname = type(first).__name__.lower()
    if hasattr(first, "vocabulary_") or "vectorizer" in cname:
        return "A"
    return "C"


def _extract_feature_cols_from_ct(ct) -> list[str]:
    """ColumnTransformer.transformers_ → list of numeric feature column names."""
    cols: list[str] = []
    for _name, _transformer, c in getattr(ct, "transformers_", []):
        if isinstance(c, (list, tuple)):
            cols.extend(str(x) for x in c)
    return cols


def _nb_build_input(pipeline, processed_text: str, add_feat_cols: list[str]):
    """Підготувати input для NB pipeline. Mode визначається з СТРУКТУРИ
    pipeline (а не з bundle metadata)."""
    mode = _detect_nb_mode(pipeline)

    if mode == "A":
        return [processed_text], mode

    if mode == "B":
        import pandas as pd
        from sklearn.compose import ColumnTransformer

        # Якщо bundle не зберіг add_feat_cols (legacy), дістаємо з самого CT.
        ct: ColumnTransformer = pipeline.steps[0][1]
        if not add_feat_cols:
            add_feat_cols = _extract_feature_cols_from_ct(ct)
        # text-колонку шукаємо у transformers_ — щоб не покладатись на
        # хардкод "text_processed".
        text_col = "text_processed"
        for _n, _t, c in getattr(ct, "transformers_", []):
            if isinstance(c, str):
                text_col = c
                break
        row: dict = {text_col: processed_text}
        for col in add_feat_cols:
            row[col] = 0.0
        return pd.DataFrame([row]), mode

    # Mode C — caller має сам повернути 400; повертаємо None як сигнал
    return None, mode


# ── DistilBERT lazy-load helper ─────────────────────────────────────────
# Спільний для /predict_distilbert та /explain_distilbert. Повертає
# (model, tokenizer, error_tuple). На успіх error_tuple == None; на провал —
# готовий (jsonify-dict, http_code) для прямого return з view-функції.

def _ensure_distilbert_loaded(model_path):
    """Lazy-load DistilBERT із .pkl bundle або HF директорії."""
    import torch

    model, tokenizer = get_distilbert_state()
    if model is not None:
        return model, tokenizer, None

    if not model_path:
        return None, None, (jsonify({
            "error": "model_not_loaded",
            "message": "No DistilBERT model in memory and no model_path provided",
            "hint": "Train a model first or provide model_path in payload",
        }), 400)

    import joblib
    from pathlib import Path as _Path

    try:
        if model_path.endswith(".pkl"):
            if not os.path.exists(model_path):
                return None, None, (jsonify({
                    "error": "pkl_not_found",
                    "model_path": model_path,
                    "message": "Bundle .pkl file not found on disk",
                }), 404)
            bundle = joblib.load(model_path)
            if not isinstance(bundle, dict) or bundle.get("type") != "distilbert":
                return None, None, (jsonify({
                    "error": "wrong_model_type",
                    "expected": "distilbert",
                    "got": bundle.get("type") if isinstance(bundle, dict) else "non-dict",
                    "hint": "This .pkl is not a DistilBERT bundle",
                }), 400)
            model_dir = bundle.get("model_dir")
            if not model_dir:
                return None, None, (jsonify({
                    "error": "missing_model_dir",
                    "message": ".pkl bundle has no 'model_dir' field",
                }), 500)
        else:
            model_dir = model_path

        if not os.path.isdir(model_dir):
            return None, None, (jsonify({
                "error": "model_dir_not_found",
                "model_dir": model_dir,
                "hint": (
                    "Directory was on local Colab disk and lost on runtime "
                    "restart. Re-train the model or save to Google Drive."
                ),
            }), 404)

        required_any = ["model.safetensors", "pytorch_model.bin"]
        if not any((_Path(model_dir) / f).exists() for f in required_any):
            return None, None, (jsonify({
                "error": "incomplete_model_dir",
                "model_dir": model_dir,
                "message": f"None of {required_any} found in directory",
            }), 500)

        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
        log.info(f"Lazy loading DistilBERT from directory: {model_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        set_distilbert_state(model, tokenizer)
        log.info("DistilBERT loaded and cached in global state")
        return model, tokenizer, None
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log.error(f"DistilBERT lazy load failed: {e}\n{tb}")
        return None, None, (jsonify({
            "error": "model_load_failed",
            "model_path": model_path,
            "message": str(e),
            "traceback": tb.splitlines()[-10:],
        }), 500)


# Уніфіковані preprocessing параметри для ВСІХ article-level моделей.
# Гарантують однаковий test set між NB / DistilBERT / GNN — обов'язково для
# ensemble compatibility (soft-align зробить fallback intersect, але кращий
# підхід — однакові тренувальні splits).
STANDARD_PREPROCESSING = {
    "test_ratio": 0.15,
    "val_ratio": 0.15,
    "min_text_length": 30,
    "seed": 42,
    "require_tweets": False,
}


# legacy `train_nb_aggregated` приймає stylistic і rhetorical окремо;
# після об'єднання у STYLISTIC_FEATURES розділяємо їх локально для роутера.
_STYLISTIC_PURE = {"caps_ratio", "ttr", "repetition_score", "avg_word_length"}
_RHETORICAL_PURE = {"clickbait_score", "authority_refs", "pronoun_ratio", "question_count"}


# ── Async jobs registry ───────────────────────────────────────────────────

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def register_routes(app: Flask):
    """Register all endpoints on Flask app."""

    # ── Health ────────────────────────────────────────────────────────────
    @app.route("/health", methods=["GET"])
    def health():
        import glob
        import torch

        from ml_server.config import MODELS_ROOT

        model, _tokenizer = get_distilbert_state()
        cuda_ok = torch.cuda.is_available()
        models_root_exists = os.path.isdir(MODELS_ROOT)

        available_distilbert: list[str] = []
        available_pkl: list[str] = []
        if models_root_exists:
            # Каталоги Hugging Face: MODELS_ROOT/user_*/distilbert_*
            available_distilbert = sorted(
                glob.glob(os.path.join(MODELS_ROOT, "user_*", "distilbert_*"))
            )[:20]
            # .pkl bundles (саме на них вказує ModelRecord.model_path)
            available_pkl = sorted(
                glob.glob(os.path.join(MODELS_ROOT, "user_*", "model_*.pkl"))
            )[:20]

        return jsonify({
            "status": "ok",
            "service": "ml_server",
            "distilbert_in_memory": model is not None,
            "cuda_available": cuda_ok,
            "device": "cuda" if cuda_ok else "cpu",
            "models_root": str(MODELS_ROOT),
            "models_root_exists": models_root_exists,
            "available_distilbert_dirs": available_distilbert,
            "available_pkl_bundles": available_pkl,
        })

    # ── Dataset management ────────────────────────────────────────────────
    @app.route("/list_datasets", methods=["GET"])
    def _list_datasets():
        return list_datasets()

    @app.route("/dataset_status", methods=["GET"])
    def _dataset_status():
        return dataset_status()

    @app.route("/list_splits", methods=["GET"])
    def _list_splits():
        return list_splits()

    @app.route("/upload_chunk", methods=["POST"])
    def _upload_chunk():
        return handle_upload_chunk()

    @app.route("/upload_finalize", methods=["POST"])
    def _upload_finalize():
        return handle_upload_finalize()

    # ── Training (sync) ───────────────────────────────────────────────────
    @app.route("/run_training", methods=["POST"])
    def run_training():
        payload = request.json or {}
        if not payload:
            return jsonify({"error": "JSON body required"}), 400
        try:
            result = _run_training_impl(payload)
            return jsonify(result)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except FileNotFoundError as e:
            log.error(f"Dataset not found: {e}")
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log.error(f"Training failed: {e}\n{tb}")
            return jsonify({"error": str(e), "traceback": tb}), 500

    # ── Training (async) ──────────────────────────────────────────────────
    @app.route("/run_training_async", methods=["POST"])
    def run_training_async():
        payload = request.json or {}
        if not payload:
            return jsonify({"error": "JSON body required"}), 400

        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {
                "status": "pending",
                "progress": "queued",
                "started_at": datetime.utcnow().isoformat(),
                "payload_summary": {
                    "model_type": payload.get("model_type"),
                    "dataset_id": payload.get("dataset_id"),
                    "experiment_id": payload.get("experiment_id"),
                },
            }

        def _bg():
            try:
                _set_job(job_id, status="running")

                def _progress(p):
                    _set_job(job_id, progress=p)

                payload["_progress_callback"] = _progress
                result = _run_training_impl(payload)
                _set_job(
                    job_id, status="done", result=result,
                    finished_at=datetime.utcnow().isoformat(),
                )
            except Exception as e:
                import traceback
                _set_job(
                    job_id, status="failed", error=str(e),
                    traceback=traceback.format_exc(),
                    finished_at=datetime.utcnow().isoformat(),
                )

        threading.Thread(target=_bg, daemon=True).start()
        log.info(f"[job {job_id[:8]}] Started async training")
        return jsonify({"job_id": job_id, "status": "pending"}), 202

    @app.route("/training_status/<job_id>", methods=["GET"])
    def training_status(job_id):
        with _jobs_lock:
            job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        response = {
            "job_id": job_id,
            "status": job["status"],
            "progress": job.get("progress"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
        }
        if job["status"] == "done":
            response["result"] = job.get("result")
        elif job["status"] == "failed":
            response["error"] = job.get("error")
            if "traceback" in job:
                response["traceback"] = job["traceback"]
        return jsonify(response)

    @app.route("/list_jobs", methods=["GET"])
    def list_jobs():
        with _jobs_lock:
            summary = {
                jid: {
                    "status": j["status"],
                    "progress": j.get("progress"),
                    "started_at": j.get("started_at"),
                    "finished_at": j.get("finished_at"),
                }
                for jid, j in _jobs.items()
            }
        return jsonify(summary)

    # ── Predictions ───────────────────────────────────────────────────────
    @app.route("/predict_distilbert", methods=["POST"])
    @app.route("/predict_deberta", methods=["POST"])  # backward compat
    def predict_distilbert():
        import torch

        data = request.json or {}
        text = data.get("text", "") or data.get("article_text", "")
        model_path = data.get("model_path")

        # ── 1. Empty-text перевірка окремо від model-loaded ─────────────
        # Раніше обидві умови повертали однакову помилку — це маскувало
        # реальну причину 400 (FE не міг розрізнити "extractor повернув ''"
        # від "модель не завантажена після Colab restart").
        if not text or not text.strip():
            return jsonify({
                "error": "empty_text",
                "message": "Text is empty or whitespace-only",
                "hint": "Check that extracted claim is not empty before sending",
            }), 400

        # ── 2. Lazy load (винесено у _ensure_distilbert_loaded — reused
        # також у /explain_distilbert) ───────────────────────────────────
        model, tokenizer, err = _ensure_distilbert_loaded(model_path)
        if err is not None:
            return err

        light_opts = {"removeUrls": True, "removeMentions": True, "cleaning": True}
        processed = preprocess_text(text, light_opts)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(
            processed, return_tensors="pt", truncation=True,
            padding="max_length", max_length=256,
        ).to(device)

        model.eval()
        with torch.no_grad():
            logits = model(**inputs).logits
            proba = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        pred = int(proba.argmax())
        return jsonify({
            "label": "FAKE" if pred == 1 else "REAL",
            "confidence": float(proba[pred]),
            "proba_fake": float(proba[1]),
            "proba_real": float(proba[0]),
        })

    @app.route("/explain_distilbert", methods=["POST"])
    def explain_distilbert_route():
        """Local explanation для DistilBERT через Layer Integrated Gradients.

        Payload: {text, model_path?, n_steps?=30, max_length?=256}.
        Reuses _ensure_distilbert_loaded → ті самі error codes що /predict_distilbert.
        """
        data = request.json or {}
        text = data.get("text", "") or ""
        model_path = data.get("model_path")
        n_steps = int(data.get("n_steps", 30))
        max_length = int(data.get("max_length", 256))

        if not text.strip():
            return jsonify({
                "error": "empty_text",
                "message": "Text is empty or whitespace-only",
            }), 400

        model, tokenizer, err = _ensure_distilbert_loaded(model_path)
        if err is not None:
            return err

        try:
            from ml_server.explainer_distilbert import explain_distilbert
        except ImportError as e:
            return jsonify({
                "error": "captum_unavailable",
                "message": f"captum not installed: {e}",
                "hint": "Run in Colab cell: !pip install captum --quiet",
            }), 500

        try:
            result = explain_distilbert(
                text, model, tokenizer,
                max_length=max_length, n_steps=n_steps,
            )
            return jsonify(result)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log.error(f"IG explain failed: {e}\n{tb}")
            return jsonify({
                "error": "explain_failed",
                "message": str(e),
                "traceback": tb.splitlines()[-10:],
            }), 500

    @app.route("/predict_nb", methods=["POST"])
    def predict_nb():
        """Inference для NB bundle (joblib pickle з `pipeline`).

        Payload: {text, model_path}. model_path вказує на .pkl bundle
        створений `nb_trainer.train_nb` ({type:"nb", pipeline, pipeline_type,
        preprocessing, ...}). Кеш по model_path (sklearn pipelines дешеві,
        але joblib.load з Drive — ~1-2s).

        Без model_path → 400: NB лежить на Drive, server не має fallback
        без явного шляху.
        """
        import joblib

        from ml_server.utils import preprocess_text

        data = request.json or {}
        text = data.get("text", "") or ""
        model_path = data.get("model_path")

        if not text.strip():
            return jsonify({
                "error": "empty_text",
                "message": "Text is empty or whitespace-only",
            }), 400
        if not model_path:
            return jsonify({
                "error": "model_path_required",
                "message": "NB inference потребує model_path до .pkl bundle",
            }), 400
        if not os.path.exists(model_path):
            return jsonify({
                "error": "pkl_not_found",
                "model_path": model_path,
            }), 404

        try:
            cached = _NB_PIPELINE_CACHE.get(model_path)
            if cached is None:
                bundle = joblib.load(model_path)
                if not isinstance(bundle, dict) or bundle.get("type") != "nb":
                    return jsonify({
                        "error": "wrong_model_type",
                        "expected": "nb",
                        "got": bundle.get("type") if isinstance(bundle, dict) else "non-dict",
                    }), 400
                pipeline = bundle.get("pipeline")
                if pipeline is None:
                    return jsonify({
                        "error": "missing_pipeline",
                        "message": "NB bundle has no 'pipeline' field",
                    }), 500
                preprocessing = bundle.get("preprocessing") or {}
                pipeline_type = bundle.get("pipeline_type", "article")
                cached = {
                    "pipeline": pipeline,
                    "preprocessing": preprocessing,
                    "pipeline_type": pipeline_type,
                    # Bundle зберігає use_text / additional_features — це
                    # надійніший спосіб визначити Mode A/B/C ніж duck-typing
                    # на named_steps (Mode C step зветься "num_scaler",
                    # не "preprocessor", і його легко переплутати з Mode A).
                    "use_text": bool(bundle.get("use_text", True)),
                    "additional_features": list(bundle.get("additional_features") or []),
                }
                if len(_NB_PIPELINE_CACHE) >= 16:
                    _NB_PIPELINE_CACHE.pop(next(iter(_NB_PIPELINE_CACHE)))
                _NB_PIPELINE_CACHE[model_path] = cached
                log.info(f"Lazy loaded NB bundle: {model_path}")

            pipeline = cached["pipeline"]
            processed = preprocess_text(text, cached["preprocessing"]) or text

            # Aggregated NB pipeline хоче DataFrame з фічами — для one-shot
            # inference без cascade-features це не підтримуємо тут.
            if cached["pipeline_type"] == "aggregated":
                return jsonify({
                    "error": "aggregated_unsupported",
                    "message": (
                        "Aggregated NB потребує emotional/social фічі — "
                        "вони не доступні для single-text inference. "
                        "Викликайте /predict_deberta з aggregated pipeline."
                    ),
                }), 400

            # Article-level NB Mode визначається з фактичної структури
            # pipeline.steps[0] (метадані bundle ненадійні у legacy моделях):
            #   Mode A — Vectorizer first → [text]
            #   Mode B — ColumnTransformer first → DataFrame з text + feat=0
            #   Mode C — Scaler first → single-text inference неможливий
            X, mode = _nb_build_input(
                pipeline, processed, cached["additional_features"]
            )
            if mode == "C" or X is None:
                return jsonify({
                    "error": "features_required",
                    "message": (
                        "Ця NB модель — features-only (pipeline починається "
                        "зі scaler-step, без text-vectorizer). Single-text "
                        "inference неможливий."
                    ),
                    "first_step_type": type(pipeline.steps[0][1]).__name__,
                    "additional_features": cached["additional_features"],
                }), 400
            log.info(f"NB predict mode={mode} model={model_path}")

            try:
                proba = pipeline.predict_proba(X)[0]
                fake_idx = (
                    list(pipeline.classes_).index(1)
                    if 1 in pipeline.classes_ else 1
                )
                prob = float(proba[fake_idx])
            except AttributeError:
                # decision_function fallback (e.g. LinearSVC у пайплайні)
                import math
                score = float(pipeline.decision_function(X)[0])
                prob = 1.0 / (1.0 + math.exp(-score))

            label = "FAKE" if prob > 0.5 else "REAL"
            return jsonify({
                "label": label,
                "confidence": abs(prob - 0.5) * 2,
                "probability": prob,
            })
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log.error(f"NB predict failed: {e}\n{tb}")
            return jsonify({
                "error": "predict_failed",
                "message": str(e),
                "traceback": tb.splitlines()[-10:],
            }), 500

    @app.route("/explain_gnn", methods=["POST"])
    def explain_gnn_route():
        """GNNExplainer для article-level classification (GIN/SAGE).

        Payload — той самий, що для /predict_gnn: article_text, tweets[],
        retweets[], replies[], model_path. Опціонально: epochs, top_k_nodes,
        top_k_edges.

        Cached 5хв за (model_path, hash payload) — GNNExplainer повільний
        (5-15s GPU, до 60s CPU).
        """
        import hashlib
        import json as _json
        import time
        import joblib
        import torch

        from ml_server.gnn_models import build_gnn_model
        from ml_server.graph_builder import build_inference_graph

        data = request.json or {}
        article_text = data.get("article_text", "") or ""
        tweets_input = data.get("tweets", []) or []
        retweets_input = data.get("retweets", []) or []
        replies_input = data.get("replies", []) or []
        model_path = data.get("model_path")
        epochs = int(data.get("epochs", 200))
        top_k_nodes = int(data.get("top_k_nodes", 10))
        top_k_edges = int(data.get("top_k_edges", 15))
        target_class = data.get("target_class")
        target_class = int(target_class) if target_class is not None else None

        if not article_text.strip():
            return jsonify({"error": "empty_text", "message": "article_text required"}), 400
        if not model_path or not os.path.exists(model_path):
            return jsonify({
                "error": "model_not_found",
                "model_path": model_path,
            }), 404

        # ── Cache lookup ──────────────────────────────────────────────
        payload_canonical = _json.dumps({
            "article_text": article_text,
            "tweets": tweets_input,
            "retweets": retweets_input,
            "replies": replies_input,
            "epochs": epochs,
            "top_k_nodes": top_k_nodes,
            "top_k_edges": top_k_edges,
            "target_class": target_class,
        }, sort_keys=True, ensure_ascii=False)
        cache_key = (
            model_path,
            hashlib.sha1(payload_canonical.encode("utf-8")).hexdigest(),
        )
        cached = _GNN_EXPLAIN_CACHE.get(cache_key)
        if cached is not None and (time.time() - cached["t"]) < _GNN_EXPLAIN_TTL:
            log.info("explain_gnn cache hit")
            return jsonify({**cached["result"], "cached": True})

        # ── Backward-compat normalization для replies (як у /predict_gnn) ──
        normalized_replies = []
        for rp in replies_input:
            if "parent_tweet_idx" in rp or "parent_reply_idx" in rp:
                normalized_replies.append(rp)
                continue
            parent_idx = rp.get("parent_idx")
            parent_type = rp.get("parent_type", "tweet")
            new_rp = {"text": rp.get("text", "")}
            if parent_idx is not None:
                if parent_type == "reply":
                    new_rp["parent_reply_idx"] = parent_idx
                else:
                    new_rp["parent_tweet_idx"] = parent_idx
            normalized_replies.append(new_rp)

        try:
            meta = joblib.load(model_path)
            if meta.get("type") != "gnn":
                return jsonify({
                    "error": "wrong_model_type",
                    "expected": "gnn",
                    "got": meta.get("type"),
                }), 400

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = build_gnn_model(
                architecture=meta["architecture"],
                in_dim=meta["in_dim"],
                hidden_dim=meta["hidden_dim"],
                dropout=meta["dropout"],
                num_layers=meta.get("num_layers", 3),
                pooling=meta.get("pooling"),
                aggregator=meta.get("aggregator", "mean"),
            ).to(device)
            model.load_state_dict(torch.load(
                meta["best_model_path"], map_location=device, weights_only=True
            ))
            model.eval()

            graph_data = build_inference_graph(
                article_text=article_text,
                tweets_input=tweets_input,
                retweets_input=retweets_input,
                replies_input=normalized_replies,
            ).to(device)

            from ml_server.explainer_gnn import (
                build_node_metadata,
                explain_gnn,
            )
            node_metadata = build_node_metadata(
                article_text=article_text,
                tweets_input=tweets_input,
                retweets_input=retweets_input,
                replies_input=normalized_replies,
            )
            result = explain_gnn(
                graph_data, model,
                target_class=target_class,
                top_k_nodes=top_k_nodes,
                top_k_edges=top_k_edges,
                epochs=epochs,
                node_metadata=node_metadata,
            )
            result["architecture"] = meta["architecture"]

            # Cache (з простим LRU evict коли > 16 записів)
            if len(_GNN_EXPLAIN_CACHE) >= 16:
                _GNN_EXPLAIN_CACHE.pop(next(iter(_GNN_EXPLAIN_CACHE)))
            _GNN_EXPLAIN_CACHE[cache_key] = {"t": time.time(), "result": result}

            return jsonify({**result, "cached": False})
        except ImportError as e:
            return jsonify({
                "error": "torch_geometric_unavailable",
                "message": f"torch_geometric / Explain API not available: {e}",
                "hint": "Run in Colab: !pip install torch_geometric --quiet",
            }), 500
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log.error(f"GNN explain failed: {e}\n{tb}")
            return jsonify({
                "error": "explain_failed",
                "message": str(e),
                "traceback": tb.splitlines()[-10:],
            }), 500

    @app.route("/predict_gnn", methods=["POST"])
    def predict_gnn():
        import joblib
        import torch
        import torch.nn.functional as F

        from ml_server.gnn_models import build_gnn_model
        from ml_server.graph_builder import build_inference_graph

        data = request.json or {}
        article_text = data.get("article_text", "")
        tweets_input = data.get("tweets", [])
        retweets_input = data.get("retweets", [])
        replies_input = data.get("replies", [])
        model_path = data.get("model_path")

        if not article_text:
            return jsonify({"error": "article_text required"}), 400
        if not model_path or not os.path.exists(model_path):
            return jsonify({"error": f"Model file not found: {model_path}"}), 400

        # Backward-compat: старі клієнти передають {parent_idx, parent_type}.
        # Нова функція очікує {parent_tweet_idx | parent_reply_idx}.
        normalized_replies = []
        for rp in replies_input:
            if "parent_tweet_idx" in rp or "parent_reply_idx" in rp:
                normalized_replies.append(rp)
                continue
            parent_idx = rp.get("parent_idx")
            parent_type = rp.get("parent_type", "tweet")
            new_rp = {"text": rp.get("text", "")}
            if parent_idx is not None:
                if parent_type == "reply":
                    new_rp["parent_reply_idx"] = parent_idx
                else:
                    new_rp["parent_tweet_idx"] = parent_idx
            normalized_replies.append(new_rp)

        try:
            meta = joblib.load(model_path)
            if meta.get("type") != "gnn":
                return jsonify({
                    "error": f"Model is not GNN: type={meta.get('type')}"
                }), 400

            architecture = meta["architecture"]
            best_model_path = meta["best_model_path"]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = build_gnn_model(
                architecture=architecture,
                in_dim=meta["in_dim"],
                hidden_dim=meta["hidden_dim"],
                dropout=meta["dropout"],
                num_layers=meta.get("num_layers", 3),
                pooling=meta.get("pooling"),
                aggregator=meta.get("aggregator", "mean"),
            ).to(device)
            model.load_state_dict(torch.load(
                best_model_path, map_location=device, weights_only=True
            ))
            model.eval()

            graph_data = build_inference_graph(
                article_text=article_text,
                tweets_input=tweets_input,
                retweets_input=retweets_input,
                replies_input=normalized_replies,
            ).to(device)

            with torch.no_grad():
                logits = model(graph_data)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()

            pred = int(probs.argmax())
            return jsonify({
                "label": "FAKE" if pred == 1 else "REAL",
                "confidence": float(probs[pred]),
                "proba_fake": float(probs[1]),
                "proba_real": float(probs[0]),
                "graph_size": int(graph_data.num_nodes),
                "graph_edges": int(graph_data.edge_index.shape[1] // 2),
                "architecture": architecture,
            })
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log.error(f"GNN prediction failed: {e}\n{tb}")
            return jsonify({"error": str(e), "traceback": tb}), 500

    @app.route("/embedding_cache/<dataset_id>", methods=["GET"])
    def get_embedding_cache_route(dataset_id):
        """Cache info для dataset (overall + per-category)."""
        from ml_server.embedding_cache import get_cache_status
        status = get_cache_status(dataset_id)
        # Backward-compat поля для старого FastAPI proxy
        status["total_size_mb"] = status.get("total_size_mb", 0)
        return jsonify(status)

    @app.route("/embedding_cache/<dataset_id>/status", methods=["GET"])
    def cache_status_route(dataset_id):
        """Per-category cache status."""
        from ml_server.embedding_cache import get_cache_status
        return jsonify(get_cache_status(dataset_id))

    @app.route("/embedding_cache/<dataset_id>", methods=["DELETE"])
    def invalidate_embedding_cache_route(dataset_id):
        """Видалити весь кеш для dataset."""
        from ml_server.embedding_cache import invalidate_all
        success = invalidate_all(dataset_id)
        return jsonify({"invalidated": success})

    @app.route("/embedding_cache/<dataset_id>/<category>", methods=["DELETE"])
    def invalidate_embedding_category_route(dataset_id, category):
        """Очистити одну категорію (articles/tweets/retweets/replies)."""
        from ml_server.embedding_cache import invalidate_category
        if category not in ("articles", "tweets", "retweets", "replies"):
            return jsonify({"error": "Invalid category"}), 400
        success = invalidate_category(dataset_id, category)
        return jsonify({"invalidated": success, "category": category})

    return app


# ── /run_training implementation ──────────────────────────────────────────

def _run_training_impl(payload: dict) -> dict:
    """Pure shared logic for sync and async training.

    Returns: dict із результатами тренування (метрики, шлях моделі, тощо).
    Raises:
        ValueError           — для bad-request випадків (HTTP 400 у sync wrapper).
        FileNotFoundError    — коли датасет не знайдено (HTTP 404).
        Exception            — будь-яка інша помилка тренування (HTTP 500).

    NB: НЕ викликати ``jsonify`` тут — у async-thread немає Flask app context.
    Sync і async ендпоїнти обертають результат / ловлять exceptions самі.
    """
    user_id = payload.get("user_id", "unknown")
    experiment_id = payload.get("experiment_id", "default")
    model_type = payload.get("model_type", "nb").lower()
    preprocessing = payload.get("preprocessing", {}) or {}
    model_params = payload.get("model_params", {}) or {}
    data_params = payload.get("data_params", {}) or {}
    dataset_id = payload.get("dataset_id")
    dataset_name = payload.get("dataset_name")
    progress_callback = payload.get("_progress_callback")

    if dataset_id is None and not dataset_name:
        raise ValueError("dataset_id або dataset_name потрібно")

    splits_subdir = data_params.get("splits_subdir")  # може бути None

    log.info(
        f"user={user_id}, exp={experiment_id}, model={model_type}, "
        f"dataset_id={dataset_id}, splits_subdir={splits_subdir}"
    )

    tmpdir = None
    try:
        # ── Branch 1: NB — article-level (DEFAULT, apples-to-apples) ──
        if model_type == "nb":
            # Витягти additional features з payload
            additional = model_params.get("additional_features", {}) or {}
            mask = additional.get("mask", {}) or {}
            enabled_features = [k for k, v in mask.items() if v]

            # Якщо є social/graph features — потрібен full_data
            needs_social = any(f in SOCIAL_FEATURES for f in enabled_features)

            train_df, val_df, test_df, full_data, data_stats, tmpdir = (
                build_article_level_data(
                    dataset_id=dataset_id, dataset_name=dataset_name,
                    **STANDARD_PREPROCESSING,
                    require_social=needs_social,
                    splits_subdir=splits_subdir,
                )
            )
            # alpha=None у model_params → auto-tune. Якщо передано — фіксований.
            alpha_param = model_params.get("alpha")
            alpha_val = float(alpha_param) if alpha_param is not None else None

            result = train_nb(
                train_df, val_df, test_df,
                user_id=user_id, experiment_id=experiment_id,
                nb_variant=model_params.get("nb_variant", "complement"),
                vectorizer_type=model_params.get("vectorizer_type", "tfidf"),
                ngram_range=model_params.get("ngram_range", "1,2"),
                alpha=alpha_val,
                tfidf_max_features=int(model_params.get("tfidf_max_features", 50000)),
                preprocessing=preprocessing,
                additional_features=enabled_features,
                full_data=full_data,
                use_text=bool(model_params.get("use_text", True)),
            )
            result["data_stats"] = data_stats

        # ── Branch 1b: NB — aggregated (legacy, для ablation) ──
        elif model_type == "nb_aggregated":
            train_df, test_df, data_stats, tmpdir = build_aggregated_data(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                top_tweet_strategy=data_params.get("top_tweet_strategy", "popularity"),
                min_text_length=int(data_params.get("min_text_length", 30)),
                test_ratio=float(data_params.get("test_ratio", 0.20)),
                seed=int(data_params.get("seed", 42)),
                splits_subdir=splits_subdir,
            )
            additional = model_params.get("additional_features", {}) or {}
            mask = additional.get("mask", {}) or {}
            enabled = [k for k, v in mask.items() if v]
            emotional_features = [f for f in enabled if f in EMOTIONAL_FEATURES]
            stylistic_features = [f for f in enabled if f in _STYLISTIC_PURE]
            rhetorical_features = [f for f in enabled if f in _RHETORICAL_PURE]
            social_features = list(additional.get("social_extra", []) or [])

            result = train_nb_aggregated(
                train_df, test_df,
                user_id=user_id, experiment_id=experiment_id,
                emotional_features=emotional_features,
                stylistic_features=stylistic_features,
                rhetorical_features=rhetorical_features,
                social_features=social_features,
                use_text=bool(payload.get("use_text", True)),
                nb_variant=model_params.get("nb_variant", "complement"),
                vectorizer_type=model_params.get("vectorizer_type", "tfidf"),
                ngram_range=model_params.get("ngram_range", "1,2"),
                alpha=float(model_params.get("alpha", 1.0)),
                preprocessing=preprocessing,
            )
            result["data_stats"] = data_stats

        # ── Branch 2: DistilBERT — article-level ──
        elif model_type in ("distilbert", "deberta", "bert"):
            train_df, val_df, test_df, full_data, data_stats, tmpdir = (
                build_article_level_data(
                    dataset_id=dataset_id, dataset_name=dataset_name,
                    **STANDARD_PREPROCESSING,
                    splits_subdir=splits_subdir,
                )
            )
            result = train_distilbert_article_level(
                train_df, val_df, test_df,
                user_id=user_id, experiment_id=experiment_id,
                model_params=model_params,
                progress_callback=progress_callback,
                full_data=full_data,
            )
            result["data_stats"] = data_stats

        # ── Branch 3: GNN — article-level + графи ──
        elif model_type == "gnn":
            # Уніфіковані splits (require_tweets=False) → той самий test set
            # що й NB/DistilBERT. require_social=True потрібен лише для
            # завантаження full_data (tweets/retweets/replies); сам test_df
            # не фільтрується по наявності твітів — статті без них отримують
            # ізольований 1-вузловий граф (див. build_graph_for_article).
            train_df, val_df, test_df, full_data, data_stats, tmpdir = (
                build_article_level_data(
                    dataset_id=dataset_id, dataset_name=dataset_name,
                    **STANDARD_PREPROCESSING,
                    require_social=True,
                    splits_subdir=splits_subdir,
                )
            )
            result = train_gnn(
                train_df, val_df, test_df, full_data,
                user_id=user_id, experiment_id=experiment_id,
                model_params=model_params,
                progress_callback=progress_callback,
            )
            result["data_stats"] = data_stats

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        log.info(f"✓ Training complete: {result['metrics']}")
        return result

    finally:
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
            log.info(f"Cleaned {tmpdir}")
