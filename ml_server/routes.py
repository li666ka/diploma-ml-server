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

        model, tokenizer = get_distilbert_state()

        # ── 2. Lazy load з model_path якщо global state втрачено ────────
        # Сценарій: Colab runtime restart → globals скинулись, але файли
        # моделі лежать у MODELS_ROOT/user_X/distilbert_<exp>/.
        # ModelRecord.model_path вказує на .pkl bundle, який містить
        # {"type":"distilbert", "model_dir": "...", "model_name": ..., "max_length": 256}.
        # AutoModel.from_pretrained() на .pkl падає ("Repo id must be in the form ..."),
        # тому спершу розпаковуємо bundle і отримуємо реальну HF-директорію.
        if model is None and model_path:
            import joblib
            from pathlib import Path as _Path

            try:
                if model_path.endswith(".pkl"):
                    if not os.path.exists(model_path):
                        return jsonify({
                            "error": "pkl_not_found",
                            "model_path": model_path,
                            "message": "Bundle .pkl file not found on disk",
                        }), 404
                    bundle = joblib.load(model_path)
                    if not isinstance(bundle, dict) or bundle.get("type") != "distilbert":
                        return jsonify({
                            "error": "wrong_model_type",
                            "expected": "distilbert",
                            "got": bundle.get("type") if isinstance(bundle, dict) else "non-dict",
                            "hint": "This .pkl is not a DistilBERT bundle",
                        }), 400
                    model_dir = bundle.get("model_dir")
                    if not model_dir:
                        return jsonify({
                            "error": "missing_model_dir",
                            "message": ".pkl bundle has no 'model_dir' field",
                        }), 500
                else:
                    # model_path вже вказує безпосередньо на HF-директорію.
                    model_dir = model_path

                if not os.path.isdir(model_dir):
                    return jsonify({
                        "error": "model_dir_not_found",
                        "model_dir": model_dir,
                        "hint": (
                            "Directory was on local Colab disk and lost on runtime "
                            "restart. Re-train the model or save to Google Drive."
                        ),
                    }), 404

                # Перевірити що HF weights присутні (інакше from_pretrained кине
                # незрозумілий KeyError всередині transformers).
                required_any = ["model.safetensors", "pytorch_model.bin"]
                if not any((_Path(model_dir) / f).exists() for f in required_any):
                    return jsonify({
                        "error": "incomplete_model_dir",
                        "model_dir": model_dir,
                        "message": f"None of {required_any} found in directory",
                    }), 500

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
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                log.error(f"DistilBERT lazy load failed: {e}\n{tb}")
                return jsonify({
                    "error": "model_load_failed",
                    "model_path": model_path,
                    "message": str(e),
                    "traceback": tb.splitlines()[-10:],
                }), 500

        if model is None:
            return jsonify({
                "error": "model_not_loaded",
                "message": (
                    "No DistilBERT model in memory and no model_path provided"
                ),
                "hint": "Train a model first or provide model_path in payload",
            }), 400

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
