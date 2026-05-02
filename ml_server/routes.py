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
        return jsonify({"status": "ok", "service": "ml_server"})

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

        model, tokenizer = get_distilbert_state()
        if not text or model is None:
            return jsonify({"error": "DistilBERT model not loaded or empty text"}), 400

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
        import numpy as np
        import torch
        from torch_geometric.data import Batch, Data

        from ml_server.encoder import encode_texts_batch
        from ml_server.gnn_models import build_gnn_model

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

        try:
            meta = joblib.load(model_path)
            if meta.get("type") != "gnn":
                return jsonify({
                    "error": f"Model is not GNN: type={meta.get('type')}"
                }), 400

            architecture = meta["architecture"]
            in_dim = meta["in_dim"]
            hidden_dim = meta["hidden_dim"]
            dropout = meta["dropout"]
            best_model_path = meta["best_model_path"]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = build_gnn_model(
                architecture, in_dim, hidden_dim, dropout
            ).to(device)
            model.load_state_dict(torch.load(
                best_model_path, map_location=device, weights_only=True
            ))
            model.eval()

            # Encode all texts in single batch
            all_texts = [article_text]
            all_texts.extend([t.get("text", "") for t in tweets_input])
            all_texts.extend([rt.get("text", "") for rt in retweets_input])
            all_texts.extend([rp.get("text", "") for rp in replies_input])

            embs_np = encode_texts_batch(
                all_texts, batch_size=128, max_chars=2000, show_progress=False
            )
            embs = torch.from_numpy(embs_np)

            # Build edge_index
            edge_src, edge_dst = [], []
            article_idx = 0
            n_tweets = len(tweets_input)
            n_retweets = len(retweets_input)

            tweet_indices = list(range(1, 1 + n_tweets))
            for tw_idx in tweet_indices:
                edge_src.append(tw_idx)
                edge_dst.append(article_idx)

            retweet_indices = list(range(1 + n_tweets, 1 + n_tweets + n_retweets))
            for i, rt in enumerate(retweets_input):
                orig_idx = rt.get("original_tweet_idx", 0)
                if 0 <= orig_idx < n_tweets:
                    edge_src.append(retweet_indices[i])
                    edge_dst.append(tweet_indices[orig_idx])

            reply_start = 1 + n_tweets + n_retweets
            for i, rp in enumerate(replies_input):
                parent_idx = rp.get("parent_idx", 0)
                parent_type = rp.get("parent_type", "tweet")
                rp_node_idx = reply_start + i
                if parent_type == "tweet" and 0 <= parent_idx < n_tweets:
                    edge_src.append(rp_node_idx)
                    edge_dst.append(tweet_indices[parent_idx])
                elif parent_type == "reply" and 0 <= parent_idx < i:
                    edge_src.append(rp_node_idx)
                    edge_dst.append(reply_start + parent_idx)

            if edge_src:
                edge_index = torch.tensor(
                    [edge_src + edge_dst, edge_dst + edge_src],
                    dtype=torch.long,
                )
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            graph_data = Data(x=embs, edge_index=edge_index)
            batch = Batch.from_data_list([graph_data]).to(device)

            with torch.no_grad():
                import torch.nn.functional as F
                logits = model(batch)
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
                    test_ratio=float(data_params.get("test_ratio", 0.15)),
                    val_ratio=float(data_params.get("val_ratio", 0.15)),
                    min_text_length=int(data_params.get("min_text_length", 30)),
                    seed=int(data_params.get("seed", 42)),
                    require_tweets=False,
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
            train_df, val_df, test_df, _, data_stats, tmpdir = (
                build_article_level_data(
                    dataset_id=dataset_id, dataset_name=dataset_name,
                    test_ratio=float(data_params.get("test_ratio", 0.15)),
                    val_ratio=float(data_params.get("val_ratio", 0.15)),
                    min_text_length=int(data_params.get("min_text_length", 30)),
                    seed=int(data_params.get("seed", 42)),
                    require_tweets=False,
                    splits_subdir=splits_subdir,
                )
            )
            result = train_distilbert_article_level(
                train_df, val_df, test_df,
                user_id=user_id, experiment_id=experiment_id,
                model_params=model_params,
                progress_callback=progress_callback,
            )
            result["data_stats"] = data_stats

        # ── Branch 3: GNN — article-level + графи ──
        elif model_type == "gnn":
            train_df, val_df, test_df, full_data, data_stats, tmpdir = (
                build_article_level_data(
                    dataset_id=dataset_id, dataset_name=dataset_name,
                    test_ratio=float(data_params.get("test_ratio", 0.15)),
                    val_ratio=float(data_params.get("val_ratio", 0.15)),
                    min_text_length=int(data_params.get("min_text_length", 30)),
                    seed=int(data_params.get("seed", 42)),
                    require_tweets=True,
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
