"""GNN end-to-end trainer.

Pipeline:
  1. MiniLM encode для articles + tweets + retweets + replies
  2. Build PyG graphs для train/val/test (по article_id)
  3. Train GIN/SAGE з early stopping по val_f1_macro
  4. Test eval, save model
"""
import gc
import os
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ml_server.config import MODELS_ROOT
from ml_server.embedding_cache import encode_incrementally_memmap
from ml_server.gnn_models import build_gnn_model
from ml_server.graph_builder import build_all_graphs
from ml_server.utils import create_download_url, log


@torch.no_grad()
def evaluate_gnn(model, loader, device):
    """Evaluate GNN. Використовує єдиний compute_metrics() з utils."""
    from ml_server.utils import compute_metrics

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        labels = batch.y
        if labels.dim() > 1:
            labels = labels.squeeze(-1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    metrics = compute_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        y_proba=all_probs,
    )

    return metrics, all_preds, all_labels, all_probs


def train_gnn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_data: dict,
    user_id: int | str,
    experiment_id: str,
    model_params: Optional[dict] = None,
    progress_callback: Optional[callable] = None,
) -> dict:
    """End-to-end GNN training."""
    from torch_geometric.loader import DataLoader as PyGDataLoader

    if model_params is None:
        model_params = {}

    # ── Reproducibility ──
    seed = int(model_params.get("seed", 42))
    log.info(f"Setting random seed: {seed}")

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    architecture = model_params.get("architecture", "gin")
    epochs = int(model_params.get("epochs", 50))
    batch_size = int(model_params.get("batch_size", 64))
    lr = float(model_params.get("learning_rate", 1e-3))
    weight_decay = float(model_params.get("weight_decay", 1e-4))
    hidden_dim = int(model_params.get("hidden_dim", 128))
    dropout = float(model_params.get("dropout", 0.3))
    patience = int(model_params.get("patience", 10))
    num_layers = int(model_params.get("num_layers", 3))
    pooling = model_params.get("pooling")
    aggregator = model_params.get("aggregator", "mean")
    use_scheduler = bool(model_params.get("use_lr_scheduler", True))
    max_grad_norm = float(model_params.get("max_grad_norm", 1.0))

    log.info(
        f"GNN params: arch={architecture}, layers={num_layers}, pooling={pooling}, "
        f"aggregator={aggregator}, hidden={hidden_dim}, dropout={dropout}, "
        f"epochs={epochs}, batch={batch_size}, lr={lr}, patience={patience}, seed={seed}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    tweets = full_data["tweets"]
    retweets = full_data["retweets"]
    replies = full_data["replies"]

    # ── 1. Encode all texts (з persistent cache) ──
    articles_all = pd.concat([
        train_df[["article_id", "combined_text"]],
        val_df[["article_id", "combined_text"]],
        test_df[["article_id", "combined_text"]],
    ]).drop_duplicates(subset=["article_id"]).reset_index(drop=True)

    dataset_id = full_data.get("dataset_id", "unknown")
    dataset_folder = full_data.get("dataset_folder", "")
    force_recompute = bool(model_params.get("force_recompute_embeddings", False))

    embeddings = encode_incrementally_memmap(
        dataset_id=dataset_id,
        dataset_folder=dataset_folder,
        articles_df=articles_all,
        tweets_df=tweets,
        retweets_df=retweets,
        replies_df=replies,
        max_chars_articles=2000,
        max_chars_tweets=500,
        batch_size=64,
        force_recompute=force_recompute,
        progress_callback=progress_callback,
    )

    # MemmapLookup objects (disk-backed); duck-typed compat з dict[id, tensor]
    article_emb = embeddings["article_lookup"]
    tweet_emb = embeddings["tweet_lookup"]
    retweet_emb = embeddings["retweet_lookup"]
    reply_emb = embeddings["reply_lookup"]

    log.info(
        f"Memmap embeddings ready: "
        f"articles={len(article_emb) if article_emb else 0:,}, "
        f"tweets={len(tweet_emb) if tweet_emb else 0:,}, "
        f"retweets={len(retweet_emb) if retweet_emb else 0:,}, "
        f"replies={len(reply_emb) if reply_emb else 0:,}, "
        f"cache_hits={embeddings['cache_hits']}, "
        f"time={embeddings['encoding_time']:.1f}s"
    )

    gc.collect()

    # Replace None lookups with empty dicts — graph_builder uses `in`/`[]`
    # на цих об'єктах, тож None спричинить TypeError.
    if tweet_emb is None:
        tweet_emb = {}
    if retweet_emb is None:
        retweet_emb = {}
    if reply_emb is None:
        reply_emb = {}

    # ── 2. Build graphs ──
    if progress_callback:
        progress_callback("building_graphs")

    log.info("Building train/val/test graphs...")
    train_graphs = build_all_graphs(
        train_df, tweets, retweets, replies,
        article_emb, tweet_emb, retweet_emb, reply_emb,
        progress_callback=progress_callback,
    )
    val_graphs = build_all_graphs(
        val_df, tweets, retweets, replies,
        article_emb, tweet_emb, retweet_emb, reply_emb,
    )
    test_graphs = build_all_graphs(
        test_df, tweets, retweets, replies,
        article_emb, tweet_emb, retweet_emb, reply_emb,
    )

    log.info(
        f"  Train: {len(train_graphs):,}, "
        f"Val: {len(val_graphs):,}, "
        f"Test: {len(test_graphs):,}"
    )

    train_loader = PyGDataLoader(train_graphs, batch_size=batch_size,
                                 shuffle=True, num_workers=0)
    val_loader = PyGDataLoader(val_graphs, batch_size=batch_size,
                               shuffle=False, num_workers=0)
    test_loader = PyGDataLoader(test_graphs, batch_size=batch_size,
                                shuffle=False, num_workers=0)

    # ── 3. Model ──
    in_dim = train_graphs[0].x.shape[1]
    model = build_gnn_model(
        architecture=architecture,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
        pooling=pooling,
        aggregator=aggregator,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable params: {n_params:,}")

    # Class weights
    n_real = (train_df["label"] == 0).sum()
    n_fake = (train_df["label"] == 1).sum()
    weight = torch.tensor([
        (n_fake + n_real) / (2 * n_real),
        (n_fake + n_real) / (2 * n_fake),
    ], dtype=torch.float, device=device)
    log.info(f"Class weights: REAL={weight[0]:.3f}, FAKE={weight[1]:.3f}")
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        log.info("Using ReduceLROnPlateau scheduler")

    # ── 4. Training loop ──
    if progress_callback:
        progress_callback("training")

    save_dir = Path(MODELS_ROOT) / f"user_{user_id}" / f"gnn_{experiment_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / "best_model.pt"

    t0 = time.time()
    best_val_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)

            labels = batch.y
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs
        train_loss = total_loss / n_samples

        val_metrics, *_ = evaluate_gnn(model, val_loader, device)

        if scheduler is not None:
            scheduler.step(val_metrics["f1_macro"])

        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        improved = val_metrics["f1_macro"] > best_val_f1
        marker = "⭐" if improved else "  "
        val_auc = val_metrics.get("roc_auc")
        val_auc_str = f"{val_auc:.4f}" if val_auc is not None else "n/a"
        log.info(
            f"  {marker} epoch {epoch:3d}  loss={train_loss:.4f}  "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}  "
            f"val_f1_score={val_metrics['f1_score']:.4f}  "
            f"val_auc={val_auc_str}"
        )

        if improved:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"  Early stopping at epoch {epoch} (best={best_epoch})")
                break

        if progress_callback:
            progress_callback(f"epoch_{epoch}_of_{epochs}")

    elapsed = round(time.time() - t0, 2)

    # ── 5. Test eval з best model ──
    log.info(f"Loading best model (epoch {best_epoch})")
    model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
    test_metrics, test_preds, test_labels, test_probs = evaluate_gnn(
        model, test_loader, device
    )

    # test_metrics вже містить повний набір з compute_metrics()
    metrics = {
        **test_metrics,
        "training_time": elapsed,
        "best_epoch": best_epoch,
        "train_size": len(train_graphs),
        "val_size": len(val_graphs),
        "test_size": len(test_graphs),
    }

    log.info(
        f"Test metrics (FAKE class for P/R/F1): "
        f"acc={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1_score']:.4f}, "
        f"f1_macro={metrics['f1_macro']:.4f}, "
        f"auc={metrics['roc_auc']}"
    )

    # ── 6. Save bundle ──
    pkl_path = Path(MODELS_ROOT) / f"user_{user_id}" / f"model_{experiment_id}.pkl"
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "type": "gnn",
        "architecture": architecture,
        "model_dir": str(save_dir),
        "best_model_path": str(best_ckpt_path),

        # Architecture params (для reload)
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "num_layers": num_layers,
        "pooling": pooling,
        "aggregator": aggregator,

        # Training metadata
        "best_epoch": best_epoch,
        "seed": seed,
        "best_val_f1_macro": best_val_f1,

        # Trained on
        "trained_on_dataset_id": full_data.get("dataset_id"),
        "training_time_seconds": elapsed,
        "n_train_graphs": len(train_graphs),
        "n_val_graphs": len(val_graphs),
        "n_test_graphs": len(test_graphs),
    }, pkl_path)

    # Save predictions для подальшого використання в ансамблях
    try:
        from ml_server.predictions_cache import save_predictions

        # article_ids у тому ж порядку що test_graphs (test_loader без shuffle).
        # build_all_graphs може пропустити статті без embedding — беремо ids
        # з самих graphs, а не з test_df, щоб гарантувати alignment.
        article_ids = [str(getattr(g, "article_id", "")) for g in test_graphs]

        assert len(article_ids) == len(test_preds) == len(test_labels), (
            f"Mismatch: aids={len(article_ids)}, preds={len(test_preds)}, "
            f"labels={len(test_labels)}"
        )

        save_predictions(
            model_dir=Path(save_dir),
            article_ids=article_ids,
            y_true=test_labels,
            y_pred=test_preds,
            y_proba_fake=test_probs,
            metrics=metrics,
            model_type=architecture,
            splits_used=full_data.get("splits_subdir", "unknown"),
            dataset_id=full_data.get("dataset_id", "unknown"),
        )
    except Exception as e:
        log.warning(f"Failed to save predictions: {e}")

    return {
        "path": str(pkl_path),
        "metrics": metrics,
        "history": history,
        "download_url": create_download_url(str(pkl_path)),
    }
