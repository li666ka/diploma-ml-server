"""GNN end-to-end trainer.

Pipeline:
  1. MiniLM encode для articles + tweets + retweets + replies
  2. Build PyG graphs для train/val/test (по article_id)
  3. Train GIN/SAGE з early stopping по val_f1_macro
  4. Test eval, save model
"""
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
from ml_server.encoder import encode_dataframe_column
from ml_server.gnn_models import build_gnn_model
from ml_server.graph_builder import build_all_graphs
from ml_server.utils import create_download_url, log


@torch.no_grad()
def evaluate_gnn(model, loader, device):
    """Returns (metrics_dict, preds, labels, probs)."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(batch.y.squeeze().cpu())
        all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro")),
        "f1_fake": float(f1_score(all_labels, all_preds, pos_label=1)),
        "f1_real": float(f1_score(all_labels, all_preds, pos_label=0)),
        "roc_auc": float(roc_auc_score(all_labels, all_probs)),
    }
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

    architecture = model_params.get("architecture", "gin")
    epochs = int(model_params.get("epochs", 50))
    batch_size = int(model_params.get("batch_size", 64))
    lr = float(model_params.get("learning_rate", 1e-3))
    weight_decay = float(model_params.get("weight_decay", 1e-4))
    hidden_dim = int(model_params.get("hidden_dim", 128))
    dropout = float(model_params.get("dropout", 0.3))
    patience = int(model_params.get("patience", 10))

    log.info(
        f"GNN params: arch={architecture}, epochs={epochs}, batch={batch_size}, "
        f"lr={lr}, hidden={hidden_dim}, dropout={dropout}, patience={patience}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    tweets = full_data["tweets"]
    retweets = full_data["retweets"]
    replies = full_data["replies"]

    # ── 1. Encode all texts ──
    if progress_callback:
        progress_callback("encoding_articles")

    log.info("Encoding articles...")
    articles_all = pd.concat([
        train_df[["article_id", "combined_text"]],
        val_df[["article_id", "combined_text"]],
        test_df[["article_id", "combined_text"]],
    ]).drop_duplicates(subset=["article_id"]).reset_index(drop=True)

    article_emb = encode_dataframe_column(
        articles_all, "article_id", "combined_text",
        max_chars=2000, batch_size=128,
    )
    log.info(f"  {len(article_emb):,} article embeddings")

    if progress_callback:
        progress_callback("encoding_tweets")
    log.info("Encoding tweets...")
    tweet_emb = {}
    if len(tweets) > 0 and "tweet_text" in tweets.columns:
        tweet_emb = encode_dataframe_column(
            tweets, "tweet_id", "tweet_text",
            max_chars=500, batch_size=512,
        )
    log.info(f"  {len(tweet_emb):,} tweet embeddings")

    if progress_callback:
        progress_callback("encoding_retweets")
    log.info("Encoding retweets...")
    retweet_emb = {}
    if len(retweets) > 0 and "retweet_text" in retweets.columns:
        retweet_emb = encode_dataframe_column(
            retweets, "retweet_id", "retweet_text",
            max_chars=500, batch_size=512,
        )
    log.info(f"  {len(retweet_emb):,} retweet embeddings")

    if progress_callback:
        progress_callback("encoding_replies")
    log.info("Encoding replies...")
    reply_emb = {}
    if len(replies) > 0 and "reply_text" in replies.columns:
        reply_emb = encode_dataframe_column(
            replies, "reply_id", "reply_text",
            max_chars=500, batch_size=512,
        )
    log.info(f"  {len(reply_emb):,} reply embeddings")

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
    model = build_gnn_model(architecture, in_dim, hidden_dim, dropout).to(device)
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
            loss = criterion(logits, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs
        train_loss = total_loss / n_samples

        val_metrics, *_ = evaluate_gnn(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        improved = val_metrics["f1_macro"] > best_val_f1
        marker = "⭐" if improved else "  "
        log.info(
            f"  {marker} epoch {epoch:3d}  loss={train_loss:.4f}  "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}  "
            f"val_f1_fake={val_metrics['f1_fake']:.4f}  "
            f"val_auc={val_metrics['roc_auc']:.4f}"
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
    test_metrics, _, _, _ = evaluate_gnn(model, test_loader, device)

    metrics = {
        "accuracy": test_metrics["accuracy"],
        "f1_macro": test_metrics["f1_macro"],
        "f1_score": test_metrics["f1_fake"],  # для UI compat
        "f1_fake": test_metrics["f1_fake"],
        "f1_real": test_metrics["f1_real"],
        "roc_auc": test_metrics["roc_auc"],
        "training_time": elapsed,
        "best_epoch": best_epoch,
        "train_size": len(train_graphs),
        "val_size": len(val_graphs),
        "test_size": len(test_graphs),
    }

    log.info(
        f"Test: f1_macro={metrics['f1_macro']:.4f}, "
        f"f1_fake={metrics['f1_fake']:.4f}, auc={metrics['roc_auc']:.4f}"
    )

    # ── 6. Save bundle ──
    pkl_path = Path(MODELS_ROOT) / f"user_{user_id}" / f"model_{experiment_id}.pkl"
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "type": "gnn",
        "architecture": architecture,
        "model_dir": str(save_dir),
        "best_model_path": str(best_ckpt_path),
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "best_epoch": best_epoch,
    }, pkl_path)

    return {
        "path": str(pkl_path),
        "metrics": metrics,
        "history": history,
        "download_url": create_download_url(str(pkl_path)),
    }
