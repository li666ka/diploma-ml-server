"""DistilBERT article-level fine-tuning."""
import os
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch

from ml_server.config import MODELS_ROOT
from ml_server.utils import compute_metrics, create_download_url, log, preprocess_text


# Globals для inference
_distilbert_model = None
_distilbert_tokenizer = None


def get_distilbert_state():
    """Returns (model, tokenizer). May be (None, None) if not trained yet."""
    return _distilbert_model, _distilbert_tokenizer


def train_distilbert_article_level(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_id: int | str,
    experiment_id: str,
    model_params: Optional[dict] = None,
    progress_callback: Optional[callable] = None,
) -> dict:
    """Fine-tune DistilBERT на article_text. БЕЗ aggregation."""
    global _distilbert_model, _distilbert_tokenizer

    if model_params is None:
        model_params = {}

    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    model_name = model_params.get("base_model", "distilbert-base-uncased")
    epochs = int(model_params.get("epochs", 3))
    batch_size = int(model_params.get("batch_size", 16))
    lr = float(model_params.get("learning_rate", 2e-5))
    max_len = int(model_params.get("max_length", 256))
    weight_decay = float(model_params.get("weight_decay", 0.01))
    warmup_ratio = float(model_params.get("warmup_ratio", 0.1))
    freeze_base = bool(model_params.get("freeze_base", True))

    log.info(
        f"DistilBERT params: model={model_name}, epochs={epochs}, "
        f"batch={batch_size}, lr={lr}, max_len={max_len}, freeze={freeze_base}"
    )

    light_opts = {"removeUrls": True, "removeMentions": True, "cleaning": True}

    df_train = train_df[["combined_text", "label"]].rename(
        columns={"combined_text": "text"}).copy()
    df_val = val_df[["combined_text", "label"]].rename(
        columns={"combined_text": "text"}).copy()
    df_test = test_df[["combined_text", "label"]].rename(
        columns={"combined_text": "text"}).copy()

    df_train["text"] = df_train["text"].apply(lambda t: preprocess_text(t, light_opts))
    df_val["text"] = df_val["text"].apply(lambda t: preprocess_text(t, light_opts))
    df_test["text"] = df_test["text"].apply(lambda t: preprocess_text(t, light_opts))

    df_train = df_train.dropna()
    df_val = df_val.dropna()
    df_test = df_test.dropna()

    df_train["label"] = df_train["label"].astype(int)
    df_val["label"] = df_val["label"].astype(int)
    df_test["label"] = df_test["label"].astype(int)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
    ).to(device)

    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze last 2 layers
        try:
            for layer in model.distilbert.transformer.layer[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True
        except AttributeError:
            try:
                for layer in model.bert.encoder.layer[-2:]:
                    for p in layer.parameters():
                        p.requires_grad = True
            except AttributeError:
                log.warning("Could not freeze — training all params")
                for p in model.parameters():
                    p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")

    # Tokenize
    train_dataset = Dataset.from_pandas(df_train.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(df_val.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(df_test.reset_index(drop=True))

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True,
            padding="max_length", max_length=max_len,
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    train_dataset.set_format("torch", columns=cols)
    val_dataset.set_format("torch", columns=cols)
    test_dataset.set_format("torch", columns=cols)

    save_dir = Path(MODELS_ROOT) / f"user_{user_id}" / f"distilbert_{experiment_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(save_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    def compute_metrics_hf(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
            "f1_fake": float(f1_score(labels, preds, pos_label=1)),
        }

    t0 = time.time()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_hf,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    elapsed = round(time.time() - t0, 2)

    # Final eval on TEST
    log.info("Final evaluation on test set...")
    test_predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(test_predictions.predictions, axis=-1)
    y_test = df_test["label"].values

    metrics = compute_metrics(y_test, y_pred, elapsed)
    metrics["train_size"] = len(df_train)
    metrics["val_size"] = len(df_val)
    metrics["test_size"] = len(df_test)

    # Save
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log.info(f"DistilBERT saved: {save_dir}")

    _distilbert_model = model
    _distilbert_tokenizer = tokenizer

    pkl_path = Path(MODELS_ROOT) / f"user_{user_id}" / f"model_{experiment_id}.pkl"
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "type": "distilbert",
        "model_dir": str(save_dir),
        "model_name": model_name,
        "max_length": max_len,
    }, pkl_path)

    return {
        "path": str(pkl_path),
        "metrics": metrics,
        "download_url": create_download_url(str(pkl_path)),
    }
