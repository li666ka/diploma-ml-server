"""Logging setup, text preprocessing, metrics."""
import logging
import re
import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ── Logger ────────────────────────────────────────────────────────────────

def setup_logger(name: str = "ml_server", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


log = setup_logger()


# ── Text preprocessing ────────────────────────────────────────────────────

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")


def preprocess_text(text: str, opts: dict | None = None) -> str:
    """Lightweight text cleaning. opts:
       removeUrls, removeMentions, cleaning, lowercase
    """
    if text is None:
        return ""
    opts = opts or {}
    s = str(text)
    if opts.get("removeUrls"):
        s = URL_RE.sub("", s)
    if opts.get("removeMentions"):
        s = MENTION_RE.sub("", s)
    if opts.get("lowercase"):
        s = s.lower()
    if opts.get("cleaning"):
        s = WHITESPACE_RE.sub(" ", s).strip()
    return s


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, training_time: float | None = None) -> dict:
    """Стандартний набір метрик для бінарної класифікації."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # labels=[0, 1] фіксує порядок: 0=REAL (negative), 1=FAKE (positive).
    # sklearn повертає [[tn, fp], [fn, tp]] — UI/API чекає dict формат.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_real": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    if training_time is not None:
        metrics["training_time"] = round(training_time, 2)
    return metrics


# ── Timing decorator ─────────────────────────────────────────────────────

def timed(label: str = ""):
    """Decorator що логує час виконання функції."""
    def deco(fn):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            log.info(f"⏱  {label or fn.__name__} starting...")
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0
            log.info(f"✓ {label or fn.__name__} done in {elapsed:.2f}s")
            return result
        return wrapper
    return deco


# ── Misc ─────────────────────────────────────────────────────────────────

def create_download_url(filepath: str) -> str | None:
    """Build download URL for Drive files (placeholder, можна розширити)."""
    if not filepath:
        return None
    return f"file://{filepath}"
