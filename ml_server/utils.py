"""Logging setup, text preprocessing, metrics."""
import logging
import re
import time
from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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

def compute_metrics(
    y_true,
    y_pred,
    training_time: Optional[float] = None,
    y_proba: Optional[np.ndarray] = None,
) -> dict:
    """Єдине джерело істини для метрик бінарної класифікації фейкових новин.

    Конвенція класів:
        REAL = 0 (negative class)
        FAKE = 1 (positive class)

    Returns dict з полями:
        accuracy:         Загальна точність (для обох класів разом)
        precision:        Precision для FAKE класу (pos_label=1)
                          = TP / (TP + FP)
                          "З передбачених FAKE — скільки дійсно FAKE"
        recall:           Recall для FAKE класу (pos_label=1)
                          = TP / (TP + FN)
                          "Зі справжніх FAKE — скільки знайшли"
        f1_score:         F1 для FAKE класу (harmonic mean precision/recall)
        f1_macro:         Незважене середнє F1 для обох класів
                          (REAL і FAKE враховуються однаково)
        roc_auc:          Area Under ROC Curve (probabilistic metric)
        confusion_matrix: {tn, fp, fn, tp} де:
                          tn = True Negative  (REAL → REAL)
                          fp = False Positive (REAL → FAKE)
                          fn = False Negative (FAKE → REAL)
                          tp = True Positive  (FAKE → FAKE)

    Args:
        y_true: ground truth labels (0=REAL, 1=FAKE)
        y_pred: predicted labels (0=REAL, 1=FAKE)
        training_time: тривалість тренування у секундах (optional)
        y_proba: probabilities для FAKE класу (optional, для ROC-AUC)

    Example:
        >>> y_true = [0, 0, 1, 1, 1]
        >>> y_pred = [0, 1, 1, 1, 0]
        >>> m = compute_metrics(y_true, y_pred)
        >>> # accuracy = 3/5 = 0.6
        >>> # precision (FAKE) = 2 правильно FAKE / 3 передбачено FAKE = 0.667
        >>> # recall (FAKE) = 2 знайдено FAKE / 3 справжніх FAKE = 0.667
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # labels=[0, 1] фіксує порядок: 0=REAL (negative), 1=FAKE (positive).
    # sklearn повертає [[tn, fp], [fn, tp]] — UI/API чекає dict формат.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    # pos_label=1 → всі precision/recall/f1 стосуються FAKE класу.
    # Це конвенція: детекція FAKE — це positive class у нашій задачі.
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }

    if y_proba is not None:
        try:
            proba = np.asarray(y_proba)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                proba = proba[:, 1]
            if len(np.unique(y_true)) >= 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
            else:
                log.warning("ROC-AUC undefined: y_true has only one class")
                metrics["roc_auc"] = None
        except Exception as e:
            log.warning(f"ROC-AUC computation failed: {e}")
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

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
