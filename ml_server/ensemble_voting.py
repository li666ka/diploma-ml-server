"""Voting strategies для ансамблів моделей.

Підтримує:
- hard      — більшість голосів (majority label)
- soft      — середнє ймовірностей FAKE
- weighted  — зважена сума ймовірностей
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def hard_vote(
    predictions: list[np.ndarray],
    tie_breaker: str = "fake",
) -> np.ndarray:
    """Hard voting: вибираємо мітку яку обрала більшість моделей.

    Args:
        predictions: list of [N] arrays з integer labels (0 або 1)
        tie_breaker: 'fake' — при tie 50/50 голосуємо FAKE (recall-oriented);
                     'real' — при tie 50/50 голосуємо REAL (precision-oriented).
    """
    if not predictions:
        raise ValueError("predictions list is empty")

    stacked = np.stack(predictions, axis=0)  # [n_models, N]
    n_models, n_samples = stacked.shape

    fake_votes = stacked.sum(axis=0)  # скільки моделей сказали FAKE
    threshold = n_models / 2

    if tie_breaker == "fake":
        result = (fake_votes >= threshold).astype(np.int64)
    else:
        result = (fake_votes > threshold).astype(np.int64)

    log.info(
        f"Hard vote: n_models={n_models}, n_samples={n_samples}, "
        f"tie_breaker={tie_breaker}, fake_rate={result.mean():.3f}"
    )
    return result


def soft_vote(
    probabilities: list[np.ndarray],
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft voting: середнє ймовірностей FAKE по всіх моделях."""
    if not probabilities:
        raise ValueError("probabilities list is empty")

    stacked = np.stack(probabilities, axis=0)  # [n_models, N]
    n_models, n_samples = stacked.shape
    avg_proba = stacked.mean(axis=0)
    predictions = (avg_proba >= threshold).astype(np.int64)

    log.info(
        f"Soft vote: n_models={n_models}, n_samples={n_samples}, "
        f"threshold={threshold}, fake_rate={predictions.mean():.3f}, "
        f"avg_proba_mean={avg_proba.mean():.3f}"
    )
    return predictions, avg_proba


def weighted_vote(
    probabilities: list[np.ndarray],
    weights: list[float],
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted voting: зважена сума ймовірностей (ваги нормалізуються до 1)."""
    if not probabilities:
        raise ValueError("probabilities list is empty")
    if len(probabilities) != len(weights):
        raise ValueError(
            f"Mismatched lengths: {len(probabilities)} probas vs {len(weights)} weights"
        )

    weights_arr = np.asarray(weights, dtype=np.float32)
    if (weights_arr <= 0).any():
        raise ValueError("All weights must be > 0")

    weights_norm = weights_arr / weights_arr.sum()
    stacked = np.stack(probabilities, axis=0)  # [n_models, N]
    weighted_proba = np.tensordot(weights_norm, stacked, axes=([0], [0]))
    predictions = (weighted_proba >= threshold).astype(np.int64)

    log.info(
        f"Weighted vote: n_models={len(probabilities)}, "
        f"weights={weights_norm.tolist()}, "
        f"fake_rate={predictions.mean():.3f}"
    )
    return predictions, weighted_proba


def _align_members_by_article_id(
    members_data: list[dict],
) -> tuple[list[dict], list[str]]:
    """Узгодити порядок article_ids між членами. Reference — перший член.

    Якщо у когось не вистачає article_id з reference — error.
    Member'ам що мають зайві article_ids — обрізаємо до reference перетину.
    """
    reference_ids = list(members_data[0]["article_ids"])

    aligned = [members_data[0]]
    for i, m in enumerate(members_data[1:], 1):
        member_ids = list(m["article_ids"])
        if member_ids == reference_ids:
            aligned.append(m)
            continue

        log.warning(
            f"Member {i} has different article_ids order. Re-aligning…"
        )
        member_map = {aid: idx for idx, aid in enumerate(member_ids)}
        new_indices: list[int] = []
        for ref_aid in reference_ids:
            if ref_aid not in member_map:
                raise ValueError(
                    f"Member {i} (model_id={m.get('model_record_id')}) "
                    f"missing article_id={ref_aid}. Members tested on "
                    "different test sets."
                )
            new_indices.append(member_map[ref_aid])

        new_indices_arr = np.asarray(new_indices, dtype=np.int64)
        aligned.append({
            **m,
            "article_ids": reference_ids,
            "y_true": np.asarray(m["y_true"])[new_indices_arr],
            "y_pred": np.asarray(m["y_pred"])[new_indices_arr],
            "y_proba_fake": np.asarray(m["y_proba_fake"])[new_indices_arr],
        })

    return aligned, reference_ids


def evaluate_ensemble(
    voting_type: str,
    members_data: list[dict],
    weights: Optional[dict] = None,
    threshold: float = 0.5,
) -> dict:
    """Об'єднана функція: predictions членів → voting → metrics.

    Args:
        voting_type: 'hard' | 'soft' | 'weighted'
        members_data: list of dicts (як з predictions_cache.load_predictions),
            кожен має включати 'model_record_id' (int).
        weights: dict {model_id_str: weight} — потрібен лише для 'weighted'.
        threshold: для soft/weighted (default 0.5).
    """
    from ml_server.utils import compute_metrics

    if not members_data:
        raise ValueError("No members data")

    # ── Validation 1: align by article_id (re-order якщо треба) ──
    aligned, reference_ids = _align_members_by_article_id(members_data)

    # ── Validation 2: y_true має співпадати ──
    reference_y_true = np.asarray(aligned[0]["y_true"])
    for i, m in enumerate(aligned[1:], 1):
        if not np.array_equal(np.asarray(m["y_true"]), reference_y_true):
            raise ValueError(
                f"Member {i} (model_id={m.get('model_record_id')}) has "
                "different y_true. Members tested on different ground truth."
            )

    # ── Voting ──
    voting_type = (voting_type or "").lower()
    if voting_type == "hard":
        preds_list = [np.asarray(m["y_pred"]) for m in aligned]
        y_pred = hard_vote(preds_list)
        y_proba = None

    elif voting_type == "soft":
        proba_list = [np.asarray(m["y_proba_fake"], dtype=np.float32) for m in aligned]
        y_pred, y_proba = soft_vote(proba_list, threshold=threshold)

    elif voting_type == "weighted":
        if not weights:
            raise ValueError("Weighted voting requires weights dict")
        proba_list: list[np.ndarray] = []
        weight_values: list[float] = []
        for m in aligned:
            mid_str = str(m["model_record_id"])
            if mid_str not in weights:
                raise ValueError(f"Missing weight for model_id={mid_str}")
            proba_list.append(np.asarray(m["y_proba_fake"], dtype=np.float32))
            weight_values.append(float(weights[mid_str]))
        y_pred, y_proba = weighted_vote(proba_list, weight_values, threshold=threshold)

    else:
        raise ValueError(f"Unknown voting_type: {voting_type}")

    metrics = compute_metrics(
        y_true=reference_y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    return {
        "predictions": y_pred,
        "probabilities": y_proba,
        "y_true": reference_y_true,
        "article_ids": reference_ids,
        "metrics": metrics,
    }
