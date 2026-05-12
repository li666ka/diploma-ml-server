"""
Кешування predictions для подальшого використання в ансамблях.
Один раз predict — потім ансамблі без перерахунку.
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def save_predictions(
    model_dir: Path,
    article_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_fake: np.ndarray,
    metrics: dict,
    model_type: str,
    splits_used: str,
    dataset_id: int | str,
    model_record_id: int | None = None,
) -> dict:
    """
    Зберегти predictions у model_dir/predictions.json (на Drive) ТА повернути
    compact dict для збереження у БД як `predictions_json`.

    Returns:
        {"path": "/abs/path/predictions.json", "compact_json": {...}}
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_record_id:
        predictions_path = model_dir / f"predictions_{model_record_id}.json"
    else:
        predictions_path = model_dir / "predictions.json"

    # Конвертувати numpy → Python native types
    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()
    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.tolist()
    if isinstance(y_proba_fake, np.ndarray):
        y_proba_fake = y_proba_fake.tolist()

    n = len(article_ids)
    if not (len(y_true) == n and len(y_pred) == n and len(y_proba_fake) == n):
        raise ValueError(
            f"Mismatched lengths: article_ids={n}, y_true={len(y_true)}, "
            f"y_pred={len(y_pred)}, y_proba={len(y_proba_fake)}"
        )

    aid_list = [str(a) for a in article_ids]
    y_true_list = [int(v) for v in y_true]
    y_pred_list = [int(v) for v in y_pred]
    y_proba_list = [float(v) for v in y_proba_fake]
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Detailed per-row form для Drive backup (зручно для debug)
    predictions = [
        {
            "article_id": aid_list[i],
            "y_true": y_true_list[i],
            "y_pred": y_pred_list[i],
            "y_proba_fake": y_proba_list[i],
        }
        for i in range(n)
    ]
    data = {
        "model_type": model_type,
        "splits_used": splits_used,
        "dataset_id": str(dataset_id),
        "test_size": n,
        "predictions": predictions,
        "metrics": metrics,
        "created_at": created_at,
    }

    tmp_path = predictions_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(predictions_path)
    log.info(f"Saved predictions: {predictions_path} ({n} items)")

    # Compact form для БД — колоночні масиви, без metrics (metrics_json окремо).
    compact_json = {
        "article_ids": aid_list,
        "y_true": y_true_list,
        "y_pred": y_pred_list,
        "y_proba_fake": y_proba_list,
        "model_type": model_type,
        "splits_used": splits_used,
        "dataset_id": str(dataset_id),
        "test_size": n,
        "created_at": created_at,
    }

    return {"path": str(predictions_path), "compact_json": compact_json}


def load_predictions(predictions_path: str | Path) -> dict:
    """
    Завантажити predictions.json.

    Returns:
        {
            "article_ids": list[str],
            "y_true": np.ndarray,
            "y_pred": np.ndarray,
            "y_proba_fake": np.ndarray,
            "metrics": dict,
            "splits_used": str,
            "dataset_id": str,
            "model_type": str,
        }
    """
    predictions_path = Path(predictions_path)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")

    with open(predictions_path) as f:
        data = json.load(f)

    preds = data["predictions"]

    return {
        "article_ids": [p["article_id"] for p in preds],
        "y_true": np.array([p["y_true"] for p in preds], dtype=np.int64),
        "y_pred": np.array([p["y_pred"] for p in preds], dtype=np.int64),
        "y_proba_fake": np.array([p["y_proba_fake"] for p in preds], dtype=np.float32),
        "metrics": data.get("metrics", {}),
        "splits_used": data.get("splits_used", ""),
        "dataset_id": data.get("dataset_id", ""),
        "model_type": data.get("model_type", ""),
        "test_size": data.get("test_size", len(preds)),
    }


def predictions_exist(model_dir: str | Path) -> bool:
    """Check if predictions.json exists for a model."""
    return (Path(model_dir) / "predictions.json").exists()
