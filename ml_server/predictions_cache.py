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
) -> Path:
    """
    Зберегти predictions у model_dir/predictions.json
    або model_dir/predictions_{model_record_id}.json (якщо id переданий).

    Args:
        model_dir: куди зберегти (директорія моделі)
        article_ids: ID статей у test set (порядок ВАЖЛИВИЙ — той самий що y_pred)
        y_true: ground truth labels
        y_pred: predicted labels
        y_proba_fake: probability що FAKE (для soft voting)
        metrics: dict з accuracy, precision, etc.
        model_type: 'nb' | 'distilbert' | 'gin' | 'sage' | 'llm'
        splits_used: 'splits_in_domain' | 'splits_cross_domain'
        dataset_id: для validation що ансамбль на тому ж dataset
        model_record_id: якщо переданий — файл називається
            predictions_{id}.json (унікальний для кожної моделі).

    Returns:
        Path до predictions.json
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

    # Validate lengths
    n = len(article_ids)
    if not (len(y_true) == n and len(y_pred) == n and len(y_proba_fake) == n):
        raise ValueError(
            f"Mismatched lengths: article_ids={n}, y_true={len(y_true)}, "
            f"y_pred={len(y_pred)}, y_proba={len(y_proba_fake)}"
        )

    # Build predictions list
    predictions = []
    for i in range(n):
        predictions.append({
            "article_id": str(article_ids[i]),
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
            "y_proba_fake": float(y_proba_fake[i]),
        })

    data = {
        "model_type": model_type,
        "splits_used": splits_used,
        "dataset_id": str(dataset_id),
        "test_size": n,
        "predictions": predictions,
        "metrics": metrics,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Atomic write через .tmp
    tmp_path = predictions_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(predictions_path)

    log.info(f"Saved predictions: {predictions_path} ({n} items)")
    return predictions_path


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
