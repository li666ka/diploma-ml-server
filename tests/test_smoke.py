"""Базовий smoke test — перевіряє що модулі імпортуються без помилок."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Перевірка що всі модулі імпортуються."""
    from ml_server import config
    from ml_server import utils
    from ml_server import data_loader
    from ml_server import encoder
    from ml_server import graph_builder
    from ml_server import gnn_models
    from ml_server import gnn_trainer
    from ml_server import distilbert_trainer
    from ml_server import nb_trainer
    from ml_server import aggregated_loader
    from ml_server import upload_handlers
    from ml_server import routes
    from ml_server import app
    print("✓ All modules imported")


def test_config():
    from ml_server.config import (
        DATASETS_ROOT, MODELS_ROOT, EMBED_DIM,
        AGGREGATED_SOCIAL_COLS,
    )
    assert EMBED_DIM == 384
    assert len(AGGREGATED_SOCIAL_COLS) == 8
    print(f"✓ Config OK (DATASETS_ROOT={DATASETS_ROOT})")


def test_app_factory():
    from ml_server.app import create_app
    app = create_app()
    rules = sorted(str(r) for r in app.url_map.iter_rules())
    expected = [
        "/health",
        "/list_datasets",
        "/dataset_status",
        "/upload_chunk",
        "/upload_finalize",
        "/run_training",
        "/run_training_async",
        "/training_status/<job_id>",
        "/list_jobs",
        "/predict_distilbert",
        "/predict_deberta",
        "/predict_gnn",
    ]
    for endpoint in expected:
        assert any(endpoint in r for r in rules), \
            f"Missing endpoint: {endpoint}"
    print(f"✓ App factory OK ({len(rules)} routes)")


if __name__ == "__main__":
    test_imports()
    test_config()
    test_app_factory()
    print("\n✅ All smoke tests passed")
