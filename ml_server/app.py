"""Flask app factory + entrypoint for Colab/local."""
import glob
import os

from flask import Flask

from ml_server.config import FLASK_PORT, IS_COLAB, MODELS_ROOT, NGROK_AUTH_TOKEN
from ml_server.routes import register_routes
from ml_server.utils import log


def _log_environment() -> None:
    """Друкуємо що бачимо у файловій системі — щоб одразу зрозуміти,
    чому lazy-load не знаходить модель після Colab restart."""
    log.info("─" * 60)
    log.info(f"IS_COLAB              = {IS_COLAB}")
    log.info(f"MODELS_ROOT           = {MODELS_ROOT}")
    log.info(f"MODELS_ROOT exists    = {os.path.isdir(MODELS_ROOT)}")
    if IS_COLAB:
        drive_mounted = os.path.isdir("/content/drive/MyDrive")
        log.info(f"Drive mounted         = {drive_mounted}")
        if not drive_mounted:
            log.warning(
                "  Google Drive не примонтований. Запусти у Colab cell: "
                "from google.colab import drive; drive.mount('/content/drive')"
            )
    if os.path.isdir(MODELS_ROOT):
        hf_dirs = sorted(glob.glob(os.path.join(MODELS_ROOT, "user_*", "distilbert_*")))
        pkl_bundles = sorted(glob.glob(os.path.join(MODELS_ROOT, "user_*", "model_*.pkl")))
        log.info(f"DistilBERT dirs       = {len(hf_dirs)}")
        for d in hf_dirs[:10]:
            log.info(f"  • {d}")
        log.info(f".pkl bundles          = {len(pkl_bundles)}")
        for p in pkl_bundles[:10]:
            log.info(f"  • {p}")
    log.info("─" * 60)


def create_app() -> Flask:
    """Build Flask app with all routes registered."""
    app = Flask(__name__)
    register_routes(app)
    _log_environment()
    return app


def start_ngrok(port: int = FLASK_PORT, auth_token: str = NGROK_AUTH_TOKEN) -> str:
    """Start ngrok tunnel and return public URL."""
    from pyngrok import ngrok

    if auth_token:
        ngrok.set_auth_token(auth_token)

    # Kill any existing tunnels (clean start)
    for tunnel in ngrok.get_tunnels():
        ngrok.disconnect(tunnel.public_url)

    public_url = ngrok.connect(port).public_url
    log.info(f"🌐 ngrok tunnel: {public_url} → http://localhost:{port}")
    log.info(f"   Copy this URL into your application backend's .env:")
    log.info(f"   COLAB_NGROK_URL={public_url}")
    return public_url


def start_server(use_ngrok: bool = True, port: int = FLASK_PORT, debug: bool = False):
    """
    Main entrypoint. Use from notebook or CLI:

        from ml_server.app import start_server
        start_server()
    """
    app = create_app()

    if use_ngrok:
        public_url = start_ngrok(port=port)
        log.info(f"Server starting on {public_url}")
    else:
        log.info(f"Server starting on http://localhost:{port}")

    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    # CLI entry: python -m ml_server.app
    start_server(use_ngrok=False)
