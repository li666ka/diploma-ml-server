"""Flask app factory + entrypoint for Colab/local."""
import os

from flask import Flask

from ml_server.config import FLASK_PORT, NGROK_AUTH_TOKEN
from ml_server.routes import register_routes
from ml_server.utils import log


def create_app() -> Flask:
    """Build Flask app with all routes registered."""
    app = Flask(__name__)
    register_routes(app)
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
