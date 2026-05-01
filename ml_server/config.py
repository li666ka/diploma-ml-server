"""Constants and configuration."""
import os
from pathlib import Path

# ── Drive paths (Colab) або локальні ──
# На Colab: /content/drive/MyDrive/...
# Локально: ./data/...
IS_COLAB = os.getenv("COLAB_RELEASE_TAG") is not None or Path("/content/drive").exists()

if IS_COLAB:
    DRIVE_ROOT = "/content/drive/MyDrive"
    DATASETS_ROOT = f"{DRIVE_ROOT}/diploma_datasets"
    MODELS_ROOT = f"{DRIVE_ROOT}/diploma_models"
    CHUNKS_ROOT = f"{DRIVE_ROOT}/diploma_chunks"
else:
    LOCAL_ROOT = Path(__file__).parent.parent / "data"
    DATASETS_ROOT = str(LOCAL_ROOT / "datasets")
    MODELS_ROOT = str(LOCAL_ROOT / "models")
    CHUNKS_ROOT = str(LOCAL_ROOT / "chunks")

# ── Aggregated NB pipeline ──
AGGREGATED_SOCIAL_COLS = [
    "tweet_count",
    "mean_followers",
    "mean_friends",
    "verified_ratio",
    "mean_statuses",
    "mean_account_age_days",
    "mean_retweets",
    "mean_favorites",
]

# ── GNN ──
EMBED_DIM = 384  # MiniLM
MAX_REPLY_DEPTH = 10
NODE_TYPE_ARTICLE = 0
NODE_TYPE_TWEET = 1
NODE_TYPE_RETWEET = 2
NODE_TYPE_REPLY = 3

# ── Server ──
FLASK_PORT = int(os.getenv("FLASK_PORT", "5050"))
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")
