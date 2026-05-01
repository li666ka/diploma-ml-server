"""Smoke test: HTTP request to running server.

Використання:
    # Локально:
    python scripts/test_inference.py --url http://localhost:5050

    # На Colab (через ngrok):
    python scripts/test_inference.py --url https://abc123.ngrok-free.app
"""
import argparse
import json

import requests


def test_health(base_url: str):
    print(f"\n→ GET {base_url}/health")
    r = requests.get(f"{base_url}/health", timeout=10)
    print(f"  {r.status_code}: {r.json()}")


def test_list_datasets(base_url: str):
    print(f"\n→ GET {base_url}/list_datasets")
    r = requests.get(f"{base_url}/list_datasets", timeout=10)
    print(f"  {r.status_code}: {json.dumps(r.json(), indent=2)[:500]}")


def test_predict_distilbert(base_url: str):
    print(f"\n→ POST {base_url}/predict_distilbert")
    payload = {"text": "Trump just signed a fake executive order!"}
    r = requests.post(
        f"{base_url}/predict_distilbert", json=payload, timeout=30
    )
    print(f"  {r.status_code}: {r.json()}")


def test_predict_gnn(base_url: str, model_path: str):
    print(f"\n→ POST {base_url}/predict_gnn")
    payload = {
        "article_text": "Scientists discovered new evidence...",
        "tweets": [
            {"text": "Wow, this is huge news!"},
            {"text": "I don't believe this story at all"},
        ],
        "retweets": [],
        "replies": [],
        "model_path": model_path,
    }
    r = requests.post(f"{base_url}/predict_gnn", json=payload, timeout=60)
    print(f"  {r.status_code}: {r.json()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:5050")
    parser.add_argument("--gnn-model-path", default=None,
                        help="Path to .pkl file for GNN test")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"Testing server at {base_url}")

    test_health(base_url)
    test_list_datasets(base_url)
    test_predict_distilbert(base_url)
    if args.gnn_model_path:
        test_predict_gnn(base_url, args.gnn_model_path)


if __name__ == "__main__":
    main()
