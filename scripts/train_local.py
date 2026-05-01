"""Локальне тренування без Flask.

Зручно для дебагу: запускаєш напряму у VSCode debugger, ставиш breakpoint, 
дивишся всі змінні. Без HTTP, без ngrok, без Drive.

Використання:
    python scripts/train_local.py --model gnn --dataset-id 1
    python scripts/train_local.py --model distilbert --dataset-id 1
    python scripts/train_local.py --model nb --dataset-id 1
"""
import argparse
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_server.aggregated_loader import build_aggregated_data
from ml_server.data_loader import build_article_level_data
from ml_server.distilbert_trainer import train_distilbert_article_level
from ml_server.gnn_trainer import train_gnn
from ml_server.nb_trainer import train_nb
from ml_server.utils import log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["nb", "distilbert", "gnn"])
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--user-id", default="local")
    parser.add_argument("--experiment-id", default="debug")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--architecture", default="gin",
                        help="GNN architecture: 'gin' or 'sage'")
    args = parser.parse_args()

    if args.model == "nb":
        train_df, test_df, stats, _ = build_aggregated_data(
            dataset_id=args.dataset_id
        )
        log.info(f"Stats: {stats}")
        result = train_nb(
            train_df, test_df,
            user_id=args.user_id, experiment_id=args.experiment_id,
        )

    elif args.model == "distilbert":
        train_df, val_df, test_df, _, stats, _ = build_article_level_data(
            dataset_id=args.dataset_id,
        )
        log.info(f"Stats: {stats}")
        model_params = {}
        if args.epochs:
            model_params["epochs"] = args.epochs
        result = train_distilbert_article_level(
            train_df, val_df, test_df,
            user_id=args.user_id, experiment_id=args.experiment_id,
            model_params=model_params,
        )

    elif args.model == "gnn":
        train_df, val_df, test_df, full_data, stats, _ = build_article_level_data(
            dataset_id=args.dataset_id, require_tweets=True,
        )
        log.info(f"Stats: {stats}")
        model_params = {"architecture": args.architecture}
        if args.epochs:
            model_params["epochs"] = args.epochs
        result = train_gnn(
            train_df, val_df, test_df, full_data,
            user_id=args.user_id, experiment_id=args.experiment_id,
            model_params=model_params,
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")

    log.info("=" * 60)
    log.info(f"✅ Training done")
    log.info(f"   Path: {result['path']}")
    log.info(f"   Metrics: {result['metrics']}")


if __name__ == "__main__":
    main()
