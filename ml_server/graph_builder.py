"""PyG graph construction.

Будуємо один граф для кожної статті:
  article (root) ← tweets ← retweets
                          ← replies (BFS до глибини MAX_REPLY_DEPTH)
"""
from typing import Optional

import pandas as pd
import torch

from ml_server.config import (
    MAX_REPLY_DEPTH,
    NODE_TYPE_ARTICLE,
    NODE_TYPE_REPLY,
    NODE_TYPE_RETWEET,
    NODE_TYPE_TWEET,
)
from ml_server.utils import log


def build_graph_for_article(
    article_id: str,
    label: int,
    article_emb: torch.Tensor,
    tweets_for_article: list[str],
    retweets_for_tweets: dict[str, list[str]],
    replies_by_parent: dict[str, list[str]],
    tweet_emb: dict[str, torch.Tensor],
    retweet_emb: dict[str, torch.Tensor],
    reply_emb: dict[str, torch.Tensor],
):
    """Build single PyG Data object for an article."""
    from torch_geometric.data import Data

    node_features: list[torch.Tensor] = []
    node_types: list[int] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []

    # 0. Article (root)
    node_features.append(article_emb)
    node_types.append(NODE_TYPE_ARTICLE)
    article_idx = 0

    # 1. Tweets
    tweet_to_idx: dict[str, int] = {}
    for tid in tweets_for_article:
        if tid not in tweet_emb:
            continue
        idx = len(node_features)
        node_features.append(tweet_emb[tid])
        node_types.append(NODE_TYPE_TWEET)
        tweet_to_idx[tid] = idx
        edge_src.append(idx)
        edge_dst.append(article_idx)

    # 2. Retweets
    for orig_tid, rt_list in retweets_for_tweets.items():
        if orig_tid not in tweet_to_idx:
            continue
        for rid in rt_list:
            if rid not in retweet_emb:
                continue
            idx = len(node_features)
            node_features.append(retweet_emb[rid])
            node_types.append(NODE_TYPE_RETWEET)
            edge_src.append(idx)
            edge_dst.append(tweet_to_idx[orig_tid])

    # 3. Replies (BFS)
    reply_to_idx: dict[str, int] = {}
    frontier = set(tweet_to_idx.keys())
    visited: set[str] = set()

    for _depth in range(MAX_REPLY_DEPTH):
        next_frontier = set()
        for parent_id in frontier:
            if parent_id in visited:
                continue
            visited.add(parent_id)
            for rid in replies_by_parent.get(parent_id, []):
                if rid in reply_to_idx or rid not in reply_emb:
                    continue
                idx = len(node_features)
                node_features.append(reply_emb[rid])
                node_types.append(NODE_TYPE_REPLY)
                reply_to_idx[rid] = idx
                if parent_id in tweet_to_idx:
                    edge_src.append(idx)
                    edge_dst.append(tweet_to_idx[parent_id])
                elif parent_id in reply_to_idx:
                    edge_src.append(idx)
                    edge_dst.append(reply_to_idx[parent_id])
                next_frontier.add(rid)
        if not next_frontier:
            break
        frontier = next_frontier

    # Build tensors. Isolated випадок (тільки article, без твітів) — додаємо
    # self-loop, щоб усі архітектури GNN не падали на 0-edge графах.
    x = torch.stack(node_features)
    if edge_src:
        edge_index = torch.tensor(
            [edge_src + edge_dst, edge_dst + edge_src],
            dtype=torch.long,
        )
    else:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        node_type=torch.tensor(node_types, dtype=torch.long),
        article_id=article_id,
        num_nodes=len(node_features),
    )


def build_all_graphs(
    articles_df: pd.DataFrame,
    tweets: pd.DataFrame,
    retweets: pd.DataFrame,
    replies: pd.DataFrame,
    article_emb_dict: dict[str, torch.Tensor],
    tweet_emb_dict: dict[str, torch.Tensor],
    retweet_emb_dict: dict[str, torch.Tensor],
    reply_emb_dict: dict[str, torch.Tensor],
    progress_callback: Optional[callable] = None,
) -> list:
    """Build PyG graphs for all articles in articles_df."""
    log.info("Building indexes...")

    tweets_by_article = (
        tweets.groupby("article_id")["tweet_id"].apply(list).to_dict()
        if len(tweets) > 0 else {}
    )
    retweets_by_orig = {}
    if len(retweets) > 0:
        retweets_by_orig = (
            retweets.groupby("original_tweet_id")["retweet_id"]
            .apply(list).to_dict()
        )
    replies_by_parent = {}
    if len(replies) > 0:
        replies_by_parent = (
            replies.groupby("parent_tweet_id")["reply_id"]
            .apply(list).to_dict()
        )

    graphs = []
    skipped_no_emb = 0
    log.info(f"Building {len(articles_df):,} graphs...")

    for i, row in enumerate(articles_df.itertuples(index=False)):
        article_id = str(row.article_id)
        label = int(row.label)

        if article_id not in article_emb_dict:
            skipped_no_emb += 1
            continue

        tweet_ids = tweets_by_article.get(article_id, [])
        rt_for_tweets = {
            tid: retweets_by_orig[tid]
            for tid in tweet_ids if tid in retweets_by_orig
        }

        graph = build_graph_for_article(
            article_id=article_id,
            label=label,
            article_emb=article_emb_dict[article_id],
            tweets_for_article=tweet_ids,
            retweets_for_tweets=rt_for_tweets,
            replies_by_parent=replies_by_parent,
            tweet_emb=tweet_emb_dict,
            retweet_emb=retweet_emb_dict,
            reply_emb=reply_emb_dict,
        )
        graphs.append(graph)

        if (i + 1) % 1000 == 0:
            if progress_callback:
                progress_callback(f"built_graphs_{i+1}_of_{len(articles_df)}")
            log.info(f"  built {i+1}/{len(articles_df)}")

    if skipped_no_emb:
        log.warning(
            f"  SKIPPED {skipped_no_emb} articles without embeddings — "
            "predictions для них не будуть створені (порушує test-set "
            "alignment для ensemble). Перетренуйте embedding cache."
        )
    log.info(f"  Done: {len(graphs):,} graphs")
    return graphs


def build_inference_graph(
    article_text: str,
    tweets_input: list[dict],
    retweets_input: list[dict],
    replies_input: list[dict],
):
    """Build PyG Data для inference. Однакова логіка з training (build_graph_for_article),
    але приймає raw text замість IDs (немає БД).

    tweets_input: [{text}]
    retweets_input: [{text, original_tweet_idx}]
    replies_input: [{text, parent_tweet_idx? OR parent_reply_idx?}]
    """
    from torch_geometric.data import Data

    from ml_server.encoder import encode_texts_batch

    all_texts = [article_text]
    all_texts.extend([t.get("text", "") for t in tweets_input])
    all_texts.extend([rt.get("text", "") for rt in retweets_input])
    all_texts.extend([rp.get("text", "") for rp in replies_input])

    embs_np = encode_texts_batch(
        all_texts, batch_size=128, max_chars=2000, show_progress=False
    )
    embs = torch.from_numpy(embs_np)

    n_tweets = len(tweets_input)
    n_retweets = len(retweets_input)
    n_replies = len(replies_input)

    article_idx = 0
    tweet_indices = list(range(1, 1 + n_tweets))
    retweet_indices = list(range(1 + n_tweets, 1 + n_tweets + n_retweets))
    reply_indices = list(range(
        1 + n_tweets + n_retweets,
        1 + n_tweets + n_retweets + n_replies,
    ))

    edge_src, edge_dst = [], []

    # 1. Tweets → Article
    for tw_idx in tweet_indices:
        edge_src.append(tw_idx)
        edge_dst.append(article_idx)

    # 2. Retweets → Tweet
    for i, rt in enumerate(retweets_input):
        orig_idx = rt.get("original_tweet_idx", 0)
        if 0 <= orig_idx < n_tweets:
            edge_src.append(retweet_indices[i])
            edge_dst.append(tweet_indices[orig_idx])

    # 3. Replies → Tweet OR Reply
    for i, rp in enumerate(replies_input):
        parent_tw = rp.get("parent_tweet_idx")
        parent_rp = rp.get("parent_reply_idx")

        if parent_tw is not None and 0 <= parent_tw < n_tweets:
            edge_src.append(reply_indices[i])
            edge_dst.append(tweet_indices[parent_tw])
        elif parent_rp is not None and 0 <= parent_rp < i:
            edge_src.append(reply_indices[i])
            edge_dst.append(reply_indices[parent_rp])

    if edge_src:
        edge_index = torch.tensor(
            [edge_src + edge_dst, edge_dst + edge_src],
            dtype=torch.long,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    batch = torch.zeros(embs.shape[0], dtype=torch.long)

    return Data(
        x=embs,
        edge_index=edge_index,
        batch=batch,
        num_nodes=embs.shape[0],
    )
