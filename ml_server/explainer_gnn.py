"""Local explanation для GIN/SAGE через PyG GNNExplainer.

GNNExplainer (Ying et al., NeurIPS 2019) знаходить мінімальну підмножину
node features та edges, яка зберігає prediction. Для FakeNewsNet каскадів
це показує які саме твіти/користувачі/коментарі найбільше вплинули на
вердикт «FAKE/REAL» для статті.

Перформанс: ~5-15s на граф з ~100 вузлами (CUDA), 30-60s на CPU.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Optional

import torch

log = logging.getLogger(__name__)


class _GraphLevelWrapper(torch.nn.Module):
    """Адаптер між PyG `Explainer` (викликає model(x=..., edge_index=...))
    і нашим `model(data)` де data — SimpleNamespace з x/edge_index/batch."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch=None, **_):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        data = SimpleNamespace(x=x, edge_index=edge_index, batch=batch)
        return self.model(data)


def explain_gnn(
    graph_data,  # PyG Data
    model: torch.nn.Module,
    *,
    target_class: Optional[int] = None,
    top_k_nodes: int = 10,
    top_k_edges: int = 15,
    epochs: int = 200,
    node_metadata: Optional[list[dict]] = None,
) -> dict:
    """GNNExplainer для graph-level classification.

    Args:
        graph_data: PyG Data з x, edge_index, batch.
        model: GIN/SAGE classifier (forward(data) → logits[1, num_classes]).
        target_class: 0/1; None → argmax prediction.
        node_metadata: per-node список метаданих ({type, text, ...}) для
            підставляння у відповідь. Довжина має дорівнювати n_nodes.
    """
    from torch_geometric.explain import Explainer, GNNExplainer

    device = next(model.parameters()).device
    model.eval()

    x = graph_data.x.to(device)
    edge_index = graph_data.edge_index.to(device)
    batch = (
        graph_data.batch.to(device)
        if getattr(graph_data, "batch", None) is not None
        else torch.zeros(x.size(0), dtype=torch.long, device=device)
    )

    # ── 1. Predict ───────────────────────────────────────────────────
    wrapped = _GraphLevelWrapper(model).to(device)
    with torch.no_grad():
        logits = wrapped(x=x, edge_index=edge_index, batch=batch)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_class = int(probs.argmax().item())
        confidence = float(probs[pred_class].item())

    if target_class is None:
        target_class = pred_class

    # ── 2. GNNExplainer ──────────────────────────────────────────────
    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",  # article-level → graph-level prediction
            return_type="raw",
        ),
    )
    explanation = explainer(
        x=x,
        edge_index=edge_index,
        batch=batch,
        target=torch.tensor([target_class], device=device),
    )

    # ── 3. Aggregate masks → per-node / per-edge importance ──────────
    node_mask = explanation.node_mask  # (n_nodes, n_features)
    node_importance = node_mask.sum(dim=-1).detach().cpu().numpy()

    edge_mask = explanation.edge_mask  # (n_edges,)
    edge_importance = edge_mask.detach().cpu().numpy()

    # ── 4. Top nodes ─────────────────────────────────────────────────
    n_nodes = int(graph_data.x.shape[0])
    top_node_idx = node_importance.argsort()[-top_k_nodes:][::-1]
    important_nodes: list[dict] = []
    for nid in top_node_idx:
        nid = int(nid)
        item = {
            "node_id": nid,
            "importance": float(node_importance[nid]),
        }
        if node_metadata and 0 <= nid < len(node_metadata):
            item["metadata"] = node_metadata[nid]
        important_nodes.append(item)

    # ── 5. Top edges (повний edge_index — без дедуплікації reverse pairs).
    # Якщо хочемо унікальні пари, треба брати пару (u,v) з max(importance(u→v),
    # importance(v→u)). Залишаємо як є — UI може дедупнути по (min, max).
    top_edge_idx = edge_importance.argsort()[-top_k_edges:][::-1]
    edge_index_np = graph_data.edge_index.cpu().numpy()
    important_edges: list[dict] = []
    for eid in top_edge_idx:
        eid = int(eid)
        important_edges.append({
            "source": int(edge_index_np[0, eid]),
            "target": int(edge_index_np[1, eid]),
            "importance": float(edge_importance[eid]),
        })

    return {
        "method": "gnn_explainer",
        "method_params": {"epochs": epochs, "algorithm": "GNNExplainer"},
        "important_nodes": important_nodes,
        "important_edges": important_edges,
        "n_nodes_total": n_nodes,
        "n_edges_total": int(graph_data.edge_index.shape[1]),
        "predicted_class": pred_class,
        "predicted_label": "FAKE" if pred_class == 1 else "REAL",
        "confidence": confidence,
    }


# ── Helper: побудувати node_metadata за тими ж input списками, що
#    `build_inference_graph` (article, tweets, retweets, replies).
def build_node_metadata(
    article_text: str,
    tweets_input: list[dict],
    retweets_input: list[dict],
    replies_input: list[dict],
    *,
    text_preview: int = 120,
) -> list[dict]:
    """Дзеркалить порядок нод у build_inference_graph:
       [article, tweets..., retweets..., replies...]."""
    def _preview(t: str) -> str:
        s = (t or "").strip()
        return (s[:text_preview] + "…") if len(s) > text_preview else s

    meta: list[dict] = [{"type": "article", "text": _preview(article_text)}]
    for t in tweets_input:
        meta.append({"type": "tweet", "text": _preview(t.get("text", ""))})
    for rt in retweets_input:
        meta.append({
            "type": "retweet",
            "text": _preview(rt.get("text", "")),
            "original_tweet_idx": rt.get("original_tweet_idx"),
        })
    for rp in replies_input:
        meta.append({
            "type": "reply",
            "text": _preview(rp.get("text", "")),
            "parent_tweet_idx": rp.get("parent_tweet_idx"),
            "parent_reply_idx": rp.get("parent_reply_idx"),
        })
    return meta
