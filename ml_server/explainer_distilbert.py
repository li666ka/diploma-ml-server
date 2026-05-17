"""Local explanation для DistilBERT через Layer Integrated Gradients (Captum).

Why IG, не attention:
  Attention weights — anti-pattern для XAI (Jain & Wallace, NAACL 2019,
  "Attention is not Explanation"). IG порівнює gradient вздовж шляху від
  baseline (PAD-токени) до фактичного input, дає axiomatically-обґрунтовану
  attribution на per-token level.

Перформанс:
  n_steps=30, seq_len=256 → ~2-5s на CUDA, 10-20s на CPU.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

log = logging.getLogger(__name__)


def explain_distilbert(
    text: str,
    model,
    tokenizer,
    max_length: int = 256,
    n_steps: int = 30,
    top_k: int = 20,
) -> dict:
    """LayerIntegratedGradients для predicted-class.

    Returns:
        {
          "method": "integrated_gradients",
          "method_params": {n_steps, baseline},
          "tokens": [{token, attribution, position, is_subword}] top-k за |attr|,
          "all_tokens_in_order": [...]  # для highlighted text у UI
          "predicted_class": 0|1,
          "predicted_label": "FAKE"|"REAL",
          "confidence": float,
        }
    """
    from captum.attr import LayerIntegratedGradients

    device = next(model.parameters()).device
    model.eval()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    # Baseline: усе PAD, але [CLS] на 0-й позиції та [SEP] на позиції
    # останнього реального токену (інакше baseline розбиває структуру речення).
    baseline = torch.full_like(input_ids, pad_id)
    if cls_id is not None:
        baseline[:, 0] = cls_id
    real_length = int(attention_mask.sum(dim=1).item())
    if sep_id is not None and real_length > 0:
        baseline[:, real_length - 1] = sep_id

    def forward_fn(inp_ids, attn_mask):
        out = model(input_ids=inp_ids, attention_mask=attn_mask)
        return torch.softmax(out.logits, dim=-1)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_class = int(probs.argmax().item())
        confidence = float(probs[pred_class].item())

    # Layer IG бере embedding layer; для DistilBERT classification head — це
    # `model.distilbert.embeddings.word_embeddings`. Якщо архітектура інша
    # (e.g. BERT/DeBERTa з тим самим класом-обгорткою), namespace зміниться —
    # підтримуємо обидва шляхи.
    embedding_layer = _resolve_embedding_layer(model)
    if embedding_layer is None:
        raise RuntimeError(
            "Cannot locate word embeddings layer on model — IG attribution "
            "unsupported for this architecture."
        )

    lig = LayerIntegratedGradients(forward_fn, embedding_layer)
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline,
        additional_forward_args=(attention_mask,),
        target=pred_class,
        n_steps=n_steps,
    )
    # (1, seq_len, hidden) → (seq_len,)
    attr_per_token = attributions.sum(dim=-1).squeeze(0)
    norm = float(torch.norm(attr_per_token).item())
    if norm > 1e-9:
        attr_per_token = attr_per_token / norm

    attr_values = attr_per_token.detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    special_tokens = {
        tokenizer.pad_token,
        tokenizer.cls_token,
        tokenizer.sep_token,
    }
    in_order: list[dict] = []
    for i, (tok, attr) in enumerate(zip(tokens, attr_values)):
        if tok in special_tokens or tok is None:
            continue
        in_order.append({
            "token": tok.replace("##", ""),
            "attribution": float(attr),
            "position": i,
            "is_subword": tok.startswith("##"),
        })

    top = sorted(in_order, key=lambda x: abs(x["attribution"]), reverse=True)[:top_k]

    return {
        "method": "integrated_gradients",
        "method_params": {"n_steps": n_steps, "baseline": "pad_tokens"},
        "tokens": top,
        "all_tokens_in_order": in_order,
        "predicted_class": pred_class,
        "predicted_label": "FAKE" if pred_class == 1 else "REAL",
        "confidence": confidence,
    }


def _resolve_embedding_layer(model) -> Optional[torch.nn.Module]:
    """Знайти word_embeddings для типових HF-classification моделей."""
    # DistilBertForSequenceClassification
    base = getattr(model, "distilbert", None)
    if base is None:
        # BertFor..., DebertaFor...
        base = getattr(model, "bert", None) or getattr(model, "deberta", None) or getattr(model, "base_model", None)
    if base is None:
        return None
    embeddings = getattr(base, "embeddings", None)
    if embeddings is None:
        return None
    return getattr(embeddings, "word_embeddings", None)
