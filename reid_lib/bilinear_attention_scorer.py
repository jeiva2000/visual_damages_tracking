"""
BilinearAttentionScorer: cabeza de atención cruzada MINIMALISTA, alternativa
independiente a LinearReIDHead (linear_reid_head.py) -- en vez de proyectar
cada embedding por separado, usa un mecanismo de atención compartido para
refinar la decisión query-vs-banco directamente sobre el CLS CRUDO de
DINOv2 (congelado).

Motivación: ya está validado que una capa lineal simple SÍ mejora sobre
DINOv2 puro, y que el transformer pesado original (set_reid_model.py:
IntraTrackEncoder con transformer completo + CrossAttentionMatcher
multi-head con FFN + MatchScorer bilineal/MLP) empeora -- evidencia de que
ese diseño memoriza en vez de generalizar en este dataset chico. Este
módulo prueba un punto intermedio deliberadamente minimalista:

  - 1 sola cabeza de atención: más cabezas ayuda a memorizar, no a
    generalizar en régimen de poco dato.
  - Sin FFN post-atención: una FFN agrega parámetros no lineales que
    pueden memorizar sin aportar nada relacional.
  - Sin encoding posicional: el banco es un conjunto, no una secuencia --
    no hay orden que codificar.
  - V SIN proyectar (self.W solo se aplica a Q/K): el "peor caso" del
    mecanismo de atención es una combinación convexa de embeddings ya
    buenos de DINOv2 -- nunca se aleja arbitrariamente de esa geometría.
  - W COMPARTIDO entre Q y K (bilineal simétrico): la mitad de los
    parámetros de una atención con Wq/Wk separadas, y fuerza que la
    similitud sea simétrica (query y banco viven en el mismo espacio).
  - Mezcla residual con el baseline de cosine-similarity vía `gate`,
    inicializado muy negativo (sigmoid(gate) ≈ 0) para que el modelo
    arranque siendo EXACTAMENTE el baseline actual -- el gradiente decide
    cuánto pesar la corrección aprendida solo si hay evidencia real.

`score_mixed` es invariante (en el argmax) a `gate` -- la identidad la
sigue decidiendo la cosine cruda; solo la decisión de umbral (match/no
match) puede beneficiarse de `score_final` (ver train_bilinear_attention_scorer.py
para cómo se entrena `gate` con una señal directa vía la BCE de confianza).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearAttentionScorer(nn.Module):
    def __init__(self, in_dim: int, low_dim: int, gate_init: float = -5.0):
        super().__init__()
        self.in_dim = in_dim
        self.low_dim = low_dim
        self.W = nn.Linear(in_dim, low_dim, bias=False)  # compartido para Q y K
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, query_embed: torch.Tensor, bank_summary: torch.Tensor, bank_mask: torch.Tensor, isolated: bool = False):
        """
        query_embed: (B, D) CLS crudo promediado de las vistas de query.
        bank_summary: (B, S, D) CLS crudo promediado por slot del banco
            (S = max_bank_size, cada slot ya es un mean sobre sus crops de
            historial -- ver _embed_batch en analyze_dinov2_nn_baseline.py).
        bank_mask: (B, S) bool, True = slot válido (no padding).
        isolated: si True, `score_mixed` devuelve los logits CRUDOS de
            atención (Q·Kᵀ/√low_dim, antes del softmax), SIN mezclar con
            cos_i ni con `gate` -- usado para aislar la hipótesis de que
            `gate` (multiplicativo, inicializado en el régimen saturado de
            la sigmoide) atenúa DOBLEMENTE el gradiente hacia W: una vez en
            el forward (sigmoid(gate)≈0 escala score_final) y otra vez en
            el propio gradiente de `gate` (sigmoid'(gate)≈0 en ese punto),
            frenando el aprendizaje de W antes de que pueda evidenciar si
            la atención aporta señal útil. Ver train_bilinear_attention_scorer.py
            (--isolated-scorer) y save_scorer/load_scorer (persiste este modo
            en el checkpoint para que la evaluación lo use automáticamente).

        Devuelve:
          score_mixed: (B, S) -- score por candidato para argmax/umbral.
              Con isolated=False (default): arranca casi idéntico a
              cos(query, bank_summary_i) porque sigmoid(gate) ≈ 0 al
              inicio. Con isolated=True: son los logits crudos de
              atención, sin ninguna mezcla. Slots inválidos en -inf.
          score_final: (B,) -- cos(query, context), score de confianza
              GLOBAL (no por candidato, NUNCA pasa por `gate`): bajo cuando
              la atención no concentra su peso en ningún candidato bueno
              (contexto "borroso"), sin necesitar una clase de rechazo
              explícita.
          attn: (B, S) -- pesos de atención (softmax), para inspección.
        """
        Q = self.W(query_embed).unsqueeze(1)  # (B, 1, low_dim)
        K = self.W(bank_summary)  # (B, S, low_dim)
        logits = torch.bmm(Q, K.transpose(1, 2)).squeeze(1) / (self.low_dim ** 0.5)  # (B, S)
        logits = logits.masked_fill(~bank_mask, float("-inf"))

        # guarda contra un episodio degenerado con bank_mask todo False
        # (no debería pasar en la práctica) -- evita NaN del softmax de una
        # fila 100% -inf; el resultado de esa fila queda descartado río
        # abajo por bank_mask de cualquier forma.
        degenerate = ~bank_mask.any(dim=1)
        if degenerate.any():
            logits = logits.clone()
            logits[degenerate, 0] = 0.0

        attn = F.softmax(logits, dim=-1)  # (B, S)
        context = torch.bmm(attn.unsqueeze(1), bank_summary).squeeze(1)  # (B, D), V sin proyectar
        score_final = F.cosine_similarity(query_embed, context, dim=-1)  # (B,)

        if isolated:
            return logits, score_final, attn

        cos_i = F.cosine_similarity(query_embed.unsqueeze(1), bank_summary, dim=-1)  # (B, S)
        g = torch.sigmoid(self.gate)
        score_mixed = (1 - g) * cos_i + g * score_final.unsqueeze(1)  # (B, S)
        score_mixed = score_mixed.masked_fill(~bank_mask, float("-inf"))

        return score_mixed, score_final, attn


def save_scorer(scorer: BilinearAttentionScorer, path: str, isolated: bool = False) -> None:
    torch.save(
        {"in_dim": scorer.in_dim, "low_dim": scorer.low_dim, "state_dict": scorer.state_dict(), "isolated": isolated},
        path,
    )


def load_scorer(path: str, device) -> BilinearAttentionScorer:
    ckpt = torch.load(path, map_location=device)
    scorer = BilinearAttentionScorer(ckpt["in_dim"], ckpt["low_dim"])
    missing, unexpected = scorer.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[WARN] load_scorer: faltan claves {missing}, sobran {unexpected} (¿checkpoint de otra versión?)")
    scorer.to(device).eval()
    for p in scorer.parameters():
        p.requires_grad = False
    # persistido por train_bilinear_attention_scorer.py --isolated-scorer --
    # analyze_dinov2_nn_baseline.py lo lee vía getattr(scorer, "isolated", False)
    # para forward(isolated=...) automáticamente, sin flag de CLI aparte.
    scorer.isolated = ckpt.get("isolated", False)
    return scorer
