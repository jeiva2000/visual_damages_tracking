"""
Cabeza liviana: una sola capa lineal sobre el embedding CONGELADO de DINOv2
(CLS de la última capa), entrenada con metric learning (batch-hard triplet,
ver hierarchical_pk_sampler.py) en vez del clasificador (k+1)-way pesado de
set_reid_model.py.

Motivación (ver analyze_dinov2_nn_baseline.py): el 1-NN crudo sobre el CLS
de DINOv2 ya supera a la cabeza entrenada completa (transformer intra-track
+ cross-attention + bilinear scorer) -- evidencia de que la cabeza pesada
pierde señal en vez de sumarla en este dataset chico. Antes de descartar
toda cabeza entrenable, se prueba la más simple posible: una capa lineal,
que en el peor caso reescala el espacio ya bueno, y en el mejor achica el
gap que quedaba en val/test frente a los casos reales de los trackers.

`load_head`/`save_head` guardan también las dimensiones (in_dim/out_dim)
junto a los pesos -- así analyze_dinov2_nn_baseline.py puede cargar el
checkpoint sin tener que adivinar la arquitectura.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearReIDHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim)
        # embedding aprendible de "no match" -- equivalente liviano al
        # new_embedding de MatchScorer (set_reid_model.py): compite como una
        # columna más en la loss InfoNCE de train_linear_reid_head.py para
        # los casos "query sin match real en el banco" (ver
        # --new-track-prob). No lo usa analyze_dinov2_nn_baseline.py
        # (esa evaluación sigue decidiendo match/no-match con el umbral
        # calibrado aparte) -- vive acá solo para que quede guardado junto
        # con el resto de los pesos del checkpoint.
        self.new_embedding = nn.Parameter(torch.randn(out_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2-normalizado -- estándar en metric learning (Hermans et al.):
        # hace que la distancia euclidiana entre embeddings sea bien
        # comportada (equivalente a similitud coseno salvo un factor fijo),
        # consistente con _cosine_sim en analyze_dinov2_nn_baseline.py.
        return F.normalize(self.proj(x), p=2, dim=-1)

    def new_embedding_normalized(self) -> torch.Tensor:
        return F.normalize(self.new_embedding, p=2, dim=-1)


def save_head(head: LinearReIDHead, path: str) -> None:
    torch.save({"in_dim": head.in_dim, "out_dim": head.out_dim, "state_dict": head.state_dict()}, path)


def load_head(path: str, device) -> LinearReIDHead:
    ckpt = torch.load(path, map_location=device)
    head = LinearReIDHead(ckpt["in_dim"], ckpt["out_dim"])
    # strict=False -- checkpoints viejos (antes de new_embedding) no tienen
    # esa clave; se avisa pero no falla, y new_embedding queda con su init
    # al azar en ese caso.
    missing, unexpected = head.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[WARN] load_head: faltan claves {missing}, sobran {unexpected} (¿checkpoint de otra versión?)")
    head.to(device).eval()
    for p in head.parameters():
        p.requires_grad = False
    return head
