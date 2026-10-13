"""
paper7_backbone_encoder.py

Envoltorio del detector RT-DETR (arquitectura Ultralytics, backbone
PP-HGNetV2) usado en el trabajo previo del grupo [7] (Perez-Zarate et al.,
"Vehicle damage detection with transformer-based multi-object tracking",
EAAI 2026, ver \\cite{perezzarate2026rtdetrvd}), como fuente de embedding de
apariencia congelado -- respondiendo M4 del checklist de revision: aplicar
NUESTRA estrategia de reid (banco/galeria, ver reid_gallery.py) sobre la
salida de tracking del transformer entrenado de [7], en vez de reemplazar
ese transformer.

MISMA interfaz que P4ROIAlignEncoder (p4_backbone_encoder.py) --
`raw_feature_map(images) -> (B, HIDDEN_SIZE, Hf, Wf)`, sin pooling propio,
para poder reusar el patron de precompute_p4_roi_cache.py (ROI-Align aguas
arriba, un solo forward por frame) sin tocar ese archivo -- ver
precompute_paper7_c4_roi_cache.py.

Eleccion de capa (decision de diseno, confirmada con el usuario): el
backbone de [7] (PP-HGNetV2, via `ultralytics.RTDETR`) NO es el mismo
backbone que el nuestro (ResNet-101-vd, `rtdetrv2_r101vd`) -- no existe un
layer "C4" compartido por nombre entre ambas arquitecturas. Se eligio la
etapa equivalente por STRIDE, mismo criterio con el que definimos C4 en
nuestro propio detector (`backbone.res_layers.2`, stride 16, ver
p4_backbone_encoder.py::LAYER_SPECS): dentro de HGNetV2
(`model.model.model[0:10]`), los modulos son:

    0 HGStem        -> stride 4
    1 HGBlock       -> stride 4
    2 DWConv        -> downsample a stride 8
    3 HGBlock       -> stride 8
    4 DWConv        -> downsample a stride 16
    5 HGBlock  \\
    6 HGBlock   >-  -> stride 16   (etapa equivalente a C4)
    7 HGBlock  /
    8 DWConv        -> downsample a stride 32
    9 HGBlock       -> stride 32

El modulo 7 (ultimo HGBlock de la etapa stride-16) es el elegido. Verificado
empiricamente contra el checkpoint real de [7] (best.pt, ultralytics 8.4.76):
input (1,3,640,640) -> salida (1,1024,40,40) en el modulo 7 -- stride 16,
1024 canales, MISMA dimensionalidad que nuestro propio C4
(backbone.res_layers.2, tambien 1024-dim) -- coincidencia util para no tener
que ajustar downstream (ROI-Align/pooling), pero no es la razon de la
eleccion (la razon es el stride, no la dimension).

A diferencia de P4ROIAlignEncoder (que hookea un submodulo NOMBRADO del
modelo, `self.model.backbone(images)`), el modelo de Ultralytics es una
lista plana (`model.model.model`, un nn.Sequential-like indexado por
posicion) -- alcanza con correr manualmente los modulos 0..7 en orden y
devolver la salida del ultimo, sin hook: los modulos 0-7 de HGNetV2 son
puramente secuenciales (sin Concat/skip-connections hacia modulos
anteriores; esas aparecen recien en el FPN/neck, modulos 15/20/23/26 en
adelante), asi que no hace falta reproducir la logica de "from index" de
`parse_model` de Ultralytics para este rango.
"""
import torch
import torch.nn as nn

# Ver docstring del modulo -- eleccion por stride, no por nombre compartido.
HGNETV2_C4_EQUIVALENT_MODULE_INDEX = 7
HGNETV2_C4_EQUIVALENT_HIDDEN_SIZE = 1024


class Paper7ROIAlignEncoder(nn.Module):
    """Carga el detector RT-DETR (Ultralytics/HGNetV2) de [7], congelado,
    exponiendo `raw_feature_map(images)` (salida del modulo
    HGNETV2_C4_EQUIVALENT_MODULE_INDEX) para ROI-Align aguas arriba -- misma
    interfaz que P4ROIAlignEncoder.raw_feature_map, ver
    precompute_paper7_c4_roi_cache.py (copia de precompute_p4_roi_cache.py
    con este encoder en vez de P4ROIAlignEncoder)."""

    def __init__(self, checkpoint_path: str):
        super().__init__()
        from ultralytics import RTDETR

        rtdetr = RTDETR(checkpoint_path)
        self.model = rtdetr.model
        self.hidden_size = HGNETV2_C4_EQUIVALENT_HIDDEN_SIZE
        # Solo los modulos 0..HGNETV2_C4_EQUIVALENT_MODULE_INDEX hacen falta
        # -- el resto (AIFI/FPN/PAN/decoder, modulos 10-28) no se corre.
        self._backbone_modules = self.model.model[: HGNETV2_C4_EQUIVALENT_MODULE_INDEX + 1]

    def raw_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W), rango [0,1] (misma convencion que
        precompute_p4_roi_cache.py: Resize + ToTensor, sin normalizacion
        mean/std) -> (B,1024,H/16,W/16), SIN normalizar (pooling/ROI-Align
        quedan aguas arriba, ver precompute_paper7_c4_roi_cache.py)."""
        y = images
        for module in self._backbone_modules:
            y = module(y)
        return y
