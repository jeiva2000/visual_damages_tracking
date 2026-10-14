"""
p4_backbone_encoder.py

Envoltorios de distintas capas de RT-DETRv2 como fuente de embedding de
apariencia congelado, con la MISMA interfaz que un `AutoModel` de
HuggingFace (`model(pixel_values=...).last_hidden_state[:, 0, :]`) para
poder usarlos como reemplazo directo de DINOv2 en
analyze_dinov2_nn_baseline.py / train_linear_reid_head.py sin tocar
`_encode_cls`/`_encode_raw_cls`/`_bank_and_query_embeds`/`train_head`.

OJO -- aclaración de nombres (confirmado empíricamente, ver conversación):
lo que este proyecto viene llamando "P4" desde el principio de la sesión es
en realidad **C4 crudo** (`backbone.res_layers.2`, salida del backbone
ANTES del encoder, 1024-dim, stride 16) -- NO la fusión FPN real del
encoder (`encoder.fpn_blocks.0`, 384-dim, que combina C5+C4 y que
`linear_test_tracking.py` en ../damages_tracking sí llama "P4" correctamente,
según la convención de nombres de RT-DETR/HybridEncoder: P3/P4/P5 son las
salidas del encoder, C2-C5 son las salidas crudas del backbone). Se
mantiene el nombre "p4"/`P4ROIAlignEncoder` para no romper compatibilidad
con caches/resultados ya generados en la sesión bajo ese nombre, pero
`LAYER_SPECS` deja explícita la distinción y agrega la fusión FPN real
como una opción nueva ("p4fpn") para comparar ambas.

A diferencia de p4_reid_finetune.py (ROI-Align sobre un frame completo con
las cajas GT), `P4BackboneEncoder` asume que el "crop" YA ES la detección
completa (mismo recorte de 224x224 que ve DINOv2 hoy, ver DEFAULT_TRANSFORM
en reid_dataloader.py) -- por eso no hace falta ROI-Align ahí: el embedding
es un global-average-pool sobre TODO el feature map de salida de esa capa.
Fue el proxy rápido inicial (después descartado por mala validez, ver
conversación) -- el camino real es `P4ROIAlignEncoder.raw_feature_map` +
ROI-Align aguas arriba (ver precompute_p4_roi_cache.py: un solo forward por
frame completo, feature map compartido entre todas las detecciones de ese
frame).
"""
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # parent of the repo: RT-DETR repository (package `src`)


# key -> (hook_layer, hidden_size, needs_encoder_forward)
# needs_encoder_forward=False: la capa vive en el backbone, alcanza con
#   correr self.model.backbone(images) para disparar el hook.
# needs_encoder_forward=True: la capa vive dentro de encoder.fpn_blocks/
#   pan_blocks, hace falta correr self.model.encoder(self.model.backbone(images))
#   completo (FPN top-down + PAN bottom-up) para que el hook dispare, aunque
#   solo nos interese un punto intermedio de ese forward.
LAYER_SPECS = {
    "c4": ("backbone.res_layers.2", 1024, False),          # "P4" tal como se usó en toda la sesión (en realidad C4 crudo)
    "c5": ("backbone.res_layers.3", 2048, False),           # última etapa cruda del backbone, pre-encoder
    "p4fpn": ("encoder.fpn_blocks.0", 384, True),           # P4 REAL (fusión FPN: C5+C4), ver linear_test_tracking.py
    "p3fpn": ("encoder.fpn_blocks.1", 384, True),           # P3 REAL (fusión FPN: P4+C3)
}


class _CLSLikeOutput:
    """Imita la salida de un AutoModel de HF lo mínimo necesario: `.last_hidden_state[:, 0, :]`
    debe devolver el embedding pooled (B, D)."""

    def __init__(self, pooled):
        self.last_hidden_state = pooled.unsqueeze(1)  # (B, 1, D)


class P4ROIAlignEncoder(nn.Module):
    """Carga del backbone+encoder de RT-DETRv2 congelado + hook en la capa
    elegida por `layer` (ver LAYER_SPECS), SIN pooling propio -- expone
    `raw_feature_map(images)` para que quien llame haga ROI-Align aguas
    arriba (ver precompute_p4_roi_cache.py: un solo forward por frame
    completo, feature map compartido entre todas las detecciones de ese
    frame). Base de `P4BackboneEncoder` (pooling global sobre un crop ya
    recortado, sin ROI-Align)."""

    def __init__(self, config_path: str, checkpoint_path: str, layer: str = "c4"):
        super().__init__()
        from src.core import YAMLConfig

        if layer not in LAYER_SPECS:
            raise ValueError(f"layer='{layer}' desconocido -- opciones: {list(LAYER_SPECS)}")
        self.layer = layer
        self.HOOK_LAYER, self.HIDDEN_SIZE, self._needs_encoder = LAYER_SPECS[layer]

        cfg = YAMLConfig(config_path, resume=checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
        cfg.model.load_state_dict(state, strict=False)
        self.model = cfg.model.deploy()
        self.config = SimpleNamespace(hidden_size=self.HIDDEN_SIZE)

        self._feat = None
        for name, mod in self.model.named_modules():
            if name == self.HOOK_LAYER:
                mod.register_forward_hook(self._capture)
                break
        else:
            raise ValueError(f"Capa '{self.HOOK_LAYER}' no encontrada en el modelo.")

    def _capture(self, module, inputs, output):
        if isinstance(output, (list, tuple)):
            output = output[-1]
        self._feat = output

    def raw_feature_map(self, images):
        """(B,3,H,W) -> (B,HIDDEN_SIZE,H/16,W/16), SIN normalizar (pooling/ROI-Align quedan aguas arriba)."""
        if self._needs_encoder:
            self.model.encoder(self.model.backbone(images))  # dispara el hook a mitad del forward del encoder
        else:
            self.model.backbone(images)  # dispara el hook; no se usa el retorno directo
        return self._feat


class P4BackboneEncoder(P4ROIAlignEncoder):
    """Interfaz estilo AutoModel de HF (`model(pixel_values=...)
    .last_hidden_state[:, 0, :]`) -- pooling GLOBAL sobre un crop que YA ES
    la detección completa (ver docstring del módulo), sin ROI-Align."""

    def forward(self, pixel_values):
        feat = self.raw_feature_map(pixel_values)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, HIDDEN_SIZE)
        pooled = F.normalize(pooled, p=2, dim=1)
        return _CLSLikeOutput(pooled)


def add_encoder_args(parser):
    """Flags para elegir el encoder congelado (DINOv2 de siempre, o una capa
    de RT-DETRv2 -- ver LAYER_SPECS/P4BackboneEncoder). Comparte
    --dinov2-model-name con el resto del script (ya definido aparte en cada
    main())."""
    parser.add_argument(
        "--encoder", choices=["dinov2", "p4", "p4fpn"], default="dinov2",
        help="Fuente del embedding congelado: 'dinov2' (CLS de --dinov2-model-name, default) / "
             "'p4' (C4 crudo del backbone, el usado en la sesión bajo el nombre 'P4' -- ver "
             "aclaración de nombres en el docstring del módulo) / 'p4fpn' (P4 REAL, fusión FPN "
             "C5+C4 del encoder).",
    )
    parser.add_argument(
        "--p4-config", default="../configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml",
        help="[--encoder p4/p4fpn] Config yaml del detector (ruta relativa a rtdetrv2_pytorch/, "
             "asumiendo cwd=visual_damages_tracking/).",
    )
    parser.add_argument(
        "--p4-checkpoint", default="../damages_tracking/models/rtdetrv2_v3_qcs_videos.pth",
        help="[--encoder p4/p4fpn] Checkpoint .pth del detector (mismo que usa p4_reid_finetune.py).",
    )
    return parser


_ENCODER_TO_LAYER = {"p4": "c4", "p4fpn": "p4fpn"}


def build_encoder(args):
    """Carga el encoder elegido por --encoder, ya movido a --device, en eval()
    y con gradientes desactivados -- reemplazo directo de
    `AutoModel.from_pretrained(args.dinov2_model_name).to(device).eval()`."""
    if args.encoder in _ENCODER_TO_LAYER:
        layer = _ENCODER_TO_LAYER[args.encoder]
        print(f"[INFO] Cargando P4BackboneEncoder (layer={layer}, config={args.p4_config}, checkpoint={args.p4_checkpoint})...")
        model = P4BackboneEncoder(args.p4_config, args.p4_checkpoint, layer=layer)
    else:
        from transformers import AutoModel
        print(f"[INFO] Cargando {args.dinov2_model_name}...")
        model = AutoModel.from_pretrained(args.dinov2_model_name)

    model = model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
