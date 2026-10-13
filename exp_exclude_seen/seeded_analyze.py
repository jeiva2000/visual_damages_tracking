"""Ejecuta analyze_dinov2_nn_baseline.py con la aleatoriedad de entrenamiento FIJADA.

En el camino --kfold-cv nunca se llama a torch.manual_seed (solo lo hace el main
de train_linear_reid_head.py), así que la inicialización de las cabezas cambia
en cada proceso (repro 2026-10-02: H3 C4 bt 0.9070 vs 0.8777 publicado). --seed
NO sirve para variarla: también fija el muestreo de los episodios de evaluación
(TrackerFailureReIDDataset), y cambiarlo rompería la comparación pareada con C4
sin cabeza. Este wrapper fija torch/numpy/random con HEAD_TORCH_SEED y deja
--seed tal cual. No modifica ningún fichero compartido.
Uso: HEAD_TORCH_SEED=1 python3 seeded_analyze.py <args de analyze_dinov2_nn_baseline.py>"""
import os
import random
import runpy
import sys

import numpy as np
import torch

APP = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking"
seed = int(os.environ["HEAD_TORCH_SEED"])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print(f"[seeded_analyze] HEAD_TORCH_SEED={seed}", flush=True)

sys.path.insert(0, APP)
script = os.path.join(APP, "analyze_dinov2_nn_baseline.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
