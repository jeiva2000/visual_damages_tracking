"""Llama a TrackEval con la MISMA configuración que eval_tracking/run_eval.py::
run_trackeval (HOTA/CLEAR/Identity, SPLIT 'all', SKIP_SPLIT_FOL, THRESHOLD 0.5,
sin paralelo ni curvas), pero fijando OUTPUT_FOLDER para no escribir dentro de
la carpeta de predicciones original.
Uso: python trackeval_api.py <gt_folder> <seqmap> <trackers_folder> <output_folder>"""
import os
import sys

sys.path.insert(0, os.environ.get("TRACKEVAL_DIR", "TrackEval"))
import trackeval  # noqa: E402

gt, seqmap, trk, out = sys.argv[1:5]
ev_cfg = trackeval.Evaluator.get_default_eval_config()
ev_cfg.update({"USE_PARALLEL": False, "PLOT_CURVES": False, "PRINT_CONFIG": False})
ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
ds_cfg.update({"GT_FOLDER": gt, "SEQMAP_FILE": seqmap, "SPLIT_TO_EVAL": "all", "SKIP_SPLIT_FOL": True,
               "TRACKERS_FOLDER": trk, "TRACKERS_TO_EVAL": [""], "TRACKER_SUB_FOLDER": "",
               "OUTPUT_FOLDER": out, "OUTPUT_SUB_FOLDER": ""})
m_cfg = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
metrics = [trackeval.metrics.HOTA(m_cfg), trackeval.metrics.CLEAR(m_cfg), trackeval.metrics.Identity(m_cfg)]
trackeval.Evaluator(ev_cfg).evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics)
