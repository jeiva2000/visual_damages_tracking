"""
Dataset de episodios (banco + query, mismo formato que ReIDEpisodeDataset)
construido a partir de casos REALES detectados por infer_track_reid_gaps.py
(ByteTrack o BotSort) -- a diferencia de ReIDEpisodeDataset, que genera
episodios sintéticos/muestreados al azar, acá cada episodio corresponde 1:1
a un caso real donde ESE tracker concreto le puso un id distinto al mismo
objeto (o, en el caso más estricto, directamente nació un id nuevo). Sin
sampling: son TODOS los casos que existen para ese tracker/split, uno por
episodio -- pensado para evaluar (el baseline de DINOv2, o más adelante el
modelo entrenado) específicamente sobre los fallos reales de un tracker
dado, no sobre casos sintéticos.

Dos fuentes de casos posibles (mismo shape, mismo consumidor):
  - load_new_id_born_cases: solo casos `new_id_born` de reid_gap_report.json
    (el tracker nació un id nuevo en una BRECHA TEMPORAL real -- definición
    estricta, ver find_real_gaps).
  - load_required_reid_cases: TODOS los cambios de id de
    required_reid_events.json (con o sin brecha temporal -- definición
    amplia, ver _count_required_reidentifications en infer_track_reid_gaps.py).

Construcción de cada episodio (misma convención que val/test):
  - identidad de referencia = la identidad de GT que el tracker perdió
  - banco: SIEMPRE incluye el historial real de esa identidad hasta el
    frame justo antes de la brecha (antes de que el tracker la perdiera) --
    slot 0, el match correcto -- + distractores del resto del índice (ver
    sample_hard_distractors/sample_easy_distractors). Con
    `same_video_only=True`, el banco SOLO se completa con distractores del
    MISMO video (nunca de otro) -- dataset 100% real, sin ninguna imagen de
    otro video mezclada.
  - query: la observación exacta del frame donde se detectó el cambio de id
    -- por construcción, SIEMPRE hay match real (nunca "no match")

Limitación conocida: banco y query usan las cajas de GT (no las cajas que
el tracker predijo), igual que new_id_born_crops en infer_track_reid_gaps.py
-- razonable porque el emparejamiento GT<->predicción exige IoU >=
--iou-match-thresh (0.5 default), pero no es 100% idéntico a lo que el
tracker "vio" en producción.

Uso:
    from train_track_query_reid import build_track_index, resolve_sequence_dirs
    from tracker_failure_dataset import load_required_reid_cases, TrackerFailureReIDDataset

    cases = load_required_reid_cases("reid_gap_report_bytetrack_test/required_reid_events.json")
    data_root, seq_dirs = resolve_sequence_dirs(None, "test")  # MISMO split que corrió infer_track_reid_gaps.py
    index = build_track_index(data_root, sequence_dirs=seq_dirs)
    dataset = TrackerFailureReIDDataset(
        index, cases, same_video_only=True,
        max_bank_size=16, num_track_history=4, query_wide_factor=9.0,
    )

También se puede correr standalone para inspeccionar visualmente los
episodios (ver --save-visuals):
    python3 tracker_failure_dataset.py \\
        --required-reid-events reid_gap_report_bytetrack_test/required_reid_events.json \\
        --split test --same-video-only --save-visuals --out-dir tracker_failures_bytetrack_test
"""
import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import torch

from reid_dataloader import DamageTrackIndex, IdentityTrack, ReIDEpisodeDataset


def load_new_id_born_cases(report_path: str) -> List[dict]:
    with open(report_path) as f:
        results = json.load(f)
    return [r for r in results if r["outcome"] == "new_id_born"]


def load_required_reid_cases(events_path: str) -> List[dict]:
    """Carga required_reid_events.json (ver _count_required_reidentifications
    en infer_track_reid_gaps.py) -- ya es una lista PLANA de casos (mismo
    shape que new_id_born: sequence/gt_identity_id/frame_before/frame_after/
    pred_id_before/pred_id_after), sin necesitar filtrar por outcome: es una
    definición más amplia que new_id_born -- CUALQUIER cambio de id
    (con o sin brecha temporal detectada), no solo cuando nace un id nuevo."""
    with open(events_path) as f:
        return json.load(f)


def _find_gap_index(track: IdentityTrack, frame_before: int, frame_after: int) -> Tuple[int, int]:
    """Devuelve (bank_end, query_idx): banco = observations[:bank_end] (todo
    el historial real hasta frame_before inclusive), query =
    observations[query_idx] (la observación en frame_after).

    frame_before/frame_after vienen de required_reid_events.json, que los
    registra consecutivos en la sub-secuencia filtrada por PREDICCIÓN
    EMPAREJADA (ver _count_required_reidentifications en
    infer_track_reid_gaps.py, que descarta observaciones sin predicción
    emparejada antes de comparar). track.observations, en cambio, solo
    filtra por visibilidad del GT -- puede tener observaciones intermedias
    (visibles pero sin predicción emparejada) entre frame_before y
    frame_after. Por eso se busca cada uno por VALOR de frame_idx, no
    asumiendo que son posiciones adyacentes en la lista (una versión
    anterior sí lo asumía: fallaba en el 100% de los huecos reales de 3-32
    frames y en la mayoría de los >32, porque casi siempre hay al menos una
    observación intermedia sin predicción emparejada en un hueco real)."""
    idx_before = idx_after = None
    for i, obs in enumerate(track.observations):
        if obs.frame_idx == frame_before:
            idx_before = i
        if obs.frame_idx == frame_after:
            idx_after = i
    if idx_before is None or idx_after is None or idx_after <= idx_before:
        raise ValueError(
            f"No se encontró el gap frame_before={frame_before} frame_after={frame_after} "
            f"en el track {track.identity_id} -- ¿el reid_gap_report.json corresponde a ESTE índice/split?"
        )
    return idx_before + 1, idx_after


class TrackerFailureReIDDataset(ReIDEpisodeDataset):
    """Un episodio por cada caso real (new_id_born o, con
    load_required_reid_cases, cualquier cambio de id) -- __len__ == cantidad
    de casos mapeados con éxito al track_index (sin sampling, sin
    repetición). Reusa _load_track_history_crops/_load_query_views de
    ReIDEpisodeDataset (mismo preprocesamiento que el resto del pipeline),
    pero arma el episodio a partir del punto de corte REAL detectado por el
    tracker, no de uno sampleado -- misma convención que val/test: el frame
    donde se detecta el fallo es la QUERY, y el historial del track ANTES
    de perderse es el banco (slot 0, el match correcto).

    same_video_only=True: el banco NUNCA se completa con distractores de
    OTRO video (sample_easy_distractors) -- solo con distractores del MISMO
    video (sample_hard_distractors), para que el dataset sea 100% real (sin
    ninguna imagen de otro video mezclada)."""

    def __init__(
        self, track_index: DamageTrackIndex, cases: List[dict], same_video_only: bool = False,
        hard_distractor_mode: str = "same_video", **kwargs,
    ):
        kwargs.pop("episodes_per_epoch", None)
        kwargs.pop("force_reid", None)
        self.same_video_only = same_video_only
        if hard_distractor_mode not in ("same_video", "same_panel"):
            raise ValueError(f"hard_distractor_mode desconocido: {hard_distractor_mode!r}")
        self.hard_distractor_mode = hard_distractor_mode  # "same_panel" = ablación M10/P4, ver sample_hard_distractors_same_panel
        super().__init__(track_index, episodes_per_epoch=max(len(cases), 1), force_reid=True, **kwargs)

        by_identity: Dict[str, IdentityTrack] = {t.identity_id: t for t in track_index.tracks}
        self.cases = []
        skipped = 0
        for case in cases:
            track = by_identity.get(case["gt_identity_id"])
            if track is None:
                skipped += 1
                continue
            try:
                bank_end, query_idx = _find_gap_index(track, case["frame_before"], case["frame_after"])
            except ValueError:
                skipped += 1
                continue
            self.cases.append((track, bank_end, query_idx, case))
        if skipped:
            print(
                f"[WARN] {skipped}/{len(cases)} casos new_id_born no se pudieron mapear al "
                f"track_index actual -- confirmá que es el mismo split que corrió infer_track_reid_gaps.py"
            )
        print(f"[OK] TrackerFailureReIDDataset: {len(self.cases)} episodios (casos new_id_born reales)")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int):
        if self.seed is not None:
            random.seed(self.seed + idx)

        reference_identity, bank_end, query_idx, case = self.cases[idx]
        before = reference_identity.observations[:bank_end]
        after_obs = reference_identity.observations[query_idx]

        bank_crops, bank_obs_masks, bank_video_ids = [], [], []
        crops, obs_mask = self._load_track_history_crops(before)
        bank_crops.append(crops)
        bank_obs_masks.append(obs_mask)
        bank_video_ids.append(reference_identity.video_id)
        true_identity_bank_pos = 0

        query_views = self._load_query_views(after_obs)
        query_video_id = reference_identity.video_id

        remaining_slots = self.max_bank_size - len(bank_crops)
        if self.same_video_only:
            n_hard, n_easy = remaining_slots, 0
        else:
            n_hard = int(remaining_slots * self.hard_distractor_ratio)
            n_easy = remaining_slots - n_hard
        if self.hard_distractor_mode == "same_panel":
            hard_distractors = self.track_index.sample_hard_distractors_same_panel(reference_identity, n_hard)
        else:
            hard_distractors = self.track_index.sample_hard_distractors(reference_identity, n_hard)
        easy_distractors = self.track_index.sample_easy_distractors(reference_identity, n_easy) if n_easy else []
        for distractor in hard_distractors + easy_distractors:
            d_crops, d_obs_mask = self._load_track_history_crops(distractor.observations)
            bank_crops.append(d_crops)
            bank_obs_masks.append(d_obs_mask)
            bank_video_ids.append(distractor.video_id)

        n_valid = len(bank_crops)
        order = list(range(n_valid))
        random.shuffle(order)
        bank_crops = [bank_crops[i] for i in order]
        bank_obs_masks = [bank_obs_masks[i] for i in order]
        bank_video_ids = [bank_video_ids[i] for i in order]
        true_identity_bank_pos = order.index(true_identity_bank_pos)

        bank_mask = torch.zeros(self.max_bank_size, dtype=torch.bool)
        bank_mask[:len(bank_crops)] = True
        while len(bank_crops) < self.max_bank_size:
            bank_crops.append(torch.zeros_like(bank_crops[0]))
            bank_obs_masks.append(torch.zeros_like(bank_obs_masks[0]))
            bank_video_ids.append("")
        bank_images = torch.stack(bank_crops, dim=0)
        bank_obs_mask_t = torch.stack(bank_obs_masks, dim=0)

        result = {
            "bank_images": bank_images,
            "bank_obs_mask": bank_obs_mask_t,
            "bank_mask": bank_mask,
            "query_views": query_views,
            "target": torch.tensor(true_identity_bank_pos, dtype=torch.long),
        }
        if self.return_video_ids:
            result["bank_video_ids"] = bank_video_ids
            result["query_video_id"] = query_video_id
        if self.return_debug_info:
            result["query0_meta"] = (after_obs.image_path, after_obs.bbox)
        return result


def main():
    from train_track_query_reid import build_track_index, resolve_sequence_dirs, visualize_episode
    from reid_dataloader import collate_episodes_with_video_ids
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reid-gap-report", help="Ruta a reid_gap_report.json -- casos new_id_born (ver infer_track_reid_gaps.py)")
    parser.add_argument("--required-reid-events", help="Ruta a required_reid_events.json -- CUALQUIER cambio de id, con o sin brecha (ver infer_track_reid_gaps.py)")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="MISMO split que se usó al correr infer_track_reid_gaps.py")
    parser.add_argument("--same-video-only", action="store_true", help="El banco NUNCA usa distractores de otro video (solo mismo-video)")
    parser.add_argument("--hard-distractor-mode", default="same_video", choices=["same_video", "same_panel"], help="same_panel prioriza distractores del mismo video que ADEMAS comparten part_label (ablación M10/P4)")
    parser.add_argument("--max-bank-size", type=int, default=16)
    parser.add_argument("--num-track-history", type=int, default=4)
    parser.add_argument("--query-wide-factor", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--out-dir", default="tracker_failure_episodes")
    args = parser.parse_args()
    if not args.reid_gap_report and not args.required_reid_events:
        raise SystemExit("Pasá --reid-gap-report o --required-reid-events")

    if args.required_reid_events:
        cases = load_required_reid_cases(args.required_reid_events)
        print(f"[INFO] {len(cases)} casos de reid requerido (cualquier cambio de id) en {args.required_reid_events}")
    else:
        cases = load_new_id_born_cases(args.reid_gap_report)
        print(f"[INFO] {len(cases)} casos new_id_born en {args.reid_gap_report}")

    data_root, seq_dirs = resolve_sequence_dirs(None, args.split)
    index = build_track_index(data_root, sequence_dirs=seq_dirs)

    dataset = TrackerFailureReIDDataset(
        index, cases, same_video_only=args.same_video_only, hard_distractor_mode=args.hard_distractor_mode,
        max_bank_size=args.max_bank_size, num_track_history=args.num_track_history,
        query_wide_factor=args.query_wide_factor, seed=args.seed, return_video_ids=True,
    )

    if args.save_visuals:
        os.makedirs(args.out_dir, exist_ok=True)
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_episodes_with_video_ids)
        for i, batch in enumerate(loader):
            out_path = os.path.join(args.out_dir, f"episode{i:04d}.jpg")
            visualize_episode(batch, 0, out_path)
        print(f"[OK] {len(dataset)} episodios guardados en {args.out_dir}/")


if __name__ == "__main__":
    main()
