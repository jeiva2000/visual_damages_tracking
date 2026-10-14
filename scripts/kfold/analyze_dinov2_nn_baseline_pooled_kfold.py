"""
COPIA EXPERIMENTAL de analyze_dinov2_nn_baseline.py (ver conversación) --
NO editar analyze_dinov2_nn_baseline.py en su lugar, este archivo existe
para no tocar el script compartido que usan embedding_analysis_dinov2_vs_p4.py,
plot_embedding_compression.py, plot_pca_crops_single_video.py,
plot_dendrogram_crops_single_video.py e investigate_c4_video6_confusion.py.

ÚNICO CAMBIO respecto al original: el protocolo de folds NO cambia (sigue
calibrando en train ~42-43 secuencias y evaluando en test ~10-11, rotando
los 6 folds, exactamente como analyze_dinov2_nn_baseline.py) -- lo que
cambia es la AGREGACIÓN final. En vez de calcular identity_acc/match_acc
por fold y promediar 6 números chicos y ruidosos (mean +/- std, el
comportamiento original), acá se acumulan los conteos crudos (aciertos y
total) de los 6 folds -- que nunca se solapan entre sí, porque los 6 test
son disjuntos y cubren el pool completo exactamente una vez -- y se calcula
UNA sola accuracy agregada sobre el total combinado, con su intervalo de
Wilson (misma técnica que ya usa el paper para proporciones con
denominadores chicos, ver Sección de significancia estadística).

Esto es "cross-validated prediction pooling" (como cross_val_predict de
scikit-learn): cada caso del pool se evalúa exactamente una vez, con un
umbral que nunca lo vio durante la calibración -- cero fuga de datos, cero
solapamiento, y una muestra del tamaño del dataset completo en vez de la
de un solo fold chico. Ver conversación: resuelve el problema que tenía la
otra copia experimental (analyze_dinov2_nn_baseline_inverted_calib.py, que
invertía los roles calibración/evaluación) -- ahí los 6 conjuntos de
evaluación quedaban ~5/6 del pool cada uno y se solapaban masivamente entre
sí, así que el "mean +/- std entre folds" ya no medía 6 muestras
independientes. Acá no hay ese problema porque no se toca qué evalúa cada
fold, solo cómo se combinan los resultados al final.

--- Docstring original de analyze_dinov2_nn_baseline.py, sin modificar ---

Baseline de 1-NN sobre embeddings CRUDOS de DINOv2 (última capa, CLS,
sin ninguna cabeza entrenada -- ni IntraTrackEncoder, ni
CrossAttentionMatcher, ni MatchScorer).

MODO RECOMENDADO -- --full-pipeline: corre todo el flujo unificado en una
sola invocación:
  1) Calibra el umbral de similitud en TRAIN (o reusa --match-threshold).
  2) Evalúa episodios sintéticos en val y test con ese umbral.
  3) Para cada tracker en --trackers (bytetrack/botsort): corre
     infer_track_reid_gaps.py sobre test (genera reid_gap_report.json +
     required_reid_events.json), arma TrackerFailureReIDDataset
     (same_video_only=True -- banco 100% real, nunca imágenes de otro
     video) con los casos reales de --case-definition, y evalúa con el
     MISMO umbral.
  4) Imprime una tabla resumen comparando todo (val, test, cada tracker).

Ejemplo:
    python3 analyze_dinov2_nn_baseline.py --full-pipeline \\
        --reid-gap-onnx checkpoints/rtdetrv2_v3_qcs_videos.onnx \\
        --output-dir dinov2_baseline_report --save-visuals

MODOS SUELTOS (para control fino / debug puntual, ver más abajo):

1) Default (sin --include-no-match): episodios con force_reid=True (SIEMPRE
   hay match real, igual que run_test_eval en train_track_query_reid.py) --
   mide solo identity_acc, sin necesitar ningún umbral de decisión.

2) --include-no-match: además evalúa la decisión binaria "hay match o no",
   igual que match_acc/identity_acc del modelo real. El coseno no tiene una
   clase "no match" como el `new_embedding` aprendido de MatchScorer, así
   que hace falta un UMBRAL de similitud -- se calibra sobre el split TRAIN
   (nunca sobre el split que se evalúa, para no hacer trampa): se generan
   episodios mezclados (REAL_REID/SYNTHETIC_SPLIT/NEW_TRACK, sin
   force_reid) y se busca el umbral que mejor separa "hay match" de
   "no hay match" en ese conjunto. Con ese umbral fijo, la decisión pasa a
   ser exactamente análoga al modelo real: la clase "no match" compite como
   un candidato más (con score = umbral) contra los candidatos reales del
   banco -- por eso identity_acc con este modo SÍ queda 100% comparable al
   oficial (antes, sin --include-no-match, había una asimetría: el baseline
   nunca podía "acobardarse" y predecir no-match habiendo match real).

Motivación: analyze_dinov2_layers.py mostró que el gap intra-identidad vs.
inter-identidad-mismo-video del embedding crudo es SIMILAR o MEJOR en
val/test que en train (train=0.2246, val=0.3874, test=0.2769) -- la señal
visual cruda generaliza bien. Si a pesar de eso el modelo entrenado (con
toda la cabeza) tiene id_acc mucho más bajo, hay evidencia de que la cabeza
está aportando ruido en vez de señal sobre este dataset chico.

Uso (modos sueltos):
    python3 analyze_dinov2_nn_baseline.py --split val
    python3 analyze_dinov2_nn_baseline.py --split test --include-no-match

    # Evaluar sobre los fallos REALES de un tracker (ver tracker_failure_dataset.py)
    # en vez de episodios sintéticos -- un episodio por cada new_id_born real:
    python3 analyze_dinov2_nn_baseline.py --split test \\
        --tracker-failures-json reid_gap_report_bytetrack_test/reid_gap_report.json

    # Version MAS AMPLIA (cualquier cambio de id, no solo new_id_born) y
    # 100% real (banco sin ninguna imagen de otro video):
    python3 analyze_dinov2_nn_baseline.py --split test --same-video-only \\
        --required-reid-events reid_gap_report_bytetrack_test/required_reid_events.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'reid_lib'))  # library modules
import argparse
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from p4_backbone_encoder import add_encoder_args, build_encoder
from reid_dataloader import ReIDEpisodeDataset, collate_episodes_with_video_ids
from train_track_query_reid import build_track_index, resolve_sequence_dirs, visualize_episode


@torch.no_grad()
def _encode_cls(model, images, device, batch_size=64, head=None):
    """head opcional (ver linear_reid_head.py): si se pasa, se aplica sobre
    el CLS antes de devolverlo -- proyección entrenada con metric learning
    en vez del CLS crudo de DINOv2."""
    outs = []
    for i in range(0, images.shape[0], batch_size):
        chunk = images[i:i + batch_size].to(device)
        cls = model(pixel_values=chunk).last_hidden_state[:, 0, :]
        if head is not None:
            cls = head(cls)
        outs.append(cls.cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def _apply_head_precomputed(embeds, device, head=None):
    """Igual que `_encode_cls` pero para embeddings YA calculados (ver
    --embedding-cache, embedding_cache en ReIDEpisodeDataset) -- no hay
    encoder que correr, `head` opcional se aplica directo sobre el embedding
    cacheado."""
    embeds = embeds.to(device)
    if head is not None:
        embeds = head(embeds)
    return embeds.cpu().numpy()


def _cosine_sim(a, b):
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return (a_n * b_n).sum(axis=-1)


def _build_loader(index, episodes, max_bank_size, num_track_history, query_wide_factor, force_reid, seed, batch_size, embedding_cache=None):
    dataset = ReIDEpisodeDataset(
        index, episodes_per_epoch=episodes, max_bank_size=max_bank_size,
        num_track_history=num_track_history, query_wide_factor=query_wide_factor,
        force_reid=force_reid, synthetic_gap_max=0, seed=seed, return_video_ids=True,
        embedding_cache=embedding_cache,
    )
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_episodes_with_video_ids)


def _fuse_query_views(query_views_embeds, query_fusion):
    """(B,2,D) -> (B,D) [o (B,2D) solo para 'concat'] combinando las 2 vistas
    de la query (índice 0=ajustada/near, índice 1=amplia/wide, ver
    _load_query_views en reid_dataloader.py) según --query-fusion. Ver el
    docstring de ese flag (main(), analyze_dinov2_nn_baseline.py) -- 'mean'
    es el comportamiento de siempre, los demás son ablaciones M10/P3."""
    if query_fusion == "mean":
        return query_views_embeds.mean(axis=1)
    if query_fusion == "near":
        return query_views_embeds[:, 0, :]
    if query_fusion == "wide":
        return query_views_embeds[:, 1, :]
    if query_fusion == "max":
        return query_views_embeds.max(axis=1)
    if query_fusion == "concat":
        B = query_views_embeds.shape[0]
        return query_views_embeds.reshape(B, -1)
    raise ValueError(f"query_fusion desconocido: {query_fusion}")


def _embed_batch(model, batch, device, head=None, query_fusion="mean"):
    """(track_summary (B,S,D) [o (B,S,2D) si query_fusion='concat'],
    query_embeds (B,D) [o (B,2D)], bank_mask (B,S) numpy, S).

    Con --embedding-cache activo, `bank_images`/`query_views` ya son
    embeddings (B,S,K,D)/(B,2,D) en vez de píxeles (B,S,K,C,H,W)/(B,2,C,H,W)
    -- ver embedding_cache en ReIDEpisodeDataset (reid_dataloader.py). Se
    distingue por dim(): 6 = píxeles (pasa por el encoder), 4 = ya
    encodeados (se usa directo, sin llamar a `model`)."""
    bank_images = batch["bank_images"]
    bank_obs_mask = batch["bank_obs_mask"].numpy()
    bank_mask = batch["bank_mask"].numpy()
    query_views = batch["query_views"]

    if bank_images.dim() == 6:
        B, S, K, C, H, W = bank_images.shape
        bank_embeds = _encode_cls(model, bank_images.view(B * S * K, C, H, W), device, head=head).reshape(B, S, K, -1)
        query_views_embeds = _encode_cls(model, query_views.view(B * 2, C, H, W), device, head=head).reshape(B, 2, -1)
    else:
        B, S, K, D = bank_images.shape
        bank_embeds = _apply_head_precomputed(bank_images.view(B * S * K, D), device, head=head).reshape(B, S, K, -1)
        query_views_embeds = _apply_head_precomputed(query_views.view(B * 2, D), device, head=head).reshape(B, 2, -1)
    query_embeds = _fuse_query_views(query_views_embeds, query_fusion)

    masked = bank_embeds * bank_obs_mask[..., None]
    counts = bank_obs_mask.sum(axis=2, keepdims=True)
    track_summary = masked.sum(axis=2) / np.clip(counts, 1, None)
    if query_fusion == "concat":
        # el banco no tiene una vista ajustada/amplia separada por slot (ya
        # es un promedio pooled de su propio historial, ver pool_history) --
        # se duplica consigo mismo para que la dimensión siga siendo
        # comparable contra la query 2xD via cosine_sim.
        track_summary = np.concatenate([track_summary, track_summary], axis=-1)
    return track_summary, query_embeds, bank_mask, S


def _best_candidate(query_vec, track_summary_b, bank_mask_b, sims=None):
    """valid_slots, sims (una por slot valido), mejor slot, mejor score.

    `sims`, si se pasa, es un array YA CALCULADO de largo S (uno por slot,
    invalidos en cualquier valor -- se restringe a valid_slots acá) --
    lo usa `_score_with_attn_scorer` (ver bilinear_attention_scorer.py)
    para reemplazar la cosine cruda por `score_mixed` sin duplicar la
    lógica de argmax/slots-validos. Default None: cosine cruda de siempre."""
    valid_slots = np.where(bank_mask_b)[0]
    if sims is None:
        sims = _cosine_sim(query_vec[None, :], track_summary_b[valid_slots])
    else:
        sims = np.asarray(sims)[valid_slots]
    best_local = int(np.argmax(sims))
    return valid_slots, sims, int(valid_slots[best_local]), float(sims[best_local])


def _score_with_attn_scorer(attn_scorer, query_embeds, track_summary, bank_mask, device):
    """Corre BilinearAttentionScorer (ver bilinear_attention_scorer.py) sobre
    un batch ENTERO de una sola vez (no por episodio) -- convierte los
    arrays numpy que ya devuelve `_embed_batch` (CLS crudo de DINOv2, sin
    head) a tensores torch, y devuelve `score_mixed` como numpy (B, S) para
    pasarlo a `_best_candidate` por episodio.

    `isolated` se lee del propio checkpoint (ver load_scorer) -- si se
    entrenó con --isolated-scorer, evalúa automáticamente con los logits
    crudos de atención (sin mezcla de coseno/gate), sin necesitar una flag
    de CLI aparte acá."""
    with torch.no_grad():
        query_t = torch.from_numpy(query_embeds).float().to(device)
        bank_t = torch.from_numpy(track_summary).float().to(device)
        mask_t = torch.from_numpy(bank_mask).to(device)
        score_mixed, _, _ = attn_scorer(query_t, bank_t, mask_t, isolated=getattr(attn_scorer, "isolated", False))
    return score_mixed.cpu().numpy()


def _calibrate_threshold(model, train_index, args, head=None, attn_scorer=None):
    """Umbral de similitud que mejor separa match/no-match en TRAIN (nunca
    en el split evaluado). Barre todos los valores de mejor-score observados
    y se queda con el que maximiza la accuracy de la decisión binaria."""
    loader = _build_loader(
        train_index, args.num_calibration_episodes, args.max_bank_size,
        args.num_track_history, args.query_wide_factor,
        force_reid=False, seed=args.calibration_seed, batch_size=args.batch_size,
        embedding_cache=args.embedding_cache,
    )
    best_sims, is_match = [], []
    for batch in loader:
        track_summary, query_embeds, bank_mask, S = _embed_batch(model, batch, args.device, head=head, query_fusion=args.query_fusion)
        sims_batch = (
            _score_with_attn_scorer(attn_scorer, query_embeds, track_summary, bank_mask, args.device)
            if attn_scorer is not None else None
        )
        target = batch["target"]
        for b in range(query_embeds.shape[0]):
            sims_b = sims_batch[b] if sims_batch is not None else None
            _, _, _, best_sim = _best_candidate(query_embeds[b], track_summary[b], bank_mask[b], sims=sims_b)
            best_sims.append(best_sim)
            is_match.append(int(target[b].item()) < S)
    best_sims = np.array(best_sims)
    is_match = np.array(is_match)

    best_thr, best_acc = 0.0, -1.0
    for t in np.unique(best_sims):
        acc = ((best_sims >= t) == is_match).mean()
        if acc > best_acc:
            best_thr, best_acc = float(t), float(acc)
    n_match = int(is_match.sum())
    print(
        f"[CALIBRACION] {len(is_match)} episodios de train ({n_match} con match, "
        f"{len(is_match) - n_match} no-match) -> umbral={best_thr:.4f} "
        f"(accuracy binaria en calibración={best_acc:.4f})"
    )
    return best_thr


def _apply_tracker_hparams(tracker_args, tracker):
    """Si se pasó --reid-gap-tracker-params-json, sobreescribe los
    --reid-gap-track-thresh/etc. FLAT de `tracker_args` con los valores
    específicos de `tracker` en ese JSON (formato
    `{"bytetrack": {...}, "botsort": {...}}`, ver
    eval_tracking/sweep_ultralytics_anchor.py) -- necesario porque ByteTrack
    y BotSort quedan tuneados con valores DISTINTOS, pero
    --reid-gap-track-thresh/etc. son un solo flag global (no hay uno por
    tracker). Sin este JSON, no hace nada (quedan los defaults/flags planos
    de siempre, iguales para ambos trackers)."""
    per_tracker = getattr(tracker_args, "reid_gap_tracker_params", None)
    if not per_tracker or tracker not in per_tracker:
        return
    overrides = per_tracker[tracker]
    for key, val in overrides.items():
        attr = f"reid_gap_{key}"
        if not hasattr(tracker_args, attr):
            raise ValueError(f"--reid-gap-tracker-params-json: clave desconocida '{key}' para tracker={tracker}")
        setattr(tracker_args, attr, val)


def _generate_reid_gap_report(args, outdir, sequence_dirs=None):
    """Corre infer_track_reid_gaps.py (detector + tracker clasico ByteTrack/
    BotSort) sobre el split indicado en args.split -- idempotente (ver
    run_reid_gap_report: si outdir/reid_gap_report.json ya existe, no
    reprocesa salvo --reid-gap-force). Devuelve la ruta al
    reid_gap_report.json (required_reid_events.json queda en el mismo
    outdir, como side-effect).

    `sequence_dirs`, si se pasa, reemplaza la resolución por args.split --
    lo usa run_kfold_cv para generar el report UNA VEZ sobre las secuencias
    del pool de folds (kfold_utils.list_all_sequences, train+test, ver
    docstring de ese módulo), en vez de una vez por fold (ver
    _filter_cases_by_sequences)."""
    from infer_track_reid_gaps import run_reid_gap_report  # import diferido -- requiere onnxruntime/boxmot

    return run_reid_gap_report(
        detector=args.reid_gap_detector, onnx_path=args.reid_gap_onnx, provider=args.reid_gap_provider,
        det_thresh=args.reid_gap_det_thresh, tracker=args.reid_gap_tracker,
        no_native_reid=args.reid_gap_no_native_reid, reid_weights=args.reid_gap_reid_weights,
        device=args.reid_gap_device, split=args.split, data_root=None,
        gt_conf_thresh=0.0, gt_class_id=-1, gt_visibility_thresh=0.0,
        min_gap_frames=args.reid_gap_min_gap_frames, iou_match_thresh=args.reid_gap_iou_match_thresh,
        max_sequences=args.reid_gap_max_sequences, outdir=outdir, force=args.reid_gap_force,
        track_thresh=args.reid_gap_track_thresh, match_thresh=args.reid_gap_match_thresh,
        track_buffer=args.reid_gap_track_buffer, new_track_thresh=args.reid_gap_new_track_thresh,
        proximity_thresh=args.reid_gap_proximity_thresh, appearance_thresh=args.reid_gap_appearance_thresh,
        cmc_method=args.reid_gap_cmc_method,
        min_hits=args.reid_gap_min_hits, iou_threshold=args.reid_gap_iou_threshold,
        asso_func=args.reid_gap_asso_func, delta_t=args.reid_gap_delta_t, inertia=args.reid_gap_inertia,
        w_association_emb=args.reid_gap_w_association_emb, alpha_fixed_emb=args.reid_gap_alpha_fixed_emb,
        aw_param=args.reid_gap_aw_param,
        sequence_dirs=sequence_dirs,
    )


def _filter_cases_by_sequences(cases, allowed_seq_names):
    """Filtra una lista de casos (new_id_born o required_reid, ambos con
    case["sequence"] = nombre de secuencia) contra un conjunto de nombres
    permitidos -- usa run_kfold_cv para recortar los casos GLOBALES (de las
    72 secuencias) a los del test de un fold en particular, sin tener que
    volver a correr detector+tracker por fold."""
    allowed = set(allowed_seq_names)
    return [c for c in cases if c["sequence"] in allowed]


def _filter_cases_by_gap_length(cases, bin_name):
    """Filtra casos required_reid por duración del hueco
    (frame_after - frame_before) -- ablación de estratificación por
    longitud (P3 paperreview.ai). bin_name='all' (default) no filtra nada,
    comportamiento idéntico al de siempre. Los bins replican los pedidos
    por el revisor (3-8/9-32/>32) más 'lt3' (cambios de id SIN hueco
    temporal real detectado, la mayoría de required_reid -- ver
    docstring de _count_required_reidentifications en
    infer_track_reid_gaps.py)."""
    if bin_name == "all":
        return cases

    def _in_bin(c):
        length = c["frame_after"] - c["frame_before"]
        if bin_name == "lt3":
            return length < 3
        if bin_name == "3-8":
            return 3 <= length <= 8
        if bin_name == "9-32":
            return 9 <= length <= 32
        if bin_name == "gt32":
            return length > 32
        raise ValueError(f"--gap-length-bin desconocido: {bin_name!r}")

    return [c for c in cases if _in_bin(c)]


def _run_reid_gap_baseline(args):
    """Genera (o reusa) el reid_gap_report.json y devuelve los conteos de
    reidentificacion (reidentified/new_id_born/id_swap/unevaluated) para
    comparar lado a lado en el resumen final -- un caso de reidentificacion
    fallida es cuando el tracker crea un id nuevo (`new_id_born`) en vez de
    reconocer el track que ya tenia."""
    outdir = args.reid_gap_outdir or f"reid_gap_report_{args.reid_gap_tracker}_{args.split}"
    report_path = _generate_reid_gap_report(args, outdir)
    with open(report_path) as f:
        results = json.load(f)
    counts = {}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    return counts


def _evaluate_loader(
    model, loader, device, include_no_match, threshold, heatmap_temperature,
    save_visuals=False, fails_dir=None, correct_dir=None, max_fails=float("inf"), max_correct=float("inf"),
    head=None, attn_scorer=None, query_fusion="mean",
):
    """Corre el 1-NN (crudo, proyectado si se pasa `head`, o vía
    BilinearAttentionScorer si se pasa `attn_scorer` -- mutuamente
    excluyentes, ver main()) sobre todos los episodios de `loader` y
    devuelve un dict de métricas -- núcleo compartido por los modos sueltos
    y por --full-pipeline (val/test/cada tracker), para no repetir la
    lógica de decisión en cada lugar."""
    if save_visuals:
        os.makedirs(fails_dir, exist_ok=True)
        os.makedirs(correct_dir, exist_ok=True)

    n_total = 0
    n_match_correct = 0
    n_identity_total = 0
    n_identity_correct = 0
    wrong_identity_intra = 0   # match real, eligió OTRO candidato del mismo video
    wrong_identity_inter = 0   # match real, eligió un candidato de OTRO video
    false_negative = 0         # match real, predijo "no match"
    false_positive_intra = 0   # no-match real (daño nuevo), predijo match con el mismo video
    false_positive_inter = 0   # no-match real (daño nuevo), predijo match con otro video
    n_fails_saved = 0
    n_correct_saved = 0

    for batch in loader:
        track_summary, query_embeds, bank_mask, S = _embed_batch(model, batch, device, head=head, query_fusion=query_fusion)
        sims_batch = (
            _score_with_attn_scorer(attn_scorer, query_embeds, track_summary, bank_mask, device)
            if attn_scorer is not None else None
        )
        target = batch["target"]
        bank_video_ids = batch["bank_video_ids"]
        query_video_id = batch["query_video_id"]

        for b in range(query_embeds.shape[0]):
            true_slot = int(target[b].item())
            true_is_match = true_slot < S
            sims_b = sims_batch[b] if sims_batch is not None else None
            valid_slots, sims, best_slot, best_sim = _best_candidate(
                query_embeds[b], track_summary[b], bank_mask[b], sims=sims_b,
            )

            if include_no_match:
                pred_is_match = best_sim >= threshold
                pred_slot = best_slot if pred_is_match else S
            else:
                if not true_is_match:
                    continue  # no debería pasar sin include_no_match (force_reid=True)
                pred_is_match = True
                pred_slot = best_slot

            n_total += 1
            if include_no_match:
                n_match_correct += int(pred_is_match == true_is_match)

            if true_is_match:
                n_identity_total += 1
                is_correct = pred_slot == true_slot
                if is_correct:
                    n_identity_correct += 1
                elif pred_is_match:
                    pred_video = bank_video_ids[b][pred_slot]
                    if pred_video == query_video_id[b]:
                        wrong_identity_intra += 1
                    else:
                        wrong_identity_inter += 1
                else:
                    false_negative += 1  # había match real, predijo "no match"
            else:
                is_correct = not pred_is_match
                if pred_is_match:  # falso positivo sobre un daño genuinamente nuevo
                    pred_video = bank_video_ids[b][pred_slot]
                    if pred_video == query_video_id[b]:
                        false_positive_intra += 1
                    else:
                        false_positive_inter += 1

            if save_visuals:
                # logits "falsos" solo para reusar el heatmap de visualize_episode:
                # -inf en slots invalidos, similitud coseno cruda en los validos, y
                # el umbral calibrado (si aplica) como score del slot "no match" --
                # asi el argmax de este vector reproduce exactamente pred_slot/
                # pred_is_match de arriba. Todo dividido por la temperatura antes
                # del softmax, porque las similitudes crudas quedan muy juntas.
                fake_logits = torch.full((S + 1,), float("-inf"))
                fake_logits[torch.from_numpy(valid_slots)] = torch.from_numpy(sims).float()
                if include_no_match:
                    fake_logits[S] = threshold
                fake_logits = fake_logits / heatmap_temperature
                if is_correct and n_correct_saved < max_correct:
                    out_path = os.path.join(correct_dir, f"correct{n_correct_saved:04d}.jpg")
                    visualize_episode(batch, b, out_path, pred=pred_slot, logits=fake_logits)
                    n_correct_saved += 1
                elif not is_correct and n_fails_saved < max_fails:
                    out_path = os.path.join(fails_dir, f"fail{n_fails_saved:04d}.jpg")
                    visualize_episode(batch, b, out_path, pred=pred_slot, logits=fake_logits)
                    n_fails_saved += 1

    identity_acc = n_identity_correct / n_identity_total if n_identity_total else float("nan")
    match_acc = (n_match_correct / n_total) if (include_no_match and n_total) else None
    n_no_match_total = n_total - n_identity_total
    return {
        "n_total": n_total,
        "identity_acc": identity_acc,
        "n_identity_total": n_identity_total,
        "n_identity_correct": n_identity_correct,  # crudo -- para poder POOLEAR entre folds (ver docstring del módulo)
        "n_no_match_total": n_no_match_total,
        "match_acc": match_acc,
        "n_match_correct": n_match_correct if include_no_match else None,  # crudo -- ídem
        "wrong_identity_intra": wrong_identity_intra,
        "wrong_identity_inter": wrong_identity_inter,
        "false_negative": false_negative,
        "false_positive_intra": false_positive_intra,
        "false_positive_inter": false_positive_inter,
        "n_correct_saved": n_correct_saved if save_visuals else None,
        "n_fails_saved": n_fails_saved if save_visuals else None,
    }


def _print_metrics(label, metrics, threshold, include_no_match, fuente=None):
    fuente_str = f" fuente={fuente}" if fuente else ""
    print(f"\n[{label}]{fuente_str} n_evaluados={metrics['n_total']}")
    if include_no_match:
        print(f"  umbral usado = {threshold:.4f}")
        print(f"  match_acc = {metrics['match_acc']:.4f}")
    print(f"  identity_acc = {metrics['identity_acc']:.4f}  (n={metrics['n_identity_total']} episodios con match real)")

    wrong_identity = metrics["wrong_identity_intra"] + metrics["wrong_identity_inter"]
    false_positive = metrics["false_positive_intra"] + metrics["false_positive_inter"]
    if wrong_identity:
        print(
            f"  identidad incorrecta (con match real, n={wrong_identity}): "
            f"intra_seq={metrics['wrong_identity_intra'] / wrong_identity:.4f}  "
            f"inter_seq={metrics['wrong_identity_inter'] / wrong_identity:.4f}"
        )
    if metrics["false_negative"]:
        print(
            f"  falso negativo (match real, predijo no-match, n={metrics['false_negative']}) -- "
            f"de {metrics['n_identity_total']} episodios con match real "
            f"({metrics['false_negative'] / metrics['n_identity_total']:.4f})"
        )
    if false_positive:
        n_no_match = metrics["n_no_match_total"]
        rate_str = f" de {n_no_match} episodios sin match ({false_positive / n_no_match:.4f})" if n_no_match else ""
        print(
            f"  falso positivo (daño nuevo, predijo match, n={false_positive}){rate_str}: "
            f"intra_seq={metrics['false_positive_intra'] / false_positive:.4f}  "
            f"inter_seq={metrics['false_positive_inter'] / false_positive:.4f}"
        )
    if metrics["n_correct_saved"] is not None:
        print(f"  [OK] {metrics['n_correct_saved']} aciertos / {metrics['n_fails_saved']} fallos guardados como imagen")


def _evaluate_synthetic_split(model, index, args, threshold, head, output_dir, label, attn_scorer=None):
    """Episodios sintéticos (force_reid=False, mezcla REAL_REID/SYNTHETIC_SPLIT/
    NEW_TRACK) sobre `index` -- extraído del paso 1 de run_full_pipeline para
    poder llamarlo también por fold (test_index) desde run_kfold_cv, con
    `output_dir`/`label` explícitos en vez de args.output_dir/split fijo (así
    cada fold guarda sus imágenes de fails/correct en su propia carpeta)."""
    max_fails = args.max_fails if args.max_fails > 0 else float("inf")
    max_correct = args.max_correct if args.max_correct > 0 else float("inf")
    loader = _build_loader(
        index, args.num_episodes, args.max_bank_size, args.num_track_history,
        args.query_wide_factor, force_reid=False, seed=args.seed, batch_size=args.batch_size,
        embedding_cache=args.embedding_cache,
    )
    metrics = _evaluate_loader(
        model, loader, args.device, include_no_match=True, threshold=threshold,
        heatmap_temperature=args.heatmap_temperature, save_visuals=args.save_visuals,
        fails_dir=os.path.join(output_dir, f"fails_{label}"),
        correct_dir=os.path.join(output_dir, f"correct_{label}"),
        max_fails=max_fails, max_correct=max_correct, head=head, attn_scorer=attn_scorer,
        query_fusion=args.query_fusion,
    )
    _print_metrics(f"BASELINE split={label}", metrics, threshold, True, fuente="episodios sintéticos")
    return metrics


def _evaluate_tracker_on_cases(
    model, args, tracker, index, cases, threshold, head, output_dir, label, fuente_suffix="", attn_scorer=None,
):
    """Evalúa episodios armados a partir de casos reales YA CARGADOS (ver
    tracker_failure_dataset.py) -- extraído del paso 2 de run_full_pipeline
    para poder alimentarse tanto de un reid_gap_report fresco de un split
    (run_full_pipeline) como de casos globales pre-filtrados por fold (ver
    _filter_cases_by_sequences en run_kfold_cv), sin volver a generar el
    reid_gap_report por cada llamada. Devuelve None (y solo imprime un
    warning) si `cases` viene vacío -- mismo comportamiento que antes."""
    from tracker_failure_dataset import TrackerFailureReIDDataset  # import diferido, igual que en run_full_pipeline

    if not cases:
        print(f"[WARN] 0 casos para {label} -- se omite la evaluación de DINOv2.")
        return None

    max_fails = args.max_fails if args.max_fails > 0 else float("inf")
    max_correct = args.max_correct if args.max_correct > 0 else float("inf")
    dataset = TrackerFailureReIDDataset(
        index, cases, same_video_only=True, hard_distractor_mode=args.hard_distractor_mode,
        max_bank_size=args.max_bank_size, num_track_history=args.num_track_history,
        query_wide_factor=args.query_wide_factor, seed=args.seed, return_video_ids=True,
        embedding_cache=args.embedding_cache,
    )
    if len(dataset) == 0:
        print(f"[WARN] 0/{len(cases)} casos de {label} se pudieron mapear al track_index actual -- se omite la evaluación.")
        return None
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_episodes_with_video_ids)
    metrics = _evaluate_loader(
        model, loader, args.device, include_no_match=True, threshold=threshold,
        heatmap_temperature=args.heatmap_temperature, save_visuals=args.save_visuals,
        fails_dir=os.path.join(output_dir, f"fails_{label}"),
        correct_dir=os.path.join(output_dir, f"correct_{label}"),
        max_fails=max_fails, max_correct=max_correct, head=head, attn_scorer=attn_scorer,
        query_fusion=args.query_fusion,
    )
    _print_metrics(
        f"BASELINE tracker={tracker} label={label}", metrics, threshold, True,
        fuente=f"casos reales ({args.case_definition}, same_video_only=True, hard_distractor_mode={args.hard_distractor_mode}, gap_length_bin={args.gap_length_bin}){fuente_suffix}",
    )
    return metrics


def run_full_pipeline(args, model, head=None, attn_scorer=None):
    """Flujo unificado: calibra en train, evalúa val/test sintéticos, y
    evalúa cada tracker de --trackers sobre sus fallos REALES en test --
    todo con el MISMO umbral, para que las cuatro (o más) evaluaciones sean
    directamente comparables entre sí."""
    if not args.reid_gap_onnx:
        raise SystemExit("--full-pipeline requiere --reid-gap-onnx (checkpoint ONNX del detector, para correr infer_track_reid_gaps.py)")

    if args.match_threshold is not None:
        threshold = args.match_threshold
        print(f"[INFO] Usando umbral fijo pasado por CLI: {threshold:.4f}")
    else:
        train_root, train_seq_dirs = resolve_sequence_dirs(None, "train")
        train_index = build_track_index(train_root, sequence_dirs=train_seq_dirs)
        threshold = _calibrate_threshold(model, train_index, args, head=head, attn_scorer=attn_scorer)

    os.makedirs(args.output_dir, exist_ok=True)
    max_fails = args.max_fails if args.max_fails > 0 else float("inf")
    max_correct = args.max_correct if args.max_correct > 0 else float("inf")
    all_metrics = {}

    # 1) val y test, episodios sinteticos -- de paso guarda (index, seq_dirs)
    # por split para reusarlos en el paso 2 (mismo split, no hace falta
    # reconstruir el track_index de nuevo por cada tracker).
    split_data = {}
    for split in ("val", "test"):
        data_root, seq_dirs = resolve_sequence_dirs(None, split)
        index = build_track_index(data_root, sequence_dirs=seq_dirs)
        split_data[split] = (index, seq_dirs)
        all_metrics[split] = _evaluate_synthetic_split(
            model, index, args, threshold, head, args.output_dir, split, attn_scorer=attn_scorer,
        )

    # 2) cada tracker, fallos reales sobre cada split de --reid-gap-splits
    # (val y test por default) + una fila "combinada" que junta los casos de
    # AMBOS splits en un solo pool -- muestra más grande, más confiable que
    # cualquiera de los dos splits por separado.
    from tracker_failure_dataset import load_new_id_born_cases, load_required_reid_cases

    for tracker in args.trackers:
        print(f"\n[INFO] ==== Tracker: {tracker} ====")
        pooled_cases = []
        pooled_seq_dirs = []

        for split in args.reid_gap_splits:
            print(f"[INFO] -- split={split} --")
            index, seq_dirs = split_data[split]
            tracker_outdir = os.path.join(args.output_dir, f"reid_gap_{tracker}_{split}")
            tracker_args = argparse.Namespace(**vars(args))
            tracker_args.reid_gap_tracker = tracker
            tracker_args.split = split
            _apply_tracker_hparams(tracker_args, tracker)
            report_path = _generate_reid_gap_report(tracker_args, tracker_outdir)

            if args.case_definition == "required_reid":
                events_path = os.path.join(tracker_outdir, "required_reid_events.json")
                cases = load_required_reid_cases(events_path)
                print(f"[INFO] {len(cases)} casos de reid requerido (cualquier cambio de id) para {tracker}/{split}")
            else:
                cases = load_new_id_born_cases(report_path)
                print(f"[INFO] {len(cases)} casos new_id_born para {tracker}/{split}")

            pooled_cases.extend(cases)
            pooled_seq_dirs.extend(seq_dirs)

            label = f"{tracker}_{split}"
            all_metrics[label] = _evaluate_tracker_on_cases(
                model, args, tracker, index, cases, threshold, head, args.output_dir, label,
                attn_scorer=attn_scorer,
            )

        # fila combinada: pool de TODOS los splits evaluados para este tracker
        if len(args.reid_gap_splits) > 1:
            combined_label = f"{tracker}_combined"
            if not pooled_cases:
                print(f"[WARN] 0 casos combinados para {tracker} -- se omite.")
                all_metrics[combined_label] = None
                continue
            combined_index = build_track_index(None, sequence_dirs=pooled_seq_dirs)
            all_metrics[combined_label] = _evaluate_tracker_on_cases(
                model, args, tracker, combined_index, pooled_cases, threshold, head, args.output_dir, combined_label,
                fuente_suffix=f", pool de {args.reid_gap_splits}", attn_scorer=attn_scorer,
            )

    # 3) tabla resumen
    print(f"\n[RESUMEN FINAL] umbral usado en todas las evaluaciones: {threshold:.4f}")
    print(f"{'fuente':<20s} {'n':>6s} {'match_acc':>10s} {'identity_acc':>13s}")
    for label, m in all_metrics.items():
        if m is None:
            print(f"{label:<20s} {'--':>6s} {'--':>10s} {'--':>13s}")
            continue
        match_str = f"{m['match_acc']:.4f}" if m["match_acc"] is not None else "-"
        id_str = "-" if math.isnan(m["identity_acc"]) else f"{m['identity_acc']:.4f}"
        print(f"{label:<20s} {m['n_total']:>6d} {match_str:>10s} {id_str:>13s}")
    print(
        "\n[INFO] Todas las filas usan el MISMO umbral -- comparables entre sí. Si "
        "match_acc/identity_acc de val/test ya superan claramente a los de la cabeza "
        "entrenada real (ver logs de train_track_query_reid.py), y los trackers muestran "
        "que el baseline rescataría la mayoría de sus fallos reales, es evidencia fuerte "
        "de simplificar la cabeza en vez de seguir invirtiendo en el encoder."
    )


def _mean_std(values):
    """Media y desvío estándar, ignorando None/NaN (folds sin ningún caso
    real para un tracker, o sin ningún episodio con match, quedan afuera del
    promedio en vez de contaminarlo con NaN)."""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return float("nan"), float("nan")
    arr = np.array(clean, dtype=float)
    return float(arr.mean()), float(arr.std())


def _print_kfold_summary(fold_results, args):
    """Tabla agregada media +/- desvío estándar entre folds -- la métrica a
    reportar en el paper es esta, no ningún fold individual (ver docstring
    de run_kfold_cv)."""
    k = len(fold_results)
    print(f"\n[RESUMEN K-FOLD] {k} folds (num_folds={args.num_folds}, seed={args.kfold_seed})")

    thr_mean, thr_std = _mean_std([r["threshold"] for r in fold_results])
    val_loss_mean, val_loss_std = _mean_std([r["best_val_loss"] for r in fold_results])
    print(f"  umbral calibrado: {thr_mean:.4f} +/- {thr_std:.4f}")
    print(f"  best_val_loss:    {val_loss_mean:.4f} +/- {val_loss_std:.4f}")

    print(f"\n{'fuente':<20s} {'match_acc':>20s} {'identity_acc':>20s}")

    def _row(label, metrics_list):
        m_mean, m_std = _mean_std([m["match_acc"] if m else None for m in metrics_list])
        i_mean, i_std = _mean_std([m["identity_acc"] if m else None for m in metrics_list])
        m_str = f"{m_mean:.4f} +/- {m_std:.4f}" if not math.isnan(m_mean) else "--"
        i_str = f"{i_mean:.4f} +/- {i_std:.4f}" if not math.isnan(i_mean) else "--"
        print(f"{label:<20s} {m_str:>20s} {i_str:>20s}")

    _row("synthetic_test", [r["synthetic"] for r in fold_results])
    for tracker in args.trackers:
        _row(tracker, [r["trackers"].get(tracker) for r in fold_results])

    print(
        f"\n[INFO] Cada uno de los {k} folds entrena/calibra/evalúa de forma independiente "
        "(nunca ve su propio test durante entrenamiento ni calibración de umbral) -- la "
        "media +/- desvío estándar de arriba es la métrica a reportar en el paper, no el "
        "resultado de ningún fold individual (ver kfold_results.json para el detalle completo)."
    )


def _wilson_interval(x, n, z=1.96):
    """Intervalo de Wilson para una proporción x/n -- misma fórmula que ya
    usa el paper para las proporciones con denominadores chicos (ver
    Sección de significancia estadística). z=1.96 -> 95%."""
    if not n:
        return float("nan"), float("nan"), float("nan")
    p = x / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return p, (center - half) / denom, (center + half) / denom


def _print_pooled_summary(fold_results, args):
    """AGREGACIÓN ALTERNATIVA a _print_kfold_summary -- en vez de promediar
    la accuracy de cada fold (6 números chicos y ruidosos), acumula los
    conteos crudos (aciertos y total) de los 6 folds -- que nunca se
    solapan entre sí, ver docstring del módulo -- y calcula UNA accuracy
    agregada sobre el total combinado, con su intervalo de Wilson. Cada
    caso del pool completo se evaluó exactamente una vez, con un umbral
    calibrado en datos que nunca lo incluyeron (cross-validated prediction
    pooling, como cross_val_predict de scikit-learn)."""
    print(f"\n[RESUMEN K-FOLD POOLEADO] {len(fold_results)} folds combinados (num_folds={args.num_folds}, seed={args.kfold_seed})")
    print(f"{'fuente':<20s} {'match_acc (n)':>28s} {'identity_acc (n)':>32s}")

    def _pooled_row(label, metrics_list):
        clean = [m for m in metrics_list if m]
        n_id_correct = sum(m["n_identity_correct"] for m in clean)
        n_id_total = sum(m["n_identity_total"] for m in clean)
        id_p, id_lo, id_hi = _wilson_interval(n_id_correct, n_id_total)

        match_terms = [m for m in clean if m.get("n_match_correct") is not None]
        if match_terms:
            n_m_correct = sum(m["n_match_correct"] for m in match_terms)
            n_m_total = sum(m["n_total"] for m in match_terms)
            m_p, m_lo, m_hi = _wilson_interval(n_m_correct, n_m_total)
            m_str = f"{m_p:.4f} [{m_lo:.4f},{m_hi:.4f}] (n={n_m_total})"
        else:
            m_str = "--"
        id_str = f"{id_p:.4f} [{id_lo:.4f},{id_hi:.4f}] (n={n_id_total})"
        print(f"{label:<20s} {m_str:>28s} {id_str:>32s}")
        return {"n_identity_correct": n_id_correct, "n_identity_total": n_id_total, "identity_acc": id_p}

    pooled = {"synthetic_test": _pooled_row("synthetic_test", [r["synthetic"] for r in fold_results])}
    for tracker in args.trackers:
        pooled[tracker] = _pooled_row(tracker, [r["trackers"].get(tracker) for r in fold_results])

    print(
        "\n[INFO] A diferencia del resumen de arriba (media +/- desvío entre folds), esta tabla "
        "junta los casos de los 6 folds en un solo pozo -- cada caso del pool completo se evaluó "
        "exactamente una vez, con un umbral calibrado sin haberlo visto (cero solapamiento, cero "
        "fuga de datos), dando 'n' del tamaño del dataset completo en vez de un solo fold chico. "
        "Intervalo de Wilson al 95%."
    )
    return pooled


def run_kfold_cv(args, model):
    """Validación cruzada de k folds a nivel de secuencia (ver kfold_utils.py):
    en vez de un train/val/test fijo (donde mirar repetidamente el mismo test
    durante el desarrollo invalida la medición, ver docstring del módulo),
    cada fold entrena su propia LinearReIDHead desde cero, calibra su propio
    umbral en SU train, y evalúa en SU test -- ningún fold decide
    hiperparámetros para sí mismo. Dispara con una sola invocación las 6 (o
    --num-folds) iteraciones completas: entrenar + calibrar + evaluar
    episodios sintéticos + evaluar fallos reales de cada tracker de
    --trackers, y termina con una tabla agregada media +/- desvío estándar.

    Optimización de cómputo: el reid_gap_report/required_reid_events de cada
    tracker se genera UNA SOLA VEZ sobre las 72 secuencias del pool de folds
    (no una vez por fold -- detector+tracker no cambian entre folds, solo la
    cabeza entrenada), y los casos reales se FILTRAN por fold (ver
    _filter_cases_by_sequences) contra el nombre de secuencia de cada caso.

    --kfold-skip-training: en vez de entrenar una LinearReIDHead por fold,
    evalúa el CLS crudo de DINOv2 (head=None) -- mismos folds/umbral
    calibrado por fold que las corridas CON cabeza entrenada, para tener el
    baseline "DINOv2 puro" con el mismo rigor estadístico (media +/-
    desvío entre folds), comparable directamente contra v6/v7/etc."""
    from kfold_utils import assign_folds, list_all_sequences, load_fold_assignment, save_fold_assignment, sequence_has_real_gap
    from tracker_failure_dataset import load_new_id_born_cases, load_required_reid_cases
    if not args.kfold_skip_training:
        from train_linear_reid_head import train_head  # import diferido -- evita el ciclo real (train_linear_reid_head importa de este módulo)

    if not args.reid_gap_onnx:
        raise SystemExit("--kfold-cv requiere --reid-gap-onnx (checkpoint ONNX del detector, para correr infer_track_reid_gaps.py)")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.fold_assignment_path and os.path.exists(args.fold_assignment_path):
        folds = load_fold_assignment(args.fold_assignment_path)
        print(f"[INFO] Fold assignment cargado de {args.fold_assignment_path} ({len(folds)} folds)")
    else:
        all_seqs = list_all_sequences()
        folds = assign_folds(all_seqs, k=args.num_folds, seed=args.kfold_seed)
        assignment_path = args.fold_assignment_path or os.path.join(args.output_dir, "fold_assignment.json")
        save_fold_assignment(folds, assignment_path)
        print(f"[INFO] Fold assignment ({len(folds)} folds, seed={args.kfold_seed}) guardado en {assignment_path}")
    k = len(folds)
    for fold_idx in sorted(folds):
        n_gap = sum(1 for s in folds[fold_idx] if sequence_has_real_gap(s))
        print(f"[INFO] fold {fold_idx}: {len(folds[fold_idx])} secuencias ({n_gap} con reid real)")

    # 1) reid-gap-report + casos GLOBALES, una vez por tracker, sobre TODAS
    # las secuencias del pool de folds -- se filtran por fold más abajo.
    all_seq_dirs = sorted({s for seqs in folds.values() for s in seqs})
    global_cases = {}
    for tracker in args.trackers:
        tracker_outdir = os.path.join(args.output_dir, f"reid_gap_{tracker}_kfold_all")
        tracker_args = argparse.Namespace(**vars(args))
        tracker_args.reid_gap_tracker = tracker
        _apply_tracker_hparams(tracker_args, tracker)
        report_path = _generate_reid_gap_report(tracker_args, tracker_outdir, sequence_dirs=all_seq_dirs)
        if args.case_definition == "required_reid":
            events_path = os.path.join(tracker_outdir, "required_reid_events.json")
            cases = load_required_reid_cases(events_path)
        else:
            cases = load_new_id_born_cases(report_path)
        print(f"[INFO] {len(cases)} casos GLOBALES ({args.case_definition}) para tracker={tracker} (pool de {len(all_seq_dirs)} secuencias)")
        global_cases[tracker] = cases

    # 2) loop de folds -- test_fold = fold_idx, val_fold = rotación fija
    # (fold_idx + 1) % k, el resto entrena. Cada fold entrena/calibra/evalúa
    # desde cero, en su propia carpeta (nunca pisa el checkpoint de otro fold).
    fold_results = []
    for fold_idx in range(k):
        print(f"\n[INFO] ==== FOLD {fold_idx}/{k - 1} ====")
        val_fold = (fold_idx + 1) % k
        test_seqs = folds[fold_idx]
        val_seqs = folds[val_fold]
        train_seqs = [s for i, seqs in folds.items() if i not in (fold_idx, val_fold) for s in seqs]
        print(
            f"[INFO] fold {fold_idx}: train={len(train_seqs)} secuencias, "
            f"val={len(val_seqs)} (fold {val_fold}), test={len(test_seqs)} (fold {fold_idx})"
        )

        train_index = build_track_index(None, sequence_dirs=train_seqs)
        val_index = build_track_index(None, sequence_dirs=val_seqs)
        test_index = build_track_index(None, sequence_dirs=test_seqs)

        fold_output_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
        if args.kfold_skip_training:
            head, best_val_loss = None, float("nan")
        else:
            head, best_val_loss = train_head(model, train_index, val_index, args, fold_output_dir)

        if args.match_threshold is not None:
            threshold = args.match_threshold
        else:
            threshold = _calibrate_threshold(model, train_index, args, head=head)

        synthetic_metrics = _evaluate_synthetic_split(model, test_index, args, threshold, head, fold_output_dir, "test")

        tracker_metrics = {}
        test_seq_names = {os.path.basename(s.rstrip("/")) for s in test_seqs}
        for tracker in args.trackers:
            cases = _filter_cases_by_sequences(global_cases[tracker], test_seq_names)
            n_before_gap_filter = len(cases)
            cases = _filter_cases_by_gap_length(cases, args.gap_length_bin)
            print(
                f"[INFO] fold {fold_idx} tracker={tracker}: {n_before_gap_filter}/{len(global_cases[tracker])} "
                f"casos globales caen en el test de este fold "
                f"({len(cases)}/{n_before_gap_filter} tras filtrar gap-length-bin={args.gap_length_bin})"
            )
            tracker_metrics[tracker] = _evaluate_tracker_on_cases(
                model, args, tracker, test_index, cases, threshold, head, fold_output_dir, f"{tracker}_fold{fold_idx}",
            )

        print(
            f"[INFO] FOLD {fold_idx} resumen: umbral={threshold:.4f} best_val_loss={best_val_loss:.4f} "
            f"synthetic_test identity_acc={synthetic_metrics['identity_acc']:.4f}"
        )
        fold_results.append({
            "fold": fold_idx, "val_fold": val_fold,
            "train_seqs": [os.path.basename(s.rstrip("/")) for s in train_seqs],
            "val_seqs": [os.path.basename(s.rstrip("/")) for s in val_seqs],
            "test_seqs": sorted(test_seq_names),
            "threshold": threshold, "best_val_loss": best_val_loss,
            "synthetic": synthetic_metrics, "trackers": tracker_metrics,
        })

    # 3) agregación final: media +/- desvío estándar entre folds (protocolo
    # original) Y accuracy pooleada con intervalo de Wilson (ver docstring
    # del módulo) -- ambas sobre los MISMOS fold_results, sin cambiar en
    # absoluto qué calibra/evalúa cada fold.
    _print_kfold_summary(fold_results, args)
    pooled = _print_pooled_summary(fold_results, args)
    detail_path = os.path.join(args.output_dir, "kfold_results.json")
    with open(detail_path, "w") as f:
        json.dump(fold_results, f, indent=2)
    print(f"[OK] Detalle completo por fold guardado en {detail_path}")
    pooled_path = os.path.join(args.output_dir, "kfold_pooled_summary.json")
    with open(pooled_path, "w") as f:
        json.dump(pooled, f, indent=2)
    print(f"[OK] Resumen pooleado guardado en {pooled_path}")
    return fold_results


def main():
    from train_linear_reid_head import add_training_args  # import diferido -- evita el ciclo real (train_linear_reid_head importa de este módulo)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dinov2-model-name", default="facebook/dinov2-small")
    parser.add_argument(
        "--full-pipeline", action="store_true",
        help="Corre TODO el flujo unificado (ver docstring del módulo): calibra en train, "
             "evalúa val/test, y evalúa cada tracker de --trackers sobre sus fallos reales en "
             "test -- todo con el mismo umbral. Ignora --split/--tracker-failures-json/"
             "--required-reid-events (se resuelven automáticamente).",
    )
    parser.add_argument("--output-dir", default="dinov2_baseline_report", help="[--full-pipeline] Carpeta base para todos los resultados/imágenes/reid_gap_reports.")
    parser.add_argument("--trackers", nargs="+", default=["bytetrack", "botsort"], choices=["bytetrack", "botsort", "ocsort", "deepocsort"], help="[--full-pipeline] Trackers a evaluar.")
    parser.add_argument("--case-definition", choices=["required_reid", "new_id_born"], default="required_reid", help="[--full-pipeline] Qué casos reales usar por tracker: required_reid (cualquier cambio de id) o new_id_born (solo brechas temporales reales).")
    parser.add_argument(
        "--reid-gap-splits", nargs="+", default=["val", "test"], choices=["val", "test"],
        help="[--full-pipeline] En qué splits buscar fallos reales de cada tracker -- con más de "
             "uno, además de una fila por split se imprime una fila 'combinada' que junta los "
             "casos de todos (muestra más grande). No incluye train (es el split de calibración).",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="[modo suelto] split a evaluar")
    parser.add_argument(
        "--num-episodes", type=int, default=400,
        help="Default 400 para igualar --val-episodes/--test-episodes del entrenamiento real "
             "(train_track_query_reid.py) -- misma cantidad de episodios, comparación en igualdad de condiciones.",
    )
    parser.add_argument("--max-bank-size", type=int, default=16)
    parser.add_argument(
        "--gap-length-bin", default="all", choices=["all", "lt3", "3-8", "9-32", "gt32"],
        help="Filtra los casos required_reid por duración del hueco (frame_after-frame_before) "
             "antes de evaluar -- ablación de estratificación por longitud (P3 paperreview.ai). "
             "'all' (default) no filtra nada, comportamiento idéntico al de siempre.",
    )
    parser.add_argument(
        "--hard-distractor-mode", default="same_video", choices=["same_video", "same_panel"],
        help="same_panel fuerza a que, si existe, al menos un distractor del banco comparta "
             "part_label (mismo panel) con la identidad correcta -- ablación de hard negatives "
             "reales (M10/P4 paperreview.ai), ver sample_hard_distractors_same_panel en "
             "reid_dataloader.py. Solo afecta la evaluación sobre casos reales "
             "(_evaluate_tracker_on_cases); no toca la calibración de umbral (_calibrate_threshold), "
             "que sigue usando ReIDEpisodeDataset con el muestreo de siempre.",
    )
    parser.add_argument("--num-track-history", type=int, default=4)
    parser.add_argument("--query-wide-factor", type=float, default=9.0)
    parser.add_argument(
        "--query-fusion", default="mean", choices=["mean", "near", "wide", "max", "concat"],
        help="[Reimplementado 2026-09-07, ver memoria feedback_isolate_experimental_pipeline_changes "
             "-- el --query-fusion original (ablacion M10/revision propia, escala 6/1.6/1.0) se perdio "
             "en algun momento sin dejar rastro en el codigo (nunca estuvo bajo git); esta es una "
             "reconstruccion, no una recuperacion bit-a-bit del original] Como combinar las DOS vistas "
             "de la query (ajustada=near, factor 1.0; amplia=wide, factor intermedio del cache o "
             "query_wide_factor sin cache -- ver _load_query_views en reid_dataloader.py) en un solo "
             "vector antes de compararlo contra el banco. mean (default, comportamiento de siempre) = "
             "promedio de las 2 vistas. near/wide = usa SOLO esa vista, descarta la otra. max = maximo "
             "elemento a elemento entre las 2 vistas. concat = concatena las 2 vistas (2xD) en vez de "
             "promediarlas -- en este modo el banco tambien se duplica consigo mismo (concat(v,v)) para "
             "mantener las dimensiones comparables en la similitud coseno, ya que el banco no tiene una "
             "nocion separada de vista-ajustada/vista-amplia como la query.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-visuals", action="store_true", help="Guarda imagenes de episodios acertados/fallados (ver visualize_episode)")
    parser.add_argument("--fails-dir", default="dinov2_nn_baseline_fails", help="[modo suelto]")
    parser.add_argument("--correct-dir", default="dinov2_nn_baseline_correct", help="[modo suelto]")
    parser.add_argument("--max-fails", type=int, default=0, help="0 = sin límite (guarda todos los fallos)")
    parser.add_argument("--max-correct", type=int, default=0, help="0 = sin límite (guarda todos los aciertos)")
    parser.add_argument(
        "--heatmap-temperature", type=float, default=0.05,
        help="Escala las similitudes coseno antes del softmax del heatmap (solo visual, no "
             "afecta la decision de matching) -- las similitudes crudas quedan muy juntas "
             "(ej. 0.75-0.85) y sin esto el softmax se ve casi plano entre candidatos.",
    )
    parser.add_argument(
        "--include-no-match", action="store_true",
        help="[modo suelto] Evalua tambien la decision binaria match/no-match (episodios "
             "mezclados, sin force_reid), calibrando un umbral de similitud en TRAIN. Sin este "
             "flag, el baseline solo mide identity_acc sobre episodios con match real.",
    )
    parser.add_argument("--match-threshold", type=float, default=None, help="Si se pasa, se usa este umbral fijo en vez de calibrar en train")
    parser.add_argument("--num-calibration-episodes", type=int, default=300)
    parser.add_argument("--calibration-seed", type=int, default=123)
    parser.add_argument(
        "--reid-gap-report", action="store_true",
        help="[modo suelto] Ademas del baseline de DINOv2, corre infer_track_reid_gaps.py sobre "
             "el mismo --split, y muestra cuantas brechas reales terminan en new_id_born -- "
             "mismos nombres de flag que --reid-gap-* en train_track_query_reid.py.",
    )
    parser.add_argument("--reid-gap-detector", choices=["rtv2", "rtv4"], default="rtv2")
    parser.add_argument("--reid-gap-onnx", default=None, help="[--reid-gap-report/--full-pipeline/--tracker-failures-json/--required-reid-events] Checkpoint ONNX del detector.")
    parser.add_argument("--reid-gap-provider", default="CUDAExecutionProvider")
    parser.add_argument("--reid-gap-det-thresh", type=float, default=0.4)
    parser.add_argument("--reid-gap-tracker", choices=["bytetrack", "botsort", "ocsort", "deepocsort"], default="bytetrack", help="[modo suelto] ignorado en --full-pipeline (usa --trackers)")
    parser.add_argument("--reid-gap-no-native-reid", action="store_true")
    parser.add_argument("--reid-gap-reid-weights", default="osnet_x1_0_msmt17.pt")
    parser.add_argument("--reid-gap-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reid-gap-track-thresh", type=float, default=0.45,
                         help="boxmot track_thresh (bytetrack) / track_high_thresh (botsort). Ver "
                              "eval_tracking/sweep.py para el valor tuneado (anclado en Ultralytics).")
    parser.add_argument("--reid-gap-match-thresh", type=float, default=0.8)
    parser.add_argument("--reid-gap-track-buffer", type=int, default=25)
    parser.add_argument("--reid-gap-new-track-thresh", type=float, default=0.6, help="[botsort]")
    parser.add_argument("--reid-gap-proximity-thresh", type=float, default=0.5, help="[botsort]")
    parser.add_argument("--reid-gap-appearance-thresh", type=float, default=0.25, help="[botsort]")
    parser.add_argument("--reid-gap-cmc-method", default="ecc", choices=["ecc", "orb", "sof", "sift"], help="[botsort]")
    parser.add_argument("--reid-gap-min-hits", type=int, default=3, help="[ocsort/deepocsort]")
    parser.add_argument("--reid-gap-iou-threshold", type=float, default=0.3, help="[ocsort/deepocsort]")
    parser.add_argument("--reid-gap-asso-func", default="iou", help="[ocsort/deepocsort]")
    parser.add_argument("--reid-gap-delta-t", type=int, default=3, help="[ocsort/deepocsort]")
    parser.add_argument("--reid-gap-inertia", type=float, default=0.2, help="[ocsort/deepocsort]")
    parser.add_argument("--reid-gap-w-association-emb", type=float, default=0.5, help="[deepocsort]")
    parser.add_argument("--reid-gap-alpha-fixed-emb", type=float, default=0.95, help="[deepocsort]")
    parser.add_argument("--reid-gap-aw-param", type=float, default=0.5, help="[deepocsort]")
    parser.add_argument(
        "--reid-gap-tracker-params-json", default=None,
        help="Ruta a un JSON {\"bytetrack\": {track_thresh, match_thresh, track_buffer}, "
             "\"botsort\": {track_thresh, match_thresh, track_buffer, new_track_thresh, "
             "proximity_thresh, appearance_thresh, cmc_method}} -- sobreescribe los "
             "--reid-gap-track-thresh/etc. planos POR TRACKER (necesario porque cada tracker "
             "queda tuneado con valores distintos, ver eval_tracking/sweep_ultralytics_anchor.py). "
             "Sin esto, se usa el mismo valor plano para ambos trackers.",
    )
    parser.add_argument("--reid-gap-min-gap-frames", type=int, default=3)
    parser.add_argument("--reid-gap-iou-match-thresh", type=float, default=0.5)
    parser.add_argument("--reid-gap-max-sequences", type=int, default=None)
    parser.add_argument("--reid-gap-outdir", default=None, help="[modo suelto] Default: reid_gap_report_<tracker>_<split>")
    parser.add_argument("--reid-gap-force", action="store_true")
    parser.add_argument(
        "--tracker-failures-json", default=None,
        help="[modo suelto] Ruta a un reid_gap_report.json (ver infer_track_reid_gaps.py o "
             "--reid-gap-report) -- si se pasa, en vez de generar episodios sintéticos, evalúa "
             "sobre TrackerFailureReIDDataset: un episodio por cada caso new_id_born REAL de ese "
             "tracker (ver tracker_failure_dataset.py, definición ESTRICTA: solo brechas "
             "temporales reales). Excluyente con --required-reid-events. Ignora --num-episodes. "
             "Fuerza --include-no-match=True -- pasá --match-threshold con el mismo valor "
             "calibrado en la corrida de --split test, para que las evaluaciones usen "
             "EXACTAMENTE el mismo umbral. Si el archivo NO existe todavía y se pasa "
             "--reid-gap-onnx, se genera automáticamente antes de evaluar.",
    )
    parser.add_argument(
        "--required-reid-events", default=None,
        help="[modo suelto] Ruta a un required_reid_events.json (ver infer_track_reid_gaps.py) -- "
             "version MAS AMPLIA que --tracker-failures-json: un episodio por CUALQUIER cambio de "
             "id en un track de GT (con o sin brecha temporal detectada), no solo new_id_born. "
             "Mismo comportamiento que --tracker-failures-json en todo lo demás. Excluyente con "
             "--tracker-failures-json.",
    )
    parser.add_argument(
        "--same-video-only", action="store_true",
        help="[modo suelto] Con --tracker-failures-json/--required-reid-events: el banco NUNCA se "
             "completa con distractores de OTRO video -- dataset 100%% real. (--full-pipeline "
             "siempre usa esto, sin necesidad de pasar el flag).",
    )
    parser.add_argument(
        "--linear-head-checkpoint", default=None,
        help="Ruta a un checkpoint de LinearReIDHead (ver linear_reid_head.py / "
             "train_linear_reid_head.py) -- si se pasa, evalúa con esa proyección entrenada en "
             "vez del CLS crudo de DINOv2 en TODOS los modos (sueltos y --full-pipeline). Ignorado "
             "con --kfold-cv (cada fold entrena la suya propia desde cero). Excluyente con "
             "--attn-scorer-checkpoint.",
    )
    parser.add_argument(
        "--attn-scorer-checkpoint", default=None,
        help="Ruta a un checkpoint de BilinearAttentionScorer (ver bilinear_attention_scorer.py / "
             "train_bilinear_attention_scorer.py) -- cabeza de atención cruzada minimalista, "
             "alternativa a LinearReIDHead, que opera sobre el CLS CRUDO de DINOv2 (reemplaza la "
             "cosine cruda por su score_mixed en TODOS los modos, sueltos y --full-pipeline). "
             "Excluyente con --linear-head-checkpoint. No soportado todavía con --kfold-cv.",
    )
    parser.add_argument(
        "--kfold-cv", action="store_true",
        help="Corre validación cruzada de --num-folds folds a nivel de secuencia (ver "
             "kfold_utils.py y run_kfold_cv) en vez de un train/val/test fijo -- cada fold "
             "entrena su propia LinearReIDHead desde cero, calibra su propio umbral en su train, "
             "y evalúa episodios sintéticos + fallos reales de --trackers en su propio test. "
             "Termina con una tabla agregada media +/- desvío estándar entre folds -- esa es la "
             "métrica a reportar, no la de ningún fold individual. Requiere --reid-gap-onnx. "
             "Ignora --full-pipeline/--linear-head-checkpoint/modo suelto.",
    )
    parser.add_argument("--num-folds", type=int, default=6, help="[--kfold-cv]")
    parser.add_argument("--kfold-seed", type=int, default=42, help="[--kfold-cv] Seed del shuffle estratificado de assign_folds.")
    parser.add_argument(
        "--fold-assignment-path", default=None,
        help="[--kfold-cv] Ruta a un fold_assignment.json ya generado (ver "
             "kfold_utils.save_fold_assignment) -- si existe, se reusa en vez de generar uno "
             "nuevo (mismo assignment entre corridas). Default: <output-dir>/fold_assignment.json.",
    )
    parser.add_argument(
        "--kfold-skip-training", action="store_true",
        help="[--kfold-cv] No entrena ninguna cabeza -- evalúa el embedding crudo del encoder "
             "elegido con --encoder (head=None) en cada fold, con la MISMA metodología (mismos "
             "folds, mismo split train/val/test rotado, mismo umbral calibrado por fold) que las "
             "corridas con cabeza entrenada -- para tener el baseline '<encoder> puro' con el "
             "mismo rigor estadístico (media +/- desvío entre folds) y comparable directamente "
             "contra v6/v7/etc. Usá un --output-dir distinto al de esas corridas.",
    )
    parser.add_argument(
        "--embedding-cache", default=None,
        help="Ruta a un PrecomputedEmbeddingCache (ver precompute_p4_roi_cache.py o "
             "train_track_query_reid.py precompute-cache) -- si se pasa, salta el crop+encoder "
             "y usa embeddings ya calculados por (image_path, bbox). Debe corresponder al MISMO "
             "encoder que --encoder/--dinov2-model-name.",
    )
    add_encoder_args(parser)
    add_training_args(parser)
    args = parser.parse_args()

    if args.linear_head_checkpoint and args.attn_scorer_checkpoint:
        raise SystemExit("--linear-head-checkpoint y --attn-scorer-checkpoint son excluyentes -- elegí uno.")

    if args.embedding_cache:
        from reid_dataloader import PrecomputedEmbeddingCache
        print(f"[INFO] Usando embedding cache: {args.embedding_cache}")
        args.embedding_cache = PrecomputedEmbeddingCache.load(args.embedding_cache)

    args.reid_gap_tracker_params = None
    if args.reid_gap_tracker_params_json:
        with open(args.reid_gap_tracker_params_json) as f:
            args.reid_gap_tracker_params = json.load(f)
        print(f"[INFO] Hiperparámetros de tracker por-tracker cargados de {args.reid_gap_tracker_params_json}: "
              f"{args.reid_gap_tracker_params}")

    model = build_encoder(args)

    if args.kfold_cv:
        if args.attn_scorer_checkpoint:
            raise SystemExit("--attn-scorer-checkpoint todavía no está soportado con --kfold-cv.")
        run_kfold_cv(args, model)
        return

    head = None
    if args.linear_head_checkpoint:
        from linear_reid_head import load_head
        head = load_head(args.linear_head_checkpoint, args.device)
        print(f"[INFO] Usando LinearReIDHead entrenada: {args.linear_head_checkpoint} (in_dim={head.in_dim}, out_dim={head.out_dim})")

    attn_scorer = None
    if args.attn_scorer_checkpoint:
        from bilinear_attention_scorer import load_scorer
        attn_scorer = load_scorer(args.attn_scorer_checkpoint, args.device)
        print(f"[INFO] Usando BilinearAttentionScorer entrenado: {args.attn_scorer_checkpoint} (in_dim={attn_scorer.in_dim}, low_dim={attn_scorer.low_dim})")

    if args.full_pipeline:
        run_full_pipeline(args, model, head=head, attn_scorer=attn_scorer)
        return

    # --- modo suelto (control fino / debug puntual) ---
    if args.reid_gap_report and not args.reid_gap_onnx:
        raise SystemExit("--reid-gap-report requiere --reid-gap-onnx (checkpoint ONNX del detector)")
    if args.tracker_failures_json and args.required_reid_events:
        raise SystemExit("--tracker-failures-json y --required-reid-events son excluyentes -- elegí uno.")
    case_source_path = args.tracker_failures_json or args.required_reid_events
    if case_source_path and not args.include_no_match:
        flag_name = "--tracker-failures-json" if args.tracker_failures_json else "--required-reid-events"
        print(f"[INFO] {flag_name} fuerza --include-no-match=True (se evalúa match/no-match, no solo identity_acc).")
        args.include_no_match = True
    max_fails = args.max_fails if args.max_fails > 0 else float("inf")
    max_correct = args.max_correct if args.max_correct > 0 else float("inf")

    threshold = None
    if args.include_no_match:
        if args.match_threshold is not None:
            threshold = args.match_threshold
            print(f"[INFO] Usando umbral fijo pasado por CLI: {threshold:.4f}")
        else:
            train_root, train_seq_dirs = resolve_sequence_dirs(None, "train")
            train_index = build_track_index(train_root, sequence_dirs=train_seq_dirs)
            threshold = _calibrate_threshold(model, train_index, args, head=head, attn_scorer=attn_scorer)
            if args.split == "train":
                print(
                    "[WARN] --split train con umbral calibrado también en train: "
                    "el número va a ser optimista, no comparable a val/test."
                )

    data_root, seq_dirs = resolve_sequence_dirs(None, args.split)
    index = build_track_index(data_root, sequence_dirs=seq_dirs)

    if case_source_path:
        report_path = case_source_path
        if not os.path.exists(report_path):
            if not args.reid_gap_onnx:
                raise SystemExit(
                    f"{report_path} no existe todavía -- pasá --reid-gap-onnx (y opcionalmente "
                    f"--reid-gap-tracker/--reid-gap-*) para generarlo automáticamente, o corré "
                    f"infer_track_reid_gaps.py antes."
                )
            outdir = os.path.dirname(report_path) or "."
            print(
                f"[INFO] {report_path} no existe -- generándolo con infer_track_reid_gaps.py "
                f"(tracker={args.reid_gap_tracker}, split={args.split})..."
            )
            _generate_reid_gap_report(args, outdir)  # escribe reid_gap_report.json Y required_reid_events.json en outdir
            if not os.path.exists(report_path):
                raise SystemExit(f"Se generó el reporte en {outdir} pero no se encontró {report_path} -- ¿nombre de archivo correcto?")

        from tracker_failure_dataset import load_new_id_born_cases, load_required_reid_cases, TrackerFailureReIDDataset
        if args.required_reid_events:
            cases = load_required_reid_cases(report_path)
            print(f"[INFO] {len(cases)} casos de reid requerido (cualquier cambio de id) en {report_path}")
        else:
            cases = load_new_id_born_cases(report_path)
            print(f"[INFO] {len(cases)} casos new_id_born en {report_path}")
        dataset = TrackerFailureReIDDataset(
            index, cases, same_video_only=args.same_video_only,
            max_bank_size=args.max_bank_size, num_track_history=args.num_track_history,
            query_wide_factor=args.query_wide_factor, seed=args.seed, return_video_ids=True,
            embedding_cache=args.embedding_cache,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_episodes_with_video_ids)
        if args.required_reid_events:
            fuente = f"casos de reid requerido reales, same_video_only={args.same_video_only} ({case_source_path})"
        else:
            fuente = f"casos new_id_born reales, same_video_only={args.same_video_only} ({case_source_path})"
    else:
        loader = _build_loader(
            index, args.num_episodes, args.max_bank_size, args.num_track_history,
            args.query_wide_factor, force_reid=not args.include_no_match,
            seed=args.seed, batch_size=args.batch_size,
            embedding_cache=args.embedding_cache,
        )
        fuente = "episodios sintéticos"

    metrics = _evaluate_loader(
        model, loader, args.device, include_no_match=args.include_no_match, threshold=threshold,
        heatmap_temperature=args.heatmap_temperature, save_visuals=args.save_visuals,
        fails_dir=args.fails_dir, correct_dir=args.correct_dir,
        max_fails=max_fails, max_correct=max_correct, head=head, attn_scorer=attn_scorer,
        query_fusion=args.query_fusion,
    )
    _print_metrics(f"BASELINE 1-NN crudo -- sin cabeza entrenada -- split={args.split}", metrics, threshold, args.include_no_match, fuente=fuente)
    print(
        "\n[INFO] Compará esto contra identity_acc/id_acc (y match_acc, si usaste "
        "--include-no-match) del modelo entrenado en el mismo split -- si este baseline sin "
        "cabeza iguala o supera al modelo, la cabeza está perdiendo señal, no sumándola."
    )

    if args.reid_gap_report:
        print(f"\n[REID-GAP] corriendo detector+{args.reid_gap_tracker} sobre split={args.split}...")
        counts = _run_reid_gap_baseline(args)
        total = sum(counts.values())
        evaluable = total - counts.get("unevaluated", 0)
        n_new_id = counts.get("new_id_born", 0)
        print(f"\n[REID-GAP {args.reid_gap_tracker}] split={args.split} {total} brechas reales de GT evaluadas:")
        for outcome in ("reidentified", "new_id_born", "id_swap", "unevaluated"):
            n = counts.get(outcome, 0)
            pct = 100 * n / total if total else 0.0
            print(f"  {outcome:15s}: {n:5d} ({pct:5.1f}%)")
        if evaluable:
            print(
                f"  -> casos de reidentificacion fallida (new_id_born): {n_new_id}/{evaluable} "
                f"evaluables ({100 * n_new_id / evaluable:.1f}%)"
            )
        print(
            "\n[INFO] Comparación: identity_acc de arriba mide si, DADO que se tiene el banco "
            "correcto de candidatos, se elige el correcto -- new_id_born mide si el tracker "
            "clásico (sin ningún banco, solo motion/IoU/ReID nativo) ni siquiera intenta "
            "reidentificar. Son dos puntos distintos del mismo pipeline, no compiten directo."
        )


if __name__ == "__main__":
    main()
