"""
Entrena LinearReIDHead (ver linear_reid_head.py): una sola capa lineal
sobre el embedding CONGELADO de DINOv2 (CLS de la última capa), con
metric learning (batch-hard triplet, Hermans et al.) -- la cabeza más
simple posible, antes de considerar algo más pesado (MLP, o volver al
transformer de set_reid_model.py).

Reusa piezas YA IMPLEMENTADAS pero nunca conectadas:
  - hierarchical_pk_sampler.py: FlatTrackViewDataset + HierarchicalPKSampler
    (batches P secuencias x K tracks x N vistas) + batch_hard_triplet_loss.
  - analyze_dinov2_nn_baseline.py: _build_loader/_evaluate_loader, para
    medir identity_acc en val cada época con el MISMO harness que ya usamos
    para el baseline crudo -- comparación directa, sin reinventar nada.

Réplica EXACTA del flujo real de producción (misma lógica de recorte que
ReIDEpisodeDataset, reusando sus métodos directo, no reimplementados):
  - El sampler P x K x N elige, para cada track, N = --num-track-history + 1
    observaciones (P secuencias x K tracks x N vistas/track).
  - De esas N, la MÁS RECIENTE (por frame_idx) hace de QUERY -- se le aplica
    `ReIDEpisodeDataset._load_query_views` (2 vistas: ajustada + amplia).
  - Las --num-track-history restantes hacen de BANCO -- se les aplica
    `ReIDEpisodeDataset._load_track_history_crops` (misma convención real:
    la más antigua con recorte AMPLIO fijo, la más reciente AJUSTADA, las
    intermedias con factor aleatorio) -- no una aproximación, el mismo
    método.
  - Por track, esto da exactamente 2 embeddings (bank_summary, query) --
    misma identidad entre sí (positivo real para batch_hard_triplet_loss),
    y compite en el batch contra los bank_summary/query de TODOS los demás
    tracks (P secuencias x K tracks) como negativos.

Loss = triplet (batch-hard, Hermans et al.) + --infonce-weight * InfoNCE
(query vs TODOS los bancos del batch a la vez, ver
_query_to_bank_infonce_loss_with_new) -- el triplet solo mira el
positivo/negativo MÁS difícil; el InfoNCE imita la decisión real de
matching (argmax de similitud contra todos los candidatos), más parecido a
_best_candidate en analyze_dinov2_nn_baseline.py. --infonce-weight 0
desactiva esta loss auxiliar (vuelve a ser solo triplet).

--new-track-prob (default 0.25): con --infonce-weight > 0, una fracción de
los tracks del batch actúa como "nuevo" (ver _select_new_tracks) -- su
banco NO entra a la galería del InfoNCE (no tiene match real), y su query
apunta a una columna extra "no match" (LinearReIDHead.new_embedding, un
embedding aprendible, equivalente liviano al new_embedding de MatchScorer
en set_reid_model.py). Sin esto, el entrenamiento NUNCA ve un caso de
no-match -- toda query siempre tenía su propio banco por construcción --
así que la cabeza no tenía ninguna señal para aprender a alejar queries
genuinamente nuevas de los bancos existentes; esa decisión quedaba 100% en
manos del umbral calibrado aparte (_calibrate_threshold en
analyze_dinov2_nn_baseline.py), desconectado de la loss de entrenamiento.

Costo de esta fidelidad: el banco tiene factores de recorte ALEATORIOS
(posiciones intermedias) y el propio sampler puede resamplear con
reemplazo -- ninguno de los dos es 100% determinista entre llamadas, así
que a diferencia de una corrida anterior de este script, **val YA NO se
puede cachear** (los crops de val cambian levemente cada vez que se pide,
aunque el split no tenga ColorJitter) -- val_loss vuelve a correr DINOv2
de verdad cada época, igual que train.

Selección de checkpoint: por val_loss (la MISMA loss triplet, sobre val),
no por identity_acc en val -- elegir por esa métrica llevó a sobreajustar
el checkpoint a val en corridas anteriores (val subía, test bajaba, de
forma consistente). identity_acc en val se sigue imprimiendo cada
--val-acc-every épocas solo como referencia informativa.

Uso:
    python3 train_linear_reid_head.py --output-dir linear_reid_head_run

    # Evaluación completa del mejor checkpoint:
    python3 analyze_dinov2_nn_baseline.py --full-pipeline \\
        --linear-head-checkpoint linear_reid_head_run/best_head.pt \\
        --reid-gap-onnx checkpoints/rtdetrv2_v3_qcs_videos.onnx
"""
import argparse
import os
import random

import torch
import torch.nn.functional as F

from hierarchical_pk_sampler import FlatTrackViewDataset, HierarchicalPKSampler, batch_hard_triplet_loss
from p4_backbone_encoder import add_encoder_args, build_encoder
from linear_reid_head import LinearReIDHead, load_head, save_head
from reid_dataloader import AUGMENTED_TRANSFORM, DEFAULT_TRANSFORM, ReIDEpisodeDataset
from train_track_query_reid import build_track_index, resolve_sequence_dirs

from analyze_dinov2_nn_baseline import _build_loader, _evaluate_loader


@torch.no_grad()
def _encode_raw_cls(model, images, device, batch_size=64):
    """CLS crudo de DINOv2 (congelado, siempre sin gradiente) -- `head` se
    aplica DESPUÉS, fuera de esta función, con o sin gradiente según haga
    falta (entrenamiento vs. validación)."""
    outs = []
    for i in range(0, images.shape[0], batch_size):
        chunk = images[i:i + batch_size].to(device)
        outs.append(model(pixel_values=chunk).last_hidden_state[:, 0, :])
    return torch.cat(outs, dim=0)


def _group_tracks(flat_dataset, batch_indices):
    """Agrupa los índices GLOBALES que devuelve HierarchicalPKSampler por
    (sequence_id, track_id), usando flat_dataset.items para recuperar las
    TrackObservation reales -- devuelve [(seq_id, track_id, [obs, ...]), ...]."""
    groups = {}
    for idx in batch_indices:
        seq_id, track_id, obs = flat_dataset.items[idx]
        groups.setdefault((seq_id, track_id), []).append(obs)
    return [(seq_id, track_id, obs_list) for (seq_id, track_id), obs_list in groups.items()]


def _bank_and_query_embeds(crop_helper, model, head, tracks, device, encode_batch_size):
    """Por cada (seq_id, track_id, observations) en `tracks`: arma el crop
    de banco (K vistas, escala por posición -- _load_track_history_crops)
    con TODAS las observaciones salvo la más reciente, y el crop de query
    (2 vistas: ajustada+amplia -- _load_query_views) con la observación MÁS
    RECIENTE (por frame_idx) -- misma convención que el antes/después real
    (el banco es "antes", la query es "después"). Reusa los métodos de
    ReIDEpisodeDataset tal cual, no reimplementa el recorte.

    Devuelve (embeds, track_ids, sequence_ids) con exactamente 2 filas por
    track: [bank_summary_0, ..., bank_summary_T, query_0, ..., query_T]."""
    bank_crops_list, bank_mask_list, query_crops_list = [], [], []
    seq_ids, track_ids = [], []

    for seq_id, track_id, observations in tracks:
        obs_sorted = sorted(observations, key=lambda o: o.frame_idx)
        query_obs = obs_sorted[-1]
        bank_obs = obs_sorted[:-1] if len(obs_sorted) > 1 else obs_sorted
        bank_crops, bank_mask = crop_helper._load_track_history_crops(bank_obs)
        query_crops = crop_helper._load_query_views(query_obs)
        bank_crops_list.append(bank_crops)
        bank_mask_list.append(bank_mask)
        query_crops_list.append(query_crops)
        seq_ids.append(seq_id)
        track_ids.append(track_id)

    bank_crops_t = torch.stack(bank_crops_list, dim=0)   # (T, K, C, H, W) o (T, K, D) con --embedding-cache
    bank_mask_t = torch.stack(bank_mask_list, dim=0)      # (T, K)
    query_crops_t = torch.stack(query_crops_list, dim=0)  # (T, 2, C, H, W) o (T, 2, D)
    T, K = bank_crops_t.shape[:2]

    if bank_crops_t.dim() == 5:
        _, _, C, H, W = bank_crops_t.shape
        bank_cls = _encode_raw_cls(model, bank_crops_t.view(T * K, C, H, W), device, encode_batch_size)
        query_cls = _encode_raw_cls(model, query_crops_t.view(T * 2, C, H, W), device, encode_batch_size)
    else:
        D = bank_crops_t.shape[-1]
        bank_cls = bank_crops_t.view(T * K, D).to(device)
        query_cls = query_crops_t.view(T * 2, D).to(device)

    bank_proj = head(bank_cls).view(T, K, -1)
    query_proj = head(query_cls).view(T, 2, -1)

    mask = bank_mask_t.unsqueeze(-1).to(bank_proj.device).float()
    bank_summary = (bank_proj * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    bank_summary = F.normalize(bank_summary, p=2, dim=-1)

    query_embed = F.normalize(query_proj.mean(dim=1), p=2, dim=-1)

    return bank_summary, query_embed, track_ids, seq_ids


def _select_new_tracks(T, new_track_prob):
    """Marca al azar qué tracks del batch actúan como "nuevos" (sin match
    real en el banco, ver --new-track-prob) -- simula NEW_TRACK de
    ReIDEpisodeDataset. Se asegura de que quede al menos 1 track "matched"
    (si no, no habría banco en absoluto para ese batch)."""
    if new_track_prob <= 0:
        return [False] * T
    is_new = [random.random() < new_track_prob for _ in range(T)]
    if all(is_new):
        is_new[0] = False
    return is_new


def _query_to_bank_infonce_loss_with_new(matched_bank, query_embed, is_new, matched_idx, new_embedding, temperature):
    """Como una InfoNCE query->banco normal, pero con una columna EXTRA en
    la galería (new_embedding, ver LinearReIDHead) -- las queries marcadas
    `is_new[i]=True` deben apuntar a esa columna en vez de a su propio
    banco (que ni siquiera está en la galería: son casos sin match real,
    ver --new-track-prob). Esto es lo que le enseña a la cabeza a alejar
    las queries genuinamente nuevas de TODOS los bancos del batch, en vez
    de solo aprender a acercar pares que siempre sabía que eran iguales."""
    gallery = torch.cat([matched_bank, new_embedding.unsqueeze(0)], dim=0)  # (M+1, D)
    logits = query_embed @ gallery.t() / temperature  # (T, M+1)

    lookup = {orig_i: mi for mi, orig_i in enumerate(matched_idx)}
    new_col = matched_bank.shape[0]
    target = torch.tensor(
        [new_col if is_new[i] else lookup[i] for i in range(query_embed.shape[0])],
        device=query_embed.device,
    )
    return F.cross_entropy(logits, target)


def _combined_loss(bank_summary, query_embed, track_ids, seq_ids, head, args):
    """triplet (batch-hard, ver docstring del módulo) + infonce (query vs
    TODOS los bancos del batch + una columna "no match" -- ver
    --new-track-prob/_select_new_tracks) -- devuelve (loss_total, triplet,
    infonce) para poder loguear el desglose. Los tracks marcados "nuevos"
    NO aportan su banco a la galería (no tienen match real), pero su query
    sigue participando -- en el triplet como negativo puro (sin positivo
    propio), en el InfoNCE apuntando a la columna new_embedding."""
    T = bank_summary.shape[0]
    is_new = _select_new_tracks(T, args.new_track_prob)
    matched_idx = [i for i, n in enumerate(is_new) if not n]
    matched_bank = bank_summary[matched_idx]
    matched_track_ids = [track_ids[i] for i in matched_idx]
    matched_seq_ids = [seq_ids[i] for i in matched_idx]

    triplet_embeds = torch.cat([matched_bank, query_embed], dim=0)
    triplet = batch_hard_triplet_loss(
        triplet_embeds, matched_track_ids + track_ids, matched_seq_ids + seq_ids,
        margin=args.margin, intra_seq_bonus=args.intra_seq_bonus,
    )
    if args.infonce_weight > 0:
        infonce = _query_to_bank_infonce_loss_with_new(
            matched_bank, query_embed, is_new, matched_idx, head.new_embedding_normalized(), args.infonce_temperature,
        )
    else:
        infonce = torch.zeros((), device=bank_summary.device)
    return triplet + args.infonce_weight * infonce, triplet, infonce


def add_training_args(parser):
    """Hiperparámetros de entrenamiento de LinearReIDHead, compartidos entre
    este script y analyze_dinov2_nn_baseline.py (modo --kfold-cv) -- así no
    quedan duplicados/desincronizados entre los dos archivos. OJO: no
    agrega --dinov2-model-name/--num-track-history/--query-wide-factor/
    --seed/--device/--output-dir -- esos ya existen en el parser de
    analyze_dinov2_nn_baseline.py (mismo nombre, mismo significado) y se
    reusan tal cual; train_linear_reid_head.py:main() los agrega aparte."""
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pk-sequences-per-batch", type=int, default=2, help="P")
    parser.add_argument("--pk-tracks-per-sequence", type=int, default=8, help="K (identidades por secuencia en el sampler)")
    parser.add_argument("--margin", type=float, default=0.3, help="<=0 usa margen suave (softplus), ver batch_hard_triplet_loss")
    parser.add_argument("--intra-seq-bonus", type=float, default=0.0)
    parser.add_argument(
        "--infonce-weight", type=float, default=0.3,
        help="Peso de la loss auxiliar InfoNCE (query vs TODOS los bancos del batch, ver "
             "_query_to_bank_infonce_loss_with_new) sumada a la triplet -- imita la decisión real "
             "de matching (argmax de similitud). 0 desactiva esta loss (solo triplet, como antes). "
             "Default bajado de 1.0 a 0.3 tras comparar corridas (con 1.0 dominaba sobre el triplet).",
    )
    parser.add_argument("--infonce-temperature", type=float, default=0.1)
    parser.add_argument(
        "--new-track-prob", type=float, default=0.25,
        help="Probabilidad de que un track del batch actúe como 'nuevo' (sin match real en el "
             "banco, ver _select_new_tracks) -- sin esto, el entrenamiento NUNCA ve un caso de "
             "no-match, así que la cabeza no aprende a alejar queries genuinamente nuevas de los "
             "bancos existentes (esa decisión hoy la toma solo el umbral, calibrado aparte). "
             "0 desactiva esto (comportamiento anterior, siempre hay match)."
    )
    parser.add_argument("--val-episodes", type=int, default=400)
    parser.add_argument("--val-max-bank-size", type=int, default=16)
    parser.add_argument("--val-num-track-history", type=int, default=4)
    parser.add_argument("--val-query-wide-factor", type=float, default=9.0)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument(
        "--encode-batch-size", type=int, default=64,
        help="Tamaño de chunk para correr DINOv2 (más alto = más VRAM, menos llamadas, más rápido si sobra memoria).",
    )
    parser.add_argument(
        "--val-acc-every", type=int, default=5,
        help="Cada cuántas épocas calcular identity_acc en val (solo informativo, NO decide el "
             "checkpoint) -- es caro (recorre ~val-episodes episodios completos con DINOv2 real). "
             "0 = nunca durante el entrenamiento, solo al final con el mejor checkpoint.",
    )
    return parser


def train_head(model, train_index, val_index, args, output_dir):
    """Cuerpo de entrenamiento propiamente dicho -- extraído de main() para
    poder llamarlo una vez por fold desde run_kfold_cv (ver
    analyze_dinov2_nn_baseline.py). `output_dir` es un parámetro EXPLÍCITO
    (nunca args.output_dir) para que cada fold guarde su checkpoint en su
    propia carpeta sin pisar a los demás. Devuelve (best_head, best_val_loss)."""
    os.makedirs(output_dir, exist_ok=True)
    embed_dim = model.config.hidden_size

    views_per_track = args.num_track_history + 1  # --num-track-history para el banco + 1 para la query

    # FlatTrackViewDataset acá solo aporta .items/.by_sequence para que
    # HierarchicalPKSampler arme los grupos P x K x N -- el recorte real lo
    # hacen los *_crop_helper de abajo (_load_track_history_crops/
    # _load_query_views), no FlatTrackViewDataset.__getitem__.
    flat_train = FlatTrackViewDataset(train_index)
    sampler = HierarchicalPKSampler(
        flat_train, P=args.pk_sequences_per_batch, K=args.pk_tracks_per_sequence, N=views_per_track,
        allow_replacement_within_sequence=True, allow_replacement_within_track=True,
    )
    print(
        f"[INFO] {len(sampler)} batches/época (P={args.pk_sequences_per_batch} x "
        f"K={args.pk_tracks_per_sequence} x N={views_per_track} [{args.num_track_history} banco + 1 query], "
        f"hasta {args.pk_sequences_per_batch * args.pk_tracks_per_sequence} tracks/batch -> "
        f"{args.pk_sequences_per_batch * args.pk_tracks_per_sequence * 2} embeddings/batch para la loss)"
    )

    flat_val = FlatTrackViewDataset(val_index)
    val_sampler = HierarchicalPKSampler(
        flat_val, P=args.pk_sequences_per_batch, K=args.pk_tracks_per_sequence, N=views_per_track,
        allow_replacement_within_sequence=True, allow_replacement_within_track=True, seed=args.seed,
    )

    # Helpers que reusan el recorte REAL (ReIDEpisodeDataset._load_track_history_crops/
    # _load_query_views) -- train con ColorJitter, val sin aumentación, igual
    # criterio que el resto del proyecto. episodes_per_epoch/max_bank_size no
    # importan acá, solo se usan sus métodos de recorte, no __getitem__.
    train_crop_helper = ReIDEpisodeDataset(
        train_index, episodes_per_epoch=1, num_track_history=args.num_track_history,
        query_wide_factor=args.query_wide_factor, transform=AUGMENTED_TRANSFORM,
        embedding_cache=args.embedding_cache,
    )
    val_crop_helper = ReIDEpisodeDataset(
        val_index, episodes_per_epoch=1, num_track_history=args.num_track_history,
        query_wide_factor=args.query_wide_factor, transform=DEFAULT_TRANSFORM,
        embedding_cache=args.embedding_cache,
    )

    head = LinearReIDHead(embed_dim, args.proj_dim).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(output_dir, "best_head.pt")

    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss, total_triplet, total_infonce, n_batches = 0.0, 0.0, 0.0, 0
        for batch_indices in sampler:
            if not batch_indices:
                continue
            tracks = _group_tracks(flat_train, batch_indices)
            bank_summary, query_embed, track_ids, seq_ids = _bank_and_query_embeds(
                train_crop_helper, model, head, tracks, args.device, args.encode_batch_size,
            )
            loss, triplet, infonce = _combined_loss(bank_summary, query_embed, track_ids, seq_ids, head, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_triplet += triplet.item()
            total_infonce += infonce.item()
            n_batches += 1
        avg_loss = total_loss / n_batches if n_batches else float("nan")
        avg_triplet = total_triplet / n_batches if n_batches else float("nan")
        avg_infonce = total_infonce / n_batches if n_batches else float("nan")

        head.eval()
        with torch.no_grad():
            total_val_loss, n_val_batches = 0.0, 0
            for batch_indices in val_sampler:
                if not batch_indices:
                    continue
                tracks = _group_tracks(flat_val, batch_indices)
                bank_summary, query_embed, track_ids, seq_ids = _bank_and_query_embeds(
                    val_crop_helper, model, head, tracks, args.device, args.encode_batch_size,
                )
                val_loss_batch, _, _ = _combined_loss(bank_summary, query_embed, track_ids, seq_ids, head, args)
                total_val_loss += val_loss_batch.item()
                n_val_batches += 1
            val_loss = total_val_loss / n_val_batches if n_val_batches else float("nan")

        # identity_acc en val: solo informativo (episodios reales completos,
        # no se usa para elegir el checkpoint -- elegir por esta métrica
        # llevó a sobreajustar el checkpoint a val en corridas anteriores).
        compute_acc = args.val_acc_every > 0 and epoch % args.val_acc_every == 0
        if compute_acc:
            val_loader = _build_loader(
                val_index, args.val_episodes, args.val_max_bank_size, args.val_num_track_history,
                args.val_query_wide_factor, force_reid=True, seed=args.seed, batch_size=args.val_batch_size,
                embedding_cache=args.embedding_cache,
            )
            val_metrics = _evaluate_loader(
                model, val_loader, args.device, include_no_match=False, threshold=None,
                heatmap_temperature=0.05, head=head,
            )
            acc_str = f" val_identity_acc={val_metrics['identity_acc']:.4f} (informativo)"
        else:
            acc_str = ""
        print(
            f"[epoch {epoch}] train_loss={avg_loss:.4f} (triplet={avg_triplet:.4f} infonce={avg_infonce:.4f}) "
            f"({n_batches} batches) val_loss={val_loss:.4f}{acc_str}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_head(head, best_ckpt_path)
            print(f"[OK] Nuevo mejor val_loss={best_val_loss:.4f} -> {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"[INFO] Sin mejora hace {epochs_no_improve}/{args.patience} epoch(s) (mejor val_loss={best_val_loss:.4f})")
            if epochs_no_improve >= args.patience:
                print(f"[STOP] Early stopping: sin mejora en {args.patience} epochs consecutivos")
                break

    print(f"\n[OK] Entrenamiento terminado. Mejor checkpoint: {best_ckpt_path} (val_loss={best_val_loss:.4f})")
    best_head = load_head(best_ckpt_path, args.device)
    return best_head, best_val_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dinov2-model-name", default="facebook/dinov2-small")
    parser.add_argument(
        "--num-track-history", type=int, default=4,
        help="Tamaño del banco por track (K de _load_track_history_crops, igual convención que "
             "el resto del proyecto) -- el sampler muestrea --num-track-history + 1 vistas por "
             "track: las --num-track-history más antiguas arman el banco, la más reciente es la query.",
    )
    parser.add_argument(
        "--query-wide-factor", type=float, default=9.0,
        help="Factor del recorte AMPLIO de la query (mismo que usa ReIDEpisodeDataset real).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="linear_reid_head_run")
    parser.add_argument(
        "--embedding-cache", default=None,
        help="Ruta a un PrecomputedEmbeddingCache (ver precompute_p4_roi_cache.py o "
             "train_track_query_reid.py precompute-cache) -- si se pasa, salta el crop+encoder "
             "y usa embeddings ya calculados por (image_path, bbox). Debe corresponder al MISMO "
             "encoder que --encoder/--dinov2-model-name. OJO: desactiva la aumentación de train "
             "(ColorJitter) -- el cache es sin augmentar.",
    )
    add_encoder_args(parser)
    add_training_args(parser)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.embedding_cache:
        from reid_dataloader import PrecomputedEmbeddingCache
        print(f"[INFO] Usando embedding cache: {args.embedding_cache}")
        args.embedding_cache = PrecomputedEmbeddingCache.load(args.embedding_cache)

    model = build_encoder(args)

    train_root, train_seq_dirs = resolve_sequence_dirs(None, "train")
    train_index = build_track_index(train_root, sequence_dirs=train_seq_dirs)
    val_root, val_seq_dirs = resolve_sequence_dirs(None, "val")
    val_index = build_track_index(val_root, sequence_dirs=val_seq_dirs)

    best_head, best_val_loss = train_head(model, train_index, val_index, args, args.output_dir)
    best_ckpt_path = os.path.join(args.output_dir, "best_head.pt")

    val_loader = _build_loader(
        val_index, args.val_episodes, args.val_max_bank_size, args.val_num_track_history,
        args.val_query_wide_factor, force_reid=True, seed=args.seed, batch_size=args.val_batch_size,
        embedding_cache=args.embedding_cache,
    )
    final_val_metrics = _evaluate_loader(
        model, val_loader, args.device, include_no_match=False, threshold=None,
        heatmap_temperature=0.05, head=best_head,
    )
    print(f"[INFO] identity_acc en val con el MEJOR checkpoint (val_loss más bajo): {final_val_metrics['identity_acc']:.4f}")
    print(
        "[INFO] Para la evaluación completa (match_acc con umbral calibrado + fallos reales de "
        "los trackers), corré:\n"
        f"  python3 analyze_dinov2_nn_baseline.py --full-pipeline "
        f"--linear-head-checkpoint {best_ckpt_path} "
        f"--reid-gap-onnx checkpoints/rtdetrv2_v3_qcs_videos.onnx"
    )


if __name__ == "__main__":
    main()
