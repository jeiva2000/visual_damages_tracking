"""
Entrena BilinearAttentionScorer (ver bilinear_attention_scorer.py): cabeza
de atención cruzada minimalista sobre el CLS CONGELADO de DINOv2,
alternativa independiente a LinearReIDHead (linear_reid_head.py /
train_linear_reid_head.py) -- comparable en la misma escala que "DINOv2
puro"/v6/v7 en la tabla de resultados, no se compone con LinearReIDHead.

A diferencia de LinearReIDHead (batches P x K x N de tracks sueltos +
triplet/InfoNCE sobre todo el batch como una sola pool), esta cabeza
necesita episodios explícitos (1 query + banco de hasta max_bank_size
candidatos + índice del match real, o "sin match" si
target==max_bank_size) -- exactamente la estructura que ya arma
ReIDEpisodeDataset/_build_loader (hoy usada solo para evaluación). Se
reusa esa misma infraestructura para entrenar, en vez de la maquinaria
HierarchicalPKSampler.

Loss (ver también el docstring de bilinear_attention_scorer.py):
  - CE identidad: cross_entropy(score_mixed, target) SOLO en episodios con
    match real -- entrena W (Q/K) a elegir bien entre candidatos, mismo
    rol que la cosine cruda del baseline actual.
  - BCE confianza: binary_cross_entropy_with_logits(score_final *
    --conf-temperature, es_match) sobre TODOS los episodios (incluye
    NEW_TRACK) -- esta es la que efectivamente entrena `gate` con una
    señal directa ligada a la decisión de match/no-match. `score_mixed`
    es invariante (en el argmax) a `gate` -- sin esta loss separada,
    `gate` solo recibiría el efecto lateral de temperatura de la CE de
    identidad, señal débil e indirecta. --conf-temperature reescala
    score_final (cosine, rango [-1,1]) al rango de logit ANTES de la BCE
    -- no toca la fórmula de score_mixed en sí.

Selección de checkpoint por val_loss (mismo criterio, ya validado, que
train_linear_reid_head.py -- elegir por identity_acc en val llevó a
sobreajustar el checkpoint a val en corridas anteriores de esa cabeza).

Uso:
    python3 train_bilinear_attention_scorer.py --output-dir bilinear_attn_run

    # Evaluación completa, comparable directo contra DINOv2 puro / v6 / v7:
    python3 analyze_dinov2_nn_baseline.py --full-pipeline \\
        --attn-scorer-checkpoint bilinear_attn_run/best_scorer.pt \\
        --reid-gap-onnx checkpoints/rtdetrv2_v3_qcs_videos.onnx
"""
import argparse
import os

import torch
import torch.nn.functional as F
from transformers import AutoModel

from analyze_dinov2_nn_baseline import _build_loader, _embed_batch
from bilinear_attention_scorer import BilinearAttentionScorer, load_scorer, save_scorer
from train_track_query_reid import build_track_index, resolve_sequence_dirs


def _step_losses(scorer, model, batch, device, conf_temperature, isolated=False):
    """CLS crudo de DINOv2 (congelado, _embed_batch con head=None) -> scorer
    -> (CE identidad, BCE confianza), ver docstring del módulo.

    `isolated=True` (ver --isolated-scorer): la CE de identidad usa los
    logits CRUDOS de atención (sin mezcla de cos_i/gate) -- aísla si W
    puede aprender algo útil cuando recibe gradiente completo desde el
    paso 1, en vez de atenuado por sigmoid(gate)≈0. La BCE de confianza no
    cambia (score_final NUNCA pasó por `gate`)."""
    track_summary, query_embeds, bank_mask, S = _embed_batch(model, batch, device, head=None)
    track_summary_t = torch.from_numpy(track_summary).float().to(device)
    query_embeds_t = torch.from_numpy(query_embeds).float().to(device)
    bank_mask_t = torch.from_numpy(bank_mask).to(device)
    target = batch["target"].to(device)

    score_mixed, score_final, _ = scorer(query_embeds_t, track_summary_t, bank_mask_t, isolated=isolated)

    is_match = target < S
    if is_match.any():
        ce_loss = F.cross_entropy(score_mixed[is_match], target[is_match])
    else:
        ce_loss = torch.zeros((), device=device)

    bce_loss = F.binary_cross_entropy_with_logits(score_final * conf_temperature, is_match.float())
    return ce_loss, bce_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dinov2-model-name", default="facebook/dinov2-small")
    parser.add_argument("--low-dim", type=int, default=128, help="Dimensión de la proyección bilineal compartida W (Q/K).")
    parser.add_argument(
        "--gate-init", type=float, default=-5.0,
        help="sigmoid(gate_init) ~ 0: el modelo arranca siendo EXACTAMENTE el baseline de cosine cruda.",
    )
    parser.add_argument(
        "--conf-temperature", type=float, default=4.0,
        help="Reescala score_final (cosine, rango [-1,1]) antes de la BCE de confianza -- NO afecta "
             "la fórmula de score_mixed, es solo para que la BCE no sature.",
    )
    parser.add_argument("--bce-weight", type=float, default=1.0, help="Peso de la BCE de confianza sumada a la CE de identidad.")
    parser.add_argument(
        "--isolated-scorer", action="store_true",
        help="Experimento de aislamiento: elimina por completo la mezcla con cos_i y con `gate` -- "
             "la CE de identidad entrena directo sobre los logits crudos de atención (gradiente NO "
             "atenuado hacia W desde el paso 1). Sirve para separar dos hipótesis cuando "
             "sigmoid(gate) colapsa a ~0: (a) el gate satura y atenúa el gradiente antes de que W "
             "aprenda algo, o (b) la atención en sí no encuentra señal aprendible con este volumen "
             "de datos. El checkpoint guarda este modo (ver save_scorer) -- "
             "analyze_dinov2_nn_baseline.py lo detecta solo al evaluar, sin flag aparte.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--episodes-per-epoch", type=int, default=400)
    parser.add_argument("--val-episodes", type=int, default=400)
    parser.add_argument("--max-bank-size", type=int, default=16)
    parser.add_argument("--num-track-history", type=int, default=4)
    parser.add_argument("--query-wide-factor", type=float, default=9.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="bilinear_attn_run")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Cargando {args.dinov2_model_name}...")
    model = AutoModel.from_pretrained(args.dinov2_model_name).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False
    embed_dim = model.config.hidden_size

    train_root, train_seq_dirs = resolve_sequence_dirs(None, "train")
    train_index = build_track_index(train_root, sequence_dirs=train_seq_dirs)
    val_root, val_seq_dirs = resolve_sequence_dirs(None, "val")
    val_index = build_track_index(val_root, sequence_dirs=val_seq_dirs)

    # force_reid=False en ambos -- episodios mixtos (match real + NEW_TRACK),
    # necesario para la BCE de confianza. train: seed=None (nueva muestra
    # cada época, ver docstring de ReIDEpisodeDataset). val: seed=args.seed
    # (mismos episodios cada época, val_loss comparable entre épocas).
    train_loader = _build_loader(
        train_index, args.episodes_per_epoch, args.max_bank_size, args.num_track_history,
        args.query_wide_factor, force_reid=False, seed=None, batch_size=args.batch_size,
    )
    val_loader = _build_loader(
        val_index, args.val_episodes, args.max_bank_size, args.num_track_history,
        args.query_wide_factor, force_reid=False, seed=args.seed, batch_size=args.batch_size,
    )

    scorer = BilinearAttentionScorer(embed_dim, args.low_dim, gate_init=args.gate_init).to(args.device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(args.output_dir, "best_scorer.pt")

    for epoch in range(1, args.epochs + 1):
        scorer.train()
        total_loss, total_ce, total_bce, n_batches = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            ce_loss, bce_loss = _step_losses(scorer, model, batch, args.device, args.conf_temperature, isolated=args.isolated_scorer)
            loss = ce_loss + args.bce_weight * bce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_bce += bce_loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches if n_batches else float("nan")
        avg_ce = total_ce / n_batches if n_batches else float("nan")
        avg_bce = total_bce / n_batches if n_batches else float("nan")

        scorer.eval()
        with torch.no_grad():
            total_val_loss, n_val_batches = 0.0, 0
            for batch in val_loader:
                ce_loss, bce_loss = _step_losses(scorer, model, batch, args.device, args.conf_temperature, isolated=args.isolated_scorer)
                val_loss_batch = ce_loss + args.bce_weight * bce_loss
                total_val_loss += val_loss_batch.item()
                n_val_batches += 1
            val_loss = total_val_loss / n_val_batches if n_val_batches else float("nan")

        gate_val = torch.sigmoid(scorer.gate).item()
        print(
            f"[epoch {epoch}] train_loss={avg_loss:.4f} (ce={avg_ce:.4f} bce={avg_bce:.4f}) "
            f"({n_batches} batches) val_loss={val_loss:.4f} sigmoid(gate)={gate_val:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_scorer(scorer, best_ckpt_path, isolated=args.isolated_scorer)
            print(f"[OK] Nuevo mejor val_loss={best_val_loss:.4f} -> {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"[INFO] Sin mejora hace {epochs_no_improve}/{args.patience} epoch(s) (mejor val_loss={best_val_loss:.4f})")
            if epochs_no_improve >= args.patience:
                print(f"[STOP] Early stopping: sin mejora en {args.patience} epochs consecutivos")
                break

    print(f"\n[OK] Entrenamiento terminado. Mejor checkpoint: {best_ckpt_path} (val_loss={best_val_loss:.4f})")
    best_scorer = load_scorer(best_ckpt_path, args.device)
    print(f"[INFO] isolated={best_scorer.isolated} -- sigmoid(gate) final = {torch.sigmoid(best_scorer.gate).item():.4f}")
    print(
        "[INFO] Para la evaluación completa (match_acc/identity_acc comparable directo contra "
        "DINOv2 puro / v6 / v7), corré:\n"
        f"  python3 analyze_dinov2_nn_baseline.py --full-pipeline "
        f"--attn-scorer-checkpoint {best_ckpt_path} "
        f"--reid-gap-onnx checkpoints/rtdetrv2_v3_qcs_videos.onnx"
    )


if __name__ == "__main__":
    main()
