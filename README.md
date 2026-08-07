# visual_damages_tracking

Contexto reducido del pipeline de ReID por track-queries (banco de tracks
perdidos + cross-attention) para reidentificar daños/vehículos que
desaparecen por oclusión y reaparecen más tarde. Extraído de
`damages_tracking` con solo los 3 archivos que hacen falta para correr
`train_track_query_reid.py` (sin otras dependencias locales):

- `reid_dataloader.py` — `ReIDEpisodeDataset` (genera episodios: banco +
  queries), `PrecomputedEmbeddingCache`.
- `reid_track_query_module.py` — `TrackQueryReIDModule` (DINOv2 + fusión
  multi-escala + memoria + cross-attention contra el banco), losses.
- `train_track_query_reid.py` — subcomandos `dump-batches` (debug visual),
  `train` (entrenamiento con early stopping + eval de test), `precompute-cache`
  (cachear embeddings DINOv2 para iterar rápido).

## Historial de entrenamiento

### Corrida 1 — bug de posición fija en el banco (encontrado por lectura de código)

Se detectó que el track que sí matchea siempre se agregaba **primero** a la
lista del banco (antes del loop de distractores), con `true_identity_bank_pos`
hardcodeado en `0`. El modelo no necesitaba aprender identidad real para
`id_acc`: le alcanzaba con aprender "si hay match, es el slot 0". Se corrigió
con un `random.shuffle` de los slots válidos del banco antes del padding,
actualizando la posición del match tras el shuffle (`reid_dataloader.py`).
Verificado con 200 episodios sintéticos: la posición del match pasó de estar
siempre en 0 a distribuirse uniformemente entre los `max_bank_size` slots.

### Corrida 2 — post-fix del shuffle, 16+ epochs

Con el fix ya aplicado (pero todavía sin aumentación ni val fijo):

- **Train**: mejora monótona sin frenar — `total_loss` 1.12→0.35-0.50,
  `id_acc` 0.61→0.86-0.91 hacia el epoch 13-16.
- **Val**: el mejor `val_loss` quedó fijo en el **epoch 1** (`1.2601`) y no
  volvió a mejorar — a partir de ahí `match_acc` se derrumbó (0.78→0.52-0.70,
  errático) y `val_loss` empeoró/osciló (1.4-1.9). Early stopping
  (`patience=15`) estaba a punto de activarse (~epoch 16).
- **Detalle importante**: `id_acc` en val, la métrica que más importa para
  el objetivo real (reidentificación), **siguió mejorando lento y sostenido**
  (0.55→0.67-0.69) incluso mientras `total_loss`/`match_acc` empeoraban. Esto
  implica que el criterio de "mejor checkpoint" (hoy: `val_loss` total,
  dominado por el componente de match) puede estar descartando modelos que
  en la práctica identifican mejor, solo porque el clasificador de match se
  desestabiliza.

**Diagnóstico**: overfitting clásico. Train mejora sin límite mientras val se
estanca/empeora — consistente con: (a) el encoder DINOv2 genérico congelado
puede no separar bien identidades de daños específicas del dominio, (b) no
había ninguna aumentación de datos en el pipeline de crops (deterministas
pixel por pixel), (c) el dataset de val se resampleaba de cero en cada epoch
(no era un benchmark estable), lo que metía ruido adicional a la curva.

### Cambios aplicados después de esta corrida (aún no probados en una corrida nueva)

1. `AUGMENTED_TRANSFORM` (ColorJitter: brillo/contraste/saturación/hue leve)
   para train — deliberadamente sin geometría (flip/resized-crop), porque
   invalidaría el bbox relativo ya calculado (`SpatialEncoding`) si no se
   ajusta en conjunto.
2. `seed` fijo en `ReIDEpisodeDataset` para val/test — mismos episodios
   epoch a epoch y entre corridas, para que la curva de val sea comparable
   de verdad (confirmado: val idéntico entre epochs con seed fijo).
3. `PrecomputedEmbeddingCache` + subcomando `precompute-cache` — cachea los
   embeddings de DINOv2 por observación (frame+bbox) para iterar rápido en
   experimentos sin correr DINOv2 en cada época. Solo válido con encoder
   congelado y sin aumentación (por eso el cache de train se genera con el
   transform sin aumentar, aunque el split sea train) — pensado para
   comparaciones rápidas, no para la corrida final.

### Pendiente / próximos pasos a probar

- Monitorear `id_acc` de val (no solo `val_loss` total) como criterio de
  mejor checkpoint, dado lo observado en la corrida 2.
- Probar el checkpoint de DINOv2 fine-tuneado (`dinov2_reid_finetune.py`) en
  vez del genérico, para ver si el techo de `id_acc` sube.
- Si el overfitting persiste con augmentación de color sola: aumentación
  geométrica (jitter de la caja del crop antes de recortar, recalculando el
  bbox relativo) y/o reducir capacidad (`--dim` más chico) o subir
  `--dropout`/`--weight-decay`.
