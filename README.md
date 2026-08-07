# visual_damages_tracking

Pipeline de ReID por matching set-to-set (banco de tracks perdidos +
cross-attention) para reidentificar daños/vehículos que desaparecen por
oclusión y reaparecen más tarde. Extraído de `damages_tracking` con solo
los archivos que hacen falta para correr `train_track_query_reid.py` (sin
otras dependencias locales):

- `reid_dataloader.py` — `ReIDEpisodeDataset` (genera episodios: banco de
  tracks candidatos + una query dual-view), `DamageTrackIndex`,
  `PrecomputedEmbeddingCache`.
- `reid_track_query_module.py` — solo se usa de acá `DINOv2Encoder` (el
  resto del archivo, `TrackQueryReIDModule` y compañía, es la arquitectura
  VIEJA, reemplazada — ver "Corrida 3" más abajo; se dejó sin borrar por
  las dudas, pero no está en uso).
- `set_reid_model.py` — arquitectura ACTUAL: `TrackReIDModel`
  (`IntraTrackEncoder` + `CrossAttentionMatcher` + `MatchScorer`, softmax
  (k+1)-way con clase explícita de "no match") y `SetTrackReIDModule`
  (wrapper end-to-end con el encoder + proyección opcional).
- `hierarchical_pk_sampler.py` — `HierarchicalPKSampler` (sampler P
  secuencias × K tracks × N vistas, con reporte de composición y
  batch-hard triplet mining) + `TrackAnchorPool`/`build_track_pk_sampler`,
  el adaptador que lo conecta a `ReIDEpisodeDataset` (`pk_driven=True`)
  para controlar qué tracks anclan cada batch de train. La loss de triplet
  mining está implementada pero TODAVÍA NO conectada al entrenamiento real
  (ver pendientes).
- `train_track_query_reid.py` — subcomandos `dump-batches` (debug visual),
  `train` (entrenamiento con early stopping + eval de test),
  `precompute-cache` (cachear embeddings DINOv2 para iterar rápido).
- `run_encoder_sweep.sh` — sweep secuencial de encoders (DINOv2/DINOv3) y
  de `--query-wide-factor` sobre el servidor con GPU dedicada.
- `train_visual_damages_tracking.slurm` — job SLURM para entrenar en un
  cluster con GPU compartida (usa `VDT_DATASET_BASE` para apuntar el
  dataset a una ruta distinta a la del servidor de GPU dedicada, sin
  perder `SPLIT_OVERRIDES`).

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

### Corrida 3 — reemplazo completo de arquitectura (set-to-set matching)

Se reemplazó `TrackQueryReIDModule` (multi-escala + memoria + fusión) por
`TrackReIDModel`/`SetTrackReIDModule` (`set_reid_model.py`): banco de
tracks candidatos, cada uno resumido por un `IntraTrackEncoder`
(self-attention + pooling sobre sus vistas de historial), cross-attention
contra una query de **dos vistas** (ajustada + amplia, para dar contexto
de en qué parte del vehículo está el daño — no hay piezas etiquetadas para
resolver esto con clasificación), y `MatchScorer` con una clase explícita
de "no match". Además: track history con K observaciones repartidas en el
tiempo (no solo la última), padding espacial por posición, rebalanceo de
splits (`SPLIT_OVERRIDES` en `train_track_query_reid.py`, mueve 2
secuencias entre train/val para no concentrar los pocos casos de reid real
en un solo split), y `history_jitter_ratio`/`synthetic_gap_max` para
romper "huellas" deterministas por track y simular oclusiones más
realistas en los positivos sintéticos.

### Corrida 4 — sweep de encoders + fixes de agregación de métricas

- Sweep de `dinov2-small/base` y `dinov3-vits16/vitb16-pretrain-lvd1689m` +
  `--query-wide-factor` (`run_encoder_sweep.sh`). Los encoders "base"
  (768-dim) hacían OOM en la GPU de 8GB sin proyección — se agregó
  `--proj-dim` (default 256) para reducir la dimensión interna del modelo
  independientemente del encoder.
- **Bug de agregación en `id_acc`/`intra_seq_error_rate`/
  `inter_seq_error_rate`**: `run_epoch` promediaba el ratio YA calculado
  por batch, dándole el mismo peso a un batch con 1 evento (ratio binario
  0/1) que a uno con 5 — con `batch_size` chico esto generaba oscilaciones
  enormes de época a época sin cambio real del modelo. Fix: acumular
  conteos crudos (aciertos/total, intra/inter) sobre toda la época y
  calcular el ratio una sola vez al final.
- `HierarchicalPKSampler` conectado a TRAIN (`--use-hierarchical-sampler
  --pk-sequences-per-batch P --pk-tracks-per-sequence K`): controla qué P
  secuencias × K tracks anclan los episodios de cada batch (en vez de
  sample_random_identity() completamente al azar), sin tocar la loss ni la
  arquitectura. El batch-hard triplet loss que motivó el sampler está
  implementado pero no conectado todavía (ver pendientes).
- **Bug de splits sintéticos degenerados**: con `synthetic_gap_max` grande
  relativo a la longitud de un track, `_try_synthetic_split` podía dejar
  el lado `before` (el que va al banco como "true match") con **1 sola
  observación real** — un slot de banco con la mayoría de las posiciones
  enmascaradas, tan pobre que el modelo aprendía a tratarlo como
  sospechoso y preferir "no match". Verificado con datos reales
  (`synthetic_gap_max=80`): ~38% de splits degenerados en test contra ~24%
  en train — como `test` usa `force_reid=True` (100% de episodios pasan
  por este código, contra ~60% en val), el sesgo se amplificaba ahí al
  punto de que `match_acc` de test caía muy por debajo del azar aun con un
  `val_loss` de checkpoint razonable. Fix: exigir un mínimo de
  observaciones reales en `before` (si no se cumple, el intento se
  descarta y cae a otro caso, en vez de aceptar el split degenerado).
  Además, `synthetic_gap_max` pasó a ser una aumentación exclusiva de
  TRAIN — val/test siempre usan 0, para que sean comparables entre
  corridas/epochs independientemente de este flag.
- Soporte para correr en un cluster SLURM con el dataset en otra ruta
  (`VDT_DATASET_BASE`, ver `train_visual_damages_tracking.slurm`) sin
  perder `SPLIT_OVERRIDES` (a diferencia de pasar
  `--train-data-root`/`--val-data-root` explícitos, que lo bypassean).

### Pendiente / próximos pasos

- Relanzar entrenamiento con el fix de splits degenerados + `--synthetic-gap-max`
  desacoplado de val/test, y confirmar que `match_acc`/`id_acc` de test
  dejan de estar muy por debajo de los de val.
- Conectar (o descartar explícitamente) el batch-hard triplet loss de
  `hierarchical_pk_sampler.py` al entrenamiento real — hoy el sampler solo
  controla composición de batch, la loss auxiliar nunca se sumó.
- Evaluación con trackers clásicos (ByteTrack/BoT-SORT vía `boxmot`) sobre
  el split de test al final del entrenamiento, para identificar los casos
  donde esos algoritmos NO logran hacer la reidentificación (justifica el
  ReID custom) — hay una versión vieja de esto en `damages_tracking`
  (`botsort_boxmot_reid_eval.py`) pero no calcula esta comparación contra
  GT explícitamente; hay que escribirla de cero. Además la API de `boxmot`
  cambió entre la versión que usaba ese script (19.x) y la actual (22.x) —
  hay que fijar versión y revalidarla antes de portar. Pendiente de diseño.
- DDP (multi-GPU) explícitamente pospuesto hasta tener más de una GPU
  disponible — ver notas de diseño ya identificadas (sharding del dataset
  generativo, reducción de métricas entre ranks) cuando llegue el momento.
- Revisar los runs del sweep viejo que fallaron por OOM antes del fix de
  `--proj-dim` (`dinov3-vitb16`, `dinov2-base`) — nunca se relanzaron.
