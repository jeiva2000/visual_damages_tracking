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

## Revisiones (paperreview.ai + propia) — estado

Checklist vivo, actualizar a medida que se resuelve cada punto. Lectura
conjunta hecha por el usuario cruzando su propia revisión crítica con la de
la herramienta externa `paperreview.ai`; se marca cada uno según coincidan
o no, y su estado real de resolución (verificado contra el código/paper,
no solo discutido).

- [x] **M1 · Coste computacional no medido** — coincidencia total entre
  ambos informes, el punto más insistente de los dos. Resuelto en buena
  parte: `paper/live_validation/section_draft.tex` (`sec:computational_cost`)
  ya tiene FPS con/sin cada módulo para los 4 trackers, breakdown por
  evento (extracción de embedding vs. búsqueda en el banco) sobre
  ByteTrack, y footprint de memoria del banco. Falta: una frase explícita
  sobre cómo escala el costo con la longitud del video y el número de
  tracks (la Q2 de paperreview.ai lo pide explícitamente) — hoy se da el
  costo medido en las secuencias del dataset (banco nunca pasa de 30-32
  identidades) pero no una afirmación de complejidad/escalado.
- [x] **M2 · Tamaño muestral del análisis de huecos (Wilson CI)** —
  coincidencia (paperreview.ai mira Tabla V, 15-29 huecos; la lectura
  propia apunta a Tabla VI, denominadores 17/22/12/10). **Resuelto**:
  Wilson CI agregado a `table_tracker_base_reid_fail_alldata.tex` (Tabla V)
  y ya estaba en `table_all_methods_same_cases.tex`/`_pct.tex` (Tabla VI).
- [x] **M5 · Sesgo del ground truth semiautomático / circularidad** —
  coincidencia por ángulos distintos (propia: circularidad, features
  RT-DETR evaluadas sobre GT generado con RT-DETR, huecos cortos
  interpolados; paperreview.ai: propagación de sesgos del detector, pide
  precisión/recall del pseudo-labeling). **Resuelto** con datos reales:
  usando `report_manual_corrections.py` (herramienta de anotación,
  `~/Documents/annotate/`, compara pseudo-track original vs. GT final
  caja por caja por IoU) se encontró que de las 72 secuencias, 55 (76%)
  se anotaron 100% a mano sin ayuda del detector; de las 17 restantes
  (asistidas con pseudo-tracks de RT-DETRv2 + interpolación PCHIP en
  huecos cortos), precisión del pseudo-labeling=51.7%, recall=38.5%, y
  solo 1 caja de 15.293 (0.007%) sobrevivió sin cambios — el resto se
  corrigió o se agregó desde cero. Escrito como cuarto punto de
  `conclusions/section_draft.tex` (`sec:conclusions_limitations`),
  incluyendo el caveat honesto de que "agregada" no distingue caja
  genuinamente nueva de frame interpolado.
- [x] **M9 · Hiperparámetros del banco y umbral sin especificar** (incluye
  **P2 · Política del banco: TTL, expulsión y falsos positivos**) —
  coincidencia, ampliada por paperreview.ai (política de TTL/expulsión del
  banco, riesgo de acumulación de falsos emparejamientos en videos largos,
  tamaño de galería al momento de reidentificar). **Resuelto**: se revisó
  `eval_tracking/reid_gallery.py` (código real, no se asumió nada) y se
  agregó un párrafo de metodología al principio de
  `live_validation/section_draft.tex` describiendo: cuándo entra/sale algo
  del banco (`bury()`, solo tracks recién perdidos), la regla de expulsión
  exacta (`age_and_prune()`, TTL puro vía `frames_lost > max_gap`, SIN tope
  de tamaño explícito — la afirmación literal de paperreview.ai de que "no
  hay TTL" era incorrecta, el mecanismo ya existía, solo faltaba
  documentarlo), el mecanismo de matching (asignación húngara + umbral de
  coseno, no vecino-más-cercano), y que el umbral es global-por-método
  (no por-tracker, no adaptativo). Tamaño de galería observado (30-32) ya
  estaba en `sec:computational_cost`.

  **2026-09-04 — acelerando la robustez empírica de P2 con un "replay" en
  vez de correr 20 veces el tracking en vivo completo**: cambiar
  `--reid-stitch-max-gap` no cambia NINGÚN embedding ni cuándo nace/muere
  un raw_id (eso lo decide el tracker de motion/IoU, sin relación con el
  banco externo) — solo cambia la regla de expulsión. Se creó, como
  COPIAS aisladas (nada del pipeline original tocado, ver
  `feedback_isolate_experimental_pipeline_changes` en memoria):
  - `eval_tracking/main_track_ttl_trace.py` (copia de `main_track.py` +
    flag `--dump-gallery-trace`: graba, por secuencia, cada evento
    bury/match del banco con su embedding crudo y frame, más el MOT SIN
    reid-stitch).
  - `eval_tracking/run_eval_ttl_trace.py` (copia de `run_eval.py`, importa
    la copia de arriba, agrega `--dump-gallery-trace-dir`).
  - `eval_tracking/replay_ttl_ablation.py` (nuevo): re-simula
    `age_and_prune`/`match_newborns` reusando `reid_gallery.ReidGallery`
    TAL CUAL (mismo código, sin reimplementar nada) alimentada por la
    traza cacheada, para cualquier `max_gap`, sin GPU ni detector — luego
    reconstruye el MOT final y vuelve a correr TrackEval (subprocess,
    rápido). Payoff esperado: ~18h (20 corridas de tracking en vivo) →
    ~1-2h (1 corrida real para grabar la traza + replays casi
    instantáneos).
  - **Validación en curso**: generando la traza de referencia
    (ByteTrack, C4, max_gap=200 default) para confirmar que el replay con
    max_gap=200 reproduce EXACTO el resultado ya conocido (HOTA=47.112,
    MOTA=20.499, IDF1=60.046, IDSW=327, de
    `c4_puro_live_bytetrack_kfoldpool.log`) antes de confiar en los demás
    valores del barrido. Si no coincide exacto, no usar los resultados del
    replay sin depurar primero.
- [ ] **M10 · Ablaciones faltantes** — coincidencia parcial y
  complementaria (paperreview.ai: umbral + tamaño de banco/TTL +
  false-recall vs. missed-recall; propia: factores de escala 6/1.6/1.0 +
  fusión de las 2 vistas). Estado por sub-punto:
  - [x] false-recall vs. missed-recall: ya resuelto, `table_live_gap_outcomes_alldata.tex`
    ya desglosa Id-swap/New-id-born.
  - [~] sensibilidad al umbral: **bug nuevo encontrado y corregido
    2026-09-03**: `run_kfold_cv` nunca revisaba `--match-threshold` (a
    diferencia de `run_full_pipeline`), así que siempre recalibraba e
    ignoraba el valor fijo pedido — los 6 puntos de C4 y los 6 de P4-FPN
    daban resultado idéntico entre sí por este motivo, no por casualidad.
    Corregido (una sola línea, diff verificado contra el remoto antes de
    sincronizar). Parte C4+P4-FPN (12 puntos, CPU, vía
    `run_threshold_c4_p4fpn_cpu.sh`) ya recorrida con el fix y validada
    (ahora sí varía: C4 estable ~0.96 hasta umbral=0.90, cae a 0.81 en
    0.96 y a 0.17 en 0.99; P4-FPN estable ~0.92 hasta 0.60, cae a 0.79 en
    0.85 y a 0.05 en 0.97 — meseta amplia, el umbral calibrado real queda
    lejos del borde). Falta la parte DINOv2 (6 puntos), todavía en la cola
    de GPU (`run_threshold_sensitivity_ablation.sh`, va a re-hacer también
    C4/P4-FPN de paso, redundante pero inofensivo, ~15 min extra).
  - [~] robustez a tamaño de banco/TTL: ablación corriendo (cadena
    nocturna, `run_ttl_ablation_live.sh`).
  - [ ] factores de escala (6/1.6/1.0) y fusión de vistas: calculados en
    una sesión anterior (antes de esta), verificar si ya están escritos
    en el paper o solo discutidos en chat.
- [x] **M3 · Falta comparación con métodos que aprenden la asociación** —
  coincidencia en el diagnóstico, nombran contrincantes distintos (propia:
  OSNet/FastReID; paperreview.ai: TBDQ-Net + ReID autosupervisado) más
  detalles de entrenamiento de las cabezas H1-H4. Estado:
  - [x] OSNet (`osnet_x0_25_msmt17`) y CLIP-VehicleID (`clip_vehicleid.pt`,
    ReID entrenado en vehículos) ya agregados a Tablas VI/VII/IX.
  - [x] TBDQ-Net (asociador aprendido sobre features del detector) —
    **decisión de scope: NO se implementa, y NO se menciona en el paper.**
    Investigado (arXiv:2411.06197): a diferencia de MOTR/MeMOTR/MOTIP (que
    reemplazan todo el paradigma tracking-by-detection), TBDQ-Net usa un
    detector congelado (YOLOX) + un asociador liviano entrenable (BII+CPA)
    encima — conceptualmente más cercano a la ablación H1-H4 de este paper
    que a un paradigma distinto, así que el argumento "es otro paradigma"
    NO es del todo preciso y no se usa. Razón real de no implementarlo:
    (1) diseñado/evaluado en DanceTrack/SportsMOT/MOT20, escenas de
    crowding denso — desafío distinto al de este dominio (oclusión
    temporal de pocos objetos, no distinguir objetos simultáneos
    parecidos); (2) sin checkpoint reusable compatible con RT-DETR +
    este dataset, reimplementar BII+CPA desde la descripción del paper
    es desarrollo nuevo sustancial, no un drop-in como OSNet/CLIP-VehicleID;
    (3) la ablación H1-H4 ya existente responde la misma pregunta de fondo
    ("¿ayuda un componente aprendido encima de la señal congelada?") con
    un head más simple, en la misma dirección de evidencia. Queda
    documentado acá como decisión de scope, deliberadamente sin rastro en
    el manuscrito.
  - [x] Detalles de entrenamiento H1-H4 (splits, LR, épocas,
    regularización): los negativos (easy/hard) ya estaban bien
    documentados en `training_methodology/section_draft.tex`. Agregada
    nueva subsubsección "Optimization details" (`sec:head_variants_optimization`,
    antes de "Checkpoint selection") con los valores numéricos reales
    (sacados de los defaults de `add_training_args` en
    `train_linear_reid_head.py`, no overrideados en `run_p4_kfold_heads.sh`
    salvo `--new-track-prob`/`--infonce-weight`, que ya distinguen H1-H4 y
    ya estaban documentados): Adam, lr=1e-3, weight\_decay=1e-4, hasta 50
    épocas con early stopping (patience=10), batch P=2×K=8, margen
    triplet=0.3, $\lambda_\text{InfoNCE}$=0.3/temperatura=0.1, proyección
    a 128-dim. Los splits usan el mismo k-fold que el resto del paper, ya
    documentado en otra sección, no duplicado acá.
- [x] **M7 · Tabla I (artefactos / falta comparación CLS vs. patch token)** —
  coincidencia débil, lectura distinta (paperreview.ai: extracción de PDF;
  propia: la tabla no contiene la comparación CLS-vs-patch que el texto
  afirma). **Resuelto, y peor de lo pensado**: `table_dinov2_layers.tex`
  (`tab:dinov2_layers`, casi seguro la Tabla I real) tenía en el caption
  "CLS pooling outperformed mean-patch pooling at every layer (omitted for
  brevity)" — al recuperar esos valores del log ya guardado
  (`analyze_dinov2_layers_all.log`), la afirmación es **falsa**: patch-mean
  gana en las capas 0-11 en las 3 particiones, CLS solo gana (por margen
  amplio) en la última capa, que es la config elegida. Tabla reconstruida
  con las 13 capas × 2 poolings (primero por split, después reformulada
  para mostrar Intra/Hard/Easy promediados en vez de train/val/test, más
  clara). También se agregó `table_hard_negative_frame_distance.tex`: los
  pares "Hard" (identidad distinta, mismo video) están separados en
  mediana por 156 frames, ~5x más que los pares "Intra" (29 frames) —
  confirma que el negativo difícil no es un artefacto de comparar frames
  casi-simultáneos.

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

### Corrida 5 — de la arquitectura pesada a una capa lineal simple (`LinearReIDHead`)

Antes de seguir invirtiendo en `set_reid_model.py` (transformer completo:
`IntraTrackEncoder` + `CrossAttentionMatcher` + `MatchScorer`), se construyó
`analyze_dinov2_nn_baseline.py`: un baseline de 1-NN sobre el CLS crudo de
DINOv2 congelado, SIN ninguna cabeza entrenada. Resultado inesperado: ese
baseline crudo superaba ampliamente al modelo pesado ya entrenado
(`identity_acc` ~0.83 vs ~0.30) — evidencia fuerte de que la cabeza pesada
estaba destruyendo señal en vez de sumarla, en este dataset chico.

Se reemplazó por `linear_reid_head.py`/`train_linear_reid_head.py`: una
sola capa lineal sobre el CLS congelado, entrenada con metric learning
(batch-hard triplet, Hermans et al. 2017, vía `hierarchical_pk_sampler.py`)
+ InfoNCE auxiliar. Iterado v1→v7 corrigiendo, en orden:

1. **Fidelidad de los crops de entrenamiento**: v1/v2 entrenaban solo con
   recortes ajustados de una sola vista, pero la evaluación promedia
   múltiples vistas del banco — se corrigió reusando directo
   `ReIDEpisodeDataset._load_track_history_crops`/`_load_query_views` (v4).
2. **Criterio de selección de checkpoint**: elegir por `val_identity_acc`
   causaba divergencia val/test (val subía, test bajaba) — se cambió a
   `val_loss` (v5+), la curva val/test mejoró de forma consistente.
3. **Peso del InfoNCE auxiliar**: bajado de 1.0 a 0.3 tras comparar
   corridas (con 1.0 dominaba sobre la triplet loss).
4. **Señal de "no match" durante entrenamiento (v7)**: el entrenamiento
   nunca veía casos de query sin match real en el banco — se agregó
   `new_embedding` (embedding aprendible de rechazo, análogo liviano al de
   `MatchScorer`) + `--new-track-prob` para simular casos `NEW_TRACK`.

Tabla final del split fijo (identity_acc; `DINOv2 puro` = baseline crudo
sin cabeza):

| fuente | DINOv2 puro | v1 | v2 | v4 | v5 | v6 | v7 |
|---|---|---|---|---|---|---|---|
| val | 0.7266 | 0.7422 | 0.7734 | 0.8008 | 0.8203 | 0.8242 | 0.7695 |
| test | 0.7349 | 0.6586 | 0.6225 | 0.6948 | 0.6747 | 0.7108 | 0.6707 |
| bytetrack_test | 0.8929 | 0.8929 | 0.7500 | 0.9286 | 0.9286 | 0.9286 | 0.9286 |
| botsort_test | 0.7857 | 0.8571 | 0.7857 | 0.9286 | 0.9286 | 0.9286 | 0.8571 |
| bytetrack_combined | – | 0.9403 | 0.8657 | 0.9104 | 0.8955 | 0.9254 | 0.9552 |
| botsort_combined | – | 0.8889 | 0.8889 | 0.8889 | 0.9630 | 0.9630 | 0.8889 |

Dato clave que motivó la corrida siguiente: v7 (con `new_embedding`)
retrocede respecto a v6 en la mayoría de las fuentes sobre este split
fijo — pero mirar repetidamente el mismo test para decidir qué versión
"ganó" (v1→v7) invalida ese test como medición insesgada.

### Corrida 6 — validación cruzada de 6 folds (k-fold) + 4 configuraciones

Para reportar resultados sin el sesgo de haber mirado el test repetidamente
durante v1-v7, se construyó `kfold_utils.py` (partición por secuencia,
estratificada por presencia de brecha temporal real) y se extendió
`train_linear_reid_head.py`/`analyze_dinov2_nn_baseline.py` con
`--kfold-cv`: 6 folds a nivel de secuencia (train/val/test rotando,
ningún fold decide sus propios hiperparámetros), comparando 4
configuraciones con el MISMO `fold_assignment.json`:

- **DINOv2 puro** (`--kfold-skip-training`): sin entrenar nada, `head=None`
  en cada fold, mismo umbral calibrado por fold.
- **v6-style** (`--new-track-prob 0`): sin `new_embedding`, todo el
  entrenamiento vía triplet + InfoNCE normal.
- **Intermedio** (`--new-track-prob 0.25 --infonce-weight 0`): sí simula
  casos "nuevos" (señal de no-match vía negativo duro en la triplet), pero
  sin `new_embedding` (InfoNCE apagado).
- **v7** (default): `new_embedding` + `--new-track-prob 0.25`.

Resultado agregado (identity_acc, media ± desvío entre 6 folds):

| fuente | DINOv2 puro | v6-style | Intermedio | v7 |
|---|---|---|---|---|
| synthetic_test | 0.7019 ± 0.0502 | 0.7059 ± 0.0461 | 0.6687 ± 0.0756 | 0.6999 ± 0.0295 |
| bytetrack | 0.8896 ± 0.0890 | 0.9357 ± 0.0501 | 0.9152 ± 0.0471 | 0.9391 ± 0.0708 |
| botsort | 0.8258 ± 0.1330 | 0.9068 ± 0.1150 | 0.9670 ± 0.0609 | 0.9416 ± 0.0610 |

Las 3 configs con cabeza entrenada superan a DINOv2 puro en los fallos
reales de tracker. Test pareado (Wilcoxon/t-test) sobre las diferencias
por fold: ningún par alcanza p<0.05 — subpotenciado con solo 6 folds (ver
Corrida 8 para el test con más poder estadístico).

### Corrida 7 — `BilinearAttentionScorer` (atención cruzada minimalista) — descartado

Como punto intermedio entre la capa lineal y el transformer pesado
original, se probó `bilinear_attention_scorer.py`: 1 sola cabeza de
atención, W compartido entre Q/K, V sin proyectar, sin FFN, sin encoding
posicional, mezclado con la cosine cruda vía un `gate` residual
(`sigmoid(gate_init=-5)≈0`, para arrancar siendo el baseline actual).

**Bug encontrado y confirmado numéricamente**: la CE de identidad sobre
`score_mixed` daba gradiente **exactamente cero** hacia `W` (no solo
atenuado por el gate) — `score_final` se suma como constante por episodio
a todos los candidatos, y el softmax de la cross-entropy es invariante a
desplazamientos. Verificado: `|dW|` en modo normal = `4.3e-12` (ruido de
punto flotante) vs. modo aislado (logits crudos, sin mezcla) = `0.081`.

Se agregó `--isolated-scorer` (entrena directo sobre los logits crudos de
atención, sin cosine ni gate) para aislar la hipótesis. Resultado: con
gradiente completo, `W` **sobreajustó severamente** (CE de train
0.83→0.002 en 12 épocas mientras `val_loss` subía sin parar de 1.86 a
2.43) y **no superó a DINOv2 puro** en la mayoría de las fuentes (test
identity_acc 0.73→0.50, bytetrack_test 0.89→0.79; solo botsort_test
mejoró, 0.79→0.93). Conclusión: no era el gate — el mecanismo de atención
en sí no encuentra señal generalizable con este volumen de datos. No se
integró a k-fold (no vale la pena sin evidencia de que funcione en un
split simple). Próximo paso sugerido si se retoma: más datos o mining de
negativos más agresivo, no más arquitectura.

### Corrida 8 — comparación caso-por-caso y test de McNemar

Para complementar el test pareado por fold (subpotenciado, Corrida 6), se
escribió `compare_kfold_cases.py`: reevalúa cada caso REAL de fallo de
tracker (ByteTrack/BoT-SORT) con el checkpoint/umbral del fold de k-fold
al que pertenece su secuencia de test, sin reentrenar nada, para las 4
configs de la Corrida 6. Test exacto de McNemar sobre pares discordantes:

| comparación vs. DINOv2 puro | bytetrack (n=313) | botsort (n=127) |
|---|---|---|
| v6-style | 17 vs 2, **p=0.00073** | 10 vs 1, **p=0.01172** |
| Intermedio | 19 vs 11, p=0.20049 | 10 vs 2, p=0.03857 |
| v7 | 16 vs 8, p=0.15159 | 9 vs 2, p=0.06543 |

**v6-style es la única config con mejora estadísticamente significativa
sobre DINOv2 puro en AMBOS trackers reales** — v7 no alcanza significancia
en ninguno de los dos pese a tener identity_acc agregado comparable o
mayor. Recomendado como el modelo a proponer (candidato a "ReIDMatchingHead"
en el paper) — más simple (sin `new_embedding`/`new-track-prob`) y con la
evidencia estadística más sólida. Visualizaciones de los casos donde
difiere de DINOv2 puro (`visualize_case_diffs.py`) guardadas en
`case_diff_visuals/`.

### Corrida 9 — extensión del tuneo de hiperparámetros a OCSort/DeepOCSort

Hasta acá el tuneo (`eval_tracking/sweep_ultralytics_anchor.py`) solo
cubría ByteTrack/BotSort. Se extendió a OCSort/DeepOCSort, ancladas en los
rangos propios que boxmot publica para su propio tuning (Ray Tune) en vez
de en defaults de Ultralytics (no hay equivalente Ultralytics para estos
dos trackers).

**Bug real encontrado y arreglado**: `main_track.py::_make_tracker()`
construía OCSort compartiendo la rama de ByteTrack, pasándole
`track_thresh`/`match_thresh`/`track_buffer`/`frame_rate` — parámetros que
**no existen** en la firma real de OCSort (`min_conf`, `delta_t`,
`inertia`, `use_byte`, `Q_xy_scaling`, `Q_s_scaling` propios +
`det_thresh`/`max_age`/`min_hits`/`iou_threshold`/`asso_func` vía
`BaseTracker`). Como `BaseTracker.__init__` descarta kwargs desconocidos en
silencio, **toda corrida previa con `--tracker ocsort` corrió siempre con
los defaults de la librería**, ignorando cualquier flag pasado. Corregido
con una rama propia para OCSort.

Hiperparámetros finales (val/test, HOTA+IDF1):

| Tracker | Configuración | val | test |
|---|---|---|---|
| ByteTrack | track_thresh=0.20, match_thresh=0.90, track_buffer=30 | 74.29 | 75.40 |
| BotSort | track_thresh=0.20, match_thresh=0.80, track_buffer=30, proximity_thresh=0.50, appearance_thresh=0.40, cmc_method=ecc | 78.76 | 77.74 |
| OCSort | max_age=60, min_hits=5, iou_threshold=0.30, delta_t=5, inertia=0.40, asso_func=iou | 83.81 | 86.73 |
| DeepOCSort | max_age=30, min_hits=5, iou_threshold=0.40, w_association_emb=0.70, alpha_fixed_emb=0.99, aw_param=0.50, delta_t=2, inertia=0.40, asso_func=iou | 81.62 | 81.40 |

**Hallazgo notable**: OCSort (sin ninguna rama de apariencia, puro
movimiento+IoU) supera a DeepOCSort (ReID nativo siempre activo) en ambos
splits — contraintuitivo, vale la pena remarcarlo en el paper.

### Corrida 10 — k-fold (DINOv2/C4/P4-FPN) extendido a OCSort/DeepOCSort + P4-FPN real para ByteTrack/BotSort

Se replicó para OCSort/DeepOCSort el mismo análisis de k-fold que ya
existía para ByteTrack/BotSort (DINOv2 puro, C4 puro + 4 cabezas), y
además se corrió por primera vez la validación real de **P4-FPN**
(`encoder.fpn_blocks.0`, la fusión FPN real — distinta de "C4", el nombre
que se usó por error durante gran parte de la sesión anterior para
`backbone.res_layers.2`) para los 4 trackers, algo que antes solo existía
como diagnóstico preliminar de embeddings (sin tracker real de por medio).

**Incidente y bug real**: la primera corrida completa (~2h de GPU)
crasheó en silencio: `analyze_dinov2_nn_baseline.py::_print_metrics`
formateaba `match_acc` sin chequear `None` (pasa cuando un fold+tracker
tiene `n_evaluados=0` — común en OCSort/DeepOCSort, que al estar bien
tuneados fallan poco). Un bug propio de bash agravó el problema:
`echo "[$(ts)] ... exit=$?"` evalúa `$(ts)` (un comando real) ANTES de
expandir `$?`, así que siempre logueaba `exit=0` sin importar el resultado
real — los 3 scripts encadenados downstream vieron "terminó bien" y
siguieron corriendo sobre una base rota. Corregido: `_print_metrics` ahora
chequea `None` explícitamente, y los scripts de orquestación capturan
`status=$?` en la línea inmediatamente siguiente al comando, con abort
explícito si falla.

Tabla completa (identity_acc media±desvío entre 6 folds; el detalle
completo con recuento de aciertos por método está en
`paper/tracker_kfold/`):

| Método | ByteTrack | BotSort | OCSort | DeepOCSort |
|---|---|---|---|---|
| DINOv2 puro | 0.9604±0.0310 | 0.9358±0.0499 | 0.8540±0.2019 | 0.9202±0.0962 |
| **C4 puro** | **0.9920±0.0156** | **0.9835±0.0202** | **0.9710±0.0451** | **0.9467±0.0767** |
| P4-FPN puro | 0.9592±0.0179 | 0.8915±0.1498 | 0.8668±0.1259 | 0.8490±0.1367 |

**C4 puro es consistentemente la mejor config en los 4 trackers**,
superando incluso a las 4 variantes de cabeza entrenada — contradice lo
que sugería el diagnóstico preliminar de embeddings (donde P4-FPN ganaba
el gap intra/inter en la mayoría de los splits). McNemar extendido con
P4-FPN para ByteTrack/BotSort: solo ByteTrack alcanza significancia clara
(C4 puro p=0.00006, C4+v6style p=0.00739; P4-FPN+v6style marginal
p=0.036); BotSort y los otros 2 trackers no alcanzan p<0.05 en ninguna
comparación, muestra insuficiente.

### Corrida 11 — validación en vivo (online) de C4 puro como ReID externo (completada)

Todo lo anterior evalúa embeddings *post-hoc* sobre fallos ya identificados
del tracker, nunca dejando que el embedding decida en vivo durante el
tracking real. Se construyó `C4PuroReID` (`eval_tracking/custom_reid.py`),
que reproduce EXACTAMENTE la metodología validada en k-fold (3 escalas de
padding `[6.0, 1.6, 1.0]`, query=promedio ajustado+intermedio del frame
actual, banco=promedio de hasta 4 frames repartidos en el tiempo con la
misma regla posicional amplio/intermedio/ajustado) conectada como
`--reid-stitch --reid-source c4_puro` en `main_track.py` — funciona para
los 4 trackers por igual (a diferencia del ReID nativo, que solo existe
para BotSort/DeepOCSort).

Resultados en el split test (8 secuencias), C4 puro en vivo vs. ReID
nativo (HOTA+IDF1):

| Tracker | C4 puro en vivo | Nativo | Δ |
|---|---|---|---|
| ByteTrack | 77.42 | 75.40 | **+2.02** |
| BotSort | 81.63 | 77.74 | **+3.88** |
| OCSort | 85.29 | 86.73 | **−1.44** |
| DeepOCSort* | 83.92 | 81.40 | **+2.52** |

*DeepOCSort usa C4PuroReID tanto interno (afecta su propia fusión
movimiento+apariencia) como externo — no es reemplazo 100% limpio como en
los otros 3.

3 de 4 trackers mejoran (sobre todo en IDF1, +2.5 a +3.8 puntos —
consistente con que el reid-stitch ataca identidad, no detección). OCSort
es la única excepción en esta muestra chica (n=8 secuencias) — se relanzó
la misma comparación sobre el pool completo de k-fold (64 secuencias, vía
un symlink farm en `eval_tracking/kfold_pool_view{,_gt}/pool/`) para tener
un número con más poder estadístico.

Resultado sobre el pool completo (64 secuencias), HOTA/MOTA/IDF1/IDSW,
nativo vs. +C4 en vivo (tabla completa en
`paper/live_validation/section_draft.tex`, `tab:live_c4_vs_native`):

| Tracker | HOTA nativo | HOTA +C4 | MOTA nativo | MOTA +C4 | IDF1 nativo | IDF1 +C4 | IDSW nativo | IDSW +C4 |
|---|---|---|---|---|---|---|---|---|
| ByteTrack | 45.96 | 47.11 | 16.24 | 20.50 | 55.98 | 60.05 | 481 | 327 |
| BotSort | 47.04 | 47.74 | 13.63 | 20.04 | 57.28 | 60.05 | 454 | 322 |
| OCSort | 49.92 | 51.23 | 40.55 | 42.21 | 61.90 | 66.15 | 382 | 139 |
| DeepOCSort* | 49.77 | 51.18 | 40.68 | 42.34 | 61.24 | 65.99 | 352 | 135 |

**Con la muestra grande, la caída de OCSort desaparece**: mejora en las 4
métricas, en los 4 trackers, sin excepción — confirma que la caída vista
en el split test de 8 secuencias era ruido de muestra chica, no un efecto
real de que C4 interactúe mal con el motion-only/Kalman puro de OCSort.
El HOTA se mueve poco (1.5–2.8% relativo) porque DetA no cambia (C4 solo
toca asociación/identidad, no detección) y la media geométrica de HOTA
diluye la mejora cuando DetA es el cuello de botella; IDF1 e IDSW sí la
aíslan bien: IDSW cae 29–32% en ByteTrack/BotSort y 62–64% en
OCSort/DeepOCSort. *DeepOCSort usa C4PuroReID tanto interno (afecta su
propia fusión movimiento+apariencia) como externo, no es reemplazo 100%
limpio como en los otros 3.

### Corrida 12 — checkpoint desactualizado en la tabla de oversmoothing por capa (completada)

La tabla de oversmoothing por capa (Inter-Sim/Intra-Sim/MAD/Discrim sobre
C2-C5/P4/P3/N4/N5/Encoder, `damages_tracking/linear_test_tracking.py`,
generada 14-jul) usaba por default un checkpoint **distinto**
(`best-2.pth`, 6-jul) al usado en todo el resto de esta sesión
(`rtdetrv2_v3_qcs_videos.pth`, 23-jul — un entrenamiento posterior,
hashes MD5 distintos). El dataset ya estaba bien (`dataset_split_2`, pese
a que la variable se llama `DANCETRACK_ROOT` por una plantilla vieja).
Relanzado con el checkpoint correcto sobre `train` (57 secuencias, a
pedido explícito, en vez del `val` de 8 secuencias que usaba la tabla
original).

Al usar la muestra más grande y variada apareció un bug latente (no una
regresión): `generate_umap_comparison` crasheaba con
`TypeError: Cannot use scipy.linalg.eigh for sparse A with k >= N` en
secuencias con muy pocas detecciones (p.ej. `video_0KIroAqMShhattx2VTH_11`,
solo 3 matches). Es una visualización de diagnóstico (UMAP P4 vs Decoder5),
no afecta las métricas de oversmoothing en sí — se envolvió la llamada en
try/except para omitir la visualización en esos casos y continuar la
corrida.

Resultado final (43 de 57 secuencias con datos suficientes para el probe
lineal; Inter-Sim↓ / Intra-Sim↑ / MAD↑ / Discrim↑):

| Capa | Inter-Sim↓ | Intra-Sim↑ | MAD↑ | Discrim↑ |
|---|---|---|---|---|
| C2 | 0.9484 | 0.9868 | 0.1780 | 1.0410 |
| C3 | 0.8818 | 0.9684 | 0.2736 | 1.1010 |
| C4 | 0.8342 | 0.9596 | 0.3242 | 1.1555 |
| C5 | 0.6263 | 0.8518 | 0.5221 | 1.3766 |
| **P4** | **0.5431** | 0.8765 | **0.5517** | **1.6709** |
| P3 | 0.7436 | 0.9344 | 0.4084 | 1.2658 |
| N4 | 0.7376 | 0.9237 | 0.4185 | 1.2624 |
| N5 / Encoder | 0.5888 | 0.8520 | 0.5343 | 1.4784 |
| Decoder Layer 0 | 0.8111 | 0.8843 | 0.3836 | 1.0918 |
| Decoder Layer 1 | 0.7926 | 0.8648 | 0.4034 | 1.0921 |
| Decoder Layer 2 | 0.7775 | 0.8511 | 0.4106 | 1.0958 |
| Decoder Layer 3 | 0.8206 | 0.8717 | 0.3497 | 1.0635 |
| Decoder Layer 4 | 0.8599 | 0.8969 | 0.2924 | 1.0440 |
| Decoder Layer 5 | 0.9091 | 0.9326 | 0.2244 | 1.0264 |

Con el checkpoint correcto y una muestra 5x más grande (train vs. el val
original de 8 secuencias) el patrón se confirma: P4 sigue siendo la capa
con menos oversmoothing (mejor Discrim, mejor Inter-Sim y MAD de todas las
capas), y los decoder layers siguen degradándose de forma consistente
frente al encoder — no era un artefacto de muestra chica ni del checkpoint
desactualizado.

### Corrida 13 — validación en vivo extendida a P4-FPN puro y DINOv2 puro (completada)

Se extendió `custom_reid.py` con `P4FpnPuroReID` (idéntica a `C4PuroReID`
pero hookeando `encoder.fpn_blocks.0` en vez de `backbone.res_layers.2`,
384-dim en vez de 1024) y `DinoV2PuroReID` (CLS crudo de
`facebook/dinov2-small`, SIN proyección entrenada, sobre recortes de
píxeles en 3 escalas — a diferencia de C4/P4-FPN, que reusan gratis el
feature map ya calculado por el detector, pasar cada recorte por DINOv2 es
caro, así que esta clase acumula los recortes SIN encodear en cada frame y
solo corre el backbone en batch al nacer (query) o perder (banco) un
track, igual principio de costo que el patrón ya usado por `--reid-source
dinov2` nativo + `--reid-stitch`). Wiring nuevo en `main_track.py`/
`run_eval.py`: `--reid-source p4fpn_puro`/`dinov2_puro`, ambos solo válidos
junto con `--reid-stitch`.

Umbrales de coseno calibrados a partir de los logs de k-fold ya existentes
(media de calibración por fold, mismo criterio que el 0.9352 de C4):
P4-FPN puro 0.7988, DINOv2 puro 0.6703.

Smoke test en el split test (8 secuencias, ByteTrack) para ambos antes de
lanzar el pool completo — sin errores, métricas sensatas. Corrida completa
sobre el pool de 64 secuencias (8 combinaciones: 2 métodos × 4 trackers),
overnight, sin errores.

IDF1 e IDSW (las métricas que aíslan el efecto, ver discusión de la
Corrida 11), nativo vs. los 3 métodos externos puros:

| Tracker | IDF1 nativo | IDF1 DINOv2 | IDF1 P4-FPN | IDF1 C4 | IDSW nativo | IDSW DINOv2 | IDSW P4-FPN | IDSW C4 |
|---|---|---|---|---|---|---|---|---|
| ByteTrack | 55.98 | 57.88 | 59.11 | **60.05** | 481 | 354 | 341 | **327** |
| BotSort | 57.28 | 57.90 | 59.35 | **60.05** | 454 | 346 | 332 | **322** |
| OCSort | 61.90 | 65.01 | **66.54** | 66.15 | 382 | 157 | 142 | **139** |
| DeepOCSort | 61.24 | 65.15 | **66.19** | 65.99 | 352 | 155 | **135** | **135** |

HOTA y MOTA también mejoran monótonamente (nativo < DINOv2 < P4-FPN ≈ C4)
en los 4 trackers, sin excepción.

**Confirma en vivo, de punta a punta, el mismo orden ya visto en el
análisis post-hoc de casos estrictos y en el k-fold: C4 puro es la mejor
señal (o empatada con P4-FPN en OCSort/DeepOCSort), P4-FPN segundo,
DINOv2 tercero — pero los 3 baten claramente al tracker nativo en las 4
métricas, en los 4 trackers.** No es un artefacto de una sola metodología
de evaluación: las 3 (casos estrictos, k-fold, validación en vivo)
coinciden.

### Corrida 14 — figuras cualitativas nativo vs +C4 (casos reales verificados por IoU)

Para el paper, se generaron ejemplos visuales de reidentificación en vivo
(no post-hoc): video anotado (`main_track.py --output`) con y sin
`--reid-stitch --reid-source c4_puro` sobre casos reales de brecha (GT
visibility 0 por ≥3 frames), usando los mismos hiperparámetros tuneados
que la validación en vivo.

**Metodología para elegir el caso**: en vez de reusar los casos ya
identificados en `reid_gap_report.json` (calculados con OTROS
hiperparámetros de tracker en el pipeline de k-fold), se buscó
directamente sobre los txt de salida de la Corrida 13 (nativo vs +C4,
mismos hiperparámetros que se van a mostrar) cruzando contra el GT por
IoU — así el caso mostrado es fiel a la config real que se está
documentando. Encontrados casos reales para ByteTrack/DeepOCSort (mismo
video, `video_3rntaZcqXY1h8ur5GcJ_11`, gt_id=2, frame 119→154) y OCSort
(`video_kYUj2rFivaSQMwnjjGp_11`, gt_id=2, frame 234→376) donde el id
nativo cambia tras el gap y +C4 lo mantiene — verificado
programáticamente (match por IoU contra GT), no solo por inspección
visual.

**Hallazgo lateral**: el mismo caso para BotSort NO se reprodujo en un
rerun aislado de una sola secuencia (el id nativo se mantuvo, cuando en
una búsqueda anterior sobre el pool completo sí cambiaba) — indica que
BotSort tiene algún componente no 100% determinista entre corridas
(candidato: `cmc_method=ecc`, registro de movimiento de cámara, o
desempates en el matching húngaro), no visto en ByteTrack/OCSort/DeepOCSort.
No investigado a fondo — no afecta las métricas agregadas ya reportadas
(Corrida 13), solo la elección de UN frame puntual para la figura
cualitativa. Pendiente si se quiere una figura de BotSort también.

**Figura final** (`paper/qualitative_examples/qualitative_native_vs_c4.png`
+ `.pdf`, con `section_draft.tex`): tras iterar sobre el formato con el
usuario, se decidió mostrar un solo caso en vez de los tres (el punto es
solo demostrar que el error se puede resolver, no comparar los 4
trackers), eligiendo **DeepOCSort** en vez de ByteTrack — es el único
tracker con ReID nativo de apariencia siempre activo, así que el ejemplo
muestra que incluso con esa señal nativa ya presente, el gap real (34
frames) sigue sin resolverse (ID3→ID17) y +C4 externo lo recupera
(marcador "REID OK"). Se usa el frame completo sin recorte/zoom (a pedido
explícito) — el video en sí ya es un plano cerrado del parachoques, no
hay pérdida de contexto al no recortar.

### Corrida 15 — extensión de toda la validación al dataset completo (72 secuencias) (completada)

**Motivo**: al revisar `tab:tracker_base_reid_fail` y
`tab:all_methods_same_cases_pct` para el paper, se detectó que mezclaban
alcances distintos sin decirlo — HOTA/IDF1 salían del split test fijo (8
secuencias), mientras que Gaps/Reidentified/Failed y toda la comparación
de métodos (DINOv2/C4/P4-FPN + heads) salían del pool de k-fold (64
secuencias, `reid_gap_{tracker}_kfold_all/reid_gap_report.json`,
verificado a fondo con un agente: `methods_on_strict_cases_results.json`
coincide byte a byte local/remoto, generado 1 minuto después del propio
script, umbrales leídos de `kfold_results.json` por fold). Se corrigió el
caption de `tab:tracker_base_reid_fail` para aclarar esto, y se decidió
además generar el equivalente de **todas** las tablas de validación sobre
el dataset completo (73 secuencias físicas train+val+test), para tener
una referencia sin mezcla de splits y más representativa.

**Secuencia rota excluida**: `video_ILWa6WrLF0UZGKKiHiv_11` tiene una
estructura corrupta (imágenes sueltas en la raíz de la secuencia, sin
subcarpeta `img1/`, sin `seqinfo.ini`) — probablemente por eso ya venía
excluida del pool original. Se descartó (no se reconstruyó), quedando
**72 secuencias** para toda esta corrida.

**Bug encontrado y corregido**: `infer_track_reid_gaps.py --device cuda`
rompe la construcción del backend de ReID nativo de BotSort/DeepOcSort
(`boxmot.reid.core.reid.ReID`) — internamente termina seteando
`CUDA_VISIBLE_DEVICES=cuda` (string, no índice), y `torch.cuda.device_count()`
pasa a devolver 0. BotSort degrada en silencio a sin-ReID-nativo (no
crashea, solo un `[WARN]`); DeepOcSort sí crashea (requiere ReID nativo sí
o sí). Reproducible al 100%, no es un problema de `main_track.py`/
`run_eval.py` (esos ya usaban `--device cuda` sin problema en corridas
anteriores). Fix: pasar `--device 0` (índice numérico) en vez de `--device
cuda` específicamente en las llamadas a `infer_track_reid_gaps.py`. Se
rehizo el gap-report de BotSort (la primera corrida había quedado inválida,
sin ReID nativo) y de DeepOcSort (nunca había terminado).

**Trabajo hecho, en orden**:
1. TrackEval nativo (HOTA/MOTA/IDF1/etc., vía `run_eval.py`) + reporte de
   gaps (`infer_track_reid_gaps.py`) para los 4 trackers sobre las 72
   secuencias (vista plana con symlinks, `all_dataset_flat`/`all_dataset_view`).
2. Precálculo de embeddings C4 y P4-FPN sobre las 72 secuencias
   (`precompute_p4_roi_cache.py --data-root all_dataset_flat`), necesarios
   para evaluar los métodos puros fuera del pool viejo.
3. `methods_puro_alldata.py` (adaptación de `methods_on_strict_cases.py`
   sin heads entrenados — los heads se entrenaron por fold sobre las 64
   secuencias viejas; extenderlos a las 8 nuevas exigiría reasignar folds o
   aceptar evaluarlas con un head arbitrario, queda pendiente): Tracker
   only / DINOv2 / C4 / P4-FPN puro sobre los gaps reales de las 72
   secuencias, con los umbrales fijos ya calibrados (0.9352 / 0.7988 /
   0.6703 — misma media de calibración por fold que ya se usaba).
4. Tracking en vivo end-to-end (`main_track.py` vía `run_eval.py`,
   `--reid-stitch`) para C4/P4-FPN/DINOv2 puro × 4 trackers = 12 corridas
   completas sobre las 72 secuencias.

**Tablas nuevas** (complementarias, NO reemplazan las de test/pool —
ya insertadas en el paper):
- `results_discussion/table_tracker_base_reid_fail_alldata.tex` — HOTA/MOTA/IDF1 + Gaps/Reidentified/Failed.
- `tracker_kfold/table_methods_puro_alldata.tex` — % Tracker only/DINOv2/C4/P4-FPN.
- `live_validation/table_live_c4_vs_native_alldata.tex` — HOTA/MOTA/IDF1/IDSW, nativo vs +C4.
- `live_validation/table_live_all_methods_idf1_idsw_alldata.tex` — IDF1/IDSW, los 4 métodos.

**Resultado**: el ranking C4 ≈ P4-FPN (empatados) > DINOv2 > nativo se
confirma exactamente igual en las 72 secuencias que en el pool de 64, en
las 4 tablas (post-hoc, tracking en vivo, todas las métricas de TrackEval)
— ninguna conclusión del paper cambia, ahora respaldada por ~12% más
datos y sin la mezcla de alcances que tenía la versión original.

**Extra**: se generó también una figura nueva para
`paper/training_methodology/` mostrando el proceso real de banco/query en
inferencia (no el episodio sintético de entrenamiento con distractores de
otros videos) — usa `reid_gallery.py` + `tracker_failure_dataset.py`
(`same_video_only=True`) sobre un caso real de fallo de BoT-SORT, con
matrículas redactadas donde aparecían en los recortes.

**Nota aparte — versión solo-porcentajes de `tab:all_methods_same_cases`**:
se guardó `tracker_kfold/table_all_methods_same_cases_pct.tex`, una
presentación más compacta (solo el %, sin el conteo "N (X%)") de la MISMA
tabla y los MISMOS datos que `table_all_methods_same_cases.tex` (pool de
64, mismos denominadores 17/22/12/10) — no es una corrida nueva, solo otro
formato. Usa a propósito el label `tab:all_methods_same_cases_pct`
(distinto de `tab:all_methods_same_cases`) para no chocar con las citas
`\ref{tab:all_methods_same_cases}` ya existentes en el texto. Por ahora
**no está insertada** vía `\input` en ningún `section_draft.tex` — si se
quiere mostrar en el paper (en vez de o además de la versión con conteos),
falta agregar el `\input{...}` correspondiente.

### Corrida 16 — investigación de brechas largas resuelta, barrido de hiperparámetros OC-SORT+C4, bug de ids acumulados corregido (completada)

**Motivo**: retomar la investigación pausada (sesión anterior) sobre si la
mayoría de errores de reid en las corridas en vivo alldata (Corrida 15) se
debían al parámetro de poda del banco (`--reid-stitch-max-gap`, default
200) o a un fallo genuino de similitud, y de paso comprobar la robustez de
esos resultados ante un barrido de hiperparámetros más exhaustivo.

**Herramienta nueva**: `analyze_reid_gap_failures.py` (`visual_damages_tracking/`,
reusa `iou_matrix`/`greedy_match` de `infer_track_reid_gaps.py` y
`build_track_index` de `train_track_query_reid.py`) clasifica cada brecha
real de GT (reidentified/new_id_born/id_swap/unevaluated) leyendo las
predicciones YA calculadas de las corridas `{c4_puro,p4fpn_puro}_live_
{tracker}_alldata` y `native_reid_{tracker}_alldata`, sin volver a correr
detector+tracker. Validado contra el caso ya confirmado en la sesión
anterior (OC-SORT/C4, `video_kYUj2rFivaSQMwnjjGp_11`, brecha 234→376) antes
de confiar en los agregados.

**Resultado — la premisa no se sostenía**: en las 8 combinaciones tracker×
{C4,P4-FPN}, **0 de 129 brechas reales** producen `new_id_born` (incluidas
las 20 que ya superan los 200 frames). En nativo (sin ningún reid externo),
solo **1 caso en las 4×129** combinaciones tracker×brecha —el mismo caso
real (`video_zKxxQLa0anaIKhbD95h_11`, track 4, brecha 178→391 = 213
frames) fallando tanto en ByteTrack como en BoT-SORT. Con C4 activo, ese
caso no se arregla pero tampoco empeora limpiamente: pasa de `new_id_born`
a `id_swap` (la identidad correcta ya fue podada del banco a los 200
frames; el id nuevo matchea, por similitud, contra otra identidad distinta
que sí seguía en el banco) — confirma que el límite de memoria acotada del
banco es real pero estructural y rarísimo en este dataset, no la causa
dominante de fallos de reid. El modo de fallo real y numeroso es
`unevaluated` (83-92%, imprecisión del detector en el borde de la
oclusión) e `id_swap`, no `new_id_born`.

**C4 reconfirmado como mejora real, no solo en el agregado de TrackEval**:
comparando nativo vs +C4 brecha-por-brecha (no en el conteo agregado de
IDSW/IDF1), `reidentified` casi se duplica y `id_swap` casi se reduce a la
mitad en los 4 trackers (ej. ByteTrack: reidentified 6→12, id_swap 15→9;
BoT-SORT: 5→12, 20→10) — corrobora caso a caso las tablas de
`live_validation` sin depender de la métrica agregada.

**Barrido conjunto de hiperparámetros** (`sweep_ocsort_c4.py`,
`eval_tracking/`, escrito aparte de `sweep_ultralytics_anchor.py` para no
tocar el que ya respalda `tab:tracker_hparams`): 6 etapas escalonadas sobre
OC-SORT+C4 con `--reid-stitch` activo desde el arranque — primero los
parámetros nativos de asociación (antes tuneados sin reid externo), después
`--reid-stitch-cosine-thresh`/`-max-gap`/`-num-track-history`, nunca antes
barridos. 87 combinaciones sobre `val` (8 secuencias), verificación en
`test`. **Resultado: la config que elige el barrido generaliza PEOR en
test (HOTA+IDF1=82.56) que la config ya usada en producción (85.29)** —
sobreajuste al `val` de 8 secuencias, no una mejora real. Confirma que la
config actual ya está en o cerca de un óptimo local; los resultados
reportados no dependen de una baseline mal tuneada.

**Bug encontrado y corregido**: `BaseTrack._count` (boxmot, ByteTrack/
BoT-SORT) es un atributo de clase compartido por todo el proceso. Desde
que `run_eval.py` corre todas las secuencias de un split en el MISMO
proceso (import directo de `main_track.run()`, ya no un subproceso por
secuencia — cambio de una sesión anterior para compartir el modelo de
ReID entre secuencias), los ids de salida de ByteTrack/BoT-SORT se
acumulan sin límite entre secuencias (confirmado: ids 6400+ para la
secuencia #30 de `native_reid_bytetrack_alldata`, vs. 1-89 en BotSort/
OcSort/DeepOcSort). Cosmético únicamente (no afecta TrackEval ni la
clasificación reidentified/new_id_born/id_swap, que solo comparan ids
dentro de una misma secuencia), pero hacía ilegibles los videos anotados.
Fix: `_reset_track_id_counter()` en `main_track.py` (mismo criterio que
`infer_track_reid_gaps.py`), llamado antes de instanciar ByteTrack/BotSort.
Verificado (ids vuelven a rangos chicos) y re-renderizados los 2 clips
afectados de `paper/qualitative_examples/rendered_sample/`
(`video_3E5cyRvJcVEaYFbCXAL_11` ByteTrack, `video_12Kt2fvxMGbX4sPNK40_11`
BoT-SORT).

**Escritura del paper**: nueva sección `paper/conclusions/section_draft.tex`
(Conclusions and Future Work, resume las 6 secciones existentes — todavía
suelta, no insertada en el documento maestro). Corregido un párrafo de
`paper/live_validation/section_draft.tex` (discusión de por qué HOTA se
mueve poco por el cuello de botella de DetA) que usaba números del pool de
64 secuencias en vez de la corrida alldata de 72 — ver
[[reference_paper_folder]] sobre la desincronización entre este repo y el
Overleaf.

### Corrida 17 — corrección de la sección de análisis de embeddings: el gap intra/inter es engañoso, C4 está comprimido pero no es más débil (completada)

**Motivo**: revisando `paper/embedding_analisis/section_draft.tex` para el
paper, la conclusión de esa sección decía que P4-FPN era "la más
prometedora de las dos capturas de RT-DETR" (por tener mejor gap
intra/inter que C4) — contradiciendo el resultado final del paper (C4≈P4-FPN
empatados, C4 levemente adelante, ver [[project_paper_reidmatchinghead]]).
Se investigó por qué el diagnóstico preliminar apuntaba en la dirección
contraria al resultado real.

**Dos problemas metodológicos en el diagnóstico original** (15 identidades,
train/val/test por separado, `embedding_analysis_dinov2_vs_p4.py`):
1. El "inter" mezclaba pares de identidades del MISMO video (el caso
   difícil, relevante en producción — la galería de tracking vive dentro de
   un único video) con pares de videos DISTINTOS (negativo trivial, fondo/
   luz/cámara ya los delatan). Se separó `inter_same_video` de
   `inter_diff_video` — confirmado en el pool completo (64 secuencias, 344
   identidades, 5.160 observaciones): el gap se achica ~30% (DINOv2) y ~26%
   (P4-FPN) al sacar el negativo trivial, pero se mantiene CASI IGUAL para
   C4 (0.145 vs 0.145) — su gap nunca estuvo inflado por el caso fácil,
   porque C4 casi no reacciona a pistas de escena entre videos.
2. Los embeddings se calculaban sobre un solo crop ajustado (factor 1.0),
   mientras que la query real del matching siempre promedia una vista
   ajustada + una amplia (contexto de pieza) antes de comparar
   (`_load_query_views`/`_embed_batch`). Se corrigió `compute_dinov2_embeddings`/
   `compute_cache_embeddings` para promediar ambas vistas, igual que en
   producción (esto causó un OOM la primera vez —duplicar los crops en
   memoria antes de batchear se comía ~14GB de RAM en el servidor de 15GB—,
   arreglado procesando por lotes de observaciones en vez de armar todos
   los crops de una vez).

**La compresión de C4 es real y se puede cuantificar** (no solo inferir de
la escala de similitud): PCA lineal (no t-SNE, que por diseño siempre
separa visualmente los clusters sin importar la distancia real —
comprobado empíricamente: t-SNE/UMAP se ven "igual de separados" para los
3 encoders sin importar cuál esté comprimido) sobre los embeddings
L2-normalizados, con ejes compartidos entre encoders: la varianza total
(traza de la covarianza, comparable entre encoders porque todos viven en
la esfera unitaria) es **0.197 para C4 vs. 0.630 (DINOv2) / 0.625
(P4-FPN)** sobre el pool completo, y se repite casi igual restringiendo a
un solo video real (video6, 35 identidades): **0.214 vs. 0.544/0.519** — no
es un artefacto de mezclar 344 identidades de 64 videos distintos.

**Pero la compresión no le cuesta accuracy de matching real**, medido con
dos métricas que no dependen de la escala absoluta de similitud (a
diferencia del gap):
- **Rank-1 same-video** (CMC, ¿el vecino más cercano ya es la identidad
  correcta?): C4 **0.9891**, P4-FPN 0.9849, DINOv2 0.9785 (n=5.160).
- **Accuracy en el umbral calibrado / AUC**, replicando el mecanismo real
  de `_calibrate_threshold` (300 episodios de galería, banco de hasta 16
  candidatos, mezcla real/sintético/nuevo): C4 gana accuracy (**0.8033**
  vs. 0.7900 P4-FPN vs. 0.7333 DINOv2); P4-FPN gana AUC (**0.8684** vs.
  0.8458 C4) — el umbral óptimo que sale de esta muestra fresca (nunca vio
  los umbrales reales) casi coincide con el calibrado en producción
  (0.9207 vs. 0.9352 real para C4, 0.7984 vs. 0.7988 para P4-FPN, 0.6746
  vs. 0.6703 para DINOv2), validando que el mecanismo replicado es fiel.

**Herramientas nuevas/extendidas**:
- `embedding_analysis_dinov2_vs_p4.py`: `intra_inter_similarities` separa
  same/diff-video; `silhouette_same_video`; `run_gallery_argmax_analysis`
  (best-of-galería, reusa `_build_loader`/`_embed_batch`/`_best_candidate`
  de `analyze_dinov2_nn_baseline.py`); `cmc_same_video` (curva CMC rank-k);
  panel de la figura combinada reemplazable por best-of-galería en vez del
  histograma intra/inter (`--gallery-episodes`).
- `plot_embedding_compression.py` (nuevo): scatter PCA de ejes compartidos
  (fondo gris = todas las identidades, color = submuestra concentrada en
  pocos videos con leyenda "Video 1/2/3" — un muestreo al azar de
  identidades cae mayormente en videos donde son la única representante,
  mostrando separación entre-video trivial en vez de dentro-del-mismo-video);
  curva de varianza cruda acumulada (NO el ratio normalizado de
  `explained_variance_ratio_`, que mide forma relativa y no sirve para
  comparar tamaño absoluto entre encoders); t-SNE/UMAP con el mismo
  esquema de color; y versión restringida a un solo video real
  (`--single-video-pca`, colorea con `nipy_spectral` para las ~35
  identidades de un video).

**Hallazgo sin resolver, anotado para después**: el UMAP de un solo video
(video6, 35 identidades) muestra algunos clusters de C4 mezclados/pegados
entre sí (varias identidades distintas en la misma zona densa) que el
t-SNE del mismo video no deja ver tan claro — podría ser evidencia real de
pares de identidades específicas más confundibles en C4 dentro de esa
galería, no capturada por el promedio agregado (rank-1=0.989). No
investigado a fondo todavía (identificar qué pares de identidades son).

**Paper**: reescrita `paper/embedding_analisis/section_draft.tex` (nueva
subsección "Same-video comparisons, embedding compression, and retrieval
accuracy at full scale"), corrigiendo la conclusión vieja que favorecía a
P4-FPN. Nuevas figuras/tabla en `paper/embedding_analisis/`:
`figure_pca_compression.tex` + `compression_pca_scatter_kfoldpool.png`,
`table_rank1_accuracy_auc.tex`, y como respaldo sin incluir todavía en el
`.tex`: `compression_pca_variance_kfoldpool.png`,
`compression_{tsne,umap}_by_identity_kfoldpool.png`,
`compression_pca_single_video.png`, `compression_{tsne,umap}_single_video.png`.

### Pendiente / próximos pasos

- Investigar el hallazgo de UMAP de Corrida 17 (video6, single-video): qué
  pares de identidades específicas se ven mezclados/pegados en C4 dentro
  de esa galería real, y si eso explica algo del 1.1% de rank-1 que C4
  pierde (los 56 casos de 5.160 donde el vecino más cercano no es la
  identidad correcta).
- Extender los heads entrenados (C4+H1..H4, P4-FPN+H1..H4) a las 72
  secuencias (Corrida 15) — hoy `table_methods_puro_alldata.tex` solo
  cubre los métodos sin entrenamiento; los heads se entrenaron por fold
  sobre las 64 secuencias viejas, así que las 8 nuevas necesitarían
  reasignar folds (o evaluarse con un head arbitrario, ya que nunca
  participaron del entrenamiento de ninguno).
- Integrar la tabla de oversmoothing actualizada (Corrida 12, ya
  completada) en la sección `paper/embedding_analisis/` del paper.
- Correr el mismo análisis de k-fold + McNemar con otra semilla de
  `assign_folds` (robustez del resultado a la partición de folds).
- Si se retoma `BilinearAttentionScorer`: probar con más datos o mining de
  negativos más agresivo antes de descartarlo definitivamente (ver
  Corrida 7) — no vale la pena seguir iterando la arquitectura en sí.
- Extender `run_kfold_cv` para soportar tipos de modelo intercambiables
  (hoy asume `LinearReIDHead`/`train_head` específicamente) si se decide
  correr `BilinearAttentionScorer` o futuras variantes bajo el mismo
  rigor de validación cruzada.
- DDP (multi-GPU) explícitamente pospuesto hasta tener más de una GPU
  disponible.
- Investigar la asimetría encontrada en `tab:all_methods_same_cases_pct`:
  entrenar una cabeza sobre P4-FPN recupera bastante headroom (ej.
  ByteTrack 64.7%→82.4% con H1), pero la misma cabeza sobre C4 nunca ayuda
  y a veces empeora — ¿es una diferencia genuina de cuánta estructura le
  falta a cada espacio de features crudo, o una limitación del volumen de
  datos de entrenamiento disponible?
- Aplicar el mismo fix de `_reset_track_id_counter()` (Corrida 16) y
  re-renderizar si se decide usar más clips de `rendered_videos_alldata/`
  (576 videos, ByteTrack/BoT-SORT nativos) en el paper — por ahora solo se
  corrigieron los 2 clips ya usados en `qualitative_examples/`.
- Aplicar el mismo criterio de robustez del barrido OC-SORT+C4 (Corrida
  16) a los otros 3 trackers, si se quiere el mismo argumento de "ya está
  en el óptimo" para todos, no solo para el combo ganador.
- Revisar el resto de `paper/` por posibles desincronizaciones con el
  Overleaf como las encontradas en Corrida 16 (`tab:kfold_identity_acc` y
  `tab:split-composition` no existen en este repo, y un párrafo de
  `live_validation` tenía números del pool en vez de alldata) — no asumir
  que el Overleaf y este repo están al día entre sí sin verificar.
- Empaquetar `dataset_split_2` (72 secuencias) y la herramienta de
  anotación para publicación pública (contribuciones complementarias del
  paper, ver `paper/conclusions/section_draft.tex`).
- Escritura del paper (Overleaf, en curso) — integrar tablas de k-fold +
  McNemar (Corridas 6, 8, 10) y la validación en vivo de los 3 métodos
  (Corridas 11 y 13) a la sección de resultados; secciones de metodología
  ya redactadas en
  `paper/training_methodology/` (construcción de batches, asociación sin
  cabeza entrenada) y `paper/tracker_hyperparameter_search/` (búsqueda de
  hiperparámetros, métrica de reid aislada por brecha, test de McNemar).
  Las 4 tablas del dataset completo (Corrida 15) ya están redactadas
  directamente en los `section_draft.tex` de `results_discussion/` y
  `live_validation/` (no quedan pendientes de integrar, a diferencia de
  las anteriores). La sección de conclusiones (Corrida 16) está escrita
  pero todavía no insertada en el documento maestro.
