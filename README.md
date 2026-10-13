# Reusing frozen detector encoder features for occlusion-robust re-identification in vehicle damage tracking

Code for the paper. A vehicle-inspection video is tracked damage by damage, and
when a damage is occluded and reappears, standard trackers often give it a new
identity. We show that a **frozen intermediate feature map of the detector's own
encoder (C4, RT-DETR)** is enough to recover the correct identity, with **no
training, no extra parameters and under 0.6 ms per re-identification event**.

![Native tracker vs. native + C4 module](docs/qualitative_native_vs_c4.png)

## Contributions

1. **Semi-supervised annotation pipeline.** A pre-trained detector proposes
   video-level pseudo-trajectories that annotators verify and correct; it extends
   the TVDT dataset with 55 new videos.
2. **Gap-isolated re-identification metric.** Measures, separately from
   frame-to-frame association (HOTA/IDF1), whether a tracker recovers the correct
   identity after an occlusion (gaps of at least 3 frames, classified as
   re-identified / id swap / new id born / unevaluated).
3. **Training-free re-identification module.** When the tracker is about to create
   a new ID, the new crop is compared (cosine similarity, Hungarian assignment,
   threshold tau) with a gallery of recently lost identities (retained up to 200
   frames), using C4 features pooled with RoIAlign from the detector's encoder.

The module is plug-in for ByteTrack, BoT-SORT, OC-SORT and DeepOC-SORT.

## Main results

Evaluation protocol: the detector was adapted with frames from 10 of the 72
sequences, so **those 10 are excluded from every evaluation**: live results use
62 sequences, the offline k-fold identity-accuracy comparison uses 54 (the 62
minus the 8 validation sequences used for tracker hyperparameters).

Live end-to-end tracking, 62 sequences (IDF1 / identity switches):

| Tracker | Native IDF1 | + C4 IDF1 | Native IDSW | + C4 IDSW |
|---|---|---|---|---|
| ByteTrack | 51.41 | 55.09 | 451 | 386 |
| BoT-SORT | 52.55 | 53.90 | 414 | 412 |
| OC-SORT | 57.37 | 61.00 | 340 | 179 |
| DeepOC-SORT | 56.18 | 61.69 | 314 | 164 |

C4 is compared with OSNet (MSMT17), CLIP-VehicleID, DINOv2 and P4-FPN in the
paper; all numbers, confidence intervals and McNemar tests are in the Results and
Ablation sections.

Gap-level identity accuracy rises from 23.5-36.4 % (native trackers) to
80.0-94.1 % with the module. Note that the native re-identification rate is
reported under three different protocols (post-hoc, train+test only, live) and
rates should only be compared within a protocol.

## Dataset

72 evaluated vehicle-inspection sequences (17 from the original 360-degree set and
55 recorded by end users), 400 annotated trajectories, 77,034 annotated boxes,
47,523 frames, six resolutions, 66 landscape / 6 portrait, 30 fps. Split into 56
train / 8 validation / 8 test sequences. All visible license plates are
anonymized. The train/validation/test labels describe annotation and tuning roles,
not an exclusion from evaluation, since the module has no trainable parameters.

### Dataset access

The dataset (annotated video sequences and ground-truth trajectories) is not
stored in this repository because of its size. It is distributed on request:

**Request form: [LINK TO THE REQUEST FORM -- TO BE ADDED]**

Access terms and the anonymization policy (all visible license plates are
blurred) will be described in the form.

## Repository layout

| Path | Contents |
|---|---|
| `live_tracking/` | Live pipeline: `run_eval_nodedup.py` (tracking + TrackEval evaluation), `main_track_nodedup.py` (trackers + module hook), `reid_gallery.py` (gallery: `bury`, `match_newborns`, ageing/pruning), `custom_reid.py` (C4 / P4-FPN / DINOv2 / OSNet / CLIP encoders), `trackeval_api.py` |
| `infer_track_reid_gaps.py`, `analyze_reid_gap_failures.py` | Gap-isolated metric: detection and classification of occlusion gaps |
| `kfold_utils.py`, `compare_kfold_cases.py`, `pool_existing_kfold_results.py` | Offline k-fold identity-accuracy comparison and McNemar tests |
| `analyze_dinov2_*.py`, `embedding_analysis_dinov2_vs_p4.py`, `plot_*_crops_*.py` | Embedding analysis (compression, PCA, isotropy) |
| `linear_reid_head.py`, `train_linear_reid_head.py`, `set_reid_model.py`, `train_track_query_reid.py` | Optional trained heads (H1-H4), reported as a separate ablation |
| `paper_fig_*.py` | Scripts that generate paper figures |
| `exp_*` | Detector-provenance experiments (no-video detectors, test_models sweep) and the exclusion of the 10 seen sequences |
| `docs/EXPERIMENT_LOG.md` | Chronological experiment log (run-by-run, in Spanish) |

## Reproducing the live evaluation

Requires an RT-DETR checkpoint and config (RT-DETR repository), `boxmot==18.0.0`,
`lap`, `onnxruntime`, `transformers`, and
[TrackEval](https://github.com/JonathonLuiten/TrackEval) (set `TRACKEVAL_DIR`).

```bash
# native BoT-SORT
python live_tracking/run_eval_nodedup.py \
  --pth-file <detector.pth> --config <rtdetrv2_r101vd_6x_coco.yml> \
  --data-root <dataset> --gt-root <dataset_gt> --split all \
  --tracker botsort --conf 0.4 --fps 10 --no-convert-gt \
  --output-dir out/native_botsort

# the same tracker + C4 module (training-free)
python live_tracking/run_eval_nodedup.py ... --tracker botsort \
  --reid-source c4_puro --reid-stitch \
  --reid-stitch-cosine-thresh 0.5 --reid-stitch-max-gap 200 \
  --output-dir out/c4_botsort
```

Tracker hyperparameters used in the paper are in the paper's Experiments section.
`--reid-source` selects the signal (`c4_puro`, `p4fpn_puro`, `dinov2_puro`,
`native_puro` for OSNet/CLIP). Sequences are evaluated one process at a time on an
8 GB GPU; `run_eval_nodedup.py` logs per-sequence errors without failing, so check
for `[ERROR]` lines and count the output files.

## Implementation note

Early versions of the module dropped one box when two tracks ended up with the
same final ID in a frame, which understated identity switches and overstated MOTA.
The `*_nodedup` files keep both boxes and assign the colliding track a new ID; all
numbers reported in the paper use this version.

## Citation

Citation details will be added upon publication.
