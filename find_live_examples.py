"""
Busca, en la corrida EN VIVO real de ByteTrack sobre las 72 secuencias
(la que respalda tab:live_c4_vs_native_alldata):
  (a) una brecha real de GT donde el tracker NATIVO perdio el id
      (new_id_born/id_swap) pero +C4 SI reidentifico -- caso de exito.
  (b) una identidad de GT genuinamente NUEVA (su primera observacion en
      todo el video, sin brecha real detras) donde +C4 la mantuvo como
      id nuevo (no la fusiono con ninguna identidad perdida) -- "predijo
      no match" correctamente.
Recorta las imagenes reales (data_root real) para poder visualizarlas.
"""
import os
import sys

sys.path.insert(0, "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking")
sys.path.insert(0, "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking")

from PIL import Image

from analyze_reid_gap_failures import classify_sequence_gaps, first_seen_by_pred_id, read_mot_predictions

DATA_ROOT = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking/all_dataset_view/all"
NATIVE_DIR = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking/native_reid_bytetrack_alldata/txt"
C4_DIR = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking/c4_puro_live_bytetrack_alldata/txt"
OUT_DIR = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking/c4_proof_visuals"
MIN_GAP_FRAMES = 3
IOU_MATCH_THRESH = 0.5


def frame_path(seq, frame_idx):
    # frames 1-indexadas en MOT, nombres tipicos %06d.jpg -- probamos variantes
    img_dir = os.path.join(DATA_ROOT, seq, "img1")
    for cand in (f"{frame_idx:06d}.jpg", f"{frame_idx:06d}.png", f"{frame_idx}.jpg", f"{frame_idx}.png"):
        p = os.path.join(img_dir, cand)
        if os.path.isfile(p):
            return p
    return None


def crop_bbox(img_path, bbox, pad=0.3):
    img = Image.open(img_path).convert("RGB")
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    x1 -= w * pad; x2 += w * pad; y1 -= h * pad; y2 += h * pad
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    return img.crop((x1, y1, x2, y2))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sequences = sorted(n for n in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, n)))

    success_case = None
    new_track_case = None

    for seq in sequences:
        native_txt = os.path.join(NATIVE_DIR, f"{seq}.txt")
        c4_txt = os.path.join(C4_DIR, f"{seq}.txt")
        if not (os.path.isfile(native_txt) and os.path.isfile(c4_txt)):
            continue
        seq_dir = os.path.join(DATA_ROOT, seq)

        native_results = classify_sequence_gaps(seq_dir, native_txt, MIN_GAP_FRAMES, IOU_MATCH_THRESH)
        c4_results = classify_sequence_gaps(seq_dir, c4_txt, MIN_GAP_FRAMES, IOU_MATCH_THRESH)
        c4_by_key = {(r["sequence"], r["gt_identity_id"], r["frame_before"], r["frame_after"]): r for r in c4_results}

        if success_case is None:
            for r in native_results:
                if r["outcome"] == "reidentified":
                    continue
                key = (r["sequence"], r["gt_identity_id"], r["frame_before"], r["frame_after"])
                c4_r = c4_by_key.get(key)
                if c4_r is not None and c4_r["outcome"] == "reidentified":
                    success_case = {"seq": seq, "seq_dir": seq_dir, "native": r, "c4": c4_r}
                    break

        if new_track_case is None:
            # identidad de GT cuya PRIMERA observacion no tiene brecha real detras
            # (naci{'o'} de verdad en este punto, no reaparece de una oclusion) --
            # buscamos en required_reid style: primera obs de cada track, chequeando
            # que el pred_id de c4 en ese frame sea un id recien nacido (first_seen
            # == ese frame exacto, no reusado de una identidad previa).
            from train_track_query_reid import build_track_index
            index = build_track_index(seq_dir, sequence_dirs=[seq_dir], min_observations=2)
            pred_by_frame_c4 = read_mot_predictions(c4_txt)
            first_seen_c4 = first_seen_by_pred_id(pred_by_frame_c4)
            for track in index.tracks:
                first_obs = track.observations[0]
                if first_obs.frame_idx <= 1:
                    continue  # aparece en el primer frame del clip, no es una "nueva identidad" util
                # matcheo iou simple contra pred_by_frame_c4 en ese frame exacto
                preds = pred_by_frame_c4.get(first_obs.frame_idx, [])
                if not preds:
                    continue
                import numpy as np
                from infer_track_reid_gaps import iou_matrix
                gt_box = np.array([first_obs.bbox], dtype=np.float32)
                pred_boxes = np.array([p[1] for p in preds], dtype=np.float32)
                iou = iou_matrix(gt_box, pred_boxes)[0]
                best = int(iou.argmax())
                if iou[best] < IOU_MATCH_THRESH:
                    continue
                pred_id = preds[best][0]
                if first_seen_c4.get(pred_id) == first_obs.frame_idx:
                    # nacio justo en este frame -- no fue "robado" de una id vieja
                    new_track_case = {
                        "seq": seq, "seq_dir": seq_dir,
                        "gt_identity_id": track.identity_id, "frame_idx": first_obs.frame_idx,
                        "bbox": first_obs.bbox, "pred_id": pred_id,
                    }
                    break

        if success_case is not None and new_track_case is not None:
            break

    print("[SUCCESS CASE]", success_case)
    print("[NEW TRACK CASE]", new_track_case)

    if success_case:
        r = success_case
        before_path = frame_path(r["seq"], r["native"]["frame_before"])
        after_path = frame_path(r["seq"], r["native"]["frame_after"])
        # bbox del GT antes/despues -- lo sacamos del track_index
        from train_track_query_reid import build_track_index
        index = build_track_index(r["seq_dir"], sequence_dirs=[r["seq_dir"]], min_observations=2)
        track = next(t for t in index.tracks if t.identity_id == r["native"]["gt_identity_id"])
        obs_by_frame = {o.frame_idx: o for o in track.observations}
        before_bbox = obs_by_frame[r["native"]["frame_before"]].bbox
        after_bbox = obs_by_frame[r["native"]["frame_after"]].bbox
        img_before = crop_bbox(before_path, before_bbox)
        img_after = crop_bbox(after_path, after_bbox)
        canvas = Image.new("RGB", (img_before.width + img_after.width + 20, max(img_before.height, img_after.height)), "black")
        canvas.paste(img_before, (0, 0))
        canvas.paste(img_after, (img_before.width + 20, 0))
        out_path = os.path.join(OUT_DIR, f"live_success_{r['seq']}.jpg")
        canvas.save(out_path, quality=90)
        print(f"[OK] {out_path}")
        print(f"  seq={r['seq']} native_outcome={r['native']['outcome']} c4_outcome={r['c4']['outcome']} "
              f"frame_before={r['native']['frame_before']} frame_after={r['native']['frame_after']}")

    if new_track_case:
        n = new_track_case
        img_path = frame_path(n["seq"], n["frame_idx"])
        img = crop_bbox(img_path, n["bbox"])
        out_path = os.path.join(OUT_DIR, f"live_newtrack_{n['seq']}.jpg")
        img.save(out_path, quality=90)
        print(f"[OK] {out_path}")
        print(f"  seq={n['seq']} gt_identity={n['gt_identity_id']} frame={n['frame_idx']} pred_id_nuevo={n['pred_id']}")


if __name__ == "__main__":
    main()
