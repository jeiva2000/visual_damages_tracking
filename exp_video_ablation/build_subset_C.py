"""
Subversión C de la ablación de vídeo: train = QCS (default_train_2, 18.133 imgs)
+ SOLO los frames de vídeo de touaregmedina (2.139); val = val_qcs.json (QCS,
solo objetos estáticos). Parte de qcs_video_ablation/annotations/train_B_qcs_video.json
y del mapeo frame->secuencia por md5 (cluster_datasets/video_frame_to_seq.json).
Genera:
  qcs_video_ablation/annotations/train_C_qcs_touaregmedina.json   (para usar con el zip grande)
  qcs_video_ablation_C/  (autocontenido: images/ + annotations/{train_C, val_qcs}.json + README)
"""
import json, os, shutil
CD = "/data/backup/serperzar/cluster_datasets"
SEQ = "touaregmedina"
B = json.load(open(f"{CD}/qcs_video_ablation/annotations/train_B_qcs_video.json"))
m = json.load(open(f"{CD}/video_frame_to_seq.json"))["train"]
keep = [i for i in B["images"] if not i["file_name"].startswith("vid_") or m[i["file_name"][4:]][0] == SEQ]
ids = {i["id"] for i in keep}
C = dict(B, images=keep, annotations=[a for a in B["annotations"] if a["image_id"] in ids])
n_vid = sum(i["file_name"].startswith("vid_") for i in keep)
print(f"train_C: {len(keep)} imgs ({len(keep)-n_vid} QCS + {n_vid} {SEQ}), {len(C['annotations'])} anns")
json.dump(C, open(f"{CD}/qcs_video_ablation/annotations/train_C_qcs_touaregmedina.json", "w"))
out = f"{CD}/qcs_video_ablation_C"
os.makedirs(f"{out}/images", exist_ok=True); os.makedirs(f"{out}/annotations", exist_ok=True)
json.dump(C, open(f"{out}/annotations/train_C_qcs_touaregmedina.json", "w"))
shutil.copy(f"{CD}/qcs_video_ablation/annotations/val_qcs.json", f"{out}/annotations/val_qcs.json")
val = json.load(open(f"{out}/annotations/val_qcs.json"))
for i in keep + val["images"]:
    dst = f"{out}/images/{i['file_name']}"
    if not os.path.lexists(dst):
        os.symlink(os.path.realpath(f"{CD}/qcs_video_ablation/images/{i['file_name']}"), dst)
open(f"{out}/README.txt", "w").write(__doc__ + f"\nPrueba C: train_C_qcs_touaregmedina.json ({len(keep)} imgs, {len(C['annotations'])} anns), "
    f"val: val_qcs.json ({len(val['images'])} imgs, solo QCS estático).\nMisma config y punto de partida que A/B. "
    f"Al evaluar, excluir también {SEQ} (y las otras 9 secuencias vistas por el detector actual si se compara con él).\n")
print("images/:", len(os.listdir(f"{out}/images")))
