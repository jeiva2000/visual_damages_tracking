"""
Construye (en teide) el dataset para la ablación "con / sin frames de vídeo"
del detector, listo para copiar al cluster:

  qcs_video_ablation/
    images/                       todas las imágenes, nombres únicos
      <QCS>.jpg                   imágenes QCS v3 (coco_v3_qcs/train), nombre original
      vid_<nnnnnn>.jpg            frames de vídeo de train (mot/damages/images/train)
      vidval_<nnnnnn>.jpg         frames de vídeo de val   (mot/damages/images/val)
    annotations/
      train_A_qcs.json            = coco_v3_qcs/default_train_2.json            (prueba A)
      train_B_qcs_video.json      = coco_v3_qcs/default_train_combine_video.json (prueba B)
      val_qcs.json                = coco_v3_qcs/default_val_2.json              (val de A y B)
      val_qcs_video.json          = coco_v3_qcs/default_val_combine_video.json  (solo referencia)
    README.txt

Por qué se renombran los frames de vídeo: en los json originales los frames de
vídeo se llaman 000001.jpg..009350.jpg y 4000 de ellos (000001..004000) chocan
con imágenes QCS del mismo nombre; en una sola carpeta se pisarían. Solo cambia
file_name; ids, anotaciones y cajas quedan idénticos a los originales.

Uso:  python3 build_cluster_dataset.py <out_dir>   (luego zip -0 -r)
"""
import hashlib
import json
import os
import sys

SRC = "/data/backup/serperzar"
QCS_IMG = f"{SRC}/coco_v3_qcs/train"
VID_TRAIN = f"{SRC}/mot/damages/images/train"
VID_VAL = f"{SRC}/mot/damages/images/val"
J = {
    "train_A_qcs.json": f"{SRC}/coco_v3_qcs/default_train_2.json",
    "train_B_qcs_video.json": f"{SRC}/coco_v3_qcs/default_train_combine_video.json",
    "val_qcs.json": f"{SRC}/coco_v3_qcs/default_val_2.json",
    "val_qcs_video.json": f"{SRC}/coco_v3_qcs/default_val_combine_video.json",
}
N_QCS_TRAIN, N_QCS_VAL = 18133, 1570  # las primeras N imágenes de los json combinados son QCS


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(out):
    img_dir, ann_dir = os.path.join(out, "images"), os.path.join(out, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    links = {}  # nombre en images/ -> origen

    def link(name, src):
        if name in links:
            assert links[name] == src, f"colisión: {name}"
            return
        assert os.path.isfile(src), src
        links[name] = src

    for out_name, src_json in J.items():
        d = json.load(open(src_json))
        n_qcs = N_QCS_VAL if out_name.startswith("val") else N_QCS_TRAIN
        vid_dir, prefix = (VID_VAL, "vidval_") if out_name.startswith("val") else (VID_TRAIN, "vid_")
        for k, im in enumerate(d["images"]):
            fn = im["file_name"]
            if out_name in ("train_A_qcs.json", "val_qcs.json") or k < n_qcs:
                link(fn, os.path.join(QCS_IMG, fn))
            else:
                im["file_name"] = prefix + fn
                link(im["file_name"], os.path.join(vid_dir, fn))
        json.dump(d, open(os.path.join(ann_dir, out_name), "w"))
        print(f"{out_name}: {len(d['images'])} imgs, {len(d['annotations'])} anns  <- {src_json}")

    for name, src in links.items():
        dst = os.path.join(img_dir, name)
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    print(f"images/: {len(links)} imágenes (symlinks; zip las empaqueta como archivos)")

    with open(os.path.join(out, "README.txt"), "w") as f:
        f.write(__doc__ + "\nmd5 de los json de origen:\n")
        for out_name, src_json in J.items():
            f.write(f"  {out_name:24s} {md5(src_json)}  {src_json}\n")
        f.write("\nPrueba A: train_A_qcs.json      val: val_qcs.json\n"
                "Prueba B: train_B_qcs_video.json val: val_qcs.json  (misma val para no elegir el\n"
                "          checkpoint mirando video1, que es una secuencia de evaluación)\n"
                "Ambas: img_folder = images/, RT-DETRv2-R101vd, 1 clase, 30 épocas, batch 8,\n"
                "       mismo punto de partida (rtdetrv2_v3_qcs_cardd_NOVIDEO.pth, md5 db5e8f47).\n")


if __name__ == "__main__":
    main(sys.argv[1])
