"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import os
import glob
import torch
import torchvision.transforms as T
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

IMG_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp')


def draw(images, labels, boxes, scores, out_dir, thrh=0.6):
    for i, im in enumerate(images):
        d = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        for j, b in enumerate(box):
            d.rectangle(list(b), outline='red')
            d.text((b[0], b[1]), text=str(lab[j].item()), fill='blue')
        im.save(os.path.join(out_dir, f'results_{i}.jpg'))


def list_images(folder):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(files)


def is_mot_root(folder):
    """True si `folder` contiene subcarpetas video_XXX/img1 (estructura MOT), en vez de imagenes sueltas."""
    return len(glob.glob(os.path.join(folder, '*', 'img1'))) > 0


def list_images_mot(folder):
    """Imagenes bajo folder/*/img1/*, clave = ruta relativa al padre de folder (ej. 'test/video_XXX/img1/000000.jpg'),
    para que coincida con file_name en el instances_{split}.json de pycocotools y no colisione entre videos."""
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, '*', 'img1', ext)))
        files.extend(glob.glob(os.path.join(folder, '*', 'img1', ext.upper())))
    root_parent = os.path.dirname(os.path.normpath(folder))
    return sorted(files), root_parent


def run_folder(sess, transforms, folder, save_images=False, thrh=0.10):
    """Run inference on every image in `folder`.

    Si `folder` tiene estructura MOT (video_XXX/img1/*), recorre todas las
    secuencias y las claves del dict resultado son la ruta relativa al padre
    de `folder` (ej. 'test/video_XXX/img1/000000.jpg'), igual que file_name
    en instances_{split}.json -- evita que colisionen frames del mismo
    numero entre videos distintos. Si no, usa solo el basename (modo plano,
    compatibilidad con datasets sin secuencias).
    """
    results = {}
    mot_mode = is_mot_root(folder)
    if mot_mode:
        image_files, root_parent = list_images_mot(folder)
    else:
        image_files = list_images(folder)

    if not image_files:
        print(f'[WARN] No images found in {folder}')
        return results

    for im_file in image_files:
        im_pil = Image.open(im_file).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None]

        im_data = transforms(im_pil)[None]

        output = sess.run(
            output_names=None,
            input_feed={
                'images': im_data.data.numpy(),
                'orig_target_sizes': orig_size.data.numpy(),
            },
        )
        labels, boxes, scores = output
        keep = scores > thrh
        labels, boxes, scores = labels[keep], boxes[keep], scores[keep]

        key = os.path.relpath(im_file, root_parent).replace(os.sep, '/') if mot_mode else os.path.basename(im_file)
        results[key] = []
        for label, box, score in zip(labels,boxes,scores):
            label , box, score = label, box, score
            xmin, ymin, xmax, ymax = box
            results[key].append([xmin,ymin,xmax,ymax,score,label])

        if save_images:
            draw([im_pil], labels, boxes, scores, folder)

    return results


def generate_infers(provider='CUDAExecutionProvider', onnx_file= '../test_models/checkpoints/rtdetrv2_v3_qcs_videos.onnx', root_path= '../datasets/dataset_split_2/test'):
    providers = [provider] if provider else ort.get_available_providers()
    sess = ort.InferenceSession(onnx_file, providers=providers)
    print(ort.get_device())

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    folders = os.listdir(root_path)
    folders = [folder for folder in folders if folder not in ['.DS_Store']]
    out_dict = {}
    for folder in folders:
        folder_path = os.path.join(root_path,folder,'img1')
        print(f'Processing folder: {folder_path}')
        folder_name = folder
        print('folder_name:',folder_name)
        detections = run_folder(sess, transforms, folder_path, save_images=False, thrh=0.4)
        #print('detections:',detections)
        out_dict[folder_name] = detections
        #break
    return out_dict


def main(args):
    providers = [args.provider] if args.provider else ort.get_available_providers()
    sess = ort.InferenceSession(args.onnx, providers=providers)
    print(ort.get_device())

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    for folder in args.folders:
        folder = folder.rstrip('/')
        print(f'Processing folder: {folder}')

        detections = run_folder(sess, transforms, folder, save_images=args.save_images, thrh=args.thrh)

        folder_name = os.path.basename(os.path.normpath(folder))
        out_path = os.path.join(args.output_dir, f'{folder_name}.npy')

        np.save(out_path, np.array(detections, dtype=object), allow_pickle=True)
        print(f'Saved {len(detections)} detections -> {out_path}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True)
    parser.add_argument(
        '--folders', type=str, nargs='+', required=True,
        help='Paths a las carpetas con imagenes a procesar (una por split, ej. test, val)',
    )
    parser.add_argument(
        '--output-dir', type=str, default='.',
        help='Where to save the .npy files',
    )
    parser.add_argument(
        '--save-images', action='store_true',
        help='Also save annotated .jpg images per detection',
    )
    parser.add_argument(
        '--thrh', type=float, default=0.10,
        help='Umbral de confianza minimo para conservar una deteccion (por defecto 0.10). '
             'Para evaluar AP con eval_ap.py usa un umbral bajo (ej. 0.001) para no truncar la curva precision/recall.',
    )
    parser.add_argument(
        '--provider', type=str, default='CUDAExecutionProvider',
        help="Execution provider de onnxruntime, ej. 'CUDAExecutionProvider' o 'CPUExecutionProvider'. "
             "Por defecto usa el primero disponible en el sistema.",
    )
    args = parser.parse_args()
    main(args)
