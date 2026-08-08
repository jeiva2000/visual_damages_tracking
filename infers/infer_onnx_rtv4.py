"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""
import os
import glob
import torch
import torchvision.transforms as T
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

IMG_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp')


def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
    """Resizes an image while maintaining aspect ratio and pads it."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), interpolation)
    new_image = Image.new("RGB", (size, size))
    new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2


def adjust_boxes(boxes, ratio, pad_w, pad_h):
    """Maps boxes from the resized/padded 640x640 space back to original image coords."""
    adjusted = boxes.copy()
    adjusted[:, 0] = (boxes[:, 0] - pad_w) / ratio
    adjusted[:, 1] = (boxes[:, 1] - pad_h) / ratio
    adjusted[:, 2] = (boxes[:, 2] - pad_w) / ratio
    adjusted[:, 3] = (boxes[:, 3] - pad_h) / ratio
    return adjusted


def draw(im, labels, boxes, scores, thrh=0.4):
    d = ImageDraw.Draw(im)
    keep = scores > thrh
    for lbl, bb in zip(labels[keep], boxes[keep]):
        d.rectangle(list(bb), outline='red')
        d.text((bb[0], bb[1]), text=str(lbl), fill='blue')
    return im


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


def process_image(sess, transforms, im_pil):
    resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, 640)
    orig_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])
    im_data = transforms(resized_im_pil).unsqueeze(0)

    output = sess.run(
        output_names=None,
        input_feed={'images': im_data.numpy(), 'orig_target_sizes': orig_size.numpy()},
    )
    labels, boxes, scores = output

    boxes_adjusted = adjust_boxes(boxes[0], ratio, pad_w, pad_h)
    return labels[0], boxes_adjusted, scores[0]


def run_folder(sess, transforms, folder, save_images=False, thrh=0.10):
    """Run inference on every image in `folder`.

    Si `folder` tiene estructura MOT (video_XXX/img1/*), recorre todas las
    secuencias y las claves del dict resultado son la ruta relativa al padre
    de `folder` (ej. 'test/video_XXX/img1/000000.jpg'), igual que file_name
    en instances_{split}.json -- evita que colisionen frames del mismo
    numero entre videos distintos. Si no, usa solo el basename (modo plano,
    compatibilidad con datasets sin secuencias).

    Solo se conservan las detecciones con score > thrh.
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
        labels, boxes, scores = process_image(sess, transforms, im_pil)

        keep = scores > thrh
        labels, boxes, scores = labels[keep], boxes[keep], scores[keep]

        key = os.path.relpath(im_file, root_parent).replace(os.sep, '/') if mot_mode else os.path.basename(im_file)
        results[key] = []
        for label, box, score in zip(labels,boxes,scores):
            label , box, score = label, box, score
            xmin, ymin, xmax, ymax = box
            results[key].append([xmin,ymin,xmax,ymax,score,label])

        if save_images:
            annotated = draw(im_pil.copy(), labels, boxes, scores, thrh=thrh)
            annotated.save(os.path.join(folder, f'{os.path.splitext(os.path.basename(im_file))[0]}_result.jpg'))

    return results


def main(args):
    providers = [args.provider] if args.provider else ort.get_available_providers()
    sess = ort.InferenceSession(args.onnx, providers=providers)
    print(f"Using device: {ort.get_device()}")

    transforms = T.Compose([
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
    parser.add_argument('--onnx', type=str, required=True, help='Path to the ONNX model file.')
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
