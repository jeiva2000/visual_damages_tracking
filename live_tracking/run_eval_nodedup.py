"""
run_eval.py
-----------
Wrapper: corre main_track.py sobre todas las secuencias de un split
y luego lanza TrackEval para obtener HOTA/CLEAR/Identity.

Uso (ONNX):
    python run_eval.py \
        --onnx-file modelo.onnx \
        --data-root /path/to/dataset \
        --split test \
        --gt-root /path/to/mot_gt \
        [--tracker bytetrack] \
        [--conf 0.25] \
        [--imgsz 640] \
        [--device cpu] \
        [--output-dir eval_output] \
        [--iou-thresh 0.5] \
        [--no-convert-gt]

Uso (checkpoint .pth de RT-DETR/RTDETRv2):
    python run_eval.py \
        --pth-file best.pth \
        --config configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml \
        --data-root /path/to/dataset \
        ... (mismos flags que arriba)

Estructura esperada del dataset:
    data-root/
    └── {split}/
        ├── seq1/
        │   └── img1/  (o directamente imágenes)
        ├── seq2/
        └── ...

    gt-root/
    └── {split}/
        ├── seq1/
        │   ├── gt/gt.txt
        │   └── seqinfo.ini
        └── ...
"""

import os
import sys
import copy
import argparse
import subprocess
import traceback
import configparser
import shutil
import hashlib
from pathlib import Path

import main_track_nodedup as main_track  # COPIA EXPERIMENTAL exp_nodedup

_TRACKEVAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TrackEval', 'scripts', 'run_mot_challenge.py')

# Cache global de detecciones: compartida entre evaluaciones, sweeps y trackers
# distintos (bytetrack, botsort, ...), porque las detecciones del detector no
# dependen del tracker ni de sus hiperparámetros. Vive fuera de --output-dir
# para que nunca se borre al re-ejecutar (run_eval.py hace rmtree de --output-dir).
_DEFAULT_DET_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'det_cache')

# Igual razonamiento para embeddings ReID: compartida entre trackers/sweeps
# (botsort, deepocsort, boosttrack, ...) porque el embedding de una caja no
# depende del tracker que lo consume, solo del modelo de ReID usado.
_DEFAULT_REID_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reid_cache')


def det_cache_path(det_cache_dir: str, img_dir: str, seq: str) -> str:
    """Ruta de cache única por carpeta de frames real (no solo por nombre de
    secuencia), para evitar colisiones entre datasets/splits distintos que
    reusen el mismo nombre de secuencia."""
    key = hashlib.sha1(os.path.abspath(img_dir).encode()).hexdigest()[:10]
    return os.path.join(det_cache_dir, f'{seq}__{key}.npz')


def reid_cache_path(reid_cache_dir: str, img_dir: str, seq: str) -> str:
    key = hashlib.sha1(os.path.abspath(img_dir).encode()).hexdigest()[:10]
    return os.path.join(reid_cache_dir, f'{seq}__{key}.npz')

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run tracking on all sequences + evaluate with TrackEval.")
    p.add_argument('--onnx-file',    type=str, default=None,
                   help='Modelo ONNX (alternativa a --pth-file/--config)')
    p.add_argument('--pth-file',     type=str, default=None,
                   help='Checkpoint de entrenamiento RT-DETR (.pth). Requiere --config')
    p.add_argument('--config',       type=str, default=None,
                   help='YAML de configuración RT-DETR usado para entrenar --pth-file')
    p.add_argument('--data-root',    type=str, required=True)
    p.add_argument('--split',        type=str, default='test')
    p.add_argument('--gt-root',      type=str, default=None,
                   help='Root con GTs MOT (default: mismo que data-root)')
    p.add_argument('--output-dir',   type=str, default='eval_output')
    p.add_argument('--tracker',      type=str, default='bytetrack',
                   choices=['bytetrack', 'cjm_bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', 'occluboost', 'boosttrack'])
    p.add_argument('--conf',         type=float, default=0.4)
    p.add_argument('--imgsz',        type=int, default=640)
    p.add_argument('--fps',          type=float, default=10.0)
    p.add_argument('--device',       type=str, default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--reid-device', type=str, default='cpu', choices=['cpu', 'cuda'],
                   help='[BotSORT/StrongSORT/DeepOCSort/OccluBoost] Device para el modelo ReID '
                        '(default: cpu)')
    p.add_argument('--iou-thresh',   type=float, default=0.5)
    p.add_argument('--no-convert-gt', action='store_true',
                   help='GTs ya están en píxeles absolutos, no convertir')
    p.add_argument('--gt-sanity',    action='store_true',
                   help='Usar GT como predicciones para verificar pipeline (debe dar métricas=1)')
    p.add_argument('--save-video',   action='store_true')
    p.add_argument('--det-cache-dir', type=str, default=_DEFAULT_DET_CACHE_DIR,
                   help='Directorio para cachear las detecciones crudas del detector y reusarlas '
                        'entre ejecuciones, sweeps y trackers distintos (default: %(default)s). '
                        'No se borra al re-ejecutar, a diferencia de --output-dir.')
    p.add_argument('--no-det-cache', action='store_true',
                   help='Desactiva el cache de detecciones y fuerza recalcular con el detector')
    p.add_argument('--reid-cache-dir', type=str, default=_DEFAULT_REID_CACHE_DIR,
                   help='Directorio para cachear embeddings ReID y reusarlos entre '
                        'ejecuciones, sweeps y trackers distintos (default: %(default)s). '
                        'Solo aplica a trackers que usan ReID.')
    p.add_argument('--no-reid-cache', action='store_true',
                   help='Desactiva el cache de embeddings ReID y fuerza recalcularlos')
    p.add_argument('--sequences',    nargs='+', default=None,
                   help='Solo evaluar estas secuencias')
    # ByteTrack params
    p.add_argument('--track-thresh', type=float, default=0.5)
    p.add_argument('--match-thresh', type=float, default=0.8)
    p.add_argument('--max-age',      type=int, default=30)
    # BotSort params
    p.add_argument('--appearance-thresh', type=float, default=0.25)
    p.add_argument('--proximity-thresh', type=float, default=0.5)
    p.add_argument('--new-track-thresh', type=float, default=0.6)
    p.add_argument('--cmc-method', type=str, default='ecc')
    p.add_argument('--no-native-reid', action='store_true')
    p.add_argument('--fuse-first-associate', action='store_true')
    # DeepOcSort params
    p.add_argument('--w-association-emb', type=float, default=0.5)
    p.add_argument('--alpha-fixed-emb', type=float, default=0.95)
    p.add_argument('--aw-param', type=float, default=0.5,
                   help='[DeepOcSort] peso de la fusion adaptativa apariencia/movimiento. '
                        'aw_off queda fijo en False en main_track.py -- siempre activa.')
    p.add_argument('--iou-threshold', type=float, default=0.3)
    # OcSort/DeepOcSort params compartidos (via BaseTracker o propios de cada clase)
    p.add_argument('--min-hits', type=int, default=3)
    p.add_argument('--delta-t', type=int, default=3)
    p.add_argument('--inertia', type=float, default=0.2)
    p.add_argument('--asso-func', type=str, default='iou',
                   choices=['iou', 'giou', 'diou', 'ciou', 'hmiou'])
    # BoostTrack params (--cmc-method ya definido arriba)
    p.add_argument('--lambda-iou', type=float, default=0.5)
    p.add_argument('--lambda-mhd', type=float, default=0.25)
    p.add_argument('--lambda-shape', type=float, default=0.25)
    p.add_argument('--dlo-boost-coef', type=float, default=0.65)
    p.add_argument('--with-reid', action='store_true')
    # Fuente del ReID (BotSort/DeepOcSort/BoostTrack) -- ver custom_reid.py
    p.add_argument('--reid-source', type=str, default='native',
                    choices=['native', 'dinov2', 'p4', 'c4_puro', 'p4fpn_puro', 'dinov2_puro', 'native_puro'])
    p.add_argument('--native-reid-weights', type=str, default=None,
                    help='[--reid-source native] Nombre/ruta del peso preentrenado a usar en vez '
                         'del default de boxmot (osnet_x0_25_msmt17.pt) -- cualquiera de '
                         'boxmot.reid.core.config.TRAINED_URLS (p.ej. osnet_x1_0_msmt17.pt, '
                         'resnet50_msmt17.pt, clip_veri.pt, clip_vehicleid.pt); boxmot lo descarga '
                         'solo si no existe localmente. None = comportamiento previo sin cambios.')
    p.add_argument('--native-reid-half', action='store_true',
                    help='[--reid-source native_puro] Corre el modelo (ej. CLIP-VehicleID) en fp16 '
                         'en vez de fp32 -- reduce a la mitad su huella de memoria GPU. Default False.')
    p.add_argument('--reid-checkpoint', type=str, default=None)
    p.add_argument('--reid-model', type=str, default='facebook/dinov2-small')
    p.add_argument('--reid-embedding-dim', type=int, default=256)
    p.add_argument('--reid-margin-ratio', type=float, default=0.2)
    # Reidentificacion externa por apariencia (ver reid_gallery.py)
    p.add_argument('--reid-stitch', action='store_true')
    p.add_argument('--reid-stitch-cosine-thresh', type=float, default=0.5)
    p.add_argument('--reid-stitch-max-gap', type=int, default=200)
    p.add_argument('--reid-stitch-bank-samples', type=int, default=1)
    p.add_argument('--reid-stitch-num-track-history', type=int, default=4)
    p.add_argument('--reid-stitch-single-crop', action='store_true',
                    help='[--reid-source c4_puro/p4fpn_puro/dinov2_puro] ALIAS de compatibilidad = '
                         '--reid-stitch-query-single-view + --reid-stitch-bank-single-view juntos '
                         '-- ver main_track.py.')
    p.add_argument('--reid-stitch-query-single-view', action='store_true',
                    help='[--reid-source c4_puro/p4fpn_puro/dinov2_puro] Ablacion factorial '
                         'single-view vs multi-view (M10/M3), eje QUERY -- ver main_track.py.')
    p.add_argument('--reid-stitch-bank-single-view', action='store_true',
                    help='[--reid-source c4_puro/p4fpn_puro/dinov2_puro] Ablacion factorial '
                         'single-view vs multi-view (M10/M3), eje BANCO -- ver main_track.py.')
    p.add_argument('--reid-stitch-debug', action='store_true')
    p.add_argument('--reid-stitch-viz-dir', type=str, default=None)
    p.add_argument('--reid-stitch-viz-top-k', type=int, default=5)
    p.add_argument('--draw-reid-events', action='store_true')
    p.add_argument('--reid-event-hold-frames', type=int, default=20)

    args = p.parse_args()
    if bool(args.onnx_file) == bool(args.pth_file):
        p.error('especifica exactamente uno de --onnx-file o --pth-file')
    if args.pth_file and not args.config:
        p.error('--pth-file requiere --config')
    if args.reid_source == 'dinov2' and not args.reid_checkpoint:
        p.error('--reid-source dinov2 requiere --reid-checkpoint')
    if args.reid_source == 'p4':
        if not args.reid_checkpoint:
            p.error('--reid-source p4 requiere --reid-checkpoint')
        if not args.pth_file:
            p.error('--reid-source p4 requiere --pth-file (no funciona con --onnx-file)')
    if args.reid_source == 'c4_puro':
        if not args.pth_file:
            p.error('--reid-source c4_puro requiere --pth-file (no funciona con --onnx-file)')
        if not args.reid_stitch:
            p.error('--reid-source c4_puro solo tiene sentido junto con --reid-stitch')
    if args.reid_source == 'p4fpn_puro':
        if not args.pth_file:
            p.error('--reid-source p4fpn_puro requiere --pth-file (no funciona con --onnx-file)')
        if not args.reid_stitch:
            p.error('--reid-source p4fpn_puro solo tiene sentido junto con --reid-stitch')
    if args.reid_source == 'dinov2_puro' and not args.reid_stitch:
        p.error('--reid-source dinov2_puro solo tiene sentido junto con --reid-stitch')
    return args


def get_sequences(data_root, split, filter_seqs=None, gt_root=None):
    split_dir = os.path.join(data_root, split)
    seqs = sorted(
        n for n in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, n))
    )
    if filter_seqs:
        seqs = [s for s in seqs if s in filter_seqs]
    if gt_root:
        gt_split = os.path.join(gt_root, split)
        seqs = [s for s in seqs if os.path.isdir(os.path.join(gt_split, s))]
    return seqs


def get_img_dir(data_root, split, seq):
    d = os.path.join(data_root, split, seq, 'img1')
    if not os.path.isdir(d):
        d = os.path.join(data_root, split, seq)
    return d


def run_tracking(args, sequences) -> str:
    """Corre el tracking de todas las secuencias EN ESTE MISMO PROCESO
    (import directo de main_track.run(), no un subprocess por secuencia).
    args ya tiene, con el mismo nombre, casi todos los atributos que
    main_track.run() necesita (se agregaron con los mismos flags a proposito
    en los dos parsers) -- solo faltan los que son propios de cada secuencia
    (frames/output/mot_output/det_cache/reid_cache), que se completan en
    seq_args mas abajo.

    Por que importa: cuando el tracker usa ReID (--reid-stitch o BotSort/
    DeepOcSort/BoostTrack con ReID), main_track.load_reid() puede tardar
    varios segundos (mas con --reid-source dinov2). Con un subprocess por
    secuencia ese costo se pagaba N veces por combinacion (una por
    secuencia); cargando el modelo una sola vez aca y pasandoselo a
    main_track.run(..., reid=reid) se paga una sola vez por combinacion."""
    txt_dir = os.path.join(args.output_dir, 'txt')
    vid_dir = os.path.join(args.output_dir, 'videos')
    os.makedirs(txt_dir, exist_ok=True)

    use_cache = not args.no_det_cache
    if use_cache:
        os.makedirs(args.det_cache_dir, exist_ok=True)
    use_reid_cache = not args.no_reid_cache
    if use_reid_cache:
        os.makedirs(args.reid_cache_dir, exist_ok=True)

    needs_reid = main_track.tracker_needs_reid(args)
    needs_reid_stitch = args.reid_stitch
    # p4/c4_puro/p4fpn_puro NO se pueden compartir entre secuencias: su
    # attach() pega el hook a una instancia concreta del detector, y esa
    # instancia se reconstruye por secuencia dentro de main_track.run()
    # (_build_sess()). Para esos tres, reid queda None aca y run() construye
    # su propia instancia por secuencia. dinov2_puro SI es compartible (sin
    # hook, igual que 'dinov2' nativo).
    shareable_reid = (needs_reid or needs_reid_stitch) and args.reid_source not in ('p4', 'c4_puro', 'p4fpn_puro')
    reid = main_track.load_reid(args) if shareable_reid else None

    for seq in sequences:
        img_dir = get_img_dir(args.data_root, args.split, seq)
        mot_out = os.path.join(txt_dir, f'{seq}.txt')

        out_video = os.path.join(vid_dir, f'{seq}.mp4') if args.save_video \
                    else os.devnull

        seq_args = copy.copy(args)
        seq_args.frames = img_dir
        seq_args.video = None
        seq_args.output = out_video
        seq_args.mot_output = mot_out
        seq_args.det_cache = det_cache_path(args.det_cache_dir, img_dir, seq) if use_cache else None
        seq_args.reid_cache = reid_cache_path(args.reid_cache_dir, img_dir, seq) if use_reid_cache else None

        print(f'\n{"="*60}')
        print(f'  Sequence: {seq}')
        print(f'{"="*60}')
        try:
            main_track.run(seq_args, reid=reid)
        except Exception:
            # Antes (subprocess por secuencia) una excepcion solo tumbaba esa
            # secuencia; aca corren todas en el mismo proceso, asi que hay que
            # atajarla a mano para no abortar el resto de la combinacion.
            print(f'  [ERROR] Failed on {seq}')
            traceback.print_exc()
    return txt_dir


def get_seq_dims(gt_root, split, seq):
    ini = os.path.join(gt_root, split, seq, 'seqinfo.ini')
    if not os.path.isfile(ini):
        return None, None
    cfg = configparser.ConfigParser()
    cfg.read(ini)
    w = int(cfg['Sequence']['imWidth'])
    h = int(cfg['Sequence']['imHeight'])
    return w, h


def prepare_gt(args, sequences):
    """Prepara GT en formato absoluto para TrackEval."""
    gt_root = args.gt_root or args.data_root
    gt_out = os.path.join(args.output_dir, 'gt_prepared', args.split)
    os.makedirs(gt_out, exist_ok=True)

    for seq in sequences:
        src_gt = os.path.join(gt_root, args.split, seq, 'gt', 'gt.txt')
        if not os.path.isfile(src_gt):
            print(f'  [WARN] No GT for {seq}: {src_gt}')
            continue

        dst_dir = os.path.join(gt_out, seq, 'gt')
        os.makedirs(dst_dir, exist_ok=True)

        # Copiar seqinfo.ini
        ini_src = os.path.join(gt_root, args.split, seq, 'seqinfo.ini')
        ini_dst = os.path.join(gt_out, seq, 'seqinfo.ini')
        if os.path.isfile(ini_src):
            shutil.copy2(ini_src, ini_dst)

        if args.no_convert_gt:
            # Copiar tal cual, filtrando tid=0; normalizar clase -1 → 1
            with open(src_gt) as f_in, open(os.path.join(dst_dir, 'gt.txt'), 'w') as f_out:
                for line in f_in:
                    p = line.strip().split(',')
                    if len(p) >= 2 and p[1] == '0':
                        continue
                    if len(p) >= 8 and p[7].strip() == '-1':
                        p[7] = '1'
                    # TrackEval (MotChallenge2DBox) lee la columna 6 ("conf" en
                    # el formato MOT) como "zero_marked": si vale 0, esa caja de
                    # GT se excluye de la evaluacion. Nuestro dataset trae la
                    # visibilidad real (0=ocluido, 1=visible) en la ULTIMA
                    # columna, que TrackEval nunca lee -- la trasladamos a donde
                    # TrackEval si la usa.
                    if len(p) >= 10:
                        p[6] = p[9]
                    f_out.write(','.join(p) + '\n')
        else:
            # Convertir cxcywh normalizado → xywh absoluto
            W, H = get_seq_dims(gt_root, args.split, seq)
            if W is None:
                print(f'  [WARN] No seqinfo.ini for {seq}, skipping GT conversion')
                continue
            with open(src_gt) as f_in, open(os.path.join(dst_dir, 'gt.txt'), 'w') as f_out:
                for line in f_in:
                    p = line.strip().split(',')
                    if len(p) < 6:
                        f_out.write(line)
                        continue
                    if p[1] == '0':
                        continue
                    fid, tid = p[0], p[1]
                    cx, cy, bw, bh = float(p[2]), float(p[3]), float(p[4]), float(p[5])
                    rest_parts = p[6:]
                    x = (cx - bw / 2) * W
                    y = (cy - bh / 2) * H
                    if len(rest_parts) >= 2 and rest_parts[1].strip() == '-1':
                        rest_parts[1] = '1'
                    # TrackEval (MotChallenge2DBox) lee la columna 6 ("conf" en
                    # el formato MOT) como "zero_marked": si vale 0, esa caja de
                    # GT se excluye de la evaluacion (ni cuenta como miss si el
                    # tracker no la encuentra, ni como FP si la encuentra). nuestro
                    # dataset trae la visibilidad real (0=ocluido, 1=visible) en la
                    # ULTIMA columna, que TrackEval nunca lee -- la trasladamos a
                    # donde TrackEval si la usa.
                    if len(rest_parts) >= 4:
                        rest_parts[0] = rest_parts[3]
                    rest = ','.join(rest_parts)
                    f_out.write(f'{fid},{tid},{x:.2f},{y:.2f},{bw*W:.2f},{bh*H:.2f},{rest}\n')

        print(f'  [GT] {seq} OK')

    return gt_out



def run_trackeval(gt_dir, tracker_dir, seqmap, split, iou_thresh):
    if not os.path.isfile(_TRACKEVAL_SCRIPT):
        print(f'\n[ERROR] TrackEval not found: {_TRACKEVAL_SCRIPT}')
        return None

    cmd = [
        sys.executable, _TRACKEVAL_SCRIPT,
        '--SPLIT_TO_EVAL',      split,
        '--METRICS',            'HOTA', 'CLEAR', 'Identity',
        '--GT_FOLDER',          gt_dir,
        '--SEQMAP_FILE',        seqmap,
        '--SKIP_SPLIT_FOL',     'True',
        '--TRACKERS_TO_EVAL',   '',
        '--TRACKER_SUB_FOLDER', '',
        '--USE_PARALLEL',       'False',
        '--PLOT_CURVES',        'False',
        '--TRACKERS_FOLDER',    tracker_dir,
        '--THRESHOLD',          str(iou_thresh),
    ]

    print('\n' + '='*60)
    print('  TrackEval')
    print('='*60 + '\n')

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print('[ERROR] TrackEval failed:')
        print((r.stderr or r.stdout)[-2000:])
        return None
    return r.stdout


def main():
    args = parse_args()
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    if args.pth_file:
        print(f'Model (pth): {args.pth_file}  (config: {args.config})')
    else:
        print(f'ONNX model : {args.onnx_file}')
    print(f'Dataset    : {args.data_root}')
    print(f'Split      : {args.split}')
    print(f'Tracker    : {args.tracker}')
    print(f'Conf thresh: {args.conf}')
    print(f'Output     : {args.output_dir}')

    gt_root = args.gt_root or args.data_root
    sequences = get_sequences(args.data_root, args.split, args.sequences, gt_root if os.path.isdir(os.path.join(gt_root, args.split)) else None)
    print(f'Sequences  : {len(sequences)}')

    # 1) Tracking (o sanity check con GT como predicciones)
    if args.gt_sanity:
        from main_track import gt_as_predictions
        gt_prepared = prepare_gt(args, sequences)
        txt_dir = gt_as_predictions(gt_prepared, args.output_dir, split=args.split)
    else:
        txt_dir = run_tracking(args, sequences)

    # 2) Preparar GT (si no se hizo ya en sanity)
    if not args.gt_sanity:
        gt_prepared = prepare_gt(args, sequences)
    gt_dir = gt_prepared

    if not os.path.isdir(os.path.join(gt_root, args.split)):
        print(f'\n[WARN] No GT found at {gt_root}/{args.split} — skipping evaluation.')
        print(f'MOT results at: {os.path.join(args.output_dir, "txt")}')
        return

    # 3) Seqmap
    seqmap = os.path.join(args.output_dir, 'seqmap.txt')
    with open(seqmap, 'w') as f:
        f.write('name\n')
        for s in sequences:
            f.write(f'{s}\n')

    # 4) TrackEval
    out = run_trackeval(gt_dir, txt_dir, seqmap, args.split, args.iou_thresh)


    if out:
        for line in out.splitlines():
            if line.strip():
                print(line)

    print(f'\nMOT results : {txt_dir}')
    print(f'GT prepared : {gt_dir}')


if __name__ == '__main__':
    main()
