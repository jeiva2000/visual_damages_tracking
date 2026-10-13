"""Compara run_inference de main_track_tm contra los scripts de /mnt/Disk2/test_models/infers en 3 frames."""
import os, sys, importlib, numpy as np, cv2, torch, onnxruntime as ort
import torchvision.transforms as T
from PIL import Image
sys.path.insert(0, "/mnt/Disk2/test_models/infers"); sys.path.insert(0, ".")
CK = "/mnt/Disk2/test_models/checkpoints/"
frames = ["all_dataset_view/all/aveo/img1/000001.jpg", "all_dataset_view/all/video2/img1/000050.jpg", "all_dataset_view/all/aveo/img1/000101.jpg"]
def ref(kind, sess, f):
    im = Image.open(f).convert("RGB"); w, h = im.size
    if kind == "rtv4":
        import infer_onnx_rtv4 as r4
        l, b, s = r4.process_image(sess, T.Compose([T.ToTensor()]), im); return b, s
    x = T.Compose([T.Resize((640, 640)), T.ToTensor()])(im)[None].numpy()
    o = sess.run(None, {"images": x, "orig_target_sizes": torch.tensor([w, h])[None].numpy()})
    return o[1][0], (o[3][0].reshape(-1) if kind == "q" else o[2][0])
for kind, model, env in [("rtv2", "rtdetrv2_qcs_cardd_default.onnx", {}), ("q", "rtdetr_quality_datasetv4_best.onnx", {"TM_SCORE_IDX": "3"}),
                         ("q", "rtdetr_quality_19_teide.onnx", {"TM_SCORE_IDX": "3"}), ("rtv4", "rtdetrv4_epoch_11.onnx", {"TM_LETTERBOX": "1"})]:
    for k in ["TM_LETTERBOX", "TM_SCORE_IDX"]: os.environ.pop(k, None)
    os.environ.update(env)
    import main_track_tm as mt; mt = importlib.reload(mt)
    sess = mt.build_session(CK + model, "cpu")
    for f in frames:
        img = cv2.imread(f); im_data, osz = mt.preprocess(img, 640)
        l, b, s = mt.run_inference(sess, im_data, osz)
        rb, rs = ref(kind, sess, f)
        k = s > 0.4; kr = rs > 0.4
        print(f"{kind:4s} {model:38s} {os.path.basename(os.path.dirname(os.path.dirname(f)))}/{os.path.basename(f)}  n>0.4 tm={k.sum()} ref={kr.sum()}  "
              f"max|dbox|={np.abs(b[k]-rb[k]).max() if k.any() else 0:.2f}px  max|dscore|={np.abs(s-rs).max():.4f}")
