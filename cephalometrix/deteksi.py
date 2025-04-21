import torch
import numpy as np
import os
import sys
import cv2
from pathlib import Path
import argparse
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, increment_path
from utils.torch_utils import select_device
from utils.dataloaders import LoadImages
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def run(weights, source, conf_thres=0.25, iou_thres=0.45, device='cpu', img_size=640):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    print(f"📌 Model dimuat dengan {len(names)} kelas: {names}")

    save_dir = increment_path(ROOT / 'result/exp', exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = LoadImages(source, img_size=img_size, stride=stride, auto=model.pt)

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        im = im[None] if im.ndimension() == 3 else im

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):
            im0 = im0s.copy()
            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = f"{names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            save_path = save_dir / Path(path).name
            cv2.imwrite(str(save_path), im0)
            print(f"✅ Saved: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Script")
    parser.add_argument('--weights', type=str, default=str(ROOT / 'best.pt'), help='Path to weights file (.pt)')
    parser.add_argument('--source', type=str, default=str(ROOT / 'data/images'), help='Image file or directory')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (e.g., cpu or 0 for GPU)')
    parser.add_argument('--img-size', type=int, default=640, help='Inference image size')
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    run(**vars(opt))



# python detect.py --weights <path_ke_model.pt> --source <path_ke_gambar/folder> --device cpu --conf-thres 0.25 --iou-thres 0.45 --img-size 640

