from pathlib import Path
from ultralytics import YOLO

MODEL = "runs/detect/yolo11n_labels_det_v2_run1/weights/best.pt"
IMG_DIR = Path("datasets/labels_det_v2/test/images")
LBL_DIR = Path("datasets/labels_det_v2/test/labels")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0.0


def load_gt(path, w, h):
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                _, x, y, bw, bh = map(float, p)
                boxes.append(((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h))
    return boxes


def edge(b, w, h, m=3):
    return b[0] <= m or b[1] <= m or b[2] >= w-m or b[3] >= h-m


model = YOLO(MODEL)
for r in model.predict(source=str(IMG_DIR), conf=0.5, imgsz=640,
                       stream=True, verbose=False):
    h, w = r.orig_shape
    name = Path(r.path).name
    gt = load_gt(LBL_DIR / (Path(r.path).stem + ".txt"), w, h)
    preds = [(b.tolist(), float(c)) for b, c in zip(r.boxes.xyxy, r.boxes.conf)]
    used = set()
    for box, conf in sorted(preds, key=lambda t: -t[1]):
        best, bi = 0.0, None
        for i, g in enumerate(gt):
            if i not in used and iou(box, g) > best:
                best, bi = iou(box, g), i
        if bi is not None and best >= 0.5:
            used.add(bi)
        else:
            print("FP", name, [round(v) for v in box], round(conf, 2),
                  "край" if edge(box, w, h) else "")
    for i, g in enumerate(gt):
        if i not in used:
            print("FN", name, [round(v) for v in g],
                  "край" if edge(g, w, h) else "")