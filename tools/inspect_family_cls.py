from pathlib import Path
from PIL import Image
from ultralytics import YOLO

SIZE = 448
EXT = {".png", ".jpg", ".jpeg", ".bmp"}

def letterbox(img):
    img = img.convert("RGB")
    k = SIZE / max(img.size)
    img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.BICUBIC)
    canvas = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    canvas.paste(img, ((SIZE - img.width) // 2, (SIZE - img.height) // 2))
    return canvas

def predict(model, img):
    r = model.predict(letterbox(img), imgsz=SIZE, verbose=False)[0]
    return model.names[r.probs.top1], float(r.probs.top1conf)

if __name__ == "__main__":
    model = YOLO("runs/classify/runs/classify/family_v1-2/weights/best.pt")
    rows = []
    for cls_dir in sorted(Path("datasets/family_cls_v1/test").iterdir()):
        for f in cls_dir.glob("*.png"):
            pred, conf = predict(model, Image.open(f))
            rows.append((cls_dir.name, pred, round(conf, 3), f.name))

    wrong = [r for r in rows if r[0] != r[1]]
    print("Ошибки:", len(wrong), "из", len(rows))
    for r in wrong:
        print("  истина=%s, предсказано=%s, conf=%s, файл=%s" % r)

    ok = sorted((r for r in rows if r[0] == r[1]), key=lambda r: r[2])
    print("Самые неуверенные верные:")
    for r in ok[:5]:
        print("  %s conf=%s %s" % (r[0], r[2], r[3]))

    print("Неизвестный макет solution_tmb_13:")
    for f in sorted(Path("datasets/solution_tmb_13").rglob("*")):
        if f.is_file() and f.suffix.lower() in EXT:
            pred, conf = predict(model, Image.open(f))
            print("  предсказано=%s conf=%.3f %s" % (pred, conf, f.name))