from pathlib import Path
from PIL import Image

SRC = Path("datasets/label_families_v1")
DST = Path("datasets/family_cls_v1")
CLASSES = ["tmb_15", "tmb_35", "stop_barcode", "fsb_t25", "stop_12"]
SPLITS = {"train": "train", "valid": "val", "test": "test"}
SIZE = 448
EXT = {".png", ".jpg", ".jpeg", ".bmp"}

def letterbox(img):
    img = img.convert("RGB")
    k = SIZE / max(img.size)
    img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.BICUBIC)
    canvas = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    canvas.paste(img, ((SIZE - img.width) // 2, (SIZE - img.height) // 2))
    return canvas

for s, out_s in SPLITS.items():
    for c in CLASSES:
        out = DST / out_s / c
        out.mkdir(parents=True, exist_ok=True)
        src_dir = SRC / s / c
        files = src_dir.rglob("*") if s == "test" else src_dir.glob("*")
        n = 0
        for f in files:
            if f.is_file() and f.suffix.lower() in EXT:
                letterbox(Image.open(f)).save(out / (f.stem + ".png"))
                n += 1
        print(out_s, c, n)