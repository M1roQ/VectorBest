import csv
import re
from collections import Counter
from pathlib import Path

from PIL import Image

OUT = Path("datasets/label_crops_v2")
SRC = Path("datasets/labels_det_v2")
EDGE = 3

with open(OUT / "crops.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

frames = {}
for r in rows:
    frames.setdefault((r["split"], r["image"]), []).append(r)

no_cand = (OUT / "frames_without_candidate.txt").read_text(
    encoding="utf-8").splitlines()

all_sizes = Counter()
for split in ("train", "valid", "test"):
    for p in (SRC / split / "images").glob("*.*"):
        with Image.open(p) as im:
            all_sizes[im.size] += 1

sizes, reasons, sides = Counter(), Counter(), Counter()
suffix = Counter()
examples = []

for item in no_cand:
    split, name = item.split("/", 1)
    with Image.open(SRC / split / "images" / name) as im:
        w, h = im.size
    sizes[(w, h)] += 1
    suffix["с суффиксом -N" if re.search(r"-\d+\.\w+$", name) else "без"] += 1
    dets = frames.get((split, name), [])
    if not dets:
        reasons["нет детекций"] += 1
        continue
    r = max(dets, key=lambda d: float(d["width_frac"]))
    x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
    s = "".join(c for c, flag in (("L", x1 <= EDGE), ("T", y1 <= EDGE),
                                  ("R", x2 >= w - EDGE), ("B", y2 >= h - EDGE))
                if flag) or "-"
    sides[s] += 1
    reasons[r["reason"]] += 1
    if len(examples) < 10:
        examples.append((name[:60], r["width_frac"], r["conf"], s, (w, h)))

print("Размеры всех кадров:", dict(all_sizes))
print("Размеры кадров без candidate:", dict(sizes))
print("Причина у самой широкой рамки:", dict(reasons))
print("Какие края касаются:", dict(sides))
print("Суффикс в имени:", dict(suffix))
for e in examples:
    print(e)