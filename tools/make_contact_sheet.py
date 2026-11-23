import csv
import random
from pathlib import Path

import cv2
import numpy as np

OUT = Path("datasets/label_crops_v2")
STATUS = "review"
REASON = "touches_edge"
N = 20
COLS = 4
TILE_W, TILE_H = 480, 270
SEED = 0

with open(OUT / "crops.csv", encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f, delimiter=";")
            if r["status"] == STATUS and (REASON is None or r["reason"] == REASON)]

random.Random(SEED).shuffle(rows)
rows = rows[:N]


def read(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


tiles = []
for r in rows:
    img = read(OUT / r["context_path"])
    if img is None:
        continue
    h, w = img.shape[:2]
    s = min(TILE_W / w, TILE_H / h)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    tile = np.full((TILE_H + 24, TILE_W, 3), 255, np.uint8)
    tile[24:24 + img.shape[0], :img.shape[1]] = img
    cv2.putText(tile, f"{r['split']} {r['image'][:28]} {r['conf']}", (4, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    tiles.append(tile)

if not tiles:
    raise SystemExit("Нет подходящих кропов")

lines = []
for i in range(0, len(tiles), COLS):
    row = tiles[i:i + COLS]
    while len(row) < COLS:
        row.append(np.full_like(tiles[0], 255))
    lines.append(np.hstack(row))
sheet = np.vstack(lines)

name = f"qa_{STATUS}_{REASON or 'all'}.jpg"
ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
buf.tofile(str(OUT / name))
print(len(tiles), OUT / name)