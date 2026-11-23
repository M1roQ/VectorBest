from pathlib import Path

ROOT = Path("datasets/labels_det_v2")
M = 0.003

for split in ("train", "valid", "test"):
    widths, total = [], 0
    for f in (ROOT / split / "labels").glob("*.txt"):
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            _, x, y, w, h = map(float, p)
            total += 1
            if x - w / 2 <= M or x + w / 2 >= 1 - M:
                widths.append(round(w, 3))
    widths.sort()
    print(split, "всего:", total, "у края:", len(widths))
    print("  ширины:", widths)