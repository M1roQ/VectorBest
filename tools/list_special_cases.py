import csv
from pathlib import Path

OUT = Path("datasets/label_crops_v2")

with open(OUT / "crops.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

by_frame = {}
for r in rows:
    by_frame.setdefault((r["split"], r["image"]), []).append(r)

print("Кадры с несколькими candidate:")
for key, dets in by_frame.items():
    cands = [d for d in dets if d["status"] == "candidates"]
    if len(cands) > 1:
        print(" ", key, [(d["crop_id"], d["width_frac"], d["conf"]) for d in cands])

print("Кадры без детекций:")
no_cand = (OUT / "frames_without_candidate.txt").read_text(
    encoding="utf-8").splitlines()
for item in no_cand:
    split, name = item.split("/", 1)
    if (split, name) not in by_frame:
        print(" ", item)

print("Кадры без candidate, но с рамками:", sum(
    1 for item in no_cand
    if tuple(item.split("/", 1)) in by_frame))