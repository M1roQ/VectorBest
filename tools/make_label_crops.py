import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT / "runs" / "detect" / "yolo11n_labels_det_v2_run1" / "weights" / "best.pt"
)
DATASET_PATH = ROOT / "datasets" / "labels_det_v2"
OUTPUT_PATH = ROOT / "datasets" / "label_crops_v2"

SPLITS = ("train", "valid", "test")
CONF = 0.5
CONTEXT_MARGIN = 0.03
EDGE_PX = 3
EDGE_FRAC = 0.01
MIN_WIDTH_FRAC = 0.2
FULL_WIDTH_FRAC = 0.5


def save_image(path: Path, image: np.ndarray, ext: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Не удалось сохранить: {path}")
    buf.tofile(str(path))


def clamp_box(x1, y1, x2, y2, w, h):
    return (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(w, int(round(x2))),
        min(h, int(round(y2))),
    )


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Папка уже существует: {OUTPUT_PATH}\n"
            "Для нового запуска измените OUTPUT_PATH."
        )
    for split in SPLITS:
        images_path = DATASET_PATH / split / "images"
        if not images_path.is_dir():
            raise FileNotFoundError(f"Нет папки: {images_path}")

    model = YOLO(str(MODEL_PATH))
    rows = []
    no_candidate = []

    for split in SPLITS:
        results = model.predict(
            source=str(DATASET_PATH / split / "images"),
            conf=CONF,
            imgsz=640,
            device=0,
            save=False,
            stream=True,
            verbose=False,
        )
        stats = {"images": 0, "boxes": 0, "candidates": 0,
                 "review": 0, "multi": 0}

        for r in results:
            stats["images"] += 1
            img = r.orig_img
            h, w = r.orig_shape
            stem = Path(r.path).stem
            dets = sorted(
                zip(r.boxes.xyxy.cpu().numpy().tolist(),
                    r.boxes.conf.cpu().numpy().tolist()),
                key=lambda d: d[0][0],
            )
            preview = img.copy()
            n_cand = 0

            for i, (box, conf) in enumerate(dets, start=1):
                x1, y1, x2, y2 = clamp_box(*box, w, h)
                if x2 <= x1 or y2 <= y1:
                    continue
                stats["boxes"] += 1

                width_frac = (x2 - x1) / w
                ex = max(EDGE_PX, int(w * EDGE_FRAC))
                ey = max(EDGE_PX, int(h * EDGE_FRAC))
                touches = (x1 <= EDGE_PX or y1 <= EDGE_PX
                        or x2 >= w - EDGE_PX or y2 >= h - EDGE_PX)
                near = (x1 <= ex or y1 <= ey or x2 >= w - ex or y2 >= h - ey)
                if touches:
                    status, reason = "review", "touches_edge"
                elif width_frac < MIN_WIDTH_FRAC:
                    status, reason = "review", "too_small"
                elif near and width_frac < FULL_WIDTH_FRAC:
                    status, reason = "review", "near_edge"
                else:
                    status, reason = "candidates", "ok_near_edge" if near else "ok"
                stats[status] += 1
                n_cand += status == "candidates"

                mx = int(round((x2 - x1) * CONTEXT_MARGIN))
                my = int(round((y2 - y1) * CONTEXT_MARGIN))
                cx1, cy1, cx2, cy2 = clamp_box(
                    x1 - mx, y1 - my, x2 + mx, y2 + my, w, h)

                name = f"{stem}_c{i:02d}.png"
                tight_rel = Path(split) / status / "tight" / name
                ctx_rel = Path(split) / status / "context" / name
                save_image(OUTPUT_PATH / tight_rel, img[y1:y2, x1:x2], ".png")
                save_image(OUTPUT_PATH / ctx_rel, img[cy1:cy2, cx1:cx2], ".png")

                color = (0, 200, 0) if status == "candidates" else (0, 165, 255)
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3)
                cv2.putText(preview, f"{conf:.2f} {reason}", (x1 + 6, y1 + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                rows.append([
                    split, Path(r.path).name, i, status, reason,
                    f"{conf:.3f}", x1, y1, x2, y2,
                    f"{width_frac:.3f}", f"{(y2 - y1) / h:.3f}", int(touches),
                    tight_rel.as_posix(), ctx_rel.as_posix(),
                ])

            save_image(OUTPUT_PATH / split / "preview" / f"{stem}.jpg",
                       preview, ".jpg")
            if n_cand == 0:
                no_candidate.append(f"{split}/{Path(r.path).name}")
            elif n_cand > 1:
                stats["multi"] += 1

        print(
            f"{split}: кадров {stats['images']}, рамок {stats['boxes']}, "
            f"candidates {stats['candidates']}, review {stats['review']}, "
            f"кадров с >1 candidate: {stats['multi']}"
        )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH / "crops.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow([
            "split", "image", "crop_id", "status", "reason", "conf",
            "x1", "y1", "x2", "y2", "width_frac", "height_frac",
            "touches_edge", "tight_path", "context_path",
        ])
        wr.writerows(rows)
    (OUTPUT_PATH / "frames_without_candidate.txt").write_text(
        "\n".join(no_candidate), encoding="utf-8")

    print(f"Кадров без candidate: {len(no_candidate)}")
    print(f"Результаты: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()