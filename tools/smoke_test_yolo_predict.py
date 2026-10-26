from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "yolo11n.pt"


def main() -> None:
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        device=device,
        conf=0.25,
        verbose=False,
    )

    boxes = results[0].boxes
    print(f"Detected objects: {len(boxes)}")

    for box in boxes:
        class_id = int(box.cls.item())
        class_name = model.names[class_id]
        confidence = float(box.conf.item())
        print(f"{class_name}: {confidence:.2f}")


if __name__ == "__main__":
    main()