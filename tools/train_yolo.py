from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "labels_det_v2" / "data.yaml"
MODEL_PATH = ROOT / "models" / "yolo11n.pt"


def main() -> None:
    model = YOLO(str(MODEL_PATH))
    model.train(
        data=str(DATA_PATH),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        name="yolo11n_labels_det_v2_run1",
    )


if __name__ == "__main__":
    main()