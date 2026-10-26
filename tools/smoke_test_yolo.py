from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolo11n.pt"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    print(f"YOLO model loaded: {MODEL_PATH}")


if __name__ == "__main__":
    main()