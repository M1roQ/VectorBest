from ultralytics import YOLO

DATA = "datasets/family_cls_v1"
model = YOLO("yolo11n-cls.pt")
model.train(
    data=DATA, imgsz=448, epochs=60, batch=32,
    fliplr=0.0, scale=0.1, workers=0,
    project="runs/classify", name="family_v1",
)
model.val(data=DATA, split="test")