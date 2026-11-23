from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/classify/runs/classify/family_v1-2/weights/best.pt")
    model.val(data="datasets/family_cls_v1", split="test", workers=0,
              project="family_cls", name="test_eval")