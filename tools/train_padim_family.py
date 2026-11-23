import csv, shutil, sys
from pathlib import Path
from anomalib.data import Folder, PredictDataset
from anomalib.engine import Engine
from anomalib.models import Padim

SRC = Path("datasets/label_families_v1")
ROOT = Path("E:/VectorBestData")
EXT = {".png", ".jpg", ".jpeg", ".bmp"}

def files(d):
    return [f for f in Path(d).glob("*") if f.is_file() and f.suffix.lower() in EXT]

def copy_all(items, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for f in items:
        shutil.copy2(f, dst / f.name)

def build(family, dst):
    if dst.exists():
        shutil.rmtree(dst)
    copy_all(files(SRC / "train" / family), dst / "train" / "good")
    copy_all(files(SRC / "valid" / family) + files(SRC / "test" / family), dst / "test" / "good")
    copy_all(files(SRC / "test" / family / "bad"), dst / "test" / "bad")

if __name__ == "__main__":
    family = sys.argv[1]
    data = ROOT / f"padim_{family}"
    build(family, data)

    dm = Folder(name=family, root=data, normal_dir="train/good",
                abnormal_dir="test/bad", normal_test_dir="test/good",
                train_batch_size=16, eval_batch_size=16, num_workers=0, seed=42)
    model = Padim(backbone="resnet18", pre_trained=True)
    engine = Engine(default_root_dir=str(ROOT / f"results_padim_{family}"))
    engine.fit(model=model, datamodule=dm)
    engine.test(model=model, datamodule=dm)

    rows = []
    for kind in ("good", "bad"):
        ds = PredictDataset(path=data / "test" / kind)
        for batch in engine.predict(model=model, dataset=ds):
            for p, s in zip(batch.image_path, batch.pred_score.flatten().tolist()):
                rows.append((kind, Path(p).name, round(s, 4)))
    with open(ROOT / f"scores_{family}.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("kind", "file", "score")] + rows)