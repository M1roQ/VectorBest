import csv
from pathlib import Path
from anomalib.data import Folder, PredictDataset
from anomalib.models import Padim
from anomalib.engine import Engine

ROOT = Path("E:/VectorBestData")
DATA = ROOT / "padim_tmb_15"

dm = Folder(
    name="tmb_15",
    root=DATA,
    normal_dir="train/good",
    abnormal_dir="test/bad",
    normal_test_dir="test/good",
    train_batch_size=16,
    eval_batch_size=16,
    num_workers=0,
    seed=42,
)
model = Padim(backbone="resnet18", pre_trained=True)
engine = Engine(default_root_dir=str(ROOT / "results_padim_tmb_15"))
engine.fit(model=model, datamodule=dm)
engine.test(model=model, datamodule=dm)

rows = []
for kind in ("good", "bad"):
    ds = PredictDataset(path=DATA / "test" / kind)
    for batch in engine.predict(model=model, dataset=ds):
        for p, s in zip(batch.image_path, batch.pred_score.flatten().tolist()):
            rows.append((kind, Path(p).name, round(s, 4)))

with open(ROOT / "scores_tmb_15.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([("kind", "file", "score")] + rows)