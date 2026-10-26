# VectorBest

Прототип системы контроля качества этикеток: детекция этикетки, проверка штрих-кода,
OCR надписи и выявление дефектов печати через PaDiM.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
```

## Структура

- `datasets/` — изображения и разметка.
- `models/` — веса YOLO и PaDiM.
- `src/` — этапы пайплайна.
- `app/` — FastAPI и веб-интерфейс.
- `tools/` — обучение, подготовка данных и тесты.