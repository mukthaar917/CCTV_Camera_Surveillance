# Training Guide

Use `epochs: 1` first for a validation run.

```powershell
python -m src.training.train --config config\training.yaml
```

Expected weights:

```text
models/training_runs/baseline_yolo11n/weights/best.pt
models/training_runs/baseline_yolo11n/weights/last.pt
```
