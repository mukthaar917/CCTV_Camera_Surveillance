# Inference Guide

```powershell
python -m src.inference.image_inference --weights models\training_runs\baseline_yolo11n\weights\best.pt --image samples\example.jpg
python -m src.inference.pipeline --weights models\training_runs\baseline_yolo11n\weights\best.pt --source samples\video.mp4
python -m src.inference.pipeline --weights models\training_runs\baseline_yolo11n\weights\best.pt --source 0
```
