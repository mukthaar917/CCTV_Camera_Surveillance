# Dataset Guide

Final classes: vehicle, person, fire, smoke, animal, package.

```powershell
python -m src.data.validate_dataset --data datasets\merged_dataset\data.yaml
python -m src.data.dataset_statistics --data datasets\merged_dataset\data.yaml
python -m src.data.visualize_labels --data datasets\merged_dataset\data.yaml --split train --count 20
python -m src.data.create_balanced_split --source datasets\merged_dataset --output datasets\balanced_dataset --overwrite
```
