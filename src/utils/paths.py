from __future__ import annotations

from pathlib import Path


# Project root:
# C:\Camera_Surveillance
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Main directories
CONFIG_DIR = PROJECT_ROOT / "config"
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCUMENTATION_DIR = PROJECT_ROOT / "documentation"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


# Dataset directories
RAW_DATASETS_DIR = DATASETS_DIR / "raw"
MERGED_DATASET_DIR = DATASETS_DIR / "merged_dataset"

TRAIN_IMAGES_DIR = MERGED_DATASET_DIR / "train" / "images"
TRAIN_LABELS_DIR = MERGED_DATASET_DIR / "train" / "labels"

VALID_IMAGES_DIR = MERGED_DATASET_DIR / "valid" / "images"
VALID_LABELS_DIR = MERGED_DATASET_DIR / "valid" / "labels"

TEST_IMAGES_DIR = MERGED_DATASET_DIR / "test" / "images"
TEST_LABELS_DIR = MERGED_DATASET_DIR / "test" / "labels"

DATASET_YAML_PATH = MERGED_DATASET_DIR / "data.yaml"
MERGE_REPORT_PATH = MERGED_DATASET_DIR / "merge_report.txt"


# Model directories
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"
WEIGHTS_DIR = MODELS_DIR / "weights"
EXPORTS_DIR = MODELS_DIR / "exports"
TRAINING_RUNS_DIR = MODELS_DIR / "training_runs"


# Default model paths
DEFAULT_BEST_WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
DEFAULT_LAST_WEIGHTS_PATH = WEIGHTS_DIR / "last.pt"


# Report directories
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
TRAINING_REPORTS_DIR = REPORTS_DIR / "training_reports"


# Configuration files
TRAINING_CONFIG_PATH = CONFIG_DIR / "training.yaml"
INFERENCE_CONFIG_PATH = CONFIG_DIR / "inference.yaml"


def ensure_project_directories() -> None:
    """
    Create the standard project directories if they do not exist.

    This function does not create dataset image or label files.
    It only creates the folder structure.
    """

    directories = [
        CONFIG_DIR,
        DATASETS_DIR,
        RAW_DATASETS_DIR,
        MERGED_DATASET_DIR,
        TRAIN_IMAGES_DIR,
        TRAIN_LABELS_DIR,
        VALID_IMAGES_DIR,
        VALID_LABELS_DIR,
        TEST_IMAGES_DIR,
        TEST_LABELS_DIR,
        MODELS_DIR,
        PRETRAINED_MODELS_DIR,
        WEIGHTS_DIR,
        EXPORTS_DIR,
        TRAINING_RUNS_DIR,
        REPORTS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        TRAINING_REPORTS_DIR,
        DOCUMENTATION_DIR,
        EXPERIMENTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def resolve_project_path(
    path: str | Path,
) -> Path:
    """
    Convert a path into an absolute project-aware path.

    Examples:

    resolve_project_path("datasets/merged_dataset/data.yaml")

    returns:

    C:\\Camera_Surveillance\\datasets\\merged_dataset\\data.yaml
    """

    candidate = Path(path)

    if candidate.is_absolute():
        return candidate.resolve()

    return (PROJECT_ROOT / candidate).resolve()


def require_file(
    path: str | Path,
    description: str = "File",
) -> Path:
    """
    Resolve a path and verify that the file exists.
    """

    resolved_path = resolve_project_path(path)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"{description} not found: {resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            f"{description} is not a file: {resolved_path}"
        )

    return resolved_path


def require_directory(
    path: str | Path,
    description: str = "Directory",
) -> Path:
    """
    Resolve a path and verify that the directory exists.
    """

    resolved_path = resolve_project_path(path)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"{description} not found: {resolved_path}"
        )

    if not resolved_path.is_dir():
        raise ValueError(
            f"{description} is not a directory: {resolved_path}"
        )

    return resolved_path


def get_training_run_directory(
    run_name: str,
) -> Path:
    """
    Return the output directory for a training run.
    """

    if not run_name.strip():
        raise ValueError("run_name cannot be empty.")

    return TRAINING_RUNS_DIR / run_name.strip()


def get_training_weights_directory(
    run_name: str,
) -> Path:
    """
    Return the weights folder for a training run.

    Example:
    models/training_runs/baseline_yolo11n/weights
    """

    return get_training_run_directory(run_name) / "weights"


def get_best_weights_path(
    run_name: str,
) -> Path:
    """
    Return the expected best.pt location for a training run.
    """

    return get_training_weights_directory(run_name) / "best.pt"


def get_last_weights_path(
    run_name: str,
) -> Path:
    """
    Return the expected last.pt location for a training run.
    """

    return get_training_weights_directory(run_name) / "last.pt"


def print_project_paths() -> None:
    """
    Print important project paths for debugging.
    """

    paths = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "DATASET_YAML_PATH": DATASET_YAML_PATH,
        "TRAINING_CONFIG_PATH": TRAINING_CONFIG_PATH,
        "INFERENCE_CONFIG_PATH": INFERENCE_CONFIG_PATH,
        "TRAINING_RUNS_DIR": TRAINING_RUNS_DIR,
        "WEIGHTS_DIR": WEIGHTS_DIR,
        "EXPORTS_DIR": EXPORTS_DIR,
        "REPORTS_DIR": REPORTS_DIR,
    }

    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    ensure_project_directories()
    print_project_paths()