from pathlib import Path

from src.data.validate_dataset import validate_label_file


def test_valid_yolo_label(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    counts, valid, invalid, issues = validate_label_file(path, {0})
    assert valid == 1
    assert invalid == 0
    assert not issues
    assert counts[0] == 1


def test_invalid_yolo_label(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_text("0 1.5 0.5 0.2 0.2\n", encoding="utf-8")
    _, valid, invalid, issues = validate_label_file(path, {0})
    assert valid == 0
    assert invalid == 1
    assert issues
