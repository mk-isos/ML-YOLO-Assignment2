from pathlib import Path

import numpy as np
import pytest

from src.evaluate import box_iou, match_detections
from src.prepare_dataset import CLASS_REMAP, TARGET_NAMES, parse_source_label


def test_class_mapping_is_exactly_three_dense_ids() -> None:
    assert CLASS_REMAP == {0: 0, 3: 1, 5: 2}
    assert TARGET_NAMES == {0: "car", 1: "pedestrian", 2: "cyclist"}


def test_parse_source_label_remaps_and_filters(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "1 0.4 0.4 0.1 0.1\n"
        "3 0.3 0.3 0.2 0.4\n"
        "5 0.7 0.7 0.1 0.2\n"
    )
    rows = parse_source_label(label)
    assert [row[0] for row in rows] == [0, 1, 2]


def test_parse_source_label_rejects_out_of_range(tmp_path: Path) -> None:
    label = tmp_path / "bad.txt"
    label.write_text("0 1.2 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="outside"):
        parse_source_label(label)


def test_iou_and_class_aware_matching() -> None:
    ground_truth = [
        (0, np.array([0, 0, 10, 10], dtype=np.float32)),
        (1, np.array([20, 20, 30, 30], dtype=np.float32)),
    ]
    predictions = [
        (0, 0.9, np.array([0, 0, 10, 10], dtype=np.float32)),
        (2, 0.8, np.array([20, 20, 30, 30], dtype=np.float32)),
    ]
    assert box_iou(ground_truth[0][1], predictions[0][2]) == 1.0
    result = match_detections(ground_truth, predictions)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
