from __future__ import annotations

import numpy as np

from src.analyze_diagnostics import match_detections, size_bin


def test_size_bin_boundaries() -> None:
    assert size_bin(32**2 - 1) == "small"
    assert size_bin(32**2) == "medium"
    assert size_bin(96**2 - 1) == "medium"
    assert size_bin(96**2) == "large"


def test_matching_is_class_consistent_and_confidence_ordered() -> None:
    ground_truth = [
        {"class_id": 0, "box": np.array([0, 0, 10, 10], dtype=np.float32)},
        {"class_id": 1, "box": np.array([20, 20, 30, 30], dtype=np.float32)},
    ]
    predictions = [
        {"class_id": 0, "confidence": 0.4, "box": np.array([0, 0, 10, 10], dtype=np.float32)},
        {"class_id": 1, "confidence": 0.9, "box": np.array([0, 0, 10, 10], dtype=np.float32)},
    ]
    result = match_detections(ground_truth, predictions)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
