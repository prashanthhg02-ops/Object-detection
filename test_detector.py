from unittest.mock import patch

import pytest

from object_detection.detector import ObjectDetector


def test_rejects_invalid_confidence() -> None:
    with patch("object_detection.detector.YOLO"):
        with pytest.raises(ValueError, match="between 0 and 1"):
            ObjectDetector(confidence=0)


def test_accepts_valid_confidence() -> None:
    with patch("object_detection.detector.YOLO") as yolo:
        detector = ObjectDetector(confidence=0.5)

    assert detector.confidence == 0.5
    yolo.assert_called_once_with("yolo11n.pt")
