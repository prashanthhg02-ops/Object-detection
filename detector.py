"""YOLO object detection implementation."""

from pathlib import Path
from typing import Iterable

import cv2
from ultralytics import YOLO


class ObjectDetector:
    """Run YOLO inference on images, videos, or webcam frames."""

    def __init__(self, model: str = "yolo11n.pt", confidence: float = 0.25) -> None:
        if not 0 < confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        self.model = YOLO(model)
        self.confidence = confidence

    def detect(self, source: str | int, output: str | None = None) -> None:
        """Detect objects in a file or camera source and display/save annotated frames."""
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open source: {source}")

        writer = None
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                annotated = self.annotate(frame)
                if writer is None and output:
                    output_path = Path(output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    height, width = annotated.shape[:2]
                    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                if writer:
                    writer.write(annotated)

                cv2.imshow("Object Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

    def annotate(self, frame):
        """Return a frame with model predictions drawn on it."""
        results = self.model.predict(source=frame, conf=self.confidence, verbose=False)
        return results[0].plot()

    def labels(self, frame) -> Iterable[str]:
        """Return detected class names for a single frame."""
        results = self.model.predict(source=frame, conf=self.confidence, verbose=False)
        names = results[0].names
        return [names[int(class_id)] for class_id in results[0].boxes.cls.tolist()]
