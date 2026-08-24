"""Command-line entry point for the object detector."""

import argparse

from object_detection import ObjectDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect objects in an image, video, or webcam stream.")
    parser.add_argument("source", help="Path to an image/video, or camera index such as 0")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model name or .pt path")
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum confidence from 0 to 1")
    parser.add_argument("--output", help="Optional path for an annotated video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source: str | int = int(args.source) if args.source.isdigit() else args.source
    ObjectDetector(args.model, args.confidence).detect(source, args.output)


if __name__ == "__main__":
    main()
