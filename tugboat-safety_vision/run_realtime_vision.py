#!/usr/bin/env python3
"""Stream a local MP4 through YOLO-Seg into Vision JSONL and an overlay MP4."""
import argparse
import json
from pathlib import Path
import sys

from vision.pipeline import run_video


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="auto", help="auto: CUDA 0 if available, else CPU; or cpu / CUDA index")
    args = parser.parse_args(argv)
    try:
        summary = run_video(args.video, args.weights, args.output_dir, args.conf, args.imgsz, args.device)
    except Exception as error:
        print(f"Vision failed: {error}\nAny partial outputs from this attempt are incomplete; use a new output directory to retry.", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
