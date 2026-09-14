"""Streaming, sequential video inference and debug output. No sensor fusion."""
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .contract import binary_mask, select_instance, timestamp_ms, vision_record


OUTPUT_NAMES = ("vision_output.jsonl", "vision_overlay.mp4", "summary.json")


def resolve_device(requested: str, cuda_available: bool, cuda_count: int,
                   mps_available: bool = False) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not mps_available:
            raise ValueError("--device mps requested, but torch.backends.mps.is_available() is False; "
                             "use --device cpu explicitly to run on CPU")
        return "mps"
    if requested == "auto":
        return "0" if cuda_available and cuda_count > 0 else "cpu"
    if not requested.isdecimal():
        raise ValueError("--device must be auto, cpu, mps, or a CUDA index such as 0")
    if not cuda_available or cuda_count == 0:
        return "cpu"
    if int(requested) >= cuda_count:
        raise ValueError(f"CUDA device {requested} does not exist ({cuda_count} available)")
    return str(int(requested))


def load_model(weights: Path, requested_device: str):
    # Imports are lazy so helpers and mock-video tests do not need torch/weights.
    cache = Path(__file__).resolve().parents[1] / "outputs" / ".cache"
    for variable, name in (("YOLO_CONFIG_DIR", "ultralytics"), ("MPLCONFIGDIR", "matplotlib")):
        os.environ.setdefault(variable, str(cache / name))
    import torch
    from ultralytics import YOLO

    device = resolve_device(requested_device, torch.cuda.is_available(), torch.cuda.device_count(),
                            mps_available=requested_device == "mps" and torch.backends.mps.is_available())
    if requested_device not in ("cpu", "auto") and device == "cpu":
        print(f"CUDA is unavailable; --device {requested_device} falls back to CPU.", file=sys.stderr)
    model = YOLO(str(weights), task="segment")
    if model.task != "segment" or model.names != {0: "towline"}:
        raise ValueError(f"Expected single-class segmentation weights (0 = towline), got {model.names}")
    return model, device


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def prediction(result, shape: tuple[int, int], threshold: float):
    """Use Ultralytics' native original-coordinate mask, preserving gaps and holes."""
    if tuple(result.orig_shape) != shape:
        raise ValueError("Prediction orig_shape does not match the video frame")
    if result.boxes is None:
        if result.masks is not None:
            raise ValueError("Segmentation masks are present without boxes")
        return None, None
    scores = _numpy(result.boxes.conf)
    classes = _numpy(result.boxes.cls)
    index = select_instance(scores, classes, threshold)
    if result.masks is None:
        if index is not None:
            raise ValueError("Qualifying towline box has no segmentation mask")
        return None, None
    masks = result.masks.data
    if len(masks.shape) != 3 or masks.shape[0] != len(scores):
        raise ValueError("Segmentation mask/box counts do not match")
    if index is None:
        return None, None
    mask = binary_mask(_numpy(masks[index]), *shape)
    if not mask.any():
        raise ValueError("Selected towline instance has an empty mask")
    return float(scores[index]), mask


def overlay_frame(frame: np.ndarray, record: dict, mask: np.ndarray | None) -> np.ndarray:
    overlay = frame.copy()
    if mask is not None:
        foreground = mask > 0
        overlay[foreground] = (
            overlay[foreground] * 0.55 + np.array([0, 255, 70]) * 0.45
        ).astype(np.uint8)
    vision = record["vision"]
    lines = [
        f"t = {record['timestamp_ms']:.1f} ms",
        "towline: detected" if vision["towline_detected"] else "towline: NOT DETECTED",
        f"conf: {vision['confidence']:.2f}" if vision["confidence"] is not None else "conf: N/A",
    ]
    scale = min(0.7, max(0.25, frame.shape[1] / 1000))
    line_height = max(14, int(34 * scale / 0.7))
    color = (70, 255, 70) if vision["towline_detected"] else (80, 180, 255)
    for line_number, text in enumerate(lines, 1):
        origin = (8, line_height * line_number)
        cv2.putText(overlay, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return overlay


def run_video(video: Path, weights: Path, output_dir: Path, conf: float = 0.25,
              imgsz: int = 960, device: str = "auto", *, model_loader=None) -> dict:
    """model_loader is an injection seam for smoke tests, never a CLI mock mode."""
    started = time.perf_counter()
    video, weights, output_dir = (Path(p).expanduser().resolve() for p in (video, weights, output_dir))
    if not video.is_file():
        raise ValueError(f"Input video does not exist: {video}")
    if not weights.is_file() or weights.suffix.lower() != ".pt":
        raise ValueError(f"Local .pt weights do not exist: {weights}")
    if not math.isfinite(conf) or not 0 <= conf <= 1 or imgsz <= 0:
        raise ValueError("Require --conf in [0, 1] and positive --imgsz")
    # Check spelling only; load_model checks actual MPS availability.
    resolve_device(device, False, 0, mps_available=True)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Use a new or empty output directory: {output_dir}")

    capture = cv2.VideoCapture(str(video))
    writer = None
    processed = detected = timing_fallbacks = 0
    confidence_sum = inference_ms_sum = predict_ms_sum = 0.0
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open input video: {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        timestamp_ms(0, fps)  # Reject unknown FPS instead of inventing a timeline.
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("Video reports invalid source dimensions")
        reported_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        expected_count = int(reported_count) if math.isfinite(reported_count) and reported_count > 0 else None
        model, actual_device = (model_loader or load_model)(weights, device)
        output_dir.mkdir(parents=True, exist_ok=True)
        # mp4v needs even dimensions. Pad only the debug video for odd-size inputs.
        overlay_width, overlay_height = width + width % 2, height + height % 2
        writer = cv2.VideoWriter(str(output_dir / OUTPUT_NAMES[1]), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (overlay_width, overlay_height))
        if not writer.isOpened():
            raise RuntimeError("Cannot open MP4 writer (mp4v codec unavailable)")
        with (output_dir / OUTPUT_NAMES[0]).open("x", encoding="utf-8") as jsonl:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = timestamp_ms(processed, fps)  # Immediately after capture, before inference.
                if frame is None or frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                    raise ValueError(f"Invalid or changing frame dimensions at frame {processed}")
                inference_started = time.perf_counter()
                results = model.predict(source=frame, conf=conf, imgsz=imgsz, device=actual_device,
                                        classes=[0], retina_masks=True, save=False, verbose=False)
                predict_ms = (time.perf_counter() - inference_started) * 1000.0
                if len(results) != 1:
                    raise ValueError("Expected one inference result per video frame")
                result = results[0]
                confidence, mask = prediction(result, (height, width), conf)
                record = vision_record(timestamp, width, height, confidence, mask)
                jsonl.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                overlay = overlay_frame(frame, record, mask)
                if overlay_width != width or overlay_height != height:
                    overlay = cv2.copyMakeBorder(overlay, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
                writer.write(overlay)
                inference_ms = (getattr(result, "speed", None) or {}).get("inference")
                if not isinstance(inference_ms, (int, float)) or not math.isfinite(inference_ms) or inference_ms < 0:
                    inference_ms = predict_ms
                    timing_fallbacks += 1
                inference_ms_sum += inference_ms
                predict_ms_sum += predict_ms
                if confidence is not None:
                    detected += 1
                    confidence_sum += confidence
                processed += 1
        if processed == 0:
            raise ValueError("Video contains no decodable frames")
        if expected_count is not None and processed != expected_count:
            raise ValueError(f"Decoded {processed} frames, container reports {expected_count}; "
                             "possible truncated/corrupt video or inaccurate container metadata")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - started
    summary = {
        "input_video": str(video), "weights": str(weights),
        "source_width": width, "source_height": height, "source_fps": fps,
        "total_frames_processed": processed, "video_duration_sec": processed / fps,
        "detected_frames": detected, "detected_ratio": detected / processed,
        "mean_confidence_detected": confidence_sum / detected if detected else None,
        "total_elapsed_sec": elapsed, "mean_inference_ms": inference_ms_sum / processed,
        "effective_processing_fps": processed / elapsed,
        "device_requested": device, "device_used": actual_device, "conf": conf, "imgsz": imgsz,
        "mean_predict_call_ms": predict_ms_sum / processed,
        "inference_timing_source": "ultralytics.speed.inference" if timing_fallbacks == 0 else "mixed_or_predict_wall_time",
        "inference_timing_fallback_frames": timing_fallbacks,
        "overlay_width": overlay_width, "overlay_height": overlay_height,
    }
    (output_dir / OUTPUT_NAMES[2]).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary
