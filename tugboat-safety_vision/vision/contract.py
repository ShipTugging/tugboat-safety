"""Pure helpers for the Vision V1 contract; no model or video required."""
import base64
import math

import cv2
import numpy as np


def timestamp_ms(frame_index: int, fps: float) -> float:
    if frame_index < 0 or not isinstance(frame_index, int):
        raise ValueError("frame_index must be a nonnegative integer")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Source FPS must be finite and positive")
    return frame_index / fps * 1000.0


def select_instance(confidences, classes, threshold: float) -> int | None:
    """Select class 0 above/equal to threshold; first instance wins a tie."""
    scores = np.asarray(confidences, dtype=float)
    labels = np.asarray(classes)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("Confidence/class arrays must be matching 1D arrays")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be in [0, 1]")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Instance confidences must be finite and in [0, 1]")
    eligible = np.flatnonzero((labels == 0) & (scores >= threshold))
    return int(eligible[np.argmax(scores[eligible])]) if eligible.size else None


def binary_mask(original_mask, height: int, width: int) -> np.ndarray:
    """Threshold an already restored mask. Never resize an internal/letterbox mask."""
    array = np.asarray(original_mask)
    if height <= 0 or width <= 0 or array.shape != (height, width):
        raise ValueError(
            f"Expected original-frame mask {(height, width)}, got {array.shape}; "
            "use retina_masks=True, never resize a letterbox mask directly"
        )
    if not np.isfinite(array).all():
        raise ValueError("Mask contains nonfinite values")
    return (array > 0.5).astype(np.uint8) * 255


def encode_mask_png(mask: np.ndarray) -> str:
    if mask.ndim != 2 or mask.dtype != np.uint8 or mask.size == 0:
        raise ValueError("Mask must be a nonempty 1-channel uint8 array")
    if not np.isin(mask, (0, 255)).all():
        raise ValueError("Mask values must be binary {0, 255}")
    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError("Failed to encode mask as PNG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def vision_record(timestamp: float, width: int, height: int,
                  confidence: float | None = None, mask: np.ndarray | None = None) -> dict:
    if not math.isfinite(timestamp) or timestamp < 0 or width <= 0 or height <= 0:
        raise ValueError("Invalid timestamp or frame dimensions")
    if (confidence is None) != (mask is None):
        raise ValueError("Confidence and mask must both be present or both be absent")
    encoded = None
    if mask is not None:
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be finite and in [0, 1]")
        if mask.shape != (height, width):
            raise ValueError("Mask dimensions must match the original frame")
        encoded = encode_mask_png(mask)
        if not mask.any():
            raise ValueError("Detected instance has an empty mask")
    return {
        "timestamp_ms": float(timestamp),
        "vision": {
            "towline_detected": mask is not None,
            "confidence": float(confidence) if confidence is not None else None,
            "mask_base64": encoded,
            "mask_width": width,
            "mask_height": height,
        },
    }
