"""Read one simulator representation; keep offline associations separate."""

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Optional

from .imu_schema import IMUSample, IMUValidationError, REQUIRED_FIELDS, numeric_value


@dataclass(frozen=True)
class OfflineSample:
    """Dataset envelope, not part of the standard sensor interface."""

    sample: IMUSample
    frame_id: Optional[str] = None
    sequence_id: Optional[str] = None
    frame_timestamp_ms: Optional[float] = None


def _identifier(value: object, name: str) -> Optional[str]:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: expected a nonblank string")
    return value


def _frame_time(sample: IMUSample, offset: object, anchor: object = None) -> Optional[float]:
    anchor = numeric_value("frame_timestamp_ms", anchor, optional=True)
    offset = numeric_value("offset_ms", offset, optional=True)
    if offset is not None:
        derived = numeric_value("frame_timestamp_ms", sample.timestamp_ms - offset)
        if anchor is not None and not math.isclose(anchor, derived, rel_tol=0, abs_tol=1e-6):
            raise ValueError("offset_ms does not agree with the frame timestamp")
        if anchor is None:
            anchor = derived
    return anchor


def _read_csv(path: Path) -> list[OfflineSample]:
    records = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate CSV column names")
        missing = [field for field in REQUIRED_FIELDS if field not in fields]
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        for row in reader:
            try:
                if None in row:
                    raise ValueError("CSV row contains more values than columns")
                sample = IMUSample.from_mapping(row)
                records.append(OfflineSample(
                    sample=sample,
                    frame_id=_identifier(row.get("frame_id"), "frame_id"),
                    sequence_id=_identifier(row.get("sequence_id"), "sequence_id"),
                    frame_timestamp_ms=_frame_time(sample, row.get("offset_ms")),
                ))
            except ValueError as exc:
                raise ValueError(f"{path}: line {reader.line_num}: {exc}") from exc
    return records


def _vector(sample: dict, key: str, size: int, *, optional: bool = False) -> list:
    value = sample.get(key)
    if value is None and optional:
        return [None] * size
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{key}: expected an array of {size} values")
    return value


def _read_json(path: Path) -> list[OfflineSample]:
    try:
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict):
            raise ValueError("expected a per-frame JSON object")
        # `sample` is a repeated endpoint when `samples` exists. Never append it.
        samples = document.get("samples") if "samples" in document else [document.get("sample")]
        if not isinstance(samples, list) or not samples:
            raise ValueError("expected a nonempty samples array or a sample object")
        frame_id = _identifier(document.get("frameId"), "frameId")
        sequence_id = _identifier(document.get("sequenceId"), "sequenceId")
        records = []
        for index, values in enumerate(samples):
            try:
                if not isinstance(values, dict):
                    raise ValueError("expected a sample object")
                accel = _vector(values, "accelerometerMps2", 3)
                gyro = _vector(values, "gyroscopeRadS", 3)
                euler = _vector(values, "orientationEulerDeg", 3, optional=True)
                mapped = dict(zip(REQUIRED_FIELDS, [values.get("timestampMs"), *accel, *gyro]))
                mapped.update(zip(("pitch_deg", "yaw_deg", "roll_deg"), euler))
                sample = IMUSample.from_mapping(mapped)
                records.append(OfflineSample(
                    sample, frame_id, sequence_id,
                    _frame_time(sample, values.get("offsetMs"), document.get("timestampMs")),
                ))
            except ValueError as exc:
                raise ValueError(f"sample index {index}: {exc}") from exc
        return records
    except (ValueError, IMUValidationError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_simulator(path: Path | str) -> list[OfflineSample]:
    """Read a CSV, a per-frame JSON, or a directory of per-frame JSON files.

    Choose one representation; a JSON directory must contain only per-frame
    JSON files (for this dataset, data/raw/imu, not data/raw).
    The declared sample rate/window length is never used to construct times.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            raise ValueError(f"{path}: no per-frame JSON files found")
        return [record for file in files for record in _read_json(file)]
    if path.suffix.lower() == ".csv":
        return _read_csv(path)
    if path.suffix.lower() == ".json":
        return _read_json(path)
    raise ValueError(f"{path}: expected CSV, per-frame JSON, or a JSON directory")
