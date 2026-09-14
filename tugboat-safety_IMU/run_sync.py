"""Synchronize Vision JSONL/CSV with standardized IMU CSV measurements."""

import argparse
import csv
from collections.abc import Mapping
from dataclasses import asdict
import json
from pathlib import Path

from src.imu_schema import REQUIRED_FIELDS, SENSOR_FIELDS
from src.synchronization import (
    DEFAULT_MAX_TIME_GAP_MS, METADATA_FIELDS, VISION_REQUIRED_FIELDS, VISION_MASK_FIELDS,
    Association, IMURecord, SynchronizedRecord, VisionSample, synchronize,
)


ROOT = Path(__file__).resolve().parent
CSV_FIELDS = (
    "vision_timestamp_ms", "imu_timestamp_ms", "sync_time_delta_ms", "sync_matched",
    "sync_reason", "sync_nearest_time_delta_ms", "towline_detected", "vision_confidence",
    *VISION_MASK_FIELDS, *SENSOR_FIELDS[1:],
    *(f"vision_{name}" for name in METADATA_FIELDS),
    *(f"imu_{name}" for name in METADATA_FIELDS),
)


def load_csv(path: Path | str, *, kind: str) -> list:
    """Read flat Vision or normalized IMU rows; preserve association metadata.

    This is deliberately not the raw simulator adapter: normalized rows already
    carry frame_timestamp_ms, and future inputs may also carry session_id.
    """
    if kind not in ("vision", "imu"):
        raise ValueError("kind must be 'vision' or 'imu'")
    required = VISION_REQUIRED_FIELDS if kind == "vision" else REQUIRED_FIELDS
    factory = VisionSample.from_mapping if kind == "vision" else IMURecord.from_mapping
    records = []
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate CSV column names")
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        for row in reader:
            try:
                if None in row:
                    raise ValueError("CSV row contains more values than columns")
                records.append(factory(row))
            except ValueError as exc:
                raise ValueError(f"{path}: line {reader.line_num}: {exc}") from exc
    return records


def flatten(record: SynchronizedRecord) -> dict:
    """Unmatched IMU cells and missing optional values serialize as blank cells."""
    features = record.vision.features()
    features["vision_confidence"] = features.pop("confidence")
    imu = record.imu.sample.to_dict() if record.imu else dict.fromkeys(SENSOR_FIELDS)
    return {
        "vision_timestamp_ms": record.vision.timestamp_ms,
        "imu_timestamp_ms": imu.pop("timestamp_ms"),
        "sync_time_delta_ms": record.time_delta_ms,
        "sync_matched": record.matched,
        "sync_reason": record.reason,
        "sync_nearest_time_delta_ms": record.nearest_time_delta_ms,
        **features,
        **imu,
        **{f"vision_{key}": value for key, value in asdict(record.vision.metadata).items()},
        **{f"imu_{key}": value for key, value in (
            asdict(record.imu.metadata) if record.imu else dict.fromkeys(METADATA_FIELDS)
        ).items()},
    }


def _merge_metadata(values: dict, additions: Mapping) -> dict:
    """Merge only explicit association fields; known conflicts are errors."""
    existing = Association.from_mapping(values)
    supplied = Association.from_mapping(additions)
    merged = asdict(existing)
    for name, value in asdict(supplied).items():
        if value is not None:
            if merged[name] is not None and merged[name] != value:
                raise ValueError(f"{name}: supplied metadata conflicts with existing value")
            merged[name] = value
    return {**values, **merged}


def _json_object(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"duplicate JSON key: {key}")
        values[key] = value
    return values


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def load_vision_jsonl(path: Path | str, *, session_id: str | None = None,
                      sequence_id: str | None = None) -> list[VisionSample]:
    """Adapt the nested Vision V1 contract. Mask strings are never transformed."""
    supplied = asdict(Association(session_id=session_id, sequence_id=sequence_id))
    path = Path(path)
    samples = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                document = json.loads(line, object_pairs_hook=_json_object,
                                      parse_constant=_invalid_constant)
                if not isinstance(document, Mapping) or not isinstance(document.get("vision"), Mapping):
                    raise ValueError("expected an object with a nested vision object")
                payload = document["vision"]
                missing = [name for name in ("timestamp_ms",) if name not in document]
                missing += [name for name in ("towline_detected", "confidence", *VISION_MASK_FIELDS)
                            if name not in payload]
                if missing:
                    raise ValueError(f"missing required fields: {', '.join(missing)}")
                if not isinstance(document["timestamp_ms"], (int, float)) or isinstance(document["timestamp_ms"], bool):
                    raise ValueError("timestamp_ms: expected a JSON number")
                if not isinstance(payload["towline_detected"], bool):
                    raise ValueError("towline_detected: expected a JSON boolean")
                confidence = payload["confidence"]
                if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))):
                    raise ValueError("confidence: expected a JSON number or null")
                for name in ("mask_width", "mask_height"):
                    if not isinstance(payload[name], int) or isinstance(payload[name], bool) or payload[name] <= 0:
                        raise ValueError(f"{name}: expected a positive JSON integer")
                if payload["towline_detected"] and payload["mask_base64"] is None:
                    raise ValueError("mask_base64: required for a detected towline")
                values = {"timestamp_ms": document["timestamp_ms"],
                          **{name: payload[name] for name in ("towline_detected", "confidence", *VISION_MASK_FIELDS)}}
                values = _merge_metadata(values, document)
                metadata = document.get("metadata", {})
                if not isinstance(metadata, Mapping):
                    raise ValueError("metadata: expected an object")
                metadata = metadata.get("vision", metadata)
                if not isinstance(metadata, Mapping):
                    raise ValueError("metadata.vision: expected an object")
                values = _merge_metadata(values, metadata)
                values = _merge_metadata(values, supplied)
                samples.append(VisionSample.from_mapping(values))
            except ValueError as exc:
                raise ValueError(f"{path}: line {line_number}: {exc}") from exc
    return samples


def _format(path: Path, requested: str) -> str:
    if requested not in ("auto", "csv", "jsonl"):
        raise ValueError("format must be auto, csv, or jsonl")
    if requested != "auto":
        return requested
    extension = path.suffix.lower().lstrip(".")
    if extension not in ("csv", "jsonl"):
        raise ValueError(f"{path}: use .csv/.jsonl or specify the format explicitly")
    return extension


def run(
    vision_path: Path | str,
    imu_path: Path | str,
    output_path: Path | str,
    diagnostics_path: Path | str,
    *,
    max_time_gap_ms: float = DEFAULT_MAX_TIME_GAP_MS,
    vision_format: str = "auto",
    output_format: str = "auto",
    session_id: str | None = None,
    sequence_id: str | None = None,
) -> dict:
    """Validate before writing; inputs/raw data and colliding outputs are protected."""
    vision_path, imu_path, output_path, diagnostics_path = map(
        Path, (vision_path, imu_path, output_path, diagnostics_path),
    )
    inputs = [vision_path.resolve(), imu_path.resolve()]
    destinations = [output_path.resolve(), diagnostics_path.resolve()]
    raw_root = (ROOT / "data/raw").resolve()
    for destination in destinations:
        if destination == raw_root or raw_root in destination.parents:
            raise ValueError("outputs must not overwrite anything under data/raw")
        if any(destination == source or destination in source.parents for source in inputs):
            raise ValueError("outputs must not overwrite inputs or their parent directories")
    if (destinations[0] == destinations[1] or destinations[0] in destinations[1].parents
            or destinations[1] in destinations[0].parents):
        raise ValueError("output and diagnostics must have distinct, non-nested paths")
    # Resolve aliases that already exist as hard links as well as symbolic links.
    for destination in (output_path, diagnostics_path):
        if destination.exists():
            if not destination.is_file():
                raise ValueError(f"{destination}: output must be a file")
            if any(destination.samefile(source) for source in (vision_path, imu_path) if source.exists()):
                raise ValueError("outputs must not overwrite inputs")
    if output_path.exists() and diagnostics_path.exists() and output_path.samefile(diagnostics_path):
        raise ValueError("output and diagnostics must have distinct paths")

    selected_input = _format(vision_path, vision_format)
    selected_output = _format(output_path, output_format)
    if selected_input == "jsonl":
        vision = load_vision_jsonl(vision_path, session_id=session_id, sequence_id=sequence_id)
    else:
        supplied = asdict(Association(session_id=session_id, sequence_id=sequence_id))
        vision = load_csv(vision_path, kind="vision")
        vision = [VisionSample.from_mapping(_merge_metadata(
            {"timestamp_ms": sample.timestamp_ms, **sample.features(), **asdict(sample.metadata)}, supplied,
        )) for sample in vision]
    imu = load_csv(imu_path, kind="imu")
    records, diagnostics = synchronize(vision, imu, max_time_gap_ms=max_time_gap_ms)
    report_text = json.dumps(diagnostics, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        if selected_output == "jsonl":
            for record in records:
                handle.write(json.dumps(record.to_dict(), allow_nan=False, separators=(",", ":")) + "\n")
        else:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(flatten(record) for record in records)
    diagnostics_path.write_text(report_text, encoding="utf-8")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision", type=Path, required=True, help="Vision V1 JSONL or legacy flat CSV")
    parser.add_argument("--imu", type=Path, default=ROOT / "data/processed/imu_normalized.csv",
                        help="standardized IMU CSV, not raw simulator JSON/windows")
    parser.add_argument("--output", type=Path, help="defaults to data/processed/vision_imu_sync with input format")
    parser.add_argument("--vision-format", choices=("auto", "csv", "jsonl"), default="auto")
    parser.add_argument("--output-format", choices=("auto", "csv", "jsonl"), default="auto")
    parser.add_argument("--session-id", help="explicit Vision recording boundary; conflicting metadata is an error")
    parser.add_argument("--sequence-id", help="explicit Vision sequence boundary; conflicting metadata is an error")
    parser.add_argument("--diagnostics", type=Path, default=ROOT / "outputs/sync_diagnostics.json")
    parser.add_argument("--max-time-gap-ms", type=float, default=DEFAULT_MAX_TIME_GAP_MS)
    args = parser.parse_args()
    try:
        if args.output is None:
            output_format = (_format(args.vision, args.vision_format)
                             if args.output_format == "auto" else args.output_format)
            args.output = ROOT / f"data/processed/vision_imu_sync.{output_format}"
        report = run(args.vision, args.imu, args.output, args.diagnostics,
                     max_time_gap_ms=args.max_time_gap_ms, vision_format=args.vision_format,
                     output_format=args.output_format, session_id=args.session_id,
                     sequence_id=args.sequence_id)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Synchronization error: {exc}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Synchronized output: {args.output}")
    print(f"Diagnostics: {args.diagnostics}")


if __name__ == "__main__":
    main()
