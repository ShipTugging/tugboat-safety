"""Synchronize upstream Vision features with standardized IMU CSV measurements."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

from src.imu_schema import REQUIRED_FIELDS, SENSOR_FIELDS
from src.synchronization import (
    DEFAULT_MAX_TIME_GAP_MS, METADATA_FIELDS, VISION_REQUIRED_FIELDS,
    IMURecord, SynchronizedRecord, VisionSample, synchronize,
)


ROOT = Path(__file__).resolve().parent
CSV_FIELDS = (
    "vision_timestamp_ms", "imu_timestamp_ms", "sync_time_delta_ms", "sync_matched",
    "sync_reason", "sync_nearest_time_delta_ms", "towline_detected", "vision_confidence",
    "towline_angle_deg", "sag_ratio", "curvature", *SENSOR_FIELDS[1:],
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


def run(
    vision_path: Path | str,
    imu_path: Path | str,
    output_path: Path | str,
    diagnostics_path: Path | str,
    *,
    max_time_gap_ms: float = DEFAULT_MAX_TIME_GAP_MS,
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
        raise ValueError("CSV and diagnostics must have distinct, non-nested paths")
    # Resolve aliases that already exist as hard links as well as symbolic links.
    for destination in (output_path, diagnostics_path):
        if destination.exists():
            if not destination.is_file():
                raise ValueError(f"{destination}: output must be a file")
            if any(destination.samefile(source) for source in (vision_path, imu_path) if source.exists()):
                raise ValueError("outputs must not overwrite inputs")
    if output_path.exists() and diagnostics_path.exists() and output_path.samefile(diagnostics_path):
        raise ValueError("CSV and diagnostics must have distinct paths")

    vision = load_csv(vision_path, kind="vision")
    imu = load_csv(imu_path, kind="imu")
    records, diagnostics = synchronize(vision, imu, max_time_gap_ms=max_time_gap_ms)
    report_text = json.dumps(diagnostics, indent=2, allow_nan=False) + "\n"
    rows = [flatten(record) for record in records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    diagnostics_path.write_text(report_text, encoding="utf-8")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision", type=Path, required=True, help="upstream Vision feature CSV")
    parser.add_argument("--imu", type=Path, default=ROOT / "data/processed/imu_normalized.csv",
                        help="standardized IMU CSV, not raw simulator JSON/windows")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/vision_imu_sync.csv")
    parser.add_argument("--diagnostics", type=Path, default=ROOT / "outputs/sync_diagnostics.json")
    parser.add_argument("--max-time-gap-ms", type=float, default=DEFAULT_MAX_TIME_GAP_MS)
    args = parser.parse_args()
    try:
        report = run(args.vision, args.imu, args.output, args.diagnostics,
                     max_time_gap_ms=args.max_time_gap_ms)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Synchronization error: {exc}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Synchronized CSV: {args.output}")
    print(f"Diagnostics: {args.diagnostics}")


if __name__ == "__main__":
    main()
