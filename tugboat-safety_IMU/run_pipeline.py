"""Small offline runner. Run `python3 run_pipeline.py --help` for inputs."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from src.imu_schema import SENSOR_FIELDS, ORIENTATION_FIELDS
from src.preprocessing import preprocess
from src.simulator_adapter import load_simulator


ROOT = Path(__file__).resolve().parent
METADATA_FIELDS = ("frame_id", "sequence_id", "frame_timestamp_ms")


def run(input_path: Path, output_path: Path, diagnostics_path: Path) -> dict:
    """Validate the complete input before writing any output."""
    input_path, output_path, diagnostics_path = map(Path, (input_path, output_path, diagnostics_path))
    raw_root = (ROOT / "data/raw").resolve()
    input_resolved = input_path.resolve()
    destinations = [output_path.resolve(), diagnostics_path.resolve()]
    for destination in destinations:
        if destination == raw_root or raw_root in destination.parents:
            raise ValueError("outputs must not overwrite anything under data/raw")
        if destination == input_resolved or (input_resolved.is_dir() and input_resolved in destination.parents):
            raise ValueError("outputs must not overwrite the input or write into its directory")
    if destinations[0] == destinations[1]:
        raise ValueError("normalized CSV and diagnostics must have different paths")

    records = load_simulator(input_path)
    groups = defaultdict(list)
    for record in records:
        # Frame IDs identify independent scenes in this simulator dataset.
        # Metadata-free input is one sequence, not one sequence per sample.
        groups[(record.sequence_id, record.frame_id)].append(record)

    output_rows, reports = [], []
    for (sequence_id, frame_id), group in groups.items():
        anchors = [r.frame_timestamp_ms for r in group if r.frame_timestamp_ms is not None]
        if anchors and any(not math.isclose(t, anchors[0], rel_tol=0, abs_tol=1e-6) for t in anchors):
            raise ValueError(f"sequence_id={sequence_id!r}, frame_id={frame_id!r}: inconsistent frame timestamps")
        ordered, report = preprocess(record.sample for record in group)
        report.update(sequence_id=sequence_id, frame_id=frame_id,
                      frame_timestamp_ms=anchors[0] if anchors else None)
        reports.append(report)
        for sample, record in zip(ordered, sorted(group, key=lambda r: r.sample.timestamp_ms)):
            output_rows.append({**sample.to_dict(), **{
                field: getattr(record, field) for field in METADATA_FIELDS
            }})

    times = [record.sample.timestamp_ms for record in records]
    rates = [r["estimated_sampling_rate_hz"] for r in reports if r["estimated_sampling_rate_hz"] is not None]
    interval_counts = Counter(interval for r in reports for interval in r["sample_intervals_ms"])
    peak_report = max((r for r in reports if r["peak_specific_force"] is not None),
                      key=lambda r: r["peak_specific_force"]["magnitude_mps2"], default=None)
    summary = {
        "input": str(input_path),
        "clock": "simulation milliseconds; zero is simulator time origin, not Unix time",
        "axis_convention": "+X lateral, +Y up, +Z bow; right-handed tug-local body frame",
        "accelerometer": "specific force in m/s^2; upright rest [0, +9.80665, 0]",
        "grouping": "independent (sequence_id, frame_id) groups; absent IDs form one sequence",
        "sensor_fields": list(SENSOR_FIELDS),
        "offline_metadata_fields": list(METADATA_FIELDS),
        "frame_id_count": len({r.frame_id for r in records if r.frame_id is not None}),
        "sequence_id_count": len({r.sequence_id for r in records if r.sequence_id is not None}),
        "group_count": len(groups),
        "total_samples": len(records),
        "samples_per_group_histogram": dict(sorted(Counter(r["sample_count"] for r in reports).items())),
        "timestamp_range_ms": [min(times), max(times)] if times else None,
        "timestamp_range_note": "inventory range only; independent scenes do not form a continuous recording",
        "within_group_interval_counts": [
            {"interval_ms": interval, "count": count} for interval, count in sorted(interval_counts.items())
        ],
        "estimated_sampling_rate_range_hz": [min(rates), max(rates)] if rates else None,
        "duplicate_timestamps_within_groups": sum(r["duplicate_timestamp_count"] for r in reports),
        "input_out_of_order_pairs_within_groups": sum(r["input_out_of_order_pairs"] for r in reports),
        "missing_required_values": 0,
        "invalid_numeric_values": 0,
        "missing_optional_orientation_values": {
            field: sum(r["missing_optional_orientation_values"][field] for r in reports)
            for field in ORIENTATION_FIELDS
        },
        "available_orientation_fields": [
            field for field in ORIENTATION_FIELDS if any(field in r["available_orientation_fields"] for r in reports)
        ],
        "peak_specific_force": {
            **peak_report["peak_specific_force"], "frame_id": peak_report["frame_id"],
            "sequence_id": peak_report["sequence_id"],
        } if peak_report else None,
        "notes": [
            "Exactly one input representation is loaded; CSV endpoints/JSON sample copies are not appended.",
            "Invalid required data or invalid numeric orientation aborts the run with source context; no rows are silently dropped.",
            "Duplicate timestamps are reported and retained; zero intervals are excluded from the rate estimate.",
            "High accelerations are retained unchanged. Peak values are diagnostics, not risk features or validity thresholds.",
            "Quaternion, simulator truth, and asset paths are outside the standardized V1 output.",
            "Ideal model-derived data; real sensor noise, bias, latency, and clock synchronization are not modeled.",
        ],
        "groups": reports,
    }
    # Serialize first so non-JSON diagnostics cannot leave a new CSV behind.
    diagnostic_text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*METADATA_FIELDS, *SENSOR_FIELDS])
        writer.writeheader()
        writer.writerows(output_rows)
    diagnostics_path.write_text(diagnostic_text, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/imu_windows.csv",
                        help="one CSV, one per-frame JSON, or a directory of per-frame JSON files")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/imu_normalized.csv")
    parser.add_argument("--diagnostics", type=Path, default=ROOT / "outputs/diagnostic_summary.json")
    args = parser.parse_args()
    try:
        summary = run(args.input, args.output, args.diagnostics)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"IMU pipeline error: {exc}\n")
    print(f"Validated {summary['total_samples']} samples in {summary['group_count']} independent groups.")
    print(f"Estimated per-group sampling rate range (Hz): {summary['estimated_sampling_rate_range_hz']}")
    print(f"Duplicate timestamps within groups: {summary['duplicate_timestamps_within_groups']}")
    peak = summary["peak_specific_force"]
    if peak:
        print(f"Peak specific force: {peak['magnitude_mps2']:.6f} m/s² ({peak['magnitude_g']:.6f} g); retained unchanged.")
    print(f"Normalized CSV: {args.output}")
    print(f"Diagnostics: {args.diagnostics}")


if __name__ == "__main__":
    main()
