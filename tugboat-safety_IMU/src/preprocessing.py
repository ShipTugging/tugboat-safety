"""Conservative validation and diagnostics for ONE sequence of IMU samples.

No windows, frame IDs, fixed sampling rates, clipping, or filtering required.
Invalid sensor values raise an indexed error. Duplicate samples are reported
and retained; positive intervals alone contribute to sampling-rate estimates.
"""

from collections import Counter
import math
from statistics import median
from typing import Iterable, Mapping

from .imu_schema import IMUSample, IMUValidationError, ORIENTATION_FIELDS


def preprocess(samples: Iterable[IMUSample | Mapping[str, object]]) -> tuple[list[IMUSample], dict]:
    """Return stably time-sorted samples and a JSON-serializable diagnostic dict.

    Mapping input is validated by the same schema, allowing explicit missing /
    invalid field detection at this boundary. Already standardized samples work
    directly. Call separately for independent scenes or recording sessions.
    """
    validated = []
    for index, sample in enumerate(samples):
        try:
            values = sample.to_dict() if isinstance(sample, IMUSample) else sample
            validated.append(IMUSample.from_mapping(values))
        except IMUValidationError as exc:
            raise IMUValidationError(exc.field, f"sample index {index}: {exc.reason}") from exc

    input_times = [sample.timestamp_ms for sample in validated]
    ordered = sorted(validated, key=lambda sample: sample.timestamp_ms)
    times = [sample.timestamp_ms for sample in ordered]
    intervals = [b - a for a, b in zip(times, times[1:])]
    if any(not math.isfinite(interval) for interval in intervals):
        raise IMUValidationError("timestamp_ms", "sample interval exceeds finite numeric range")
    positive_intervals = [interval for interval in intervals if interval > 0]
    median_interval = median(positive_intervals) if positive_intervals else None
    rate = 1000.0 / median_interval if median_interval is not None else None
    if rate is not None and not math.isfinite(rate):
        raise IMUValidationError("timestamp_ms", "estimated sampling rate exceeds finite numeric range")
    counts = Counter(times)
    duplicates = [{"timestamp_ms": time, "count": count}
                  for time, count in sorted(counts.items()) if count > 1]
    peak_sample = max(ordered, key=lambda s: math.hypot(s.ax_mps2, s.ay_mps2, s.az_mps2), default=None)
    peak = None
    if peak_sample is not None:
        magnitude = math.hypot(peak_sample.ax_mps2, peak_sample.ay_mps2, peak_sample.az_mps2)
        if not math.isfinite(magnitude):
            raise IMUValidationError("accelerometer", "magnitude exceeds finite numeric range")
        peak = {
            "timestamp_ms": peak_sample.timestamp_ms,
            "vector_mps2": [peak_sample.ax_mps2, peak_sample.ay_mps2, peak_sample.az_mps2],
            "magnitude_mps2": magnitude,
            "magnitude_g": magnitude / 9.80665,
        }
    report = {
        "sample_count": len(ordered),
        "timestamp_range_ms": [times[0], times[-1]] if times else None,
        "duration_ms": times[-1] - times[0] if times else None,
        "input_out_of_order_pairs": sum(b < a for a, b in zip(input_times, input_times[1:])),
        "input_timestamps_nondecreasing": all(b >= a for a, b in zip(input_times, input_times[1:])),
        "input_timestamps_strictly_increasing": all(b > a for a, b in zip(input_times, input_times[1:])),
        "output_timestamps_strictly_increasing": all(interval > 0 for interval in intervals),
        "duplicate_timestamp_count": sum(item["count"] - 1 for item in duplicates),
        "duplicate_timestamps": duplicates,
        "sample_intervals_ms": intervals,
        "positive_interval_stats_ms": {
            "count": len(positive_intervals),
            "min": min(positive_intervals) if positive_intervals else None,
            "median": median_interval,
            "max": max(positive_intervals) if positive_intervals else None,
        },
        "estimated_sampling_rate_hz": rate,
        "sampling_rate_method": "1000 / median of positive consecutive timestamp differences in ms",
        "missing_required_values": 0,
        "invalid_numeric_values": 0,
        "validation_policy": "fail on missing required or non-finite/invalid numeric values; no rows dropped",
        "missing_optional_orientation_values": {
            field: sum(getattr(sample, field) is None for sample in ordered)
            for field in ORIENTATION_FIELDS
        },
        "available_orientation_fields": [
            field for field in ORIENTATION_FIELDS
            if any(getattr(sample, field) is not None for sample in ordered)
        ],
        "acceleration_component_ranges_mps2": {
            field: [min(getattr(s, field) for s in ordered), max(getattr(s, field) for s in ordered)]
            if ordered else None for field in ("ax_mps2", "ay_mps2", "az_mps2")
        },
        "peak_specific_force": peak,
        "acceleration_policy": "all values retained unchanged; peaks are descriptive, with no rejection threshold",
    }
    if report["duration_ms"] is not None and not math.isfinite(report["duration_ms"]):
        raise IMUValidationError("timestamp_ms", "sequence duration exceeds finite numeric range")
    return ordered, report
