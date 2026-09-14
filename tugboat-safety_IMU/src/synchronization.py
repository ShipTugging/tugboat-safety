"""Timestamp-only nearest matching within compatible session/sequence boundaries.

Inputs must already use a common clock in milliseconds. No file I/O, clock
correction, interpolation, filtering, or feature extraction occurs here.
"""

from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import math
from statistics import mean
from typing import Iterable, Mapping

from .imu_schema import IMUSample, numeric_value


DEFAULT_MAX_TIME_GAP_MS = 20.0
VISION_REQUIRED_FIELDS = ("timestamp_ms", "towline_detected", "confidence")
VISION_GEOMETRY_FIELDS = ("towline_angle_deg", "sag_ratio", "curvature")
METADATA_FIELDS = ("session_id", "sequence_id", "frame_id", "frame_timestamp_ms")


@dataclass(frozen=True)
class Association:
    """Optional metadata, separate from sensor/features; frame_id is debug only."""

    session_id: str | None = None
    sequence_id: str | None = None
    frame_id: str | None = None
    frame_timestamp_ms: float | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "sequence_id", "frame_id"):
            value = getattr(self, name)
            if value is None or value == "":
                object.__setattr__(self, name, None)
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name}: expected a nonblank string")
        object.__setattr__(self, "frame_timestamp_ms", numeric_value(
            "frame_timestamp_ms", self.frame_timestamp_ms, optional=True,
        ))

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "Association":
        return cls(**{name: values.get(name) for name in METADATA_FIELDS})


def _mapping(values: object) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise ValueError("sample: expected a field mapping")
    return values


def _boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1"):
            return True
        if normalized in ("false", "0"):
            return False
    raise ValueError("towline_detected: expected a boolean or CSV true/false/1/0")


@dataclass(frozen=True)
class VisionSample:
    """Already computed upstream features; unavailable geometry stays None."""

    timestamp_ms: float
    towline_detected: bool
    confidence: float
    towline_angle_deg: float | None = None
    sag_ratio: float | None = None
    curvature: float | None = None
    metadata: Association = field(default_factory=Association)

    def __post_init__(self) -> None:
        for name in ("timestamp_ms", "confidence", *VISION_GEOMETRY_FIELDS):
            object.__setattr__(self, name, numeric_value(
                name, getattr(self, name), optional=name in VISION_GEOMETRY_FIELDS,
            ))
        object.__setattr__(self, "towline_detected", _boolean(self.towline_detected))
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence: expected a value between 0 and 1")
        if not isinstance(self.metadata, Association):
            raise ValueError("metadata: expected Association")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "VisionSample":
        values = _mapping(values)
        return cls(**{
            name: values.get(name) for name in (*VISION_REQUIRED_FIELDS, *VISION_GEOMETRY_FIELDS)
        }, metadata=Association.from_mapping(values))

    def features(self) -> dict:
        return {name: getattr(self, name) for name in (
            "towline_detected", "confidence", *VISION_GEOMETRY_FIELDS,
        )}


@dataclass(frozen=True)
class IMURecord:
    """The existing standardized IMUSample plus optional association metadata."""

    sample: IMUSample
    metadata: Association = field(default_factory=Association)

    def __post_init__(self) -> None:
        if not isinstance(self.sample, IMUSample):
            raise ValueError("sample: expected IMUSample")
        if not isinstance(self.metadata, Association):
            raise ValueError("metadata: expected Association")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "IMURecord":
        values = _mapping(values)
        return cls(IMUSample.from_mapping(values), Association.from_mapping(values))


@dataclass(frozen=True)
class SynchronizedRecord:
    vision: VisionSample
    imu: IMURecord | None
    time_delta_ms: float | None
    reason: str | None = None
    nearest_time_delta_ms: float | None = None

    @property
    def matched(self) -> bool:
        return self.imu is not None

    def to_dict(self) -> dict:
        return {
            "timestamp_ms": self.vision.timestamp_ms,
            "vision": self.vision.features(),
            "imu": self.imu.sample.to_dict() if self.imu else None,
            "sync": {
                "matched": self.matched,
                "time_delta_ms": self.time_delta_ms,
                "reason": self.reason,
                # Only populated for a rejected nearest candidate, not an IMU state.
                "nearest_time_delta_ms": self.nearest_time_delta_ms,
            },
            "metadata": {
                "vision": asdict(self.vision.metadata),
                "imu": asdict(self.imu.metadata) if self.imu else None,
            },
        }


class NearestTimestampMatcher:
    """Reusable matcher over a finite IMU snapshot, independent of CSV/batching.

    Sessions are primary boundaries; sequences further restrict them. A missing
    ID is unspecified, not evidence of equality. A query must resolve to exactly
    one compatible (session_id, sequence_id) group before timestamps are searched.
    With no IDs, the caller is responsible for supplying one continuous session.
    frame_id and frame_timestamp_ms NEVER restrict or replace timestamps.

    IMU order may be arbitrary. Equal-distance ties select the earlier time;
    timestamp duplicates select the first input record. Records can be reused
    for multiple Vision samples; matching does not consume IMU data.
    """

    def __init__(
        self,
        imu_samples: Iterable[IMURecord | IMUSample | Mapping[str, object]],
        *,
        max_time_gap_ms: float = DEFAULT_MAX_TIME_GAP_MS,
    ) -> None:
        gap = numeric_value("max_time_gap_ms", max_time_gap_ms)
        if gap < 0:
            raise ValueError("max_time_gap_ms: must be nonnegative")
        self._max_time_gap_ms = gap
        groups = defaultdict(list)
        for index, value in enumerate(imu_samples):
            try:
                if isinstance(value, IMURecord):
                    record = value
                elif isinstance(value, IMUSample):
                    record = IMURecord(value)
                else:
                    record = IMURecord.from_mapping(value)
            except ValueError as exc:
                raise ValueError(f"IMU sample index {index}: {exc}") from exc
            groups[(record.metadata.session_id, record.metadata.sequence_id)].append(record)
        self._groups = {}
        for key, records in groups.items():
            ordered = sorted(records, key=lambda record: record.sample.timestamp_ms)
            self._groups[key] = ([r.sample.timestamp_ms for r in ordered], ordered)

    @property
    def max_time_gap_ms(self) -> float:
        return self._max_time_gap_ms

    def match(self, vision: VisionSample | Mapping[str, object]) -> SynchronizedRecord:
        """Match one sample, preserving its features even when no match exists."""
        if not isinstance(vision, VisionSample):
            vision = VisionSample.from_mapping(vision)
        if not self._groups:
            return SynchronizedRecord(vision, None, None, "no_imu_samples")
        query_key = (vision.metadata.session_id, vision.metadata.sequence_id)
        compatible = [group for key, group in self._groups.items() if all(
            query is None or candidate is None or query == candidate
            for query, candidate in zip(query_key, key)
        )]
        if not compatible:
            return SynchronizedRecord(vision, None, None, "metadata_mismatch")
        if len(compatible) != 1:
            return SynchronizedRecord(vision, None, None, "ambiguous_metadata")

        times, records = compatible[0]
        position = bisect_left(times, vision.timestamp_ms)
        indices = []
        if position < len(times):
            indices.append(position)
        if position:
            # bisect_left also selects the first input among earlier duplicates.
            indices.append(bisect_left(times, times[position - 1]))
        chosen = min(indices, key=lambda i: (abs(times[i] - vision.timestamp_ms), times[i]))
        delta = times[chosen] - vision.timestamp_ms
        if not math.isfinite(delta):
            raise ValueError("timestamp_ms: time delta exceeds finite numeric range")
        if abs(delta) > self.max_time_gap_ms:
            return SynchronizedRecord(vision, None, None, "max_time_gap_exceeded", delta)
        return SynchronizedRecord(vision, records[chosen], delta)


def synchronize(
    vision_samples: Iterable[VisionSample | Mapping[str, object]],
    imu_samples: Iterable[IMURecord | IMUSample | Mapping[str, object]],
    *,
    max_time_gap_ms: float = DEFAULT_MAX_TIME_GAP_MS,
) -> tuple[list[SynchronizedRecord], dict]:
    """Preserve Vision input order; return records and matched-only gap statistics.

    Invalid inputs abort with sample context, without dropping invalid rows.
    Empty Vision has an undefined (None) match rate; no matches have None gap
    statistics. Missing geometry/orientation is valid and stays None.
    """
    matcher = NearestTimestampMatcher(imu_samples, max_time_gap_ms=max_time_gap_ms)
    records = []
    for index, vision in enumerate(vision_samples):
        try:
            records.append(matcher.match(vision))
        except ValueError as exc:
            raise ValueError(f"Vision sample index {index}: {exc}") from exc
    gaps = sorted(abs(record.time_delta_ms) for record in records if record.matched)
    count = len(gaps)
    middle = count // 2
    median_gap = None
    if gaps:
        median_gap = gaps[middle]
        if not count % 2:
            lower, upper = gaps[middle - 1], gaps[middle]
            median_gap = lower + (upper - lower) / 2
    return records, {
        "total_vision_samples": len(records),
        "matched_samples": count,
        "unmatched_samples": len(records) - count,
        "match_rate": count / len(records) if records else None,
        "mean_absolute_time_gap_ms": mean(gaps) if gaps else None,
        "median_absolute_time_gap_ms": median_gap,
        "maximum_absolute_time_gap_ms": max(gaps) if gaps else None,
        "max_time_gap_ms": matcher.max_time_gap_ms,
        "unmatched_reasons": dict(Counter(record.reason for record in records if not record.matched)),
    }
