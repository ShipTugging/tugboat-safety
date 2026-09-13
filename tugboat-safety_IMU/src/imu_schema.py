"""Physical-IMU-compatible sensor values only.

Right-handed tug-local axes: +X lateral, +Y up, +Z bow. Acceleration is
specific force (m/s²); upright rest is [0, +9.80665, 0]. Gyro components
are body angular velocity (rad/s). Optional Euler XYZ is [pitch, yaw, roll]
in degrees. timestamp_ms is milliseconds in the source's documented clock,
not implicitly Unix time. Adapters must establish units, axes, and clock.
"""

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Optional


REQUIRED_FIELDS = (
    "timestamp_ms", "ax_mps2", "ay_mps2", "az_mps2",
    "gx_rad_s", "gy_rad_s", "gz_rad_s",
)
ORIENTATION_FIELDS = ("roll_deg", "pitch_deg", "yaw_deg")
SENSOR_FIELDS = REQUIRED_FIELDS + ORIENTATION_FIELDS


class IMUValidationError(ValueError):
    """A missing or invalid value, identified by field name."""

    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


def numeric_value(field: str, value: object, *, optional: bool = False) -> Optional[float]:
    """Accept finite numbers/numeric strings; blank optional values become None."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if optional:
            return None
        raise IMUValidationError(field, "missing required numeric value")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise IMUValidationError(field, "expected a finite numeric value")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise IMUValidationError(field, "expected a finite numeric value") from exc
    if not math.isfinite(number):
        raise IMUValidationError(field, "expected a finite numeric value")
    return number


@dataclass(frozen=True)
class IMUSample:
    """One validated sample; no frame IDs, simulator truth, or quaternions."""

    timestamp_ms: float
    ax_mps2: float
    ay_mps2: float
    az_mps2: float
    gx_rad_s: float
    gy_rad_s: float
    gz_rad_s: float
    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    yaw_deg: Optional[float] = None

    def __post_init__(self) -> None:
        for field in SENSOR_FIELDS:
            object.__setattr__(self, field, numeric_value(
                field, getattr(self, field), optional=field in ORIENTATION_FIELDS,
            ))

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "IMUSample":
        """Allowlist sensor fields. Unrelated input keys never enter the sample."""
        if not isinstance(values, Mapping):
            raise IMUValidationError("sample", "expected a field mapping")
        return cls(**{field: values.get(field) for field in SENSOR_FIELDS})

    def to_dict(self) -> dict:
        return asdict(self)
