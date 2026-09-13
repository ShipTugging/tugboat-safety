# Simulator IMU source inspection

Inspected all 33 source files: `data/raw/imu.csv`, `imu_windows.csv`,
`imu_schema.json`, and `imu/frame_0001.json` through `imu/frame_0030.json`.
This is an inspection record; `run_pipeline.py` regenerates the normalized CSV
and machine-readable diagnostics, not this source interpretation document.

**Exact CSV columns**

Both CSVs have the same 28 columns in this order:

```text
frame_id,sequence_id,image,label,mask,imu_file,timestamp_ms,offset_ms,ax_mps2,ay_mps2,az_mps2,gx_rad_s,gy_rad_s,gz_rad_s,pitch_deg,yaw_deg,roll_deg,qx,qy,qz,qw,world_ax_mps2,world_ay_mps2,world_az_mps2,simulator_roll_rate_deg_s,tow_position,line_tension_kn,girting_risk_pct
```

The schema describes conventions rather than defining every column. Definitions
not supplied by the schema are identified below rather than assumed.

| CSV field | Meaning / units | Classification |
|---|---|---|
| `frame_id` | Image/scene ID; matches JSON `frameId` | Metadata |
| `sequence_id` | Sequence ID; always equals `frame_id` here | Metadata |
| `image` | Associated image path | Metadata |
| `label` | Associated label path; format/content not defined | Metadata |
| `mask` | Associated mask path; mask semantics not defined | Metadata |
| `imu_file` | Path to corresponding per-frame IMU JSON | Metadata |
| `timestamp_ms` | Sample time in simulation milliseconds | Required physical-compatible |
| `offset_ms` | Sample time minus frame time in ms; verified from data | Metadata |
| `ax_mps2` | Body X specific force, m/s² | Required physical-compatible |
| `ay_mps2` | Body Y specific force, m/s² | Required physical-compatible |
| `az_mps2` | Body Z specific force, m/s² | Required physical-compatible |
| `gx_rad_s` | Body angular velocity X component, rad/s | Required physical-compatible |
| `gy_rad_s` | Body angular velocity Y component, rad/s | Required physical-compatible |
| `gz_rad_s` | Body angular velocity Z component, rad/s | Required physical-compatible |
| `pitch_deg` | X component of Euler XYZ, degrees | Optional physical-compatible |
| `yaw_deg` | Y component of Euler XYZ, degrees | Optional physical-compatible |
| `roll_deg` | Z component of Euler XYZ, degrees | Optional physical-compatible |
| `qx` | Quaternion X, dimensionless | Optional physical-compatible; omitted in V1 |
| `qy` | Quaternion Y, dimensionless | Optional physical-compatible; omitted in V1 |
| `qz` | Quaternion Z, dimensionless | Optional physical-compatible; omitted in V1 |
| `qw` | Quaternion scalar, dimensionless | Optional physical-compatible; omitted in V1 |
| `world_ax_mps2` | Simulator world X linear acceleration, m/s² | Simulator-only / exclude |
| `world_ay_mps2` | Simulator world Y linear acceleration, m/s² | Simulator-only / exclude |
| `world_az_mps2` | Simulator world Z linear acceleration, m/s² | Simulator-only / exclude |
| `simulator_roll_rate_deg_s` | Name indicates degrees/s; derivation unspecified | Simulator-only / exclude |
| `tow_position` | `ahead`, `astern`, `port`, `starboard`; reference unspecified | Simulator-only / exclude |
| `line_tension_kn` | Name suggests kN; definition not supplied by schema | Simulator-only / exclude |
| `girting_risk_pct` | Name suggests percentage; calculation unspecified | Simulator-only / exclude |

Physical-compatible refers to the kind of measurement, not its provenance:
every measurement in these files is simulated. Only the seven required sensor
fields and three optional Euler fields enter `IMUSample`. Frame/sequence/time
associations use a separate offline envelope. The runner exports those three
metadata columns alongside the ten sensor columns; other metadata is not exported.

**All JSON fields**

Every per-frame file has the same top-level keys:

```text
frameId, sequenceId, timestampMs, sampleRateHz, windowMs, sample, samples, image, label, mask
```

- `frameId`, `sequenceId`, and file paths are association metadata.
- Top-level `timestampMs` is frame-time metadata; it equals the final sample time.
- `sampleRateHz` (100 Hz) and `windowMs` (200 ms) are generation metadata.
- `sample` and `samples` are structural containers. `sample` repeats `samples[-1]`.

Every nested sample has the following keys:

| JSON sample field | Meaning / CSV mapping | Classification |
|---|---|---|
| `timestampMs` | `timestamp_ms` | Required physical-compatible |
| `offsetMs` | `offset_ms` | Metadata |
| `accelerometerMps2` | `[ax_mps2, ay_mps2, az_mps2]` | Required physical-compatible |
| `gyroscopeRadS` | `[gx_rad_s, gy_rad_s, gz_rad_s]` | Required physical-compatible |
| `linearAccelerationWorldMps2` | `[world_ax_mps2, world_ay_mps2, world_az_mps2]` | Simulator-only / exclude |
| `orientationQuaternion` | `[qx, qy, qz, qw]` | Optional physical-compatible; omitted in V1 |
| `orientationEulerDeg` | `[pitch_deg, yaw_deg, roll_deg]` | Optional physical-compatible |
| `positionWorldM` | Name indicates world XYZ position in metres; origin unspecified; absent from CSV | Simulator-only / exclude |
| `simulatorRollRateDegS` | `simulator_roll_rate_deg_s` | Simulator-only / exclude |

**Schema and coordinate conventions**

`imu_schema.json` contains `source`, `sampleRateHz`, `windowMs`,
`samplesPerFrame`, `clock`, `synchronization`, `sensorFrame`, `rotation`,
`accelerometer`, `gyroscope`, `gravityWorldMps2`, and `limitations`. These describe
provenance, generation settings, units, reference frames, and limitations;
they are not additional per-sample sensor fields.

- Source: `ideal_simulated_imu`.
- Right-handed tug-local XYZ: **+X lateral, +Y up, +Z bow**. +X is not explicitly
  labeled port/starboard. Sensor is aligned at model-local origin `[0, 0, 0]`.
- Quaternion `[x, y, z, w]` maps body to simulator world. Euler XYZ is
  `[pitch, yaw, roll]` in degrees.
- Body specific force is the inverse-rotated world linear acceleration minus
  gravity. World gravity is `[0, -9.80665, 0]` m/s². Upright stationary
  acceleration is **`[0, +9.80665, 0]` m/s²**; it is not gravity-removed acceleration.
- Gyro is body angular velocity in rad/s, from the shortest quaternion rotation
  over a centered 20 ms difference. This is not an Euler-rate interface, and the
  simulator roll-rate variable must not substitute for a gyro axis.
- Values are noise-free and model-derived. Bias, latency, and real-vessel
  validation are not modeled. Centered derivatives use neighboring virtual
  poses; the schema says future images are not included.

All 630 JSON quaternion/Euler pairs agree with the stated rotation convention
to floating-point precision. Quaternion norm is effectively 1 throughout.
Rotating specific force to world and comparing against world acceleration minus
gravity gives a maximum residual of about `5.7e-14` m/s². These comparisons were
inspection checks only; excluded fields are not used by preprocessing.

**Timing and scene relationships**

The clock is simulation milliseconds from simulator origin, not Unix time.
CSV timestamps have nine decimal places; JSON numbers preserve additional
serialized precision. That precision is not proof of physical timing accuracy.
There is no wall-clock epoch mapping, hardware clock mapping, or documented
world-heading reference.

The schema explicitly states that each image is an independent scene and
windows from different frame IDs must not be concatenated. Each `sequence_id`
equals its `frame_id`. Frame indices, nearby timestamps, and overlapping windows
do not imply temporal continuity between scenes.

Every CSV sample satisfies exactly at serialized decimal precision:

```text
timestamp_ms = frame_timestamp_ms + offset_ms
```

Offsets run from -200 to 0 ms, including both endpoints, in 10 ms steps. JSON
top-level frame time equals the final IMU sample time, which the schema identifies
as the RGB telemetry timestamp. The actual RGB/label/mask files are not included,
so image telemetry agreement cannot be independently checked here.

**File differences and diagnostic results**

| Observation | Result |
|---|---|
| Independent frame IDs / sequence IDs | 30 / 30 |
| `imu.csv` rows | 30; one endpoint per scene |
| `imu_windows.csv` rows | 630; 21 samples per scene |
| Per-frame JSON files | 30; each repeats the 21 samples and a separate endpoint copy |
| Distinct measurements across representations | 630, not the sum of exported copies |
| Window span | 200 ms in every scene |
| Observed within-window intervals | 600 intervals, all 10 ms at CSV precision |
| Estimated per-window rate | 100 Hz; derived from timestamps |
| Endpoint CSV timestamp range (ms) | 66936.655324455 to 104281.218846329 |
| Full window timestamp range (ms) | 66736.655324455 to 104281.218846329 |
| Missing CSV cells / missing-null JSON values | None found |
| Invalid / non-finite numeric values | None found |
| Duplicate timestamps within scenes | None |
| Duplicate timestamps within either individual CSV | None |
| Within-scene ordering | Strictly increasing |
| Backward transitions across scenes in each CSV | 13; not within-scene corruption |
| Available orientation | All three Euler fields and four quaternion components |

Every complete row of `imu.csv` exactly equals the corresponding last row in
`imu_windows.csv`. All mapped JSON/CSV values agree within approximately
`5e-10` in their respective units, consistent with nine-decimal CSV rounding.
`imu.csv` alone cannot establish a rate: each scene has only one sample.
Global timestamp ranges above are inventory ranges, not continuous durations.

The largest specific-force magnitude is approximately **435.051254 m/s²
(44.362882 g)** in `frame_0013`, at CSV timestamp `75261.291037407` ms, with vector
`[-343.617997363, 252.491982780, -86.278990010]` m/s². All samples, including this
one, are retained unchanged. This magnitude merits checking the source motion
generator; agreement with simulator equations does not establish vessel realism.
V1 reports per-scene component ranges and peaks without rejection thresholds.

**Validation and remaining limitations**

The automated suite covers the supplied dataset, missing fields, invalid numeric
values, optional orientation, metadata-free samples, irregular/non-100 Hz
intervals, ordering, duplicate retention, independent scene boundaries, raw-data
write protection, and sensor allowlisting. Dataset checks compare every exported
sensor value against the original CSV using decimal numeric equality and compare
all 630 CSV/JSON sensor representations within serialization precision.

All 33 raw files were fingerprinted with SHA-256 before implementation. Their
hashes matched on resume, and the dataset integration test verifies no raw changes
through pipeline execution. The final deliverable audit also compares against
the original fingerprint manifest.

The pipeline intentionally provides finite-batch validation only. Sensor clock
calibration, packet-loss inference, coordinate conversion for hardware, causal
live processing, window generation, Vision synchronization, and risk modeling
are outside V1. Unusually large simulator accelerations and incomplete definitions
of excluded simulator variables remain source-data limitations, not silently
corrected inputs.
