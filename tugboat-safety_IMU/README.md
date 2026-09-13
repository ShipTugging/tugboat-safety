# IMU Foundation V1

A standalone, timestamp-based IMU input and preprocessing pipeline for the
tugboat safety prototype. The current adapter reads simulated IMU measurements;
future physical IMU adapters can produce the same sensor interface.

```text
Simulator CSV / JSON → simulator adapter → IMUSample → preprocessing
                                                        ↓
                                           normalized CSV + diagnostics
```

**Run**

Requires Python 3.10+ with no third-party dependencies. Run from the project root:

```sh
python3 -B -m unittest discover -s tests -v
python3 -B run_pipeline.py
```

The default input is `data/raw/imu_windows.csv`. Outputs are
`data/processed/imu_normalized.csv` and `outputs/diagnostic_summary.json`.
Successful runs replace these outputs. Validation errors exit nonzero with a
field and source row/sample location, before writing outputs. Existing output
files from a previous successful run remain in place after a validation failure;
they are not a report of the failed run. Nothing under `data/raw/` is written.

Select one alternative representation at a time:

```sh
python3 -B run_pipeline.py --input data/raw/imu.csv --output data/processed/anchors.csv --diagnostics outputs/anchors.json
python3 -B run_pipeline.py --input data/raw/imu --output data/processed/from_json.csv --diagnostics outputs/from_json.json
python3 -B run_pipeline.py --input data/raw/imu/frame_0001.json --output data/processed/frame_0001.csv --diagnostics outputs/frame_0001.json
```

The directory option expects per-frame JSON files (`data/raw/imu`), not the
parent directory containing `imu_schema.json`. The two CSVs and per-frame JSONs
repeat measurements; the runner never combines these representations. If a
JSON contains both `samples` and `sample`, only `samples` is ingested.

**Standard sensor interface**

`src/imu_schema.py` defines the immutable `IMUSample`. Only these fields enter it:

| Fields | Units / meaning | Required? |
|---|---|---|
| `timestamp_ms` | Milliseconds in the documented source clock | Yes |
| `ax_mps2`, `ay_mps2`, `az_mps2` | Body-frame specific force, m/s² | Yes |
| `gx_rad_s`, `gy_rad_s`, `gz_rad_s` | Body angular velocity components, rad/s | Yes |
| `roll_deg`, `pitch_deg`, `yaw_deg` | Orientation, degrees; Euler XYZ = `[pitch, yaw, roll]` | No |

Required fields must be finite numbers or numeric strings. Missing values,
booleans, invalid numeric strings, NaN, and infinity are rejected. Optional
orientation fields can be omitted, null, or blank; missing values become `None`
and blank CSV cells. Invalid *present* orientation values are rejected. Unknown
input keys are excluded by an explicit sensor-field allowlist.

The right-handed tug-local convention is **+X lateral, +Y up, +Z bow**, with the
simulator sensor aligned at the model origin. The schema does not explicitly
label +X port or starboard. Acceleration is **specific force**, not gravity-removed
linear acceleration. An upright stationary sensor reads approximately
**`[0, +9.80665, 0]` m/s²**. V1 preserves this convention and performs no rotation,
gravity subtraction, or orientation estimation. The simulator is already in
the required units, so its adapter needs no unit conversion.

**Offline associations and timestamps**

`OfflineSample` in `src/simulator_adapter.py` holds an `IMUSample` and optional
`frame_id`, `sequence_id`, `frame_timestamp_ms`. These are dataset association
metadata, not required sensor fields. The normalized CSV includes these three
metadata columns first, followed by the ten sensor columns above. Quaternion
components are outside the V1 output.

Simulator time is decimal milliseconds from the simulator origin, **not Unix
time**. CSV frame time is recovered as `timestamp_ms - offset_ms`; JSON also
provides it at the top level. Unknown frame times remain absent. When redundant
frame timestamps are supplied, consistency checks allow an absolute 0.000001 ms
tolerance for serialization/float rounding; timestamps are not rounded to that
tolerance or snapped to a grid. Source timestamps are parsed as Python floats;
CSV formatting can change, while the provided CSV's numeric sensor values are
preserved exactly on decimal round-trip (covered by the dataset test).

Each image in this dataset is an **independent scene**. The runner processes
each `(sequence_id, frame_id)` group separately; a shared sequence ID does not
override distinct frame IDs. Group order follows first appearance, and samples
are stably sorted within groups. A file with no association IDs is treated as
one sequence; do not put multiple independent scenes in such a file. Global
timestamp ranges describe inventory, not the duration of one continuous run.

**Preprocessing and diagnostics**

`preprocess(samples)` in `src/preprocessing.py` accepts an iterable of sensor
samples (or sensor mappings for validation) from **one sequence**. It does not
require frames, windows, or a declared sampling frequency. It returns sorted
samples and diagnostics containing:

- Required/numeric validation status and missing optional orientation counts.
- Original ordering checks, strict monotonicity flags, and timestamp ranges.
- Duplicate timestamps and counts; all duplicate samples remain in stable order.
- Actual sorted consecutive intervals, including zero intervals for duplicates.
- Estimated Hz = `1000 / median(positive intervals in ms)`; no rate is available
  for fewer than two distinct timestamps. This is an estimate for irregular
  sampling, not a reconstructed hardware frequency or a missing-packet detector.
- Specific-force component ranges and the peak vector, magnitude, and timestamp.

There are no value thresholds, clipping, interpolation, duplicate averaging,
sample removal, or filtering. Invalid input aborts validation; successful reports
therefore have zero missing required/invalid numeric counts. These zero counts
do not represent skipped rows. Diagnostics do not infer packet loss without a
known expected cadence. Monotonicity flags are vacuously true for empty/single
sequences; their rate remains unavailable. Sorting requires a finite batch in
V1, although the input iterable need not be pre-windowed.

**Simulator-only data**

The adapter excludes `world_ax_mps2`, `world_ay_mps2`, `world_az_mps2`,
`simulator_roll_rate_deg_s`, `tow_position`, `line_tension_kn`, `girting_risk_pct`,
JSON `positionWorldM`, and all other non-allowlisted values. Asset paths and
generation settings remain source documentation rather than sensor features.
Quaternion data is optional physical-compatible orientation in the source,
but is deliberately omitted from standardized V1 output.

The supplied dataset has 30 independent scenes, 630 samples, 21 samples per
200 ms window, and observed 10 ms intervals (100 Hz). These are observations,
not constants in preprocessing. All three Euler fields are present. A peak of
about 435.05 m/s² (44.36 g) in `frame_0013` is retained unchanged and reported
descriptively. The simulator uses ideal model-derived values, including centered
numerical derivatives; it does not model hardware bias, noise, or latency.
See [the inspection report](outputs/data_inspection.md) and
[generated diagnostics](outputs/diagnostic_summary.json).

**Future adapters and Vision connection**

A future hardware adapter will map its source units, axes, and clock into this
same interface, then pass `IMUSample` values to `preprocess` per recording
sequence. Optional orientation may stay absent. For example, the downstream
boundary can already be used without any offline metadata:

```python
from src.imu_schema import IMUSample
from src.preprocessing import preprocess

samples = [
    IMUSample(t, 0, 9.80665, 0, 0, 0, 0)
    for t in (0.0, 25.0, 50.0)
]
clean, diagnostics = preprocess(samples)  # Observed rate: 40 Hz
```

Later timestamp-based Vision synchronization must establish a common clock
and retain recording/scene boundaries. Frame indices alone cannot synchronize
sensors. Timestamp window generation can be added later without changing
`IMUSample`. No synchronization, Vision processing, risk logic, risk features,
hardware drivers, transport integrations, live streaming, or multimodal model
is implemented in V1.

**Project files**

- `src/imu_schema.py`: sensor representation and value validation.
- `src/simulator_adapter.py`: simulator CSV/JSON ingestion and offline metadata.
- `src/preprocessing.py`: per-sequence sorting, validation, and diagnostics.
- `run_pipeline.py`: offline grouping, CLI, CSV and JSON report writing.
- `tests/test_imu_pipeline.py`: unit tests and full supplied-dataset regression checks.
- `outputs/data_inspection.md`: source field inventory and interpretation.
- `outputs/diagnostic_summary.json`: reproducible run diagnostics.
- `data/processed/imu_normalized.csv`: example normalized measurements.
