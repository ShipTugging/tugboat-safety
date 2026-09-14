# IMU Foundation V1 and Vision–IMU Synchronization V1

A standalone, timestamp-based IMU input and preprocessing pipeline for the
tugboat safety prototype. The current adapter reads simulated IMU measurements;
future physical IMU adapters can produce the same sensor interface.
An additional synchronization layer pairs upstream Vision features with these
standardized measurements by timestamp. The Foundation pipeline is unchanged.

```text
Simulator CSV / JSON → simulator adapter → IMUSample → preprocessing
                                                        ↓
                                           normalized CSV + diagnostics
```

**Run**

Requires Python 3.10+ with no third-party dependencies. Run from
`tugboat-safety_IMU/` (the commands below use this directory):

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

Each image in this dataset is an **independent scene**. The Foundation runner processes
each `(sequence_id, frame_id)` group separately; a shared sequence ID does not
override distinct frame IDs. Group order follows first appearance, and samples
are stably sorted within groups. A file with no association IDs is treated as
one sequence; do not put multiple independent scenes in such a file. Global
timestamp ranges describe inventory, not the duration of one continuous run.
This grouping is specific to Foundation's simulator ingestion. The generic
Vision matcher described below uses session/sequence boundaries, never frame
equality. The supplied data already has one unique `sequence_id` per scene.

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

The synchronization layer below consumes these standardized samples. Both
sources must establish a common clock upstream and retain session/sequence
boundaries. Timestamp window generation could be added later without changing
`IMUSample`. Vision processing, risk logic, risk features, hardware drivers,
transport integrations, live streaming, and multimodal models are not implemented.

**Vision–IMU synchronization API**

`src/synchronization.py` contains no file I/O. Its public interfaces are:

- `VisionSample(timestamp_ms, towline_detected, confidence, towline_angle_deg=None,
  sag_ratio=None, curvature=None, metadata=Association())`: upstream features.
- `IMURecord(sample: IMUSample, metadata=Association())`: the existing IMU type
  with optional metadata. Plain `IMUSample` values also work directly.
- `Association(session_id=None, sequence_id=None, frame_id=None,
  frame_timestamp_ms=None)`: metadata kept separate from sensors/features.
- `NearestTimestampMatcher(imu_samples, max_time_gap_ms=20).match(vision_sample)`:
  one Vision query against an indexed, finite IMU snapshot.
- `synchronize(vision_samples, imu_samples, max_time_gap_ms=20)`:
  returns `(list[SynchronizedRecord], diagnostics_dict)` in Vision input order.
  Both inputs accept iterables of typed records or flat field mappings.
- `record.to_dict()`: nested `timestamp_ms`, `vision`, `imu`, `sync`, and
  `metadata` output. Metadata preserves separate Vision and matched IMU values.

`VisionSample.from_mapping()` and `IMURecord.from_mapping()` accept the flat
CSV field layout, including optional association columns. Required Vision
fields are `timestamp_ms`, `towline_detected`, and `confidence`. Confidence is
finite and in `[0, 1]`; no confidence cutoff affects synchronization. Detection
is a boolean (CSV also accepts case-insensitive `true`/`false` and `1`/`0`).
Geometry fields may be absent, null, or blank. Present numeric values must be
finite; no geometric meaning is inferred. IMU validation reuses `IMUSample`.
Unknown fields are excluded. Invalid samples fail with index/CSV line context,
while valid samples lacking a match remain in the output.

Here is a small in-memory example, with features supplied as mock values:

```python
from src.imu_schema import IMUSample
from src.synchronization import Association, IMURecord, VisionSample, synchronize

scope = Association(session_id="demo")
imu = [
    IMURecord(IMUSample(t, 0.12, 9.71, -0.05, 0.021, -0.004, 0.031), scope)
    for t in (1230.0, 1240.0)
]
vision = [
    VisionSample(1234.5, True, 0.94, metadata=Association(session_id="demo", frame_id="camera_1")),
    VisionSample(1300.0, False, 0.1, metadata=Association(session_id="demo", frame_id="camera_2")),
]
records, diagnostics = synchronize(vision, imu, max_time_gap_ms=20)
matched, unmatched = [record.to_dict() for record in records]
assert matched["imu"]["timestamp_ms"] == 1230.0
assert matched["sync"]["time_delta_ms"] == -4.5
assert unmatched["imu"] is None
assert unmatched["sync"]["reason"] == "max_time_gap_exceeded"
assert unmatched["vision"]["towline_detected"] is False
```

For an existing `OfflineSample` from the simulator adapter, wrap `record.sample`
in `IMURecord` with `Association(sequence_id=record.sequence_id,
frame_id=record.frame_id, frame_timestamp_ms=record.frame_timestamp_ms)`.
Foundation's `OfflineSample` and `IMUSample` definitions remain unchanged.

**Matching and association rules**

IMU samples are stably sorted within `(session_id, sequence_id)` groups. For
each Vision sample, compatible groups are resolved before a binary search finds
the nearest IMU timestamp. No row index, frame timestamp, or sampling-rate
constant is used. A sample may match an earlier or later IMU sample. Equal
distances choose the earlier timestamp, and duplicate timestamps choose the
first original IMU input record. No duplicate averaging or interpolation occurs;
the selected IMU values are preserved. A measurement can serve multiple Vision
samples. Detection being false or geometry being unavailable does not prevent
timestamp matching.

`time_delta_ms = imu_timestamp_ms - vision_timestamp_ms`. Acceptance is exactly
`abs(time_delta_ms) <= max_time_gap_ms`, including equality. The default **20 ms**
is a configurable initial tolerance: an ideal 100 Hz stream has at most 5 ms
nearest-sample error inside its time range, leaving some margin for timing
variation. It is not a fixed-frequency assumption or a validated hardware
latency allowance. Set the tolerance for the clock accuracy and staleness
requirements of each deployment. Zero allows exact matches only; negative or
non-finite tolerances are rejected.

Metadata rules are deliberately independent of camera frame numbering:

- `session_id` is the primary boundary. Conflicting known sessions cannot match,
  even if their sequence/frame IDs and timestamps are identical.
- `sequence_id` further restricts matching when supplied by both inputs.
- `frame_id` is association/debug metadata only. Different frame IDs, or absent
  IMU frame IDs, are valid within a continuous session. `frame_timestamp_ms` is
  also preserved only as metadata; `timestamp_ms` remains the matching clock.
- Missing session/sequence IDs are unspecified. A query can proceed only if
  exactly one compatible IMU group remains. If several groups remain, even an
  exact timestamp cannot resolve the ambiguity. Known conflicts never fall
  back to another group merely because it is closer in time.
- When IDs are absent on one or both sides, the caller must supply inputs from
  the same single session/sequence or add explicit boundary metadata upstream.
  Missing IDs are never filled in or inferred from frame IDs or timestamps.

The current synthetic data contains 30 scenes with distinct `sequence_id`
values, so Vision must carry the corresponding scene sequence ID. Future
synthetic exports that reuse a sequence ID across independent scenes must map
their scene identifier into `sequence_id` on **both** inputs before matching.
The generic matcher cannot infer that an ordinary frame ID represents a scene.

**Unmatched output and diagnostics**

Unmatched records retain the Vision timestamp, features, and metadata, with
`imu=null`, `sync.matched=false`, and `sync.time_delta_ms=null`. Reasons are:

| `sync.reason` | Meaning |
|---|---|
| `no_imu_samples` | Empty IMU input |
| `metadata_mismatch` | No compatible session/sequence group |
| `ambiguous_metadata` | More than one compatible group; IDs are insufficient |
| `max_time_gap_exceeded` | Nearest sample in the compatible group is too far away |

For a rejected nearest candidate only, `sync.nearest_time_delta_ms` records its
signed gap as a diagnostic; it does not supply an IMU state. Matched records
have `reason=null` and `nearest_time_delta_ms=null`.

Diagnostics contain `total_vision_samples`, `matched_samples`,
`unmatched_samples`, `match_rate` (a fraction), `mean_absolute_time_gap_ms`,
`median_absolute_time_gap_ms`, `maximum_absolute_time_gap_ms`,
`max_time_gap_ms`, and `unmatched_reasons` counts. Gap statistics include only
accepted matches. With no accepted matches, all three gap statistics are null.
With empty Vision input, counts are zero and match rate is null (undefined).
Rates are not estimated by this layer or used in matching.

**CSV synchronization**

Create/provide a small upstream Vision CSV using this header (geometry columns
may be omitted or left blank); add `session_id`, `sequence_id`, `frame_id`, and
`frame_timestamp_ms` as needed:

```csv
timestamp_ms,towline_detected,confidence,towline_angle_deg,sag_ratio,curvature
1234.5,true,0.94,,,
1300.0,false,0.1,,,
```

These mock timestamps require corresponding mock IMU data, as in the Python
example; they are not timestamps from the supplied simulator dataset. To run
against actual normalized IMU data, provide Vision timestamps and scene
sequence IDs from the same source clock:

```sh
python3 -B run_sync.py --vision vision_features.csv \
  --imu data/processed/imu_normalized.csv \
  --output data/processed/vision_imu_sync.csv \
  --diagnostics outputs/sync_diagnostics.json \
  --max-time-gap-ms 20
```

The CLI defaults the IMU/output paths relative to this module, so it also works
when invoked by absolute path from another directory. `run_sync.load_csv()`
reads standardized CSV directly, preserving `session_id` and
`frame_timestamp_ms`; raw simulator JSON adaptation remains in Foundation.
`run_sync.flatten()` provides the flattened output representation separately
from the matching logic.

Output columns are `vision_timestamp_ms`, `imu_timestamp_ms`,
`sync_time_delta_ms`, `sync_matched`, `sync_reason`,
`sync_nearest_time_delta_ms`, `towline_detected`, `vision_confidence`, the three
geometry fields, the nine IMU value/orientation fields, and separate
`vision_`/`imu_`-prefixed association fields. Nulls become blank CSV cells;
booleans become `True`/`False`. An empty Vision CSV with required headers writes
an output header and empty diagnostics. A completely empty file without the
required header is invalid.

The runner validates both inputs and computes results before writing. Validation
failure preserves previous outputs and exits nonzero. Successful runs replace
the requested output files. Input files, raw data paths, and colliding output
paths are protected. The CSV and diagnostic JSON are not a transactional pair;
a filesystem failure during writing can leave partial output.

**Synchronization verification and limits**

Run `python3 -B -m unittest discover -s tests -v` for all Foundation and
synchronization tests. Supplied-dataset regressions require the existing
`data/raw/` fixtures; derived CSVs are regenerated in temporary directories.
`tests/test_synchronization.py::mock_inputs` supplies only seven mock Vision
rows and four irregular IMU samples. Its expected result at 20 ms is 5 matched,
2 unmatched (one gap, one sequence mismatch), match rate `5/7`, mean absolute
gap `9.1 ms`, median `7.5 ms`, and maximum `20 ms`. Other small fixtures cover
non-100 Hz inputs, optional values, boundaries, duplicates, malformed data,
empty inputs, CLI execution, and the unchanged Foundation dataset output.

The single-sample API can be reused against a future live buffer, but V1 builds
a finite IMU snapshot; it does not accept an unbounded iterator, incrementally
update buffers, or wait for later packets. Offline nearest matching can choose a
future IMU sample. A live caller must define look-ahead/waiting and late-packet
policies before interpreting a result as final. Common-clock conversion,
offset/drift correction, and timestamp provenance remain upstream obligations.
The matching strategy is isolated in `NearestTimestampMatcher`, so future
strategies can produce the same output without changing file I/O or IMU types.
There is no Vision feature extraction, YOLO modification, risk logic,
interpolation, filtering, window aggregation, live server, API, or UI.

**Project files**

- `src/imu_schema.py`: sensor representation and value validation.
- `src/simulator_adapter.py`: simulator CSV/JSON ingestion and offline metadata.
- `src/preprocessing.py`: per-sequence sorting, validation, and diagnostics.
- `run_pipeline.py`: offline grouping, CLI, CSV and JSON report writing.
- `src/synchronization.py`: typed multimodal state, nearest matching, and diagnostics.
- `run_sync.py`: standardized Vision/IMU CSV I/O and synchronization CLI.
- `tests/test_imu_pipeline.py`: unit tests and full supplied-dataset regression checks.
- `tests/test_synchronization.py`: small mock features, synchronization and integration tests.
- `outputs/data_inspection.md`: source field inventory and interpretation.
- `outputs/diagnostic_summary.json`: reproducible run diagnostics.
- `data/processed/imu_normalized.csv`: example normalized measurements.
