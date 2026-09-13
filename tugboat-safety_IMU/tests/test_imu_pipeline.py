"""Unit tests and regression checks against the actual supplied simulator data."""

from collections import Counter, defaultdict
import csv
from dataclasses import FrozenInstanceError
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from run_pipeline import METADATA_FIELDS, ROOT, run
from src.imu_schema import IMUSample, IMUValidationError, ORIENTATION_FIELDS, REQUIRED_FIELDS, SENSOR_FIELDS
from src.preprocessing import preprocess
from src.simulator_adapter import load_simulator


def sample_values(timestamp=0, **changes):
    values = dict(zip(REQUIRED_FIELDS, [timestamp, 0, 9.80665, 0, 0, 0, 0]))
    values.update(changes)
    return values


def write_csv(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_sample(timestamp=100):
    return {"timestampMs": timestamp, "accelerometerMps2": [0, 9.80665, 0],
            "gyroscopeRadS": [0.1, -0.2, 0.3]}


class SchemaTests(unittest.TestCase):
    def test_sensor_interface_needs_no_metadata(self):
        sample = IMUSample(**sample_values())
        self.assertEqual(tuple(sample.to_dict()), SENSOR_FIELDS)
        self.assertEqual([sample.ax_mps2, sample.ay_mps2, sample.az_mps2], [0, 9.80665, 0])
        self.assertFalse(hasattr(sample, "frame_id"))
        with self.assertRaises(FrozenInstanceError):
            sample.ax_mps2 = 1

    def test_numeric_strings_and_fractional_timestamp(self):
        sample = IMUSample.from_mapping({k: str(v) for k, v in sample_values(100.125).items()})
        self.assertEqual(sample.timestamp_ms, 100.125)
        self.assertEqual(sample.ay_mps2, 9.80665)

    def test_each_required_field_missing_or_blank(self):
        for field in REQUIRED_FIELDS:
            for value in (None, "", "   "):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(IMUValidationError, field + ": missing"):
                        IMUSample.from_mapping(sample_values(**{field: value}))
            values = sample_values()
            del values[field]
            with self.assertRaisesRegex(IMUValidationError, field + ": missing"):
                IMUSample.from_mapping(values)

    def test_invalid_numeric_values_including_timestamp_and_orientation(self):
        for field in SENSOR_FIELDS:
            for value in ("bad", "NaN", "Infinity", float("nan"), float("inf"), -float("inf"), True, [], {}):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(IMUValidationError, field):
                        IMUSample.from_mapping(sample_values(**{field: value}))

    def test_optional_orientation_absent_blank_partial_and_present(self):
        absent = IMUSample.from_mapping(sample_values())
        self.assertTrue(all(getattr(absent, field) is None for field in ORIENTATION_FIELDS))
        partial = IMUSample.from_mapping(sample_values(roll_deg="", pitch_deg="  ", yaw_deg="12.5"))
        self.assertIsNone(partial.roll_deg)
        self.assertIsNone(partial.pitch_deg)
        self.assertEqual(partial.yaw_deg, 12.5)
        complete = IMUSample.from_mapping(sample_values(roll_deg=1, pitch_deg=2, yaw_deg=3))
        self.assertEqual((complete.roll_deg, complete.pitch_deg, complete.yaw_deg), (1, 2, 3))

    def test_sensor_allowlist_excludes_metadata_truth_and_quaternion(self):
        values = sample_values(frame_id="scene", sequence_id="scene", frame_timestamp_ms=0,
                               world_ax_mps2=4, simulator_roll_rate_deg_s=5, tow_position="port",
                               line_tension_kn=6, girting_risk_pct=7, qx=0, qy=0, qz=0, qw=1)
        self.assertEqual(set(IMUSample.from_mapping(values).to_dict()), set(SENSOR_FIELDS))

    def test_non_mapping_input_rejected(self):
        with self.assertRaisesRegex(IMUValidationError, "expected a field mapping"):
            IMUSample.from_mapping([1, 2, 3])


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_csv_units_timestamps_and_separate_metadata(self):
        path = self.directory / "input.csv"
        write_csv(path, [sample_values(123.25, ax_mps2=-343.617997363, gx_rad_s=0.25,
                                      frame_id="f", sequence_id="s", offset_ms=-17.5,
                                      pitch_deg=1, yaw_deg=2, roll_deg=3, girting_risk_pct=99, qw=1)])
        record, = load_simulator(path)
        self.assertEqual((record.frame_id, record.sequence_id, record.frame_timestamp_ms), ("f", "s", 140.75))
        self.assertEqual(record.sample.timestamp_ms, 123.25)
        self.assertEqual(record.sample.ax_mps2, -343.617997363)
        self.assertEqual(record.sample.gx_rad_s, 0.25)
        self.assertEqual(set(record.sample.to_dict()), set(SENSOR_FIELDS))

    def test_csv_without_orientation_or_metadata(self):
        path = self.directory / "input.csv"
        write_csv(path, [sample_values(0), sample_values(25)])
        records = load_simulator(path)
        self.assertEqual(len(records), 2)
        self.assertIsNone(records[0].frame_id)
        self.assertIsNone(records[0].sequence_id)
        self.assertIsNone(records[0].frame_timestamp_ms)
        self.assertIsNone(records[0].sample.roll_deg)

    def test_csv_missing_column_and_invalid_row_have_context(self):
        path = self.directory / "input.csv"
        row = sample_values()
        del row["gx_rad_s"]
        write_csv(path, [row])
        with self.assertRaisesRegex(ValueError, r"input.csv: missing required columns: gx_rad_s"):
            load_simulator(path)
        for bad in ("", "nan", "nonsense"):
            write_csv(path, [sample_values(), sample_values(10, ax_mps2=bad)])
            with self.assertRaisesRegex(ValueError, r"input.csv: line 3: ax_mps2"):
                load_simulator(path)

    def test_duplicate_headers_and_extra_csv_cells_rejected(self):
        path = self.directory / "input.csv"
        path.write_text(",".join([*REQUIRED_FIELDS, "ax_mps2"]) + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate CSV column"):
            load_simulator(path)
        path.write_text(",".join(REQUIRED_FIELDS) + "\n0,0,9.8,0,0,0,0,EXTRA\n")
        with self.assertRaisesRegex(ValueError, "more values than columns"):
            load_simulator(path)

    def test_json_vectors_and_endpoint_are_not_duplicated(self):
        path = self.directory / "frame.json"
        first, last = json_sample(75), json_sample(100)
        first.update(offsetMs=-25, orientationEulerDeg=[2, 3, 1])
        last.update(offsetMs=0, orientationEulerDeg=[5, 6, 4], orientationQuaternion=[0, 0, 0, 1],
                    positionWorldM=[1, 2, 3], linearAccelerationWorldMps2=[0, 0, 0])
        # Declared 100 Hz deliberately differs from the actual 25 ms interval.
        path.write_text(json.dumps({"frameId": "f", "sequenceId": "s", "timestampMs": 100,
                                    "sampleRateHz": 100, "windowMs": 200, "sample": last,
                                    "samples": [first, last]}))
        records = load_simulator(path)
        self.assertEqual(len(records), 2)
        self.assertEqual([r.sample.timestamp_ms for r in records], [75, 100])
        self.assertEqual(records[1].frame_timestamp_ms, 100)
        sample = records[1].sample
        self.assertEqual((sample.pitch_deg, sample.yaw_deg, sample.roll_deg), (5, 6, 4))
        self.assertEqual((sample.gx_rad_s, sample.gy_rad_s, sample.gz_rad_s), (0.1, -0.2, 0.3))
        self.assertEqual(set(sample.to_dict()), set(SENSOR_FIELDS))
        _, report = preprocess(r.sample for r in records)
        self.assertEqual(report["estimated_sampling_rate_hz"], 40)

    def test_json_single_sample_and_missing_optional_orientation(self):
        path = self.directory / "frame.json"
        path.write_text(json.dumps({"sample": json_sample()}))
        record, = load_simulator(path)
        self.assertIsNone(record.sample.roll_deg)
        self.assertIsNone(record.frame_id)
        self.assertIsNone(record.frame_timestamp_ms)

    def test_json_partial_orientation(self):
        path = self.directory / "frame.json"
        values = json_sample()
        values["orientationEulerDeg"] = [None, 12, ""]
        path.write_text(json.dumps({"sample": values}))
        sample = load_simulator(path)[0].sample
        self.assertIsNone(sample.pitch_deg)
        self.assertEqual(sample.yaw_deg, 12)
        self.assertIsNone(sample.roll_deg)

    def test_json_invalid_vectors_values_and_offsets(self):
        path = self.directory / "frame.json"
        cases = [
            ({"accelerometerMps2": [0, 1]}, "accelerometerMps2"),
            ({"gyroscopeRadS": [0, None, 0]}, "gy_rad_s"),
            ({"orientationEulerDeg": [0, 1]}, "orientationEulerDeg"),
            ({"timestampMs": "nan"}, "timestamp_ms"),
            ({"offsetMs": -10}, "offset_ms does not agree"),
        ]
        for changes, message in cases:
            with self.subTest(changes=changes):
                values = json_sample()
                values.update(changes)
                path.write_text(json.dumps({"timestampMs": 100, "sample": values}))
                with self.assertRaisesRegex(ValueError, "frame.json: sample index 0: .*" + message):
                    load_simulator(path)

    def test_json_directory_and_invalid_shape(self):
        path = self.directory / "a.json"
        path.write_text(json.dumps({"sample": json_sample(10)}))
        (self.directory / "b.json").write_text(json.dumps({"sample": json_sample(20)}))
        self.assertEqual(len(load_simulator(self.directory)), 2)
        for value in ([], {}, {"samples": []}, {"samples": [None]}):
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_simulator(path)


class PreprocessingTests(unittest.TestCase):
    def test_sorting_monotonicity_and_fractional_timestamps(self):
        ordered, report = preprocess(sample_values(t) for t in (30.125, 10.125, 20.125))
        self.assertEqual([s.timestamp_ms for s in ordered], [10.125, 20.125, 30.125])
        self.assertEqual(report["input_out_of_order_pairs"], 1)
        self.assertFalse(report["input_timestamps_nondecreasing"])
        self.assertFalse(report["input_timestamps_strictly_increasing"])
        self.assertTrue(report["output_timestamps_strictly_increasing"])

    def test_duplicate_values_retained_in_stable_order(self):
        ordered, report = preprocess([sample_values(20), sample_values(0, ax_mps2=1),
                                      sample_values(0, ax_mps2=2), sample_values(10)])
        self.assertEqual(len(ordered), 4)
        self.assertEqual([s.ax_mps2 for s in ordered[:2]], [1, 2])
        self.assertEqual(report["duplicate_timestamp_count"], 1)
        self.assertEqual(report["duplicate_timestamps"], [{"timestamp_ms": 0, "count": 2}])
        self.assertEqual(report["sample_intervals_ms"], [0, 10, 10])
        self.assertEqual(report["estimated_sampling_rate_hz"], 100)
        self.assertFalse(report["output_timestamps_strictly_increasing"])

    def test_non_100_hz_and_irregular_intervals(self):
        for times, intervals, rate in [
            ([0, 5, 10], [5, 5], 200),
            ([0, 25, 50], [25, 25], 40),
            ([0, 8, 20, 40], [8, 12, 20], 1000 / 12),
        ]:
            with self.subTest(times=times):
                ordered, report = preprocess(sample_values(t) for t in times)
                self.assertEqual(report["sample_intervals_ms"], intervals)
                self.assertAlmostEqual(report["estimated_sampling_rate_hz"], rate)
                self.assertEqual([s.timestamp_ms for s in ordered], times)

    def test_empty_singleton_and_all_duplicate_have_no_rate(self):
        for times in ([], [7.5], [7.5, 7.5]):
            with self.subTest(times=times):
                ordered, report = preprocess(sample_values(t) for t in times)
                self.assertEqual(len(ordered), len(times))
                self.assertIsNone(report["estimated_sampling_rate_hz"])
                json.dumps(report, allow_nan=False)

    def test_invalid_data_rejected_with_index(self):
        for bad in (sample_values(timestamp_ms=None), sample_values(ax_mps2="NaN"), sample_values(roll_deg="bad")):
            with self.assertRaisesRegex(IMUValidationError, "sample index 1"):
                preprocess([sample_values(), bad])

    def test_missing_orientation_counted_per_field(self):
        _, report = preprocess([sample_values(0, yaw_deg=0), sample_values(20, roll_deg=0, pitch_deg=2)])
        self.assertEqual(report["missing_optional_orientation_values"], {"roll_deg": 1, "pitch_deg": 1, "yaw_deg": 1})
        self.assertEqual(report["available_orientation_fields"], list(ORIENTATION_FIELDS))
        self.assertEqual(report["missing_required_values"], 0)
        self.assertEqual(report["invalid_numeric_values"], 0)

    def test_high_acceleration_retained_and_reported(self):
        values = sample_values(75261.291037407, ax_mps2=-343.617997363,
                               ay_mps2=252.491982780, az_mps2=-86.278990010)
        original = IMUSample.from_mapping(values)
        ordered, report = preprocess([original])
        self.assertEqual(ordered[0].to_dict(), original.to_dict())
        self.assertAlmostEqual(report["peak_specific_force"]["magnitude_g"], 44.3628817169, places=8)

    def test_invalid_timestamp_arithmetic_reported(self):
        with self.assertRaisesRegex(IMUValidationError, "sample interval exceeds"):
            preprocess([sample_values(-1e308), sample_values(1e308)])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.input = self.directory / "input.csv"
        self.output = self.directory / "normalized.csv"
        self.diagnostics = self.directory / "diagnostics.json"

    def test_independent_scenes_with_overlapping_times_remain_separate(self):
        rows = [sample_values(t, sequence_id="shared", frame_id=frame, offset_ms=t-20)
                for frame, t in [("b", 20), ("a", 0), ("b", 0), ("a", 20)]]
        write_csv(self.input, rows)
        summary = run(self.input, self.output, self.diagnostics)
        self.assertEqual(summary["group_count"], 2)
        self.assertEqual(summary["total_samples"], 4)
        self.assertEqual(summary["duplicate_timestamps_within_groups"], 0)
        self.assertEqual(summary["within_group_interval_counts"], [{"interval_ms": 20, "count": 2}])
        with self.output.open(newline="") as handle:
            result = list(csv.DictReader(handle))
        self.assertEqual([(r["frame_id"], float(r["timestamp_ms"])) for r in result],
                         [("b", 0), ("b", 20), ("a", 0), ("a", 20)])

    def test_metadata_free_input_is_one_sequence(self):
        write_csv(self.input, [sample_values(0), sample_values(40), sample_values(80)])
        summary = run(self.input, self.output, self.diagnostics)
        self.assertEqual(summary["group_count"], 1)
        self.assertEqual(summary["frame_id_count"], 0)
        self.assertEqual(summary["sequence_id_count"], 0)
        self.assertEqual(summary["estimated_sampling_rate_range_hz"], [25, 25])

    def test_conflicting_frame_times_fail_before_outputs(self):
        write_csv(self.input, [sample_values(0, frame_id="f", offset_ms=0),
                              sample_values(10, frame_id="f", offset_ms=0)])
        with self.assertRaisesRegex(ValueError, "inconsistent frame timestamps"):
            run(self.input, self.output, self.diagnostics)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.diagnostics.exists())

    def test_invalid_input_preserves_existing_outputs(self):
        write_csv(self.input, [sample_values(0), sample_values(10, gx_rad_s="bad")])
        self.output.write_text("previous CSV")
        self.diagnostics.write_text("previous report")
        with self.assertRaisesRegex(ValueError, "line 3: gx_rad_s"):
            run(self.input, self.output, self.diagnostics)
        self.assertEqual(self.output.read_text(), "previous CSV")
        self.assertEqual(self.diagnostics.read_text(), "previous report")

    def test_input_and_raw_output_overwrite_protection(self):
        write_csv(self.input, [sample_values()])
        before = self.input.read_bytes()
        for output, diagnostics in [
            (self.input, self.diagnostics), (self.output, self.input),
            (self.output, self.output), (ROOT / "data/raw/imu.csv", self.diagnostics),
            (self.output, ROOT / "data/raw/new_report.json"),
        ]:
            with self.subTest(output=output, diagnostics=diagnostics):
                with self.assertRaises(ValueError):
                    run(self.input, output, diagnostics)
        self.assertEqual(self.input.read_bytes(), before)

    def test_duplicate_samples_remain_in_csv(self):
        write_csv(self.input, [sample_values(0, ax_mps2=1), sample_values(0, ax_mps2=2)])
        summary = run(self.input, self.output, self.diagnostics)
        self.assertEqual(summary["duplicate_timestamps_within_groups"], 1)
        with self.output.open(newline="") as handle:
            result = list(csv.DictReader(handle))
        self.assertEqual([float(r["ax_mps2"]) for r in result], [1, 2])


class SuppliedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = ROOT / "data/raw"
        cls.windows = load_simulator(cls.raw / "imu_windows.csv")

    def test_all_csv_and_json_measurements_agree_without_double_counting(self):
        json_records = load_simulator(self.raw / "imu")
        self.assertEqual(len(json_records), 630)
        self.assertEqual(len(self.windows), 630)
        for csv_record, json_record in zip(self.windows, json_records):
            self.assertEqual(csv_record.frame_id, json_record.frame_id)
            self.assertEqual(csv_record.sequence_id, json_record.sequence_id)
            self.assertAlmostEqual(csv_record.frame_timestamp_ms, json_record.frame_timestamp_ms, delta=5.1e-10)
            for field in SENSOR_FIELDS:
                self.assertAlmostEqual(getattr(csv_record.sample, field), getattr(json_record.sample, field), delta=5.1e-10)

    def test_anchor_csv_equals_window_endpoints(self):
        anchors = load_simulator(self.raw / "imu.csv")
        last_by_frame = {record.frame_id: record for record in self.windows}
        self.assertEqual(len(anchors), 30)
        for record in anchors:
            self.assertEqual(record, last_by_frame[record.frame_id])
            _, report = preprocess([record.sample])
            self.assertIsNone(report["estimated_sampling_rate_hz"])

    def test_real_run_output_values_schema_diagnostics_and_raw_integrity(self):
        def raw_hashes():
            return {str(p.relative_to(self.raw)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in self.raw.rglob("*") if p.is_file()}
        before = raw_hashes()
        self.assertEqual(len(before), 33)
        with tempfile.TemporaryDirectory() as temp:
            output, diagnostic = Path(temp) / "normalized.csv", Path(temp) / "report.json"
            summary = run(self.raw / "imu_windows.csv", output, diagnostic)
            self.assertEqual(json.loads(diagnostic.read_text())["total_samples"], 630)
            self.assertEqual(summary["frame_id_count"], 30)
            self.assertEqual(summary["sequence_id_count"], 30)
            self.assertEqual(summary["group_count"], 30)
            self.assertEqual(summary["samples_per_group_histogram"], {21: 30})
            self.assertEqual(summary["within_group_interval_counts"], [{"interval_ms": 10, "count": 600}])
            self.assertEqual(summary["estimated_sampling_rate_range_hz"], [100, 100])
            self.assertEqual(summary["timestamp_range_ms"], [66736.655324455, 104281.218846329])
            self.assertEqual(summary["duplicate_timestamps_within_groups"], 0)
            self.assertEqual(summary["input_out_of_order_pairs_within_groups"], 0)
            self.assertEqual(summary["missing_required_values"], 0)
            self.assertEqual(summary["invalid_numeric_values"], 0)
            self.assertEqual(summary["missing_optional_orientation_values"], dict.fromkeys(ORIENTATION_FIELDS, 0))
            self.assertEqual(summary["peak_specific_force"]["frame_id"], "frame_0013")
            self.assertAlmostEqual(summary["peak_specific_force"]["magnitude_g"], 44.3628817169, places=8)
            self.assertTrue(all(g["duration_ms"] == 200 for g in summary["groups"]))
            with output.open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, [*METADATA_FIELDS, *SENSOR_FIELDS])
                normalized = list(reader)
            with (self.raw / "imu_windows.csv").open(newline="") as handle:
                original = list(csv.DictReader(handle))
            self.assertEqual(len(normalized), 630)
            self.assertEqual(Counter(r["frame_id"] for r in normalized),
                             Counter({f"frame_{i:04d}": 21 for i in range(1, 31)}))
            for source, clean in zip(original, normalized):
                for field in SENSOR_FIELDS:
                    self.assertEqual(Decimal(source[field]), Decimal(clean[field]), (source["frame_id"], field))
                self.assertEqual(clean["frame_id"], source["frame_id"])
                self.assertEqual(clean["sequence_id"], source["sequence_id"])
                self.assertEqual(Decimal(clean["frame_timestamp_ms"]),
                                 Decimal(source["timestamp_ms"]) - Decimal(source["offset_ms"]))
        self.assertEqual(raw_hashes(), before)


if __name__ == "__main__":
    unittest.main()
