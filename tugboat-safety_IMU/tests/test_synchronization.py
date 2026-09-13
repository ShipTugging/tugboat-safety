"""Small mock Vision inputs test alignment only; features are never calculated."""

import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from run_sync import CSV_FIELDS, ROOT, flatten, load_csv, run
from run_pipeline import run as run_imu_pipeline
from src.imu_schema import IMUSample, REQUIRED_FIELDS, SENSOR_FIELDS
from src.preprocessing import preprocess
from src.simulator_adapter import load_simulator
from src.synchronization import (
    Association, IMURecord, NearestTimestampMatcher, VisionSample, synchronize,
)


def imu(timestamp, **changes):
    values = dict(zip(REQUIRED_FIELDS, [timestamp, 0.12, 9.71, -0.05, 0.021, -0.004, 0.031]))
    values.update(changes)
    return values


def vision(timestamp, **changes):
    values = {"timestamp_ms": timestamp, "towline_detected": True, "confidence": 0.94}
    values.update(changes)
    return values


def mock_inputs():
    """Seven mock feature rows and four irregular IMU measurements, no risk data."""
    features = [vision(t, sequence_id="scene_a", frame_id=f"camera_{i}")
                for i, t in enumerate((0, 7, 17.5, 45, 80, 81))]
    features.append(vision(25, sequence_id="scene_b", towline_detected=False, confidence=0.1))
    measurements = [imu(t, sequence_id="scene_a", frame_id="offline_debug")
                    for t in (0, 10, 25, 60)]
    return features, measurements


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class MatchingTests(unittest.TestCase):
    def test_exact_fractional_timestamp_and_unchanged_sensor_values(self):
        sample = IMUSample.from_mapping(imu(1234.125, roll_deg=3.4, pitch_deg=0.8, yaw_deg=42.1))
        record = NearestTimestampMatcher([sample]).match(vision(1234.125))
        self.assertTrue(record.matched)
        self.assertEqual(record.time_delta_ms, 0)
        self.assertIs(record.imu.sample, sample)
        self.assertEqual(record.to_dict()["imu"], sample.to_dict())

    def test_nearest_sample_and_signed_delta_in_both_directions(self):
        matcher = NearestTimestampMatcher([imu(1230), imu(1240)])
        for timestamp, selected, delta in [(1234.5, 1230, -4.5), (1237, 1240, 3)]:
            with self.subTest(timestamp=timestamp):
                record = matcher.match(vision(timestamp))
                self.assertEqual(record.imu.sample.timestamp_ms, selected)
                self.assertEqual(record.time_delta_ms, delta)

    def test_equal_distance_selects_earlier_without_interpolation(self):
        record = NearestTimestampMatcher([imu(20, ax_mps2=9), imu(0, ax_mps2=1)]).match(vision(10))
        self.assertEqual(record.imu.sample.timestamp_ms, 0)
        self.assertEqual(record.imu.sample.ax_mps2, 1)
        self.assertEqual(record.time_delta_ms, -10)

    def test_duplicate_timestamp_selects_first_input_on_either_side(self):
        measurements = [imu(20, ax_mps2=2), imu(0, ax_mps2=1),
                        imu(20, ax_mps2=200), imu(0, ax_mps2=100)]
        matcher = NearestTimestampMatcher(measurements)
        for timestamp, expected in [(-1, 1), (0, 1), (1, 1), (10, 1), (19, 2), (20, 2), (21, 2)]:
            with self.subTest(timestamp=timestamp):
                self.assertEqual(matcher.match(vision(timestamp)).imu.sample.ax_mps2, expected)
        self.assertEqual([row["ax_mps2"] for row in measurements], [2, 1, 200, 100])

    def test_gap_is_configurable_inclusive_and_zero_allows_exact_only(self):
        for gap in (0, 5.5, 20, 100):
            matcher = NearestTimestampMatcher([imu(0)], max_time_gap_ms=gap)
            self.assertTrue(matcher.match(vision(gap)).matched)
            self.assertTrue(matcher.match(vision(-gap)).matched)
            self.assertFalse(matcher.match(vision(math.nextafter(gap, math.inf))).matched)
        self.assertFalse(NearestTimestampMatcher([imu(0)]).match(vision(21)).matched)
        self.assertTrue(NearestTimestampMatcher([imu(0)], max_time_gap_ms=21).match(vision(21)).matched)

    def test_unmatched_preserves_all_vision_features_and_metadata(self):
        features = VisionSample.from_mapping(vision(
            90, towline_detected=False, confidence=0.1, towline_angle_deg=42.1,
            sag_ratio=0.051, curvature=0.17, session_id="s", frame_id="camera_9",
        ))
        record = NearestTimestampMatcher([imu(0, session_id="s")]).match(features)
        self.assertIs(record.vision, features)
        self.assertFalse(record.matched)
        self.assertIsNone(record.imu)
        self.assertIsNone(record.time_delta_ms)
        self.assertEqual(record.reason, "max_time_gap_exceeded")
        self.assertEqual(record.nearest_time_delta_ms, -90)
        self.assertEqual(record.to_dict()["vision"], features.features())
        self.assertEqual(record.to_dict()["metadata"]["vision"], asdict(features.metadata))

    def test_non_100_hz_input_and_10_hz_vision(self):
        for interval in (5, 25, 40):  # 200, 40, and 25 Hz; no declared rate.
            with self.subTest(interval=interval):
                samples = [imu(t) for t in range(0, 401, interval)]
                records, _ = synchronize([vision(t) for t in range(0, 401, 100)], samples)
                self.assertTrue(all(record.matched for record in records))
                self.assertEqual(len(records), 5)
                self.assertTrue(all(abs(record.time_delta_ms) <= interval / 2 for record in records))

    def test_irregular_imu_and_unsorted_vision_order_are_preserved(self):
        features = [vision(t) for t in (38, 1, 18, 18, 90)]
        samples = [imu(t) for t in (40, 0, 21, 8)]
        records, _ = synchronize(iter(features), iter(samples))
        self.assertEqual([record.vision.timestamp_ms for record in records], [38, 1, 18, 18, 90])
        self.assertEqual([record.imu.sample.timestamp_ms if record.matched else None for record in records],
                         [40, 0, 21, 21, None])
        sorted_records, _ = synchronize(features, sorted(samples, key=lambda row: row["timestamp_ms"]))
        self.assertEqual(records, sorted_records)
        self.assertEqual([row["timestamp_ms"] for row in samples], [40, 0, 21, 8])

    def test_single_match_calls_do_not_consume_imu_samples(self):
        sample = IMUSample.from_mapping(imu(10))
        matcher = NearestTimestampMatcher([sample])
        records = [matcher.match(vision(t)) for t in (9, 11, 9)]
        self.assertTrue(all(record.imu.sample is sample for record in records))
        self.assertEqual(records[0], records[2])

    def test_optional_orientation_and_nullable_geometry(self):
        for values in ({}, {"roll_deg": "", "pitch_deg": None, "yaw_deg": "12.5"},
                       {"roll_deg": 1, "pitch_deg": 2, "yaw_deg": 3}):
            with self.subTest(values=values):
                record = NearestTimestampMatcher([imu(0, **values)]).match(vision(
                    0, towline_angle_deg=None, sag_ratio="", curvature=None,
                ))
                self.assertEqual(record.imu.sample, IMUSample.from_mapping(imu(0, **values)))
                self.assertTrue(all(record.vision.features()[key] is None
                                    for key in ("towline_angle_deg", "sag_ratio", "curvature")))

    def test_existing_preprocessing_result_can_be_used_directly(self):
        samples, before = preprocess([imu(30), imu(0), imu(15)])
        records, _ = synchronize([vision(14)], samples)
        self.assertIs(records[0].imu.sample, samples[1])
        _, after = preprocess(samples)
        self.assertEqual(before["sample_intervals_ms"], after["sample_intervals_ms"])

    def test_empty_imu_preserves_vision_and_empty_vision_has_undefined_rate(self):
        records, report = synchronize([vision(0), vision(1)], [])
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.reason == "no_imu_samples" for record in records))
        self.assertEqual(report["match_rate"], 0)
        self.assertIsNone(report["mean_absolute_time_gap_ms"])
        for samples in ([], [imu(0)]):
            records, report = synchronize([], samples)
            self.assertEqual(records, [])
            self.assertEqual(report["total_vision_samples"], 0)
            self.assertIsNone(report["match_rate"])
            self.assertIsNone(report["median_absolute_time_gap_ms"])
            self.assertIsNone(report["maximum_absolute_time_gap_ms"])
            json.dumps(report, allow_nan=False)

    def test_mock_diagnostics_use_accepted_gaps_only(self):
        records, report = synchronize(*mock_inputs())
        self.assertEqual([record.time_delta_ms for record in records], [0, 3, -7.5, 15, -20, None, None])
        self.assertEqual(report, {
            "total_vision_samples": 7, "matched_samples": 5, "unmatched_samples": 2,
            "match_rate": 5 / 7, "mean_absolute_time_gap_ms": 9.1,
            "median_absolute_time_gap_ms": 7.5, "maximum_absolute_time_gap_ms": 20,
            "max_time_gap_ms": 20,
            "unmatched_reasons": {"max_time_gap_exceeded": 1, "metadata_mismatch": 1},
        })
        json.dumps([record.to_dict() for record in records], allow_nan=False)

    def test_even_median_and_large_finite_gap_statistics(self):
        _, report = synchronize([vision(1), vision(3)], [imu(0)])
        self.assertEqual(report["median_absolute_time_gap_ms"], 2)
        _, report = synchronize([vision(1e308), vision(1e308)], [imu(0)], max_time_gap_ms=1e308)
        self.assertEqual(report["mean_absolute_time_gap_ms"], 1e308)
        self.assertEqual(report["median_absolute_time_gap_ms"], 1e308)
        json.dumps(report, allow_nan=False)
        tiny = math.nextafter(0, math.inf)
        _, report = synchronize([vision(tiny), vision(tiny)], [imu(0)])
        self.assertEqual(report["median_absolute_time_gap_ms"], tiny)

    def test_search_agrees_with_brute_force_over_unsorted_duplicate_times(self):
        rng = random.Random(1043)
        measurements = [imu(rng.randrange(-100, 101), ax_mps2=i) for i in range(100)]
        matcher = NearestTimestampMatcher(measurements, max_time_gap_ms=12)
        for timestamp in [rng.randrange(-140, 141) / 2 for _ in range(100)]:
            expected_index, expected = min(enumerate(measurements), key=lambda pair: (
                abs(pair[1]["timestamp_ms"] - timestamp), pair[1]["timestamp_ms"], pair[0],
            ))
            record = matcher.match(vision(timestamp))
            delta = expected["timestamp_ms"] - timestamp
            self.assertEqual(record.matched, abs(delta) <= 12)
            if record.matched:
                self.assertEqual(record.imu.sample.ax_mps2, expected_index)
                self.assertEqual(record.time_delta_ms, delta)


class BoundaryTests(unittest.TestCase):
    def test_session_is_primary_even_when_sequence_and_frame_are_identical(self):
        records, _ = synchronize([vision(10, session_id="b", sequence_id="same", frame_id="same")], [
            imu(10, session_id="a", sequence_id="same", frame_id="same", ax_mps2=1),
            imu(12, session_id="b", sequence_id="same", frame_id="different", ax_mps2=2),
        ])
        self.assertEqual(records[0].imu.sample.ax_mps2, 2)
        self.assertEqual(records[0].time_delta_ms, 2)

    def test_sequence_isolates_scenes_with_overlapping_times_in_one_session(self):
        matcher = NearestTimestampMatcher([
            imu(10, session_id="s", sequence_id="scene_a", ax_mps2=1),
            imu(15, session_id="s", sequence_id="scene_b", ax_mps2=2),
        ])
        record = matcher.match(vision(10, session_id="s", sequence_id="scene_b"))
        self.assertEqual(record.imu.sample.ax_mps2, 2)
        self.assertEqual(record.time_delta_ms, 5)

    def test_distinct_camera_frames_match_in_a_continuous_session(self):
        matcher = NearestTimestampMatcher([
            imu(0, session_id="live", frame_id="imu_debug_1"),
            imu(10, session_id="live", frame_id="imu_debug_2"),
        ])
        for timestamp, camera_frame in [(0, "camera_20"), (9, "camera_21")]:
            record = matcher.match(vision(timestamp, session_id="live", frame_id=camera_frame))
            self.assertTrue(record.matched)
            self.assertEqual(record.vision.metadata.frame_id, camera_frame)
            self.assertNotEqual(record.imu.metadata.frame_id, camera_frame)
        self.assertTrue(matcher.match(vision(10, session_id="live")).matched)

    def test_frame_only_ids_are_debug_metadata_not_groups_or_fallback_clocks(self):
        record = NearestTimestampMatcher([
            imu(0, frame_id="other", frame_timestamp_ms=1000),
            imu(100, frame_id="same", frame_timestamp_ms=5),
        ]).match(vision(5, frame_id="same", frame_timestamp_ms=1000))
        self.assertEqual(record.imu.sample.timestamp_ms, 0)
        self.assertEqual(record.time_delta_ms, -5)
        self.assertEqual(record.imu.metadata.frame_id, "other")

    def test_mismatched_ids_do_not_fall_back_to_other_groups(self):
        for query, candidate in [
            ({"session_id": "a"}, {"session_id": "b"}),
            ({"sequence_id": "a"}, {"sequence_id": "b"}),
            ({"session_id": "s", "sequence_id": "a"}, {"session_id": "s", "sequence_id": "b"}),
        ]:
            with self.subTest(query=query):
                record = NearestTimestampMatcher([imu(0, **candidate)]).match(vision(0, **query))
                self.assertEqual(record.reason, "metadata_mismatch")
                self.assertIsNone(record.imu)

    def test_missing_ids_with_multiple_compatible_groups_are_ambiguous(self):
        cases = [
            ({}, [{"session_id": "a"}, {"session_id": "b"}]),
            ({"session_id": "s"}, [{"session_id": "s", "sequence_id": "a"},
                                   {"session_id": "s", "sequence_id": "b"}]),
            ({"sequence_id": "same"}, [{"session_id": "a", "sequence_id": "same"},
                                        {"session_id": "b", "sequence_id": "same"}]),
            ({"session_id": "a"}, [{"session_id": "a"}, {}]),
        ]
        for query, candidates in cases:
            with self.subTest(query=query, candidates=candidates):
                matcher = NearestTimestampMatcher([imu(i * 1000, **ids) for i, ids in enumerate(candidates)])
                # Even an exact time cannot choose between unidentified groups.
                record = matcher.match(vision(0, **query))
                self.assertEqual(record.reason, "ambiguous_metadata")
                self.assertIsNone(record.imu)

    def test_partial_metadata_can_resolve_one_compatible_group(self):
        cases = [
            ({}, {"session_id": "only"}),
            ({"session_id": "only"}, {}),
            ({"session_id": "s"}, {"session_id": "s", "sequence_id": "only"}),
        ]
        for query, candidate in cases:
            with self.subTest(query=query):
                record = NearestTimestampMatcher([imu(0, **candidate)]).match(vision(0, **query))
                self.assertTrue(record.matched)
                self.assertEqual(record.vision.metadata, Association(**query))
                self.assertEqual(record.imu.metadata, Association(**candidate))

    def test_compatible_group_gap_exceeded_does_not_match_other_session(self):
        record = NearestTimestampMatcher([
            imu(0, session_id="a"), imu(100, session_id="b"),
        ]).match(vision(100, session_id="a"))
        self.assertEqual(record.reason, "max_time_gap_exceeded")
        self.assertEqual(record.nearest_time_delta_ms, -100)


class ValidationTests(unittest.TestCase):
    def test_invalid_max_gap_rejected_even_for_empty_inputs(self):
        for value in (-1, None, True, "", "bad", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_time_gap_ms"):
                synchronize([], [], max_time_gap_ms=value)

    def test_invalid_vision_is_not_silently_dropped(self):
        for field, value in [("timestamp_ms", "NaN"), ("timestamp_ms", True),
                             ("confidence", -0.1), ("confidence", 1.1), ("confidence", None),
                             ("towline_detected", None), ("towline_detected", "maybe"),
                             ("towline_angle_deg", "nan"), ("sag_ratio", "bad"),
                             ("curvature", float("inf")), ("session_id", 1),
                             ("sequence_id", "  "), ("frame_id", []),
                             ("frame_timestamp_ms", "nan")]:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, f"Vision sample index 1: {field}"):
                    synchronize([vision(0), vision(1, **{field: value})], [imu(0)])

    def test_required_vision_and_imu_fields_reuse_validation(self):
        for name in ("timestamp_ms", "towline_detected", "confidence"):
            values = vision(0)
            del values[name]
            with self.assertRaisesRegex(ValueError, name):
                synchronize([values], [])
        for name in SENSOR_FIELDS:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"IMU sample index 1: {name}"):
                synchronize([], [imu(0), imu(1, **{name: "bad"})])

    def test_csv_boolean_strings_and_numeric_strings(self):
        for token, expected in [("true", True), (" FALSE ", False), ("1", True), ("0", False)]:
            sample = VisionSample.from_mapping(vision("1.25", confidence="0.94", towline_detected=token))
            self.assertIs(sample.towline_detected, expected)
            self.assertEqual(sample.timestamp_ms, 1.25)

    def test_unrelated_input_fields_are_excluded(self):
        record = NearestTimestampMatcher([imu(0, unused="not a sensor")]).match(vision(0, unused="not a feature"))
        self.assertNotIn("unused", json.dumps(record.to_dict()))
        self.assertEqual(set(record.to_dict()["imu"]), set(SENSOR_FIELDS))

    def test_invalid_types_and_overflow_have_explicit_errors(self):
        with self.assertRaisesRegex(ValueError, "IMU sample index 0: sample"):
            synchronize([], [None])
        with self.assertRaisesRegex(ValueError, "Vision sample index 0: sample"):
            synchronize([None], [])
        with self.assertRaisesRegex(ValueError, "time delta exceeds finite"):
            synchronize([vision(1e308)], [imu(-1e308)])


class CSVTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.vision_path = self.directory / "vision.csv"
        self.imu_path = self.directory / "imu.csv"
        self.output = self.directory / "sync.csv"
        self.diagnostics = self.directory / "diagnostics.json"
        features, measurements = mock_inputs()
        write_csv(self.vision_path, features)
        write_csv(self.imu_path, measurements)

    def run_sync(self, **kwargs):
        return run(self.vision_path, self.imu_path, self.output, self.diagnostics, **kwargs)

    def test_csv_output_schema_values_metadata_and_diagnostics(self):
        before = (self.vision_path.read_bytes(), self.imu_path.read_bytes())
        report = self.run_sync()
        self.assertEqual(json.loads(self.diagnostics.read_text()), report)
        with self.output.open(newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(tuple(reader.fieldnames), CSV_FIELDS)
            rows = list(reader)
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[2]["sync_time_delta_ms"], "-7.5")
        self.assertEqual(rows[2]["vision_frame_id"], "camera_2")
        self.assertEqual(rows[2]["imu_frame_id"], "offline_debug")
        self.assertEqual(rows[2]["vision_confidence"], "0.94")
        self.assertEqual(rows[2]["roll_deg"], "")
        self.assertEqual(rows[2]["curvature"], "")
        self.assertEqual(rows[5]["sync_matched"], "False")
        self.assertEqual(rows[5]["sync_reason"], "max_time_gap_exceeded")
        self.assertEqual(rows[5]["sync_nearest_time_delta_ms"], "-21.0")
        self.assertEqual(rows[5]["imu_timestamp_ms"], "")
        self.assertTrue(all(rows[5][name] == "" for name in SENSOR_FIELDS[1:]))
        self.assertEqual(rows[6]["towline_detected"], "False")
        self.assertEqual((self.vision_path.read_bytes(), self.imu_path.read_bytes()), before)

    def test_csv_preserves_frame_time_and_session_without_using_them_as_sensor_fields(self):
        write_csv(self.imu_path, [imu(10, session_id="s", frame_timestamp_ms=999)])
        records = load_csv(self.imu_path, kind="imu")
        self.assertEqual(records[0].metadata.frame_timestamp_ms, 999)
        self.assertEqual(records[0].metadata.session_id, "s")
        self.assertEqual(set(records[0].sample.to_dict()), set(SENSOR_FIELDS))

    def test_header_only_inputs_produce_valid_empty_or_unmatched_output(self):
        write_csv(self.imu_path, [], fields=REQUIRED_FIELDS)
        report = self.run_sync()
        self.assertEqual(report["unmatched_reasons"], {"no_imu_samples": 7})
        write_csv(self.vision_path, [], fields=("timestamp_ms", "towline_detected", "confidence"))
        report = self.run_sync()
        self.assertIsNone(report["match_rate"])
        with self.output.open(newline="") as handle:
            self.assertEqual(list(csv.DictReader(handle)), [])

    def test_invalid_input_or_gap_preserves_existing_outputs(self):
        self.output.write_text("previous CSV")
        self.diagnostics.write_text("previous diagnostics")
        for gap in (-1, float("nan")):
            with self.assertRaises(ValueError):
                self.run_sync(max_time_gap_ms=gap)
        write_csv(self.vision_path, [vision(0), vision(1, confidence="bad")])
        with self.assertRaisesRegex(ValueError, r"vision.csv: line 3: confidence"):
            self.run_sync()
        self.assertEqual(self.output.read_text(), "previous CSV")
        self.assertEqual(self.diagnostics.read_text(), "previous diagnostics")

    def test_malformed_csv_has_source_context(self):
        cases = [
            ("timestamp_ms,confidence\n", "missing required columns: towline_detected"),
            ("timestamp_ms,towline_detected,confidence,confidence\n", "duplicate CSV column"),
            ("timestamp_ms,towline_detected,confidence\n0,true,0.9,extra\n", "line 2: CSV row"),
        ]
        for text, message in cases:
            self.vision_path.write_text(text)
            with self.assertRaisesRegex(ValueError, message):
                self.run_sync()
        self.assertFalse(self.output.exists())

    def test_input_raw_and_output_collision_protection(self):
        before = (self.vision_path.read_bytes(), self.imu_path.read_bytes())
        for output, diagnostics in [
            (self.vision_path, self.diagnostics), (self.output, self.imu_path),
            (self.output, self.output), (self.output, self.output / "child.json"),
            (ROOT / "data/raw/sync.csv", self.diagnostics),
        ]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                run(self.vision_path, self.imu_path, output, diagnostics)
        self.assertEqual((self.vision_path.read_bytes(), self.imu_path.read_bytes()), before)

    def test_input_alias_protection(self):
        for alias_type in ("symlink", "hardlink"):
            alias = self.directory / alias_type
            if alias_type == "symlink":
                alias.symlink_to(self.imu_path)
            else:
                alias.hardlink_to(self.imu_path)
            with self.assertRaisesRegex(ValueError, "overwrite inputs"):
                run(self.vision_path, self.imu_path, alias, self.diagnostics)

    def test_cli_works_from_another_directory_and_honors_gap(self):
        result = subprocess.run([
            sys.executable, "-B", str(ROOT / "run_sync.py"),
            "--vision", str(self.vision_path), "--imu", str(self.imu_path),
            "--output", str(self.output), "--diagnostics", str(self.diagnostics),
            "--max-time-gap-ms", "0",
        ], cwd=self.directory, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.diagnostics.read_text())
        self.assertEqual(report["matched_samples"], 1)
        self.assertEqual(report["max_time_gap_ms"], 0)

    def test_cli_invalid_input_exits_nonzero(self):
        result = subprocess.run([
            sys.executable, "-B", str(ROOT / "run_sync.py"), "--vision", str(self.directory / "absent.csv"),
            "--imu", str(self.imu_path), "--output", str(self.output),
            "--diagnostics", str(self.diagnostics),
        ], cwd=self.directory, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Synchronization error", result.stderr)
        self.assertFalse(self.output.exists())


class SuppliedDatasetSyncTests(unittest.TestCase):
    def test_existing_normalized_dataset_uses_scene_sequences_and_is_unchanged(self):
        # Regenerate through the unchanged Foundation runner in a temporary
        # directory; tests need raw fixtures, not pre-existing derived outputs.
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "normalized.csv"
        run_imu_pipeline(ROOT / "data/raw/imu_windows.csv", path, Path(temp.name) / "imu_report.json")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        records = load_csv(path, kind="imu")
        self.assertEqual(len(records), 630)
        scenes = {}
        for record in records:
            self.assertIsNotNone(record.metadata.sequence_id)
            scenes.setdefault(record.metadata.sequence_id, set()).add(record.metadata.frame_id)
        self.assertEqual(len(scenes), 30)
        self.assertTrue(all(len(frames) == 1 for frames in scenes.values()))
        # Only three mock Vision samples, with unrelated camera frame identifiers.
        features = []
        expected = []
        for sequence_id in list(scenes)[:3]:
            group = [r for r in records if r.metadata.sequence_id == sequence_id]
            anchor = group[0].metadata.frame_timestamp_ms
            features.append(vision(anchor, sequence_id=sequence_id, frame_id="live_camera_debug"))
            expected.append(next(r for r in group if r.sample.timestamp_ms == anchor))
        synced, report = synchronize(features, records)
        self.assertEqual([r.imu for r in synced], expected)
        self.assertEqual(report["matched_samples"], 3)
        self.assertEqual(report["maximum_absolute_time_gap_ms"], 0)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_offline_sample_can_be_wrapped_without_changing_foundation(self):
        source = load_simulator(ROOT / "data/raw/imu.csv")[0]
        wrapped = IMURecord(source.sample, Association(
            sequence_id=source.sequence_id, frame_id=source.frame_id,
            frame_timestamp_ms=source.frame_timestamp_ms,
        ))
        record = NearestTimestampMatcher([wrapped]).match(vision(
            source.sample.timestamp_ms, sequence_id=source.sequence_id, frame_id="different",
        ))
        self.assertIs(record.imu.sample, source.sample)
        self.assertEqual(flatten(record)["imu_frame_timestamp_ms"], source.frame_timestamp_ms)


if __name__ == "__main__":
    unittest.main()
