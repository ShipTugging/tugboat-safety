import base64
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_realtime_vision import main
from vision.contract import binary_mask, encode_mask_png, select_instance, timestamp_ms, vision_record
from vision.pipeline import load_model, overlay_frame, prediction, resolve_device, run_video


def decode_mask(encoded):
    data = np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def result_for(shape, confidences=(), classes=(), masks=None, speed=2.5):
    return SimpleNamespace(
        orig_shape=shape,
        boxes=SimpleNamespace(conf=np.array(confidences), cls=np.array(classes)),
        masks=SimpleNamespace(data=masks) if masks is not None else None,
        speed={"inference": speed},
    )


class ContractTests(unittest.TestCase):
    def test_timestamp_uses_zero_based_source_clock(self):
        self.assertEqual(timestamp_ms(25, 25), 1000.0)
        self.assertEqual(timestamp_ms(0, 30), 0.0)
        self.assertAlmostEqual(timestamp_ms(1, 30), 33.333333333333336)
        self.assertAlmostEqual(timestamp_ms(30, 29.97), 1001.001001001001)
        for fps in (0, -25, float("nan"), float("inf")):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                timestamp_ms(0, fps)

    def test_png_base64_round_trip_binary_grayscale_original_size(self):
        mask = np.zeros((720, 1280), np.uint8)
        mask[10:13, 100:900] = 255
        mask[300:330, 1100:1110] = 255
        encoded = encode_mask_png(mask)
        self.assertTrue(base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n"))
        decoded = decode_mask(encoded)
        np.testing.assert_array_equal(decoded, mask)
        self.assertEqual(decoded.dtype, np.uint8)
        self.assertEqual(decoded.shape, (720, 1280))
        self.assertEqual(set(np.unique(decoded)), {0, 255})

    def test_no_detection_json_has_nulls_and_exact_contract(self):
        record = json.loads(json.dumps(vision_record(1200, 1280, 720)))
        self.assertEqual(record, {"timestamp_ms": 1200.0, "vision": {
            "towline_detected": False, "confidence": None, "mask_base64": None,
            "mask_width": 1280, "mask_height": 720,
        }})

    def test_selection_highest_confidence_filters_class_threshold_and_ties(self):
        self.assertEqual(select_instance([0.4, 0.95, 0.8], [0, 0, 0], 0.25), 1)
        self.assertEqual(select_instance([0.99, 0.25, 0.2], [1, 0, 0], 0.25), 1)
        self.assertEqual(select_instance([0.8, 0.8], [0, 0], 0.25), 0)
        self.assertIsNone(select_instance([0.2, 0.9], [0, 1], 0.25))
        self.assertIsNone(select_instance([], [], 0.25))

    def test_binary_threshold_and_invalid_masks(self):
        result = binary_mask([[0, 0.5, 0.6, 1]], 1, 4)
        np.testing.assert_array_equal(result, [[0, 0, 255, 255]])
        self.assertEqual(result.dtype, np.uint8)
        with self.assertRaisesRegex(ValueError, "letterbox"):
            binary_mask(np.zeros((544, 960)), 720, 1280)
        for mask in (np.ones((2, 2), np.uint8), np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), float)):
            with self.subTest(shape=mask.shape), self.assertRaises(ValueError):
                encode_mask_png(mask)

    def test_record_rejects_inconsistent_detection(self):
        mask = np.full((2, 2), 255, np.uint8)
        for confidence, supplied in ((None, mask), (0.5, None), (0.5, np.zeros_like(mask))):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                vision_record(0, 2, 2, confidence, supplied)


class PredictionTests(unittest.TestCase):
    def test_selected_mask_preserves_holes_fragments_and_coordinates(self):
        masks = np.zeros((2, 72, 128), np.uint8)
        masks[0, :, :10] = 1
        masks[1, 20:40, 70:90] = 1
        masks[1, 25:30, 75:80] = 0
        masks[1, 55:57, 110:120] = 1
        score, mask = prediction(result_for((72, 128), [0.4, 0.9], [0, 0], masks), (72, 128), 0.25)
        self.assertEqual(score, 0.9)
        np.testing.assert_array_equal(mask, masks[1] * 255)
        self.assertFalse(mask[:, :10].any())

    def test_missing_misaligned_and_empty_selected_masks_fail(self):
        for masks in (None, np.zeros((1, 36, 64)), np.zeros((2, 72, 128)), np.zeros((1, 72, 128))):
            with self.subTest(shape=None if masks is None else masks.shape), self.assertRaises(ValueError):
                prediction(result_for((72, 128), [0.9], [0], masks), (72, 128), 0.25)

    def test_no_detection_and_below_threshold(self):
        self.assertEqual(prediction(result_for((72, 128)), (72, 128), 0.25), (None, None))
        self.assertEqual(prediction(result_for((72, 128), [0.2], [0]), (72, 128), 0.25), (None, None))

    def test_device_resolution(self):
        self.assertEqual(resolve_device("auto", True, 2), "0")
        self.assertEqual(resolve_device("auto", False, 0), "cpu")
        self.assertEqual(resolve_device("0", False, 0), "cpu")
        self.assertEqual(resolve_device("cpu", True, 2), "cpu")
        self.assertEqual(resolve_device("1", True, 2), "1")
        with self.assertRaises(ValueError):
            resolve_device("2", True, 2)
        with self.assertRaises(ValueError):
            resolve_device("cuda:bad", False, 0)

    def test_mps_resolution_preserves_cpu_cuda_and_auto_policy(self):
        self.assertEqual(resolve_device("mps", False, 0, mps_available=True), "mps")
        self.assertEqual(resolve_device("mps", True, 2, mps_available=True), "mps")
        self.assertEqual(resolve_device("auto", False, 0, mps_available=True), "cpu")
        self.assertEqual(resolve_device("auto", True, 2, mps_available=True), "0")
        self.assertEqual(resolve_device("cpu", True, 2, mps_available=True), "cpu")
        self.assertEqual(resolve_device("0", False, 0, mps_available=True), "cpu")
        self.assertEqual(resolve_device("1", True, 2, mps_available=True), "1")

    def test_mps_unavailable_fails_without_fallback(self):
        for cuda_available, cuda_count in ((False, 0), (True, 2)):
            with self.subTest(cuda=cuda_available), self.assertRaisesRegex(
                ValueError, r"--device mps requested.*is_available\(\) is False"
            ):
                resolve_device("mps", cuda_available, cuda_count, mps_available=False)

    def test_model_loader_checks_mps_availability_without_inference(self):
        for available in (True, False):
            model = SimpleNamespace(task="segment", names={0: "towline"})
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: available)),
            )
            with self.subTest(available=available), patch.dict(os.environ), patch.dict(sys.modules, {
                "torch": fake_torch,
                "ultralytics": SimpleNamespace(YOLO=lambda *args, **kwargs: model),
            }):
                if available:
                    self.assertEqual(load_model(Path("mock-only.pt"), "mps"), (model, "mps"))
                else:
                    with self.assertRaisesRegex(ValueError, r"--device mps requested.*is_available\(\) is False"):
                        load_model(Path("mock-only.pt"), "mps")

    def test_overlay_changes_only_mask_and_hud(self):
        frame = np.full((180, 320, 3), 70, np.uint8)
        original = frame.copy()
        mask = np.zeros((180, 320), np.uint8)
        mask[100:110, 140:220] = 255
        overlay = overlay_frame(frame, vision_record(1000, 320, 180, 0.94, mask), mask)
        np.testing.assert_array_equal(frame, original)
        self.assertGreater(overlay[105, 150, 1], frame[105, 150, 1])
        np.testing.assert_array_equal(overlay[140:, :], frame[140:, :])

    @unittest.skipUnless(importlib.util.find_spec("ultralytics"), "Optional installed-Ultralytics coordinate check")
    def test_real_ultralytics_letterbox_restoration_without_weights(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "YOLO_CONFIG_DIR": tmp, "MPLCONFIGDIR": tmp,
        }):
            import torch
            from ultralytics.models.yolo.segment.predict import SegmentationPredictor

            # Original 128x72 is letterboxed to 640x640: scale 5, y padding 140.
            # This box should restore to x=[32,96), y=[10,50), without padding drift.
            predictor = SimpleNamespace(args=SimpleNamespace(retina_masks=True),
                                        model=SimpleNamespace(names={0: "towline"}))
            result = SegmentationPredictor.construct_result(
                predictor, torch.tensor([[160., 190., 480., 390., .9, 0., 1.]]),
                torch.zeros((1, 3, 640, 640)), np.zeros((72, 128, 3), np.uint8), "mock.png",
                torch.ones((1, 160, 160)),
            )
            _, mask = prediction(result, (72, 128), 0.25)
            expected = np.zeros((72, 128), np.uint8)
            expected[10:50, 32:96] = 255
            np.testing.assert_array_equal(mask, expected)


class VideoSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.video = self.root / "synthetic.mp4"
        self.weights = self.root / "mock-only.pt"
        self.weights.write_bytes(b"Mock loader fixture; never passed to YOLO")
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (1280, 720))
        self.assertTrue(writer.isOpened(), "mp4v writer must be available for smoke test")
        try:
            for i in range(6):
                frame = np.full((720, 1280, 3), 35 + i * 20, np.uint8)
                cv2.line(frame, (100, 600), (1100, 400), (160, 180, 210), 4)
                writer.write(frame)
        finally:
            writer.release()

    def mock_loader(self, all_missing=False, speed=2.5):
        calls = []

        def predict(**kwargs):
            index = len(calls)
            # Store only scalars, not frames, even in the mock.
            calls.append({key: value for key, value in kwargs.items() if key != "source"})
            shape = kwargs["source"].shape[:2]
            if all_missing or index % 2:
                return [result_for(shape, speed=speed)]
            masks = np.zeros((3, *shape), np.uint8)
            masks[0, 400:500, 5:10] = 1
            masks[1, 400:404, 100:1100] = 1
            masks[2, 500:550, 50:100] = 1
            return [result_for(shape, [0.4, 0.9, 0.99], [0, 0, 1], masks, speed=speed)]

        def loader(weights, device):
            self.assertEqual(weights, self.weights)
            return SimpleNamespace(predict=predict), "cpu"

        return loader, calls

    def test_streaming_720p_mp4_jsonl_overlay_summary(self):
        loader, calls = self.mock_loader()
        output = self.root / "output"
        events = []
        real_timestamp = timestamp_ms

        def timed(index, fps):
            # Called before this frame's predict (and once for FPS validation).
            events.append((index, len(calls)))
            return real_timestamp(index, fps)

        with patch("vision.pipeline.timestamp_ms", side_effect=timed):
            summary = run_video(self.video, self.weights, output, device="cpu", model_loader=loader)
        self.assertEqual(events, [(0, 0)] + [(i, i) for i in range(6)])
        records = [json.loads(line) for line in (output / "vision_output.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 6)
        for index, record in enumerate(records):
            self.assertEqual(record["timestamp_ms"], index * 40.0)
            self.assertEqual(record["vision"]["towline_detected"], index % 2 == 0)
            self.assertEqual((record["vision"]["mask_width"], record["vision"]["mask_height"]), (1280, 720))
            if index % 2 == 0:
                mask = decode_mask(record["vision"]["mask_base64"])
                expected = np.zeros((720, 1280), np.uint8)
                expected[400:404, 100:1100] = 255
                np.testing.assert_array_equal(mask, expected)
                self.assertEqual(record["vision"]["confidence"], 0.9)
            else:
                self.assertIsNone(record["vision"]["mask_base64"])
                self.assertIsNone(record["vision"]["confidence"])
        for call in calls:
            self.assertEqual(call, dict(conf=0.25, imgsz=960, device="cpu", classes=[0],
                                       retina_masks=True, save=False, verbose=False))
        overlay = cv2.VideoCapture(str(output / "vision_overlay.mp4"))
        self.assertTrue(overlay.isOpened())
        count = 0
        try:
            self.assertEqual(overlay.get(cv2.CAP_PROP_FPS), 25)
            while True:
                ok, frame = overlay.read()
                if not ok:
                    break
                self.assertEqual(frame.shape, (720, 1280, 3))
                count += 1
        finally:
            overlay.release()
        self.assertEqual(count, 6)
        self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
        self.assertEqual(summary["total_frames_processed"], 6)
        self.assertEqual(summary["detected_frames"], 3)
        self.assertEqual(summary["detected_ratio"], 0.5)
        self.assertEqual(summary["source_fps"], 25)
        self.assertEqual(summary["video_duration_sec"], 6 / 25)
        self.assertAlmostEqual(summary["mean_confidence_detected"], 0.9)
        self.assertEqual(summary["mean_inference_ms"], 2.5)
        self.assertEqual(summary["inference_timing_fallback_frames"], 0)
        self.assertAlmostEqual(summary["effective_processing_fps"], 6 / summary["total_elapsed_sec"])
        with self.assertRaisesRegex(ValueError, "new or empty"):
            run_video(self.video, self.weights, output, model_loader=loader)

    def test_no_detection_summary_and_inference_timing_fallback(self):
        loader, _ = self.mock_loader(all_missing=True, speed=None)
        summary = run_video(self.video, self.weights, self.root / "none", model_loader=loader)
        self.assertEqual(summary["detected_frames"], 0)
        self.assertEqual(summary["detected_ratio"], 0)
        self.assertIsNone(summary["mean_confidence_detected"])
        self.assertEqual(summary["inference_timing_fallback_frames"], 6)
        self.assertGreater(summary["mean_inference_ms"], 0)

    def test_missing_weights_rejected_before_output_or_loader(self):
        with self.assertRaisesRegex(ValueError, "weights do not exist"):
            run_video(self.video, self.root / "missing.pt", self.root / "missing")
        self.assertFalse((self.root / "missing").exists())

    def test_mid_run_failure_has_no_success_summary(self):
        def broken(**kwargs):
            raise RuntimeError("mock inference failure")

        output = self.root / "failed"
        with self.assertRaisesRegex(RuntimeError, "mock inference failure"):
            run_video(self.video, self.weights, output,
                      model_loader=lambda *args: (SimpleNamespace(predict=broken), "cpu"))
        self.assertFalse((output / "summary.json").exists())

    def test_cli_mps_parsing_reaches_loader_and_predictor(self):
        mock_loader, calls = self.mock_loader()

        def mps_loader(weights, requested):
            self.assertEqual(requested, "mps")
            model, _ = mock_loader(weights, requested)
            return model, resolve_device(requested, False, 0, mps_available=True)

        stdout = io.StringIO()
        with patch("vision.pipeline.load_model", mps_loader), contextlib.redirect_stdout(stdout):
            status = main(["--video", str(self.video), "--weights", str(self.weights),
                           "--output-dir", str(self.root / "mps-cli"), "--device", "mps"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout.getvalue())["device_used"], "mps")
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(call["device"] == "mps" for call in calls))

    def test_cli_stdout_is_summary_and_bad_input_exits_nonzero(self):
        loader, _ = self.mock_loader()
        stdout = io.StringIO()
        with patch("vision.pipeline.load_model", loader), contextlib.redirect_stdout(stdout):
            status = main(["--video", str(self.video), "--weights", str(self.weights),
                           "--output-dir", str(self.root / "cli"), "--device", "cpu"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout.getvalue())["total_frames_processed"], 6)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = main(["--video", str(self.root / "absent.mp4"), "--weights", str(self.weights),
                           "--output-dir", str(self.root / "bad-cli")])
        self.assertEqual(status, 1)
        self.assertIn("does not exist", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
