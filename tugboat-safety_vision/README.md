# Real-time Vision V1

Local MP4 → sequential YOLO-Seg inference → per-frame Vision JSONL, debug overlay,
and run summary. This is an offline video prototype: processing speed depends on
hardware and is not guaranteed to match playback FPS.

**Run**

From the workspace root, use the existing model environment:

```sh
cd tugboat-safety_vision
../tugboat-safety_model/.venv/bin/python run_realtime_vision.py \
  --video /path/to/input.mp4 \
  --weights /path/to/best.pt \
  --output-dir ./outputs/run-001 \
  --conf 0.25 --imgsz 960 --device 0
```

Alternatively create a virtual environment here and install `requirements.txt`:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_realtime_vision.py \
  --video /path/to/input.mp4 --weights /path/to/best.pt \
  --output-dir ./outputs/run-001 --conf 0.25 --imgsz 960 --device cpu
```

The pinned inference packages match the existing model pipeline (validated with
Python 3.13). Paths above are placeholders: supply your actual MP4 and trained
YOLO26n-seg checkpoint. The local `.pt` must exist and declare segmentation with
exactly `{0: 'towline'}`. The runner does not download substitute weights.
`--device auto` (default) chooses CUDA 0 when available, otherwise CPU. `--device 0`
also falls back to CPU with a stderr message if CUDA is unavailable; an invalid
index on a CUDA machine is an error. `--device cpu` always uses CPU.

Use a new or empty output directory. A successful run creates:

| File | Purpose |
| --- | --- |
| `vision_output.jsonl` | One Vision object per decoded frame, in source order. |
| `vision_overlay.mp4` | Translucent green selected mask and timestamp/detection/confidence HUD, at source FPS. |
| `summary.json` | Source properties, detection totals, mean confidence and measured processing performance; also printed to stdout. |

**Contract and coordinates**

```json
{"timestamp_ms":1200.0,"vision":{"towline_detected":false,"confidence":null,"mask_base64":null,"mask_width":1280,"mask_height":720}}
```

Detected frames have `towline_detected: true`, the selected instance confidence,
and a Base64-encoded grayscale PNG in `mask_base64`. Selection uses the highest
confidence class-0 segmentation instance at or above `--conf`; ties keep the
first. Other instances are not merged. Below-threshold results count as absent.

Masks are **original frame height × width, uint8, one channel, values 0/255**.
`retina_masks=True` asks Ultralytics to remove letterbox padding and restore native
coordinates. The runner checks the dimensions and uses those raster masks
directly, preserving disconnected fragments and holes. It never directly
resizes a padded internal mask or bridges fragments through polygon conversion.
A qualifying detection with a missing, empty, or wrong-size mask fails explicitly.

Timestamp is calculated immediately after reading each frame, before inference:
`timestamp_ms = frame_index / source_fps * 1000`, starting at index 0. For 25 FPS,
frame 25 is 1000 ms. It is relative to the start of this MP4, not a wall clock or
an inference completion time. For variable-frame-rate files this deliberately
uses nominal FPS as specified in V1, not individual presentation timestamps.

No detection means **Vision geometry unavailable**. It makes no safety claim.
Small, thin, or low-visibility towlines may be missed. V1 outputs no angle, sag,
curvature, tension, risk, IMU synchronization, API, WebSocket, or Three.js features.

**Summary interpretation and first checks**

- `source_fps` is playback cadence. `video_duration_sec` is processed frames / FPS.
- `mean_inference_ms` uses Ultralytics `result.speed['inference']`, excluding
  preprocessing and postprocessing. If unavailable, prediction-call wall time is
  substituted and counted in `inference_timing_fallback_frames`. Check
  `inference_timing_source` before comparing timings.
- `mean_predict_call_ms` includes prediction preprocessing/postprocessing and
  first-call setup. `total_elapsed_sec` includes model loading, capture, all frame
  processing, JSONL writing and video finalization (excluding summary writing).
  `effective_processing_fps` is processed frames / that elapsed time.
- `mean_confidence_detected` is null if no frames have detections; otherwise it
  averages only the selected detected instances. No per-frame history is held
  in memory; masks/frames are processed one at a time.
- First watch the overlay for mask alignment, fragmentation and missed thin
  ropes. Check source dimensions, frame count and duration against the input;
  JSONL should have exactly that many lines and start at timestamp 0. Decode a
  few masks and confirm original dimensions and binary values. Then check
  detection ratio and `device_used`, before considering processing speed.

MP4 uses OpenCV `mp4v`. For odd-size inputs only the overlay receives up to one
black pixel of padding at the right/bottom; JSON mask dimensions remain exact.
Summary includes `overlay_width`/`overlay_height`. Missing codec, invalid FPS,
inconsistent frame dimensions, no decodable frames or a decoded/container frame
count mismatch are errors. On failure the CLI exits nonzero and writes no success
summary; partial JSONL/video may remain for diagnosis. Do not consume them as a
completed run. OpenCV cannot distinguish every decoder failure from normal EOF;
the container frame count check detects premature EOF when that count is known.

**Tests (no trained weights or user video required)**

```sh
../tugboat-safety_model/.venv/bin/python -B -m unittest discover -s tests -v
```

Tests cover timestamp order, PNG/Base64 round-trip, exact no-detection JSON,
instance selection, original coordinates/binary masks, CPU fallback and errors.
The smoke test creates a six-frame 1280×720 MP4 at 25 FPS, injects a mock predictor,
then decodes every JSON mask and all overlay frames and checks the summary.
It does not run actual YOLO inference. When Ultralytics is installed, an additional
test exercises its real letterbox-to-original mask postprocessing with synthetic
tensors (no checkpoint). All smoke files are temporary and cleaned up.

**Files**

- `run_realtime_vision.py`: CLI and stdout/error handling.
- `vision/contract.py`: pure timestamp, selection, binary mask, PNG/Base64 and JSON helpers.
- `vision/pipeline.py`: model/device loading, streaming inference, overlay and summary.
- `tests/test_realtime_vision.py`: unit, coordinate and MP4 smoke tests.
- `requirements.txt`: inference dependencies; `.gitignore`: local outputs/media/weights.

The existing dataset, model and IMU modules are not imported or modified.
