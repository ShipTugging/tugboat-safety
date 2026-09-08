# Towline Vision AI baseline

## 1. Purpose

The first baseline implements only **RGB → YOLO instance segmentation → predicted towline mask → segmentation evaluation**. It reuses the existing Three.js-generated dataset and leaves simulator source unchanged. It does not extract a centerline, estimate sag/curvature, infer tension, fuse IMU, train GRU/LSTM, or classify girting risk.

## 2. Current dataset

The source `towline_dataset/` contains 500 synchronized 1280×720 RGB PNGs, binary masks, and metadata: **350 train, 100 validation, 50 test**. White mask pixels are visible physical towline. Occluded geometry is not part of the mask. There is one class: **0 = towline**.

Source names are `images/<split>/sample_XXXXXX.png`, `masks/<split>/sample_XXXXXX_mask.png`, `metadata/<split>/sample_XXXXXX.json`. Existing metadata is used only for dimensions, identity/split integrity and choosing QA examples across generation bins. Sag bins are not training classes.

## 3. Why YOLO-Seg is the first baseline

Selected checkpoint: **`yolo26n-seg.pt`**, downloaded and successfully loaded using **Ultralytics 8.4.142**. It is a lightweight COCO-pretrained nano instance segmentation model, allowing a standard training/validation/inference workflow and future runtime evaluation. The one-class head is fine-tuned from this pretrained model; loading an architecture-only YAML is rejected by these scripts.

The official [segmentation workflow](https://docs.ultralytics.com/tasks/segment/) and [YOLO26 model guide](https://docs.ultralytics.com/models/yolo26/) describe this checkpoint and task. Model selection here was confirmed against the installed package and an actual smoke run rather than assuming a filename exists. `--model` accepts another compatible pretrained `.pt` segmentation checkpoint. No fallback to a detection-only model occurs silently.

## 4. Thin-object limitation and polygon conversion

Towline foreground occupies roughly 0.1–0.54% of a frame. A one-pixel lateral error, mask downsampling or a short missing segment can matter greatly even if a bounding box looks correct.

Conversion uses OpenCV `RETR_CCOMP` with `CHAIN_APPROX_SIMPLE`, which removes redundant collinear contour points but does not apply an approximation epsilon or curve smoothing. Shapely validates every polygon. Invalid self-touching contours are repaired with `make_valid`; only positive-area polygon parts are retained. The converter then measures any removed pixels and fails if reconstruction IoU falls below **0.98** or if any new foreground pixels appear. Repairs are never silent.

Coordinates are `(contour_x + 0.5)/width`, `(contour_y + 0.5)/height`, serialized to nine decimal places. Pixel-center coordinates avoid systematic truncation bias during Ultralytics' segment interpolation. Both the actual on-disk float32 labels and the installed Ultralytics rasterizer are verified. Validation also simulates the loader's 1,000-or-more-point resampling and applies the same IoU floor. Resize/augmentation during actual training still adds further raster error.

YOLO text format has one simple polygon per row, with `0` followed by normalized coordinate pairs. **Each disconnected visible component gets its own class-0 row.** The format has no shared instance/group identifier for disconnected polygons, so fragments of one rope become separate training instances. Components are never joined through an occluder. During pixel evaluation/inference all class-0 predictions are unioned into one semantic towline mask. Consequently polygon mAP is component/instance-based, while PNG IoU/Dice measure the union of the visible rope. See the [official annotation format](https://docs.ultralytics.com/datasets/segment/).

Holes and isolated one-pixel/zero-area components cannot be represented reliably as simple positive-area YOLO rows. These scripts **stop with an explicit error** for those cases instead of filling holes, inventing a bridge or silently dropping a component. The supplied 500 masks contain no such cases. Tests cover two disconnected curved components, invalid polygons, empty masks, holes and degenerate components.

## 5. Folder structure

```text
ai/
├── requirements.txt
├── README_VISION.md
├── prepare_yolo_seg.py
├── validate_yolo_seg.py
├── train_yolo_seg.py
├── evaluate_yolo_seg.py
├── predict_yolo_seg.py
├── utils/{common,dataset,polygons}.py
├── tests/test_pipeline.py
├── .venv/                         # ignored local environment
└── outputs/                       # ignored weights, runs, reports, caches
    ├── checkpoints/yolo26n-seg.pt
    ├── preparation/validation.json
    ├── preparation/validated_contact.png
    ├── smoke/run_config.json
    ├── smoke/train/weights/{best,last}.pt
    ├── smoke_val_predictions/
    ├── smoke_test_evaluation/
    └── baseline_plan/              # dry-run only

towline_yolo_seg/                   # ignored derived dataset
├── images/{train,val,test}/*.png   # relative symlinks to original RGBs
├── labels/{train,val,test}/*.txt
├── dataset.yaml
├── preparation_summary.json
├── preparation_samples.csv
└── conversion_contact.png
```

Scripts default to this repository's data paths even when called from another working directory. Output names are relative to `ai/outputs/`; absolute/traversing output names are rejected. A training/prediction/evaluation run refuses an existing destination so previous results are preserved. The preparation command also requires a fresh destination. Original RGB/masks/metadata are never written. Symlinks save approximately 350 MiB of duplicate RGB data. Ultralytics writes its label caches only into the derived dataset.

`preparation_summary.json` records sample counts, environment, per-source SHA256 hashes, per-label hashes, reconstruction fidelity, repair counts and class convention. `preparation_samples.csv` records per-sample split/seed/bin, dimensions, TP/FP/FN/TN, IoU/Dice, pixel precision/recall, polygon/vertex counts and repaired-contour count. `validation.json` adds actual Ultralytics rasterization/resampling fidelity and source-immutability checks.

## 6. Environment setup

Verified environment: **Python 3.13.7**, macOS arm64, Apple M3 Pro, PyTorch **2.14.0**, torchvision **0.29.0**, Ultralytics **8.4.142**. Direct package versions are pinned in `requirements.txt`; every training run also saves the complete `pip freeze` result and runtime package versions.

From the repository root:

```sh
python3 -m venv ai/.venv
ai/.venv/bin/python -m pip install -r ai/requirements.txt
```

This environment was created and the listed package versions were installed for development. The simulator's npm dependencies are unaffected. The existing local `.venv` is already ready; rerunning installation is unnecessary unless rebuilding the environment.

`--device auto` selects available Apple MPS, otherwise CUDA device 0, otherwise CPU. Explicit `--device cpu`, `--device mps` or `--device 0` is supported. No CUDA assumption is made. The actual smoke run used **MPS**. Restricted execution environments can hide Metal availability; use a normal local terminal or explicit CPU when needed.

Ultralytics settings, Matplotlib cache, Torch cache and downloaded checkpoints are directed under `ai/outputs/`. Dataset/model uploads and deployment are not performed. The first pretrained checkpoint load requires network access. Training uses `workers=0`, batch 2 and AMP disabled for a conservative local Mac setup.

## 7. Dataset preparation

```sh
ai/.venv/bin/python ai/prepare_yolo_seg.py
ai/.venv/bin/python ai/validate_yolo_seg.py --overlay ai/outputs/preparation/validated_contact.png
```

These commands were run successfully. The default output already exists in this workspace. To perform a separate conversion without overwriting it:

```sh
ai/.venv/bin/python ai/prepare_yolo_seg.py --output towline_yolo_seg_next
ai/.venv/bin/python ai/validate_yolo_seg.py --dataset towline_yolo_seg_next
```

Alternate output arguments use the same tested conversion path. `--image-mode copy` is available if the filesystem does not support symlinks; the default is `symlink`. `--expected-counts TRAIN VAL TEST` defaults to `350 100 50` and must match both the source manifest and observed files. There is no random re-splitting. Full source preparation validates every RGB/mask/metadata name, dimension, class, coordinate bound and polygon, as well as sample ID/scenario uniqueness across splits.

Preparation result for this pilot:

| Check | Result |
|---|---:|
| Train / val / test | 350 / 100 / 50 |
| Polygons / empty labels | 500 / 0 |
| Mean direct reconstruction IoU | 0.999992879 |
| Minimum direct reconstruction IoU | 0.999377722 |
| Added foreground pixels | 0 |
| Removed foreground pixels, all 500 images | 9 |
| Repaired contours | 9 |
| Mean IoU after Ultralytics resampling | 0.999943120 |
| Minimum IoU after Ultralytics resampling | 0.997277677 |

The 9 removed pixels are zero-area tip/spur pixels from 9 repaired contours. Those losses are explicitly recorded. RGB / original GT / green polygon contact sheets include validation examples from all five sag bins and were visually inspected. Preparation validation can inspect test annotations for format integrity, but model tuning never uses test predictions or scores.

## 8. Exact commands and checks

All entry points implement `--help` and a main guard. The execution sequence used was preparation → label validation/visual QA → syntax/import and unit checks → smoke training → validation-image prediction → held-out test evaluation → baseline configuration dry-run.

```sh
ai/.venv/bin/python -m unittest discover -s ai/tests -v
ai/.venv/bin/python ai/train_yolo_seg.py --mode smoke --name smoke
ai/.venv/bin/python ai/predict_yolo_seg.py --weights ai/outputs/smoke/train/weights/best.pt --source towline_yolo_seg/images/val --limit 5 --imgsz 640 --name smoke_val_predictions
ai/.venv/bin/python ai/evaluate_yolo_seg.py --weights ai/outputs/smoke/train/weights/best.pt --split test --imgsz 640 --name smoke_test_evaluation
ai/.venv/bin/python ai/train_yolo_seg.py --mode baseline --name baseline_plan --dry-run
```

These run names already exist. Use new `--name` values to repeat. The long baseline command in §10 is supported and its configuration was dry-run checked; **the long training job was not executed**. CPU and alternative checkpoint options are available, but no CPU training or alternative checkpoint experiment is claimed here.

## 9. Smoke-test training

Default smoke mode: **1 epoch**, **640** image size, **batch 2**, seed **1043**, deterministic flags, fixed **32 train / 20 val** subsets. The subset is selected with a local seeded shuffle from each existing split; no example changes split. Exact IDs and absolute derived-image paths are saved in `run_config.json` and `train.txt`/`val.txt`. These lists retain the derived path rather than resolving RGB symlinks, so Ultralytics finds the correct derived labels.

The smoke run completed on MPS in **47.05 seconds including model loading/setup/final validation**, and produced `ai/outputs/smoke/train/weights/best.pt`. It is a **pipeline check, not a useful trained detector**. A pretrained 80-class model is adapted to the single towline class; one short epoch is insufficient to learn that class.

Train arguments include `mask_ratio=1`, `overlap_mask=False`, mosaic/mixup/copy-paste off, mild rotation/translation/scaling and color changes, no random flips, AdamW at 0.001, no disk image cache. This reduces aggressive augmentation/downsampling of the thin foreground. Actual resolved Ultralytics settings are also saved in `train/args.yaml`.

## 10. Baseline training

The first full experiment defaults to **30 epochs**, **960** image size, **batch 2**, all **350 train / 100 val** images, seed 1043, early-stopping patience 10. Its exact settings were validated using:

```sh
ai/.venv/bin/python ai/train_yolo_seg.py --mode baseline --name baseline_plan --dry-run
```

The next actual training command is:

```sh
ai/.venv/bin/python ai/train_yolo_seg.py --mode baseline --name baseline
```

This command has not been run as a long training job. Use `--imgsz`, `--epochs`, `--batch`, `--device`, `--model` and `--seed` to make deliberate experiments, recording each under a fresh name. Do not select these values using the smoke test-set scores. If memory is constrained, reduce batch/image size or choose CPU and document the resulting experiment.

Only train and validation paths are placed in each run's **training-specific YAML**. The test key is omitted completely. `best.pt` is selected by Ultralytics on validation results. Evaluation does not feed back into training or change checkpoints.

## 11. Evaluation

Evaluation accepts only weights whose model class mapping is `{0: 'towline'}`. It defaults to the complete test split. The smoke evaluation command in §8 was run on all 50 test images, with no filtering by confidence, sag or sample outcome.

For a future trained baseline:

```sh
ai/.venv/bin/python ai/evaluate_yolo_seg.py --weights ai/outputs/baseline/train/weights/best.pt --split test --name baseline_test_evaluation
```

This future command requires the baseline checkpoint and was not executed. Use `--split val` for development/model selection; reserve test for a frozen final experiment.

Outputs inside the requested evaluation directory:

- `evaluation.json`: checkpoint hash, dataset preparation hash, runtime/settings, segmentation P/R/mAP, pixel micro/macro averages, confusion counts and connectivity summary.
- `evaluation.csv`: one row per input image, sample/split/bin, TP/FP/FN/TN, IoU/Dice/pixel P/R, predicted component count/largest-component fraction, GT component count.
- `metrics_summary.csv`: a compact one-row aggregate.
- `predictions/*_mask.png`, `predictions/*_overlay.png`: original-size predicted masks and bright green overlays.
- `evaluation_contact.png`: RGB / original PNG GT / predicted overlay for examples across bins.
- `yolo_validation/`: standard Ultralytics validation plots and diagnostics.

**Observed smoke test-set result, fractions rather than percentages:**

| Metric | Result |
|---|---:|
| Segmentation precision | 0.0001333333 |
| Segmentation recall | 0.04 |
| Segmentation mAP50 | 0.00000505963 |
| Segmentation mAP50–95 | 0.00000112035 |
| Pixel micro/macro IoU | 0 |
| Pixel micro/macro Dice | 0 |
| Pixel micro/macro precision / recall | 0 / 0 |

At fixed pixel confidence **0.25**, this smoke model produces empty masks on all 50 test images and the five inspected validation images. The model has **not** learned reliable towline detection. The pipeline records these failures rather than drawing ground truth as a prediction or treating empty predictions as success. No further training configuration was selected using these test results.

## 12. Prediction visualization

The executed validation example:

```sh
ai/.venv/bin/python ai/predict_yolo_seg.py --weights ai/outputs/smoke/train/weights/best.pt --source towline_yolo_seg/images/val --limit 5 --imgsz 640 --name smoke_val_predictions
```

`--source` accepts one image or a recursive image directory. Outputs preserve subdirectories and include the original extension to avoid name collisions, e.g. `sample_000001.png_mask.png` and `sample_000001.png_overlay.png`. Source RGB is never overwritten. `predictions.json` records the checkpoint/hash, settings, input path, output names, pixel count and connectivity.

Inference uses `retina_masks=True` and requires masks at the original image dimensions. All class-0 instance masks are unioned and thresholded at 0.5. If no instances survive confidence filtering, a correctly sized all-black mask and unchanged RGB overlay are saved. Foreground, when predicted, is shown bright green; a unit test covers the nonempty/disconnected union path. Empty masks from the smoke run are expected failures of the short-trained model, not substituted GT masks.

## 13. Metrics and their meanings

**Ultralytics segmentation metrics:** precision/recall and mask mAP at IoU 0.50 and averaged 0.50–0.95. These compare predicted instances with converted YOLO polygons. Validation uses confidence floor **0.001** to retain a precision-recall curve; reported P/R are the library's selected curve operating point. They are not the fixed-threshold pixel precision/recall.

**Original-PNG pixel metrics:** use the union of class-0 predictions at fixed confidence **0.25** by default, original image dimensions, and original binary masks directly. No polygon approximation, bounding box or centerline is used as the GT for these metrics.

```text
IoU             = TP / (TP + FP + FN)
Dice            = 2 TP / (2 TP + FP + FN)
pixel precision = TP / (TP + FP)
pixel recall    = TP / (TP + FN)
```

Micro averages pool pixel counts across all images. Macro averages give each image equal weight. True negatives are recorded but do not inflate IoU/Dice; background-dominated pixel accuracy is intentionally not the headline score. Empty prediction against nonempty GT gets zero IoU/Dice/P/R. Both empty gets IoU/Dice 1; if GT is empty, recall is defined as 1. The supplied source dataset has no empty GT.

Connectivity uses 8-connected foreground pixels: component count and fraction of predicted foreground in the largest component (zero for empty prediction). It applies no area threshold, smoothing or postprocessing, so isolated false-positive pixels count. Occlusion can legitimately create multiple GT components; these values are diagnostics, not safety or tracking scores. mAP alone does not assess continuity, missing narrow sections or geometric usefulness. Inspect original-resolution overlays and pixel scores together.

## 14. Train / val / test policy

Original scenario-level 350/100/50 allocation is preserved exactly. Source IDs, seeds and groups are checked for duplication during preparation. No script resplits data or moves a sample. Smoke subsets retain original splits. All three partitions are prepared for format validation, but training/model selection receives only train/val.

The smoke test evaluation verifies evaluation code and is explicitly an inadequate-model diagnostic. Do not tune hyperparameters, confidence, postprocessing or epochs against those results. Future model selection belongs on validation. After repeated development, use newly held-out scenarios/real footage for an unbiased final assessment. Cross-run dataset deduplication remains necessary if more synthetic data is generated.

## 15. Reproducibility

Keep source code, direct dependency pins, the original dataset, `preparation_summary.json`, `run_config.json`, `requirements-frozen.txt`, `train/args.yaml`, selected IDs and weight hashes together. Source SHA256s verify that conversion/training did not edit originals. Label hashes reject accidental derived-label edits.

Seed 1043 and deterministic Ultralytics flags are saved. **MPS emitted a warning that `index_put_with_accumulate_mps` has no deterministic implementation**, so bitwise-identical training is not promised even with these flags. CPU/GPU/driver or package changes can alter results; `--device cpu` is supported but was not used for the smoke training. Dataset YAML contains an absolute derived-root path; after moving the project, prepare a fresh derived directory so that paths and reports describe the new location. RGB symlinks themselves are relative.

The pretrained checkpoint is saved in `ai/outputs/checkpoints/`, and its SHA256 plus the trained checkpoint SHA256 are recorded in each completed run. There is no automatic experiment upload, push or deployment. Large generated data, `.venv`, weights and caches are gitignored.

## 16. Known limitations

- Smoke training verified execution but did not produce a usable model; the full baseline remains to be trained.
- Only 500 synthetic positive images; there are no negative/background-only examples for false-positive robustness. Synthetic-to-real gap and restricted camera views remain.
- Raster-to-polygon repairs remove 9 zero-area spur pixels in this pilot. Loader interpolation, letterboxing and finite-resolution mask heads introduce additional errors.
- Disconnected visible components become separate YOLO instances, without a shared physical-rope identity. Holes/degenerate components fail conversion explicitly.
- MPS training includes nondeterministic operations despite requested deterministic flags.
- No sensor calibration, lens distortion, temporal tracking, runtime deployment or validated maritime safety conclusions are included.
- Default evaluation writes both AP and pixel metrics; their confidence/GT conventions differ and must not be conflated.

## 17. Next step

Once a useful segmentation model is trained and validated, the next separate task is **predicted mask → centerline extraction → Sag / curvature**. That stage is deliberately not implemented here.
