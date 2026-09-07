# Synthetic Towline Dataset

## 1. Purpose

This dataset extends the existing TUG GUARD Three.js tugboat simulator. It provides synchronized RGB images, visible towline segmentation masks, and geometric ground truth for a future Vision AI model. That model should learn to locate the physical rope and recover its image-space shape. Known simulation geometry supplies Sag Ratio automatically; no manual sag annotation is needed.

**Vision ground truth:** mask and centerline. **Geometric ground truth:** Sag Ratio, length ratio, and discrete curvature. No AI training, inference, tension estimation, or girting-risk prediction is implemented by this generator. It replaces the developer panel's previous JPEG/YOLO capture workflow; legacy YOLO utility modules and tests remain for compatibility, but the new samples have no taut/slack or danger classification labels.

## 2. Current Prototype Assumption

RGB comes from a camera rigidly mounted on the tug, near the port side of the forward bridge. It looks toward the midpoint of the towline. The mount is fixed in tug-local coordinates; only small orientation/FOV variations are allowed.

> The preliminary dataset assumes a camera configuration in which the projected towline curvature is sufficiently observable. Full multi-view / 3D towline reconstruction is outside the scope of the preliminary prototype.

This is proof-of-concept synthetic Three.js data, **not real-world validated maritime accident data**. It does not provide real physical towline tension. The existing scene is approximately meter-scaled: the tug is roughly 11.4 units long, merchant vessel roughly 70 units long, original tube radius 0.085 units. Dataset geometry uses world units (approximately meters), rather than asserting calibrated dimensions.

## 3. Dataset Generation Pipeline

1. Validate the central configuration and allocate independent scenario slots to splits and sag bins.
2. Derive a unique scenario seed from the run seed and attempt slot; sample continuous parameters with the existing seeded PRNG.
3. Place the existing tug/merchant models; rotate the merchant chock consistently with its hull; build the existing Catmull–Rom towline with an explicit dataset sag override.
4. Let React commit the scenario and wait two render callbacks. Pause live physics, fix ocean/radar/propeller time, and disable analysis graphics.
5. At the final render callback, draw RGB into an offscreen render target from a separate onboard camera.
6. Without advancing time or transforms, render the same scene/camera/rope into a mask pass. Read both buffers and encode lossless PNG.
7. Sample the actual tube path, transform into world coordinates, project with the export camera, calculate geometric features, and check visibility.
8. Reject invalid candidates and retry within the **same split and bin**, using a fresh seed. Save accepted RGB/mask/JSON together under their allocated split.
9. Write the summary CSV, full configuration manifest, rejection log and this README. Only a completed run is marked `complete`.

The local Node command drives the same React application through Playwright and writes each sample before requesting the next. It does not replace or reimplement the simulator. Files are written via temporary files followed by renaming; a failed run may have an unfinished sample, and must not be treated as a complete dataset. The validator detects missing/orphan files.

## 4. Dataset Directory Structure

```text
towline_dataset/
├── images/{train,val,test}/sample_000001.png
├── masks/{train,val,test}/sample_000001_mask.png
├── metadata/{train,val,test}/sample_000001.json
├── dataset_summary.csv
├── generation_manifest.json
├── rejection_log.json
├── runtime.json                 # local CLI only
└── README_DATASET.md
```

IDs are unique across the run; each ID appears in exactly one split. Folders for an empty split in a very small run need not exist. The CLI defaults to `./towline_dataset` relative to the command's working directory and refuses to overwrite **any existing output directory**. Use a new `--out` path for each run. Browser folder export creates `towline_dataset` under the selected parent and refuses a nonempty destination. Small ZIP exports contain the same root directory.

## 5. RGB Images

Default: 1280 × 720 lossless RGB/RGBA PNG from the tug-mounted camera. The scene includes the sea, sky, merchant stern, towline and some tug deck/equipment. DOM controls, status text, risk overlays, bounding boxes and tracking graphics are not rendered into these PNGs. Ship-name markings are part of the existing physical vessel texture, not annotation labels.

Both passes use the same single-sample offscreen raster grid (no MSAA), size, near/far planes and camera projection. This gives crisp binary ground truth, with some aliasing at rope edges. RGB uses ACES filmic tone mapping and randomized exposure; there is no JPEG compression. Mild blur/noise/contrast postprocessing is intentionally omitted in version 1 to keep the labels unambiguous.

## 6. Towline Masks

- White `(255,255,255)`: visible physical towline tube.
- Black `(0,0,0)`: everything else, including end shackles, winch rope coils, sea, hulls and sky.

The dedicated rendering pass gives the tube an unlit white material. Other renderable objects use copies of their materials whose final RGB output is black. Their original vertex shaders, alpha testing, geometry, side selection and depth settings remain. Thus the ocean's displaced wave geometry still occludes a submerged rope; a flat replacement ocean is not used. Scene background and fog are black/disabled for this pass; lighting and tone mapping cannot gray the white rope. Original renderer and material state is restored in `finally` even after capture failure.

The readback is flipped vertically into top-left image order, then thresholded at 128 to enforce exact 0/255 values. With opaque geometry and no MSAA this is a direct binary rasterization. Transparent foreground surfaces retain their original alpha blending; the 128 cutoff defines their visibility convention. These are **visible-surface masks**, not amodal masks of hidden rope. Fog attenuation does not change the geometric label. The current scenarios avoid active propeller foam and limit fog.

## 7. Centerline Ground Truth

The actual `TubeGeometry.parameters.path` is sampled at **50 equally spaced arc-length positions**, including both endpoints. The tube uses 128 longitudinal segments and 8 radial segments. Ground-truth lengths and sag are numerical estimates from the stored 50-point polyline; they are not exact analytic curve integrals.

- `centerlineWorld`: `[x,y,z]`; right-handed Three.js world, +Y up, +Z nominal vessel forward, units approximately meters. First point = merchant chock, last point = tug staple.
- `centerlineImage`: `[u,v]` pixel-edge coordinates, origin at the upper-left, +u right, +v down. Pixel centers are `(x+0.5,y+0.5)`. Projection uses `u=(ndcX+1)*width/2`, `v=(1-ndcY)*height/2`.
- Offscreen projected points are retained without clamping. Invalid or behind-camera/clipped-depth centerlines are rejected.
- `centerlineInFrame`: whether each projected point is within image bounds.
- `centerlineVisible`: whether there is a white mask pixel within a 5 × 5 neighborhood around the floored projection. This is a practical support check, **not an exact ray visibility or visible arc-length measurement**.

Centerline coordinates include occluded geometry; use the visibility flags when evaluating only visible predictions. Do not assume every stored point lies on a white pixel, especially at the attachment equipment.

## 8. Sag Ratio Definition

Let A and B be the first and last stored centerline points; let `L = ||B−A||` and `e = (B−A)/L`. For each stored point P:

```text
d(P) = ||(P−A) − dot(P−A,e) e||
maxSagDistance = max d(P)
SagRatioGT = maxSagDistance / L
```

In words:

```text
SagRatio = Maximum perpendicular distance from towline centerline to endpoint chord
           / Endpoint chord length
```

Distance is to the infinite chord line through AB, rather than vertical distance or distance to the camera. For these curves the perpendicular foot lies within the segment. Sloping endpoints are handled correctly. `chordLength` is L. `sagParameter` is the input downward sine displacement amplitude at the control points; it is **not** the ground-truth ratio and may differ from `maxSagDistance`.

Sag Ratio is a dimensionless geometric visual feature, **not direct tension**. The saved ratio is calculated in world space; perspective and occlusion mean an image-space sag ratio is not generally equal to it. Recovering this world-space ratio from a single uncalibrated RGB frame is not guaranteed. The stored camera and world centerline permit separate projected-feature evaluation.

## 9. Length Ratio

```text
curveLength = sum ||P[i+1]−P[i]||
LengthRatioGT = curveLength / chordLength
```

A nearly straight rope has length ratio near 1; curved rope is greater than 1. Both lengths use world units. This is geometric slackness, not a physical rope tension measurement.

Optional curvature is included using the inverse circumcircle radius of each consecutive point triple: `2*||a×b|| / (||a||*||b||*||a+b||)`, where a and b are consecutive edges. The mean is the arithmetic mean over 48 interior points; max is their maximum. Units are inverse world units. Degenerate triples yield zero; zero endpoint chord or invalid coordinates reject the sample.

## 10. Domain Randomization

All ranges live in **`src/dataset/config.ts`** and are sampled uniformly and continuously, except for stratified sag. Defaults:

| Parameter | Range / fixed setting |
|---|---|
| Dataset size / run seed | 500 / 1043 |
| Image dimensions / centerline count | 1280 × 720 / 50 |
| Sag input amplitude | 0.02–2.00 world units |
| Tube radius (thickness is twice radius) | 0.06–0.11 world units |
| Rope material brightness b | 0.35–0.85; linear RGB `(b,0.9b,0.72b)` |
| Merchant chock forward offset from tug origin | 16–28 world units (+Z) |
| Merchant chock lateral offset from tug origin | 8–18 world units (+X) |
| Merchant heading | −0.06 to +0.06 radians about +Y |
| Tug position / Euler rotation | `[0,0.4,0]` / `[0,0,0]` |
| Camera tug-local mount | `[-2.7,4.8,1]` |
| Camera local pitch/yaw/roll perturbation | each −0.025 to +0.025 radians (about ±1.43°) |
| Vertical camera FOV | 64–72 degrees |
| Main light intensity multiplier | 0.8–1.2; base daylight intensity 2.7 |
| Sun position / main light direction | X −100–80, Y 90–180, Z 20–100 |
| Exposure (scene/sky brightness) | 0.90–1.15 |
| Exponential fog density | 0.0002–0.003 inverse world units |
| Wave strength | 0.4–0.9; changes wave geometry and surface normals |
| Simulation time | 60–120 seconds |
| Time of day / vessel speed / propeller RPM | daylight / 0 / 0 |

Merchant `distance` specifically means chock **forward offset**, not Euclidean hull-center distance. `lateralOffset` is the chock's X offset. Hull translation compensates for the rotated local chock so it remains attached. Sun direction changes sky illumination and the main directional light; the existing ocean shader retains its simplified daylight specular direction. Exposure/light/waves/time jointly vary sea appearance; no independent sea-color model is added.

The sag range is narrower than the live simulator's 4.5-unit maximum because the attachment heights are around 3 units and large sag would put much of the rope underwater. Values are chosen for this existing scale and visible curvature, not physical danger thresholds. Randomization reduces repetition of identical backgrounds, vessel positions, material brightness and camera projections.

## 11. Sag Sampling Strategy

Five **generation bins only**, numbered 0–4, span the configured sag input range equally. For defaults: `[0.020,0.416)`, `[0.416,0.812)`, `[0.812,1.208)`, `[1.208,1.604)`, `[1.604,2.000)`.

A continuous value is sampled within the assigned bin. Slots rotate through bins within each split and are deterministically shuffled. A 500-sample run has exactly 100 samples per bin: each bin has 70 train, 20 validation and 10 test samples. Small-run split/bin rounding is described below. Retries preserve the assigned bin, avoiding rejection-driven loss of high-sag coverage.

These bins are **not AI classification labels**, risk classes, tension levels or calibrated physical categories. Ground-truth Sag Ratio remains continuous and also depends on chord length; balanced input bins do not mean equal-width Sag Ratio histograms.

## 12. Train / Validation / Test Split

Default **70% / 20% / 10%**. Counts use largest-remainder rounding (ties in train/val/test order), so they always sum to the requested size. Sag bin counts within any split differ by at most one.

One independently seeded scenario currently generates one frame. Split assignment occurs before rendering; accepted and rejected scenario seeds cannot repeat within a run. Each metadata record includes `scenarioGroup`. Any future temporal frames must inherit that group's split. Never randomly split sequential frames from one trajectory.

Run seed is not a globally unique dataset identifier. Re-running a seed/config deliberately recreates the same scenarios. Changing size or split ratios can move scenario slots into different splits. Do not combine separately generated datasets blindly; deduplicate scenario seeds/groups across all runs and freeze group-to-split assignment before training. The validator checks uniqueness **within the supplied run**.

## 13. Metadata Schema

One JSON per accepted sample. Every field is listed below. Unless explicitly degrees, angles are radians. Array matrices are Three.js column-major.

| Field | Meaning |
|---|---|
| `generatorVersion` | Algorithm/schema version, currently `towline-1.0.0` |
| `sampleId`, `split` | Unique run-local filename stem; `train`, `val` or `test` |
| `scenarioSeed`, `scenarioGroup` | Unsigned 32-bit scenario PRNG seed; `scenario_<seed>` grouping key |
| `simulationTime` | Frozen scene time in seconds |
| `sagBin` | Generation-only bin index 0–4 |
| `paths.image`, `paths.mask`, `paths.metadata` | Paths relative to dataset root |
| `image.width`, `image.height`, `image.format` | Export size and `PNG` |
| `image.coordinateConvention` | Human-readable pixel convention from §7 |
| `camera.mode`, `camera.mount` | `TUG_AFT_DECK`; dataset-specific tug-local mount in world units |
| `camera.jitter` | Applied local perturbations `[pitch,yaw,roll]`, radians; rotations applied X then Y then Z |
| `camera.fov`, `camera.aspect` | Vertical FOV in degrees; width/height |
| `camera.position` | World-space camera origin |
| `camera.rotation`, `camera.rotationOrder` | Actual final camera Euler angles `[x,y,z]` and `XYZ` order |
| `camera.quaternion` | Actual world orientation `[x,y,z,w]` |
| `camera.near`, `camera.far` | Clipping distances, 0.15 and 1800 world units |
| `camera.projectionMatrix` | Actual 16-element projection matrix |
| `tug.position`, `tug.rotation`, `tug.heading` | Tug world origin, Euler XYZ angles, and yaw |
| `merchantVessel.position`, `merchantVessel.heading` | Hull world origin and yaw |
| `merchantVessel.distance`, `merchantVessel.lateralOffset` | Chock forward/lateral offsets from tug origin (§10), world units |
| `towline.sagParameter` | Input vertical displacement amplitude, world units |
| `towline.radius`, `towline.brightness` | Tube radius and linear material brightness parameter |
| `towline.chordLength`, `towline.curveLength` | Endpoint distance and 50-point polyline length, world units |
| `towline.maxSagDistance` | Maximum perpendicular chord deviation, world units |
| `towline.sagRatioGT`, `towline.lengthRatioGT` | Dimensionless geometric ratios (§8–9) |
| `towline.meanCurvatureGT`, `towline.maxCurvatureGT` | Discrete curvature statistics, inverse world units |
| `towline.centerlineWorld`, `towline.centerlineImage` | 50 points in world / pixel coordinates (§7) |
| `towline.centerlineInFrame`, `towline.centerlineVisible` | 50 corresponding bounds/support flags (§7) |
| `environment.timeOfDay` | `day` |
| `environment.lightIntensity` | Main light multiplier (base intensity 2.7) |
| `environment.sunPosition` | Main light and sky sun world position |
| `environment.exposure` | RGB tone-mapping exposure |
| `environment.fog`, `environment.waveStrength` | Exponential fog density; wave multiplier |
| `environment.shipSpeed`, `environment.propellerRpm` | Both 0 for stationary snapshots |
| `validation.maskPixels` | Count of white pixels in saved mask |
| `validation.inFrameFraction` | Fraction of projected centerline samples inside frame |
| `validation.visibleFraction` | Fraction with mask support within 2 pixels (§7) |
| `validation.projectedChordPixels` | Euclidean image distance between projected endpoints |

Example record excerpt below is illustrative; arrays are shortened to keep it readable. Actual records contain all fields above and all 50 points. Exact values are available in each generated JSON.

```json
{
  "generatorVersion": "towline-1.0.0",
  "sampleId": "sample_000001",
  "split": "train",
  "scenarioSeed": 2654436804,
  "scenarioGroup": "scenario_2654436804",
  "simulationTime": 85.2,
  "sagBin": 3,
  "image": {"width": 1280, "height": 720, "format": "PNG"},
  "camera": {"mode": "TUG_AFT_DECK", "fov": 68, "position": [-2.7, 5.2, 1]},
  "tug": {"position": [0, 0.4, 0], "heading": 0},
  "merchantVessel": {"position": [8.5, 0, 56.2], "heading": 0, "distance": 22, "lateralOffset": 12},
  "towline": {
    "sagParameter": 1.4,
    "centerlineWorld": [[12, 2.7, 22], [0, 3.2, 3.2]]
  },
  "environment": {"timeOfDay": "day", "lightIntensity": 1, "fog": 0.001}
}
```

`generation_manifest.json` documents `generatorVersion`, `startedAt`/`finishedAt` (UTC ISO timestamps), `status` (`complete` or `incomplete`), full `configuration`, `generatedSamples`, `rejectedSamples`, `splitCounts`, `sagBinCounts`. Configuration contains the §10 ranges, split ratios, dimensions, point count, run seed, size, and §16 validation thresholds. Every `*Range` is `[minimum,maximum]`.

`rejection_log.json` is an array of `sampleIndex` (zero-based intended output slot), `attempt` (zero-based retry), `scenarioSeed` and textual `reasons`. Rejected images are not included. `runtime.json` (CLI) records Node version, browser version, platform, source Git commit and whether its working tree was modified. An output marked `incomplete` is diagnostic only.

## 14. dataset_summary.csv

One row per accepted sample; JSON is the authoritative full record. Column meanings:

| CSV column | Metadata / meaning |
|---|---|
| `sample_id`, `split`, `scenario_seed` | `sampleId`, `split`, `scenarioSeed` |
| `image_path`, `mask_path`, `metadata_path` | Corresponding `paths` entries |
| `sag_parameter` | `towline.sagParameter`, world units |
| `sag_ratio_gt`, `length_ratio_gt` | Dimensionless geometric ratios |
| `curve_length`, `chord_length`, `max_sag_distance` | Corresponding towline lengths, world units |
| `merchant_distance`, `merchant_lateral_offset` | Forward/lateral chock offsets, world units |
| `camera_fov` | Vertical FOV, degrees |
| `camera_pitch`, `camera_roll`, `camera_yaw` | Final camera Euler X, Z, Y respectively, radians; **not just the jitter** |
| `mean_curvature_gt`, `max_curvature_gt` | Curvature statistics, inverse world units |
| `sag_bin` | Generation-only input bin |
| `mask_pixels`, `visible_fraction` | Mask count and point-support fraction |

Use these columns for sag distribution plots, split checks and downstream evaluation. Do not use generation bins as safety labels.

## 15. Dataset Generation Instructions

Run commands from the repository root. Node 22+ and npm are recommended; development verification used Node 24.15.0. Locked runtime versions include Three.js 0.165.0, React 18.3.1, React Three Fiber 8.18.0, Drei 9.122.0, Vite 5.4.21 and Playwright 1.63.0. Install the committed lockfile:

```sh
npm ci
npm run dev
```

Open the local URL printed by Vite (normally port 5173). In the right-hand developer section, enable **Dataset Generation**.

1. Set **Random Seed**, **Image Width**, **Image Height** as desired.
2. Click **Preview Random Scenario**. Inspect the expanded RGB/mask preview, Sag Ratio GT and JSON parameters. This renders one scenario without saving. Repeated preview clicks advance the run seed deterministically and rotate the sag bin; read the actual `scenarioSeed` in the preview. Preview does not move the operator's live camera.
3. Click **Generate Single Sample** to download a ZIP using the entered seed. It is independent of the preview click counter.
4. For a pilot, set **Dataset Size = 500** and click **Generate Pilot Dataset**. In Chrome on localhost, select an empty parent folder; the app creates its `towline_dataset` child and writes files sequentially. Keep the tab active. The browser's directory permission prompt is required by its security model.
5. Progress shows accepted samples and rejected candidates. Cancel produces an incomplete run in folder mode; don't train on it. The original live camera/settings resume after completion/cancel/failure.

Browsers without the File System Access API support ZIP export for at most 10 samples (128 MiB encoded input cap). They display an explicit instruction to use the local CLI for larger runs. Large ZIP archives are not silently accumulated in memory.

**Preferred repeatable local workflow** (Chrome installed; no need to start a separate dev server):

```sh
# Single-sample inspection, then a small batch
npm run dataset:generate -- --count 1 --out towline_dataset_single
npm run dataset:generate -- --count 10 --out towline_dataset_smoke

# Validate both before running the pilot; Pillow is the validator's only Python dependency
python3 -m pip install Pillow
npm run dataset:validate -- towline_dataset_single --count 1 --preview artifacts/single-contact.png
npm run dataset:validate -- towline_dataset_smoke --count 10 --preview artifacts/smoke-contact.png

# Default pilot: 500 samples, seed 1043, 1280x720
npm run dataset:generate
npm run dataset:validate -- towline_dataset --count 500 --preview artifacts/pilot-contact.png

# Change size, seed, dimensions and destination
npm run dataset:generate -- --count 300 --seed 2026 --width 1280 --height 720 --out towline_dataset_300

# After validating the pilot, the same streaming path supports larger runs
npm run dataset:generate -- --count 2000 --seed 2048 --out towline_dataset_2000
```

If Chrome is unavailable, install Playwright's Chromium and select it:

```sh
npx playwright install chromium
npm run dataset:generate -- --browser chromium --count 10 --out towline_dataset_chromium
```

To change geometry/environment ranges, edit `src/dataset/config.ts` or pass `--config settings.json`. A JSON config may override top-level values and partial sections, for example:

```json
{"datasetSize": 500, "seed": 1043, "camera": {"fovRange": [65, 70]}}
```

```sh
npm run dataset:generate -- --config settings.json --out towline_dataset_custom
```

CLI flags take precedence. `generation_manifest.json.configuration` is a complete settings object that can be copied to a standalone config JSON for replay. The CLI starts a loopback-only Vite server, waits for the scene, streams PNG/JSON files via a Playwright binding, then closes the browser/server. The automation binding exists only in development builds and does not expose a filesystem-writing HTTP endpoint. Files stay on your machine.

```sh
npm test
npm run build
```

The original simulator physics and legacy dataset utility tests remain, with new geometric/reproducibility/split tests added.

## 16. Dataset Validation

Before accepting a candidate, the generator rejects invalid centerline coordinates, nearly zero endpoint chord (< 1e−6), nonfinite Sag Ratio, unprojectable/behind-camera/clipped-depth centerline, or failures of these configurable defaults:

| Check | Threshold |
|---|---|
| White mask pixels | at least 12 |
| Centerline points in frame | at least 70% |
| Centerline points supported by visible mask | at least 60% |
| Projected endpoint span | at least 60 pixels |
| Candidate attempts per accepted output slot | at most 20 |

These are practical dataset checks, not safety criteria. Low resolution or custom camera/sag ranges may need adjusted thresholds. On retry exhaustion, generation stops with an explicit error and incomplete manifest; it does not silently deliver fewer samples as a success. Renderer failures are fatal rather than repeatedly retried.

`npm run dataset:validate -- <directory>` independently loads every PNG/JSON/CSV and checks:

- Exact 0/255 mask channels, PNG formats/dimensions, nonempty masks and duplicate RGB frames.
- Matching IDs, paths, metadata and CSV numerical fields; no missing/orphan image/mask/JSON files.
- World-space chord, curve length, sag and curvature recomputed in Python.
- Image coordinates reprojected independently using saved camera quaternion and projection matrix.
- Recomputed bounds/mask-support flags and acceptance thresholds.
- Unique scenario seeds/groups, actual split counts, balanced sag bins per split, manifest counts and rejection log counts.

`--preview artifacts/contact.png` produces one row per represented sag bin: RGB, green mask overlay, binary mask. Visually verify that the overlay follows the physical rope and stops at occluders. This contact sheet is a diagnostic artifact; green markings are never saved into training RGB. Also inspect the histogram of `sag_ratio_gt` in the CSV and verify the highest sag cases. Numerical projection checks alone cannot guarantee visual realism or solve the single-view ambiguity.

## 17. Known Limitations

- Synthetic-to-real appearance gap; simplified materials, lighting, water and vessel details.
- Restricted onboard viewpoints and intentional observable-sag assumption; the projected curve can still be strongly foreshortened.
- No real tension measurement or calibrated risk labels; live simulator tension thresholds do not label this dataset.
- Sine-displaced Catmull–Rom rope, not a physical catenary solver; no rope dynamics, wind, stiffness or hydrodynamics.
- Full world centerline is available as simulator ground truth; no 3D reconstruction model is implemented. World-space sag is not identical to projected sag.
- Partial occlusion at hull/attachment points; transparent-surface threshold convention and approximate point visibility flags.
- Curvature/length metrics approximate the smooth curve with 50 points; eight-sided rope and single-sample rasterization can alias.
- Fixed daylight, stationary tug, no active wake, no sensor blur/noise or lens distortion. Sea specular direction is simplified.
- GPU/browser/driver differences can change pixels. Source/config changes may change geometry, bin assignment or split allocation.
- The validator only checks leakage within one run; cross-run datasets require independent deduplication.
- Multi-thousand-image runs are supported by streaming but should follow pilot review. Interrupted runs are not resumable in version 1; use a fresh output directory.

## 18. Future Work

Multi-view datasets; 3D towline geometry evaluation; more realistic rope physics; Unity or higher-fidelity simulation if required; depth maps; real towing footage; domain adaptation; real-world validation; IMU synchronization; temporal sag-change features; and future multimodal girting-risk estimation. None of these models or integrations is included in this task.

## 19. Reproducibility

Generator version: **`towline-1.0.0`**. Keep the source snapshot/commit, `package-lock.json`, complete configuration, run size/seed and browser/GPU environment together. Bump the generator version when changing scenario sampling, camera, geometry or mask semantics.

For attempt slot `slot = sampleIndex * maxAttemptsPerSample + attempt`, scenario seed is unsigned 32-bit:

```text
scenarioSeed = (runSeed + imul(slot + 1, 2654435761)) >>> 0
```

The odd multiplier makes the mapping bijective over uint32; allowed run sizes/attempt counts cannot exhaust it. Each scenario starts the existing seeded PRNG from this seed, independent of previous scenarios' random draws. A separate seeded shuffle allocates the plan. The same generator version, full configuration and run seed reproduce parameters, geometry and split assignments. Run size matters to the shuffled plan, so generating a one-image run is not the way to replay an arbitrary member of a 500-image run.

For exact scenario replay/debugging, reconstruct its plan entry from metadata `{split, sagBin}` and use `makeScenario(config, sampleIndex, attempt, planEntry)`; sampleIndex is the numeric sample suffix minus one. For an accepted sample, attempt is zero unless that slot appears in the rejection log; then it is one greater than the last rejected attempt. To reproduce a whole run without code, copy the manifest configuration into a JSON file and rerun the CLI with `--config` and a fresh `--out` path.

Simulation time is seeded. Both captures occur synchronously between animation updates; radar and propeller angles use that fixed time. Generation dates only describe execution and are not inputs to randomization. Local `runtime.json` records the tested browser/Node/source state; a dirty working tree requires preserving its changes as well as the commit. Pixel-identical results across GPUs or versions are not promised.
