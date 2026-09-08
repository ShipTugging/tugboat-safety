# Towline dataset implementation and validation

Implemented and verified locally on 2026-09-07. Repository: `tugboat-safety-twin`, cloned from `https://github.com/cozymob/tugboat-safety-twin.git`. The changes are local; no deployment or push was performed.

## Architecture inspected and reused

- `Scene3D.tsx`: React Three Fiber Canvas, live camera controller, object refs, existing capture bridge and analysis controls.
- `Tugboat.tsx` / `LargeShip.tsx`: existing procedural vessel models, attachments and animated equipment.
- `dataset/camera.ts`: existing deck/bridge CCTV conventions and independent export-camera pattern.
- `TowingLine.tsx` / `simulation/towline.ts`: tube mesh, five-point sine-displaced Catmull–Rom curve, ship/tug anchors. The input sag override is scoped to dataset scenarios; original live tension-based shape remains.
- `DatasetCaptureBridge.tsx`: final-priority `useFrame`, two-frame readiness handshake, cancellation and renderer restoration.
- `useDatasetExporter.ts` / `DatasetControls.tsx`: existing developer controls and export lifecycle.
- `dataset/environment.ts`: existing deterministic PRNG, reused without modification.

The repository had JPEG/YOLO bounding-box export, but no pixel mask, stored centerline or geometric Sag Ratio output. The new developer UI produces PNG/mask/geometric metadata; legacy YOLO utilities/tests remain. No simulator rebuild, AI model, physical tension estimator or unrelated dependency migration was performed.

## Added files

| File | Purpose |
|---|---|
| `src/dataset/config.ts` | Central defaults, ranges, validation thresholds, generator version and configuration validation |
| `src/dataset/scenarios.ts` | Seed allocation, split/bin plan, frozen independent scenarios and onboard camera orientation |
| `src/dataset/geometry.ts` | World/image centerline, sag/length/curvature, metadata and acceptance checks |
| `src/dataset/mask.ts` | Synchronized offscreen RGB/mask renderer, original occluder vertex/depth/alpha behavior, state restoration |
| `src/dataset/export.ts` | Sequential sample acceptance, file sink, summary, manifest and rejection log |
| `src/vite-env.d.ts` | Vite development/raw-asset types |
| `scripts/generate-towline-dataset.mjs` | Local Playwright/Vite batch exporter with sequential filesystem writes |
| `scripts/verify-towline-dataset.py` | Independent PNG, projection, geometry, CSV, split and distribution verifier; contact sheet |
| `scripts/test-towline-ui.mjs` | Browser preview/download/cancel/error tests and pixel mask oracle |
| `tests/towline-dataset.test.ts` | Numeric geometry, reproducibility, seed uniqueness, rotated anchor, split/bin and config tests |
| `README_DATASET.md` | Full formation process, schema, exact commands, assumptions and reproducibility instructions |
| `docs/towline-dataset-validation.md` | This implementation/validation record |

## Modified files

| File | Change |
|---|---|
| `src/components/Scene3D.tsx` | Dataset scenario parameters, rotated merchant parent, frozen component times, rope scenario |
| `src/components/TowingLine.tsx` | Optional sag/radius/material override; 128 dataset segments, original 64 live segments |
| `src/simulation/towline.ts` | Optional sag override and merchant-heading-aware anchor transform |
| `src/components/HarborEnvironment.tsx` | Optional randomized sun position |
| `src/components/Tugboat.tsx` | Frozen radar angle for dataset time |
| `src/components/LargeShip.tsx` | Frozen radar/propeller angles for dataset time |
| `src/components/DatasetCaptureBridge.tsx` | Towline-specific paired PNG capture path using existing handshake |
| `src/dataset/types.ts` | Optional towline scenario/capture payloads |
| `src/hooks/useDatasetExporter.ts` | New generator lifecycle, browser folder sink, small ZIP fallback, development-only local automation binding |
| `src/components/DatasetControls.tsx` | Size/seed/dimensions, RGB/mask/JSON preview, single/pilot actions, progress and error feedback |
| `package.json`, `package-lock.json` | Playwright development dependency and generation/validation/browser-test scripts |
| `.gitignore` | Generated datasets and validation artifacts excluded from version control |
| `README.md` | Current dataset workflow and link to the full dataset guide |

## Actual checks run

- `npm ci`, followed by adding Playwright. Installed versions: Three.js 0.165.0, Fiber 8.18.0, Drei 9.122.0, React 18.3.1, Vite 5.4.21, TypeScript 5.9.3, Playwright 1.63.0. Node 24.15.0; macOS Google Chrome.
- `npm test`: **34 passed** (existing tests plus four dataset tests).
- `npm run build`: TypeScript and production Vite build succeeded. Vite reports its large-bundle warning; this is not a build failure.
- `git diff --check`: passed.
- Single 1280×720 sample rendered and visually inspected before batch testing. Binary mask excludes the visible shackle/staple occluder and aligns with the physical tube.
- 10-sample 1280×720 batch: **7 train / 2 val / 1 test**, zero rejected. Independent verifier passed and five-bin RGB/overlay/mask contact sheet inspected.
- Only after the single/batch checks, default `npm run dataset:generate`: **500 samples**, generated 2026-09-07 08:50:28–08:51:11 UTC (about 43 seconds on this machine).
- `npm run dataset:validate -- towline_dataset --count 500 --preview artifacts/pilot-contact.png`: all **500** accepted samples verified. **350 train / 100 val / 50 test**, exactly **100 per sag bin**, zero rejected. White mask pixels range **1,062–4,930**; geometric Sag Ratio range **0.0008337393–0.1077016339**. No duplicate RGBs or scenario seeds/groups, no orphan files, no mismatch in metadata/CSV, and all reprojected centerline coordinates and recomputed geometry passed.
- `npm run test:dataset-ui`: actual Chrome preview, next preview, single-sample ZIP, 640×360 size override, invalid-size feedback, folder writer, cancellation, subsequent preview and forced retry exhaustion passed. Folder writer was exercised with a **mock browser directory API** for 11 filesets; the native OS directory chooser itself was not automated. Actual bulk filesystem writes were verified with the local CLI.
- Browser mask oracle: a magenta tube behind a black opaque box, rendered through the actual mask pass. **208 white pixels, zero pixel mismatches** between visible RGB tube support and binary mask; original rope material/render target/tone mapping restored. This checks occlusion and synchronization independently of the exported centerline.
- Forced invalid-mask threshold with two allowed attempts: zero accepted, two distinct rejected scenario seeds, explicit error, rejection log and `incomplete` manifest verified.
- Replayed the 10-sample run with identical seed/config: **all 10 RGB PNGs, 10 mask PNGs, 10 metadata JSONs and the CSV were byte-identical** on this machine. Dates/runtime descriptors are not expected to match across runs.

The initial sandbox blocked the local server and package/network access. Required installation and Chrome/server commands were rerun with explicit execution approval. These were environment restrictions, not simulated test results.

## Outputs and use

- `towline_dataset/`: completed 500-sample pilot, approximately **367 MB decimal / 350 MiB**.
- `artifacts/pilot-contact.png`: one example per bin, RGB / green mask overlay / binary mask.
- `artifacts/ui/controls.png`: verified developer UI screenshot.
- `artifacts/single-verified/`, `artifacts/batch-10/`, `artifacts/batch-10-replay/`: smaller test runs, excluded from Git. The documentation was finalized after the initial single/batch runs and copied into those test folders; their captures/labels were not changed.

Run `npm run dev` from the repository root to use the UI. For another 500-sample run use `npm run dataset:generate -- --out towline_dataset_next`; the existing pilot is protected from overwriting. See `README_DATASET.md` for installation, Chromium fallback, custom config, schema and validation instructions.

## Remaining scope and limitations

The pilot assumes visible curvature from the configured onboard mount and simplified, stationary daylight scenarios. There is no real physical tension or real-world risk calibration, AI training, full 3D reconstruction, sensor noise/blur model or multi-view coverage. Mask alpha thresholding and the 5×5 centerline visibility support convention are documented. GPU/browser changes may alter pixels despite local byte-identical replay. No 2,000–5,000-sample run was executed; the streaming path supports those configured sizes, pending pilot review. Native browser filesystem permission dialogs were not tested end-to-end; CLI disk output was.
