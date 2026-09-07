#!/usr/bin/env python3
"""Independent PNG, geometry, camera projection, split and CSV validation."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    check(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7), f'{a} != {b}')


def subtract(a, b):
    return [x-y for x, y in zip(a, b)]


def norm(a):
    return math.sqrt(sum(x*x for x in a))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def project(point, camera, width, height):
    # Inverse unit quaternion rotates world-camera into camera coordinates.
    p = subtract(point, camera['position'])
    x, y, z, w = camera['quaternion']
    q = [-x, -y, -z]
    t = [2*v for v in cross(q, p)]
    v = [p[i]+w*t[i]+cross(q, t)[i] for i in range(3)] + [1]
    matrix = camera['projectionMatrix']
    clip = [sum(matrix[col*4+row]*v[col] for col in range(4)) for row in range(4)]
    return [(clip[0]/clip[3]+1)*width/2, (1-clip[1]/clip[3])*height/2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--count', type=int)
    parser.add_argument('--preview', type=Path, help='RGB, green mask overlay, and mask contact sheet')
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root/'generation_manifest.json').read_text())
    config = manifest['configuration']
    check(manifest['status'] == 'complete', 'Incomplete run: inspect rejection_log.json')
    rows = list(csv.DictReader((root/'dataset_summary.csv').open()))
    expected = args.count if args.count is not None else config['datasetSize']
    check(len(rows) == expected == manifest['generatedSamples'], 'Sample count mismatch')
    check((root/'README_DATASET.md').stat().st_size > 1000, 'Missing complete dataset documentation')
    seeds, groups, ids, hashes = set(), set(), set(), set()
    counts, bins, by_split = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    masks, geometries, views = [], [], []
    expected_paths = {'images': set(), 'masks': set(), 'metadata': set()}
    seen_bins = set()
    for row in rows:
        m = json.loads((root/row['metadata_path']).read_text())
        sid, split = m['sampleId'], m['split']
        check(sid not in ids and m['scenarioSeed'] not in seeds and m['scenarioGroup'] not in groups, 'Duplicate sample or scenario / split leakage')
        ids.add(sid); seeds.add(m['scenarioSeed']); groups.add(m['scenarioGroup'])
        check(sid == row['sample_id'] and split == row['split'] and str(m['scenarioSeed']) == row['scenario_seed'], 'CSV identity mismatch')
        check(m['generatorVersion'] == manifest['generatorVersion'], 'Version mismatch')
        for key, column, category in [('image', 'image_path', 'images'), ('mask', 'mask_path', 'masks'), ('metadata', 'metadata_path', 'metadata')]:
            check(m['paths'][key] == row[column] and f'/{split}/' in row[column], 'Path/split mismatch')
            expected_paths[category].add(row[column])
        width, height = m['image']['width'], m['image']['height']
        check((width, height) == (config['imageWidth'], config['imageHeight']), 'Config dimensions mismatch')
        with Image.open(root/row['image_path']) as source:
            check(source.format == 'PNG' and source.size == (width, height), 'Invalid RGB PNG')
            rgb = source.convert('RGB')
        with Image.open(root/row['mask_path']) as source:
            check(source.format == 'PNG' and source.size == rgb.size, 'Mask dimensions mismatch')
            mask_rgb = source.convert('RGB')
            colors = mask_rgb.getcolors(maxcolors=3)
            check(colors is not None and {color for _, color in colors} <= {(0, 0, 0), (255, 255, 255)}, 'Nonbinary mask')
            mask = mask_rgb.convert('L')
        pixels = mask.histogram()[255]
        check(pixels >= config['validation']['minMaskPixels'], 'Empty/tiny mask')
        check(pixels == m['validation']['maskPixels'], 'Mask pixel count mismatch')
        masks.append(pixels); hashes.add(hashlib.sha256(rgb.tobytes()).hexdigest())
        t = m['towline']; points = t['centerlineWorld']
        check(len(points) == config['centerlinePoints'] == len(t['centerlineImage']), 'Point count mismatch')
        check(all(math.isfinite(v) for p in points+t['centerlineImage'] for v in p), 'Invalid coordinates')
        chord = subtract(points[-1], points[0]); length = norm(chord)
        check(length > 1e-6, 'Degenerate chord')
        unit = [v/length for v in chord]
        sag = max(norm(subtract(d, [dot(d, unit)*v for v in unit])) for d in (subtract(p, points[0]) for p in points))
        curve_length = sum(norm(subtract(b, a)) for a, b in zip(points, points[1:]))
        curvature = []
        for a, b, c in zip(points, points[1:], points[2:]):
            left, right = subtract(b, a), subtract(c, b)
            curvature.append(2*norm(cross(left, right))/(norm(left)*norm(right)*norm(subtract(c, a))))
        for key, value in [('chordLength', length), ('curveLength', curve_length), ('maxSagDistance', sag), ('sagRatioGT', sag/length), ('lengthRatioGT', curve_length/length), ('meanCurvatureGT', sum(curvature)/len(curvature)), ('maxCurvatureGT', max(curvature))]:
            close(t[key], value)
        geometries.append(t['sagRatioGT'])
        for world, image in zip(points, t['centerlineImage']):
            for actual, wanted in zip(project(world, m['camera'], width, height), image):
                close(actual, wanted)
        visibility = []
        in_frame = []
        for u, v in t['centerlineImage']:
            inside = 0 <= u < width and 0 <= v < height
            in_frame.append(inside)
            visibility.append(inside and any(mask.getpixel((x, y)) == 255 for x in range(math.floor(u)-2, math.floor(u)+3) for y in range(math.floor(v)-2, math.floor(v)+3) if 0 <= x < width and 0 <= y < height))
        check(visibility == t['centerlineVisible'] and in_frame == t['centerlineInFrame'], 'Visibility flags mismatch')
        close(sum(visibility)/len(points), m['validation']['visibleFraction'])
        close(sum(in_frame)/len(points), m['validation']['inFrameFraction'])
        for name, minimum in [('visibleFraction','minVisibleFraction'),('inFrameFraction','minInFrameFraction'),('projectedChordPixels','minProjectedChordPixels')]:
            check(m['validation'][name] >= config['validation'][minimum], f'Visibility threshold failed: {sid}')
        numeric = dict(sag_parameter=t['sagParameter'], sag_ratio_gt=t['sagRatioGT'], length_ratio_gt=t['lengthRatioGT'], curve_length=t['curveLength'], chord_length=t['chordLength'], max_sag_distance=t['maxSagDistance'], merchant_distance=m['merchantVessel']['distance'], merchant_lateral_offset=m['merchantVessel']['lateralOffset'], camera_fov=m['camera']['fov'], camera_pitch=m['camera']['rotation'][0], camera_roll=m['camera']['rotation'][2], camera_yaw=m['camera']['rotation'][1], mean_curvature_gt=t['meanCurvatureGT'], max_curvature_gt=t['maxCurvatureGT'], sag_bin=m['sagBin'], mask_pixels=pixels, visible_fraction=m['validation']['visibleFraction'])
        for name, value in numeric.items():
            close(float(row[name]), value)
        lo, hi = config['towline']['sagRange']; bin_id = m['sagBin']
        check(lo+(hi-lo)*bin_id/5 <= t['sagParameter'] <= lo+(hi-lo)*(bin_id+1)/5, 'Sag bin mismatch')
        counts[split] += 1; bins[bin_id] += 1; by_split[split][bin_id] += 1
        if args.preview and bin_id not in seen_bins:
            seen_bins.add(bin_id)
            overlay = rgb.copy()
            overlay.paste(Image.new('RGB', rgb.size, '#1cff65'), (0, 0), mask)
            panel = Image.new('RGB', (960, 204), '#14202a')
            for i, im in enumerate([rgb, overlay, mask_rgb]):
                im.thumbnail((320, 180)); panel.paste(im, (320*i, 24))
            ImageDraw.Draw(panel).text((8, 5), f'{sid} | bin {bin_id} | sag GT {t["sagRatioGT"]:.5f} | RGB / mask overlay / mask', fill='white')
            views.append((bin_id, panel))
    for category, paths in expected_paths.items():
        actual = {str(p.relative_to(root)) for p in (root/category).rglob('*') if p.is_file()}
        check(actual == paths, f'Orphan or missing files in {category}')
    for split, count in manifest['splitCounts'].items():
        check(counts[split] == count, 'Manifest split mismatch')
    # Largest-remainder allocation independently recomputed from the configuration.
    names = ['train','val','test']; target = {s:math.floor(expected*config['split'][s]) for s in names}
    order = sorted(names, key=lambda s: -(expected*config['split'][s]-target[s]))
    for s in order[:expected-sum(target.values())]: target[s] += 1
    check(all(counts[s] == target[s] for s in names), 'Incorrect train/val/test allocation')
    for split in names:
        values = [by_split[split][i] for i in range(5)]
        check(max(values)-min(values) <= 1, 'Unbalanced sag bins within split')
    check([bins[i] for i in range(5)] == manifest['sagBinCounts'], 'Manifest bin mismatch')
    log = json.loads((root/'rejection_log.json').read_text())
    check(len(log) == manifest['rejectedSamples'], 'Rejection log mismatch')
    check(len(hashes) == expected, 'Duplicate RGB images')
    if args.preview:
        sheet = Image.new('RGB', (960, len(views)*204))
        for i, (_, panel) in enumerate(sorted(views)): sheet.paste(panel, (0, i*204))
        args.preview.parent.mkdir(parents=True, exist_ok=True); sheet.save(args.preview)
    print(json.dumps({'validatedSamples':expected,'splitCounts':dict(counts),'sagBinCounts':[bins[i] for i in range(5)],'rejectedSamples':len(log),'maskPixelsRange':[min(masks),max(masks)],'sagRatioRange':[min(geometries),max(geometries)]}, indent=2))


if __name__ == '__main__':
    main()
