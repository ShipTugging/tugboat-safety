#!/usr/bin/env python3
"""Derive YOLO-Seg labels without modifying or resplitting the source dataset."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import yaml
from utils.common import ROOT, SPLITS, require, read_mask, sha256, write_json, write_csv, contact_sheet, versions
from utils.polygons import mask_to_labels, parse_labels, rasterize


def prepare(source, output, expected, image_mode, minimum_iou):
    source, output = source.resolve(), output.absolute()
    require(not output.exists(), f'Output exists; use a new directory: {output}')
    require(output != source and source not in output.parents, 'Derived data must be outside the source dataset')
    manifest = json.loads((source / 'generation_manifest.json').read_text())
    require(manifest['status'] == 'complete', 'Source generation is incomplete')
    summary = list(csv.DictReader((source / 'dataset_summary.csv').open()))
    require(len(summary) == sum(expected.values()), 'Source CSV count does not match expected counts')
    by_id = {r['sample_id']: r for r in summary}
    require(len(by_id) == len(summary), 'Duplicate CSV sample IDs')
    # Complete all conversion and integrity checks before creating the derived dataset.
    records, labels, examples, seen_bins = [], {}, [], set()
    ids, seeds, groups = set(), set(), set()
    fingerprints = {}
    for split in SPLITS:
        images = sorted((source / 'images' / split).glob('*.png'))
        require(len(images) == expected[split] == manifest['splitCounts'][split], f'{split} count mismatch')
        stems = {p.stem for p in images}
        require({p.stem.removesuffix('_mask') for p in (source/'masks'/split).glob('*.png')} == stems, f'{split} mask filenames mismatch')
        require({p.stem for p in (source/'metadata'/split).glob('*.json')} == stems, f'{split} metadata filenames mismatch')
        for image in images:
            sid = image.stem
            require(sid not in ids, f'Cross-split duplicate sample: {sid}')
            ids.add(sid)
            mask_path, meta_path = source/'masks'/split/f'{sid}_mask.png', source/'metadata'/split/f'{sid}.json'
            metadata = json.loads(meta_path.read_text())
            require(metadata['sampleId'] == sid and metadata['split'] == split == by_id[sid]['split'], f'Identity mismatch: {sid}')
            require(metadata['scenarioSeed'] not in seeds and metadata['scenarioGroup'] not in groups, f'Duplicate scenario: {sid}')
            seeds.add(metadata['scenarioSeed']); groups.add(metadata['scenarioGroup'])
            for key, path in [('image',image),('mask',mask_path),('metadata',meta_path)]:
                require(metadata['paths'][key] == str(path.relative_to(source)) == by_id[sid][key+'_path'], f'Path mismatch: {sid}')
                fingerprints[str(path.relative_to(source))] = sha256(path)
            with Image.open(image) as im:
                rgb = im.convert('RGB')
            mask = read_mask(mask_path)
            require(rgb.size == (metadata['image']['width'], metadata['image']['height']) == (manifest['configuration']['imageWidth'], manifest['configuration']['imageHeight']), f'Image dimensions mismatch: {sid}')
            require(mask.shape == (rgb.height, rgb.width), f'Mask dimensions mismatch: {sid}')
            text, reconstructed, metrics = mask_to_labels(mask, minimum_iou)
            labels[(split, sid)] = text
            records.append({'sample_id':sid,'split':split,'scenario_seed':metadata['scenarioSeed'],'sag_bin':metadata['sagBin'],
                            'width':rgb.width,'height':rgb.height,**metrics})
            if split == 'val' and metadata['sagBin'] not in seen_bins:
                seen_bins.add(metadata['sagBin'])
                examples.append((f'{sid} | val bin {metadata["sagBin"]} | RGB / original GT / polygon overlay | IoU {metrics["iou"]:.5f}',rgb,mask,reconstructed))
    for filename in ['dataset_summary.csv','generation_manifest.json','README_DATASET.md']:
        fingerprints[filename] = sha256(source/filename)
    output.mkdir(parents=True)
    try:
        for split in SPLITS:
            (output/'images'/split).mkdir(parents=True)
            (output/'labels'/split).mkdir(parents=True)
        for (split, sid), label in labels.items():
            destination, original = output/'images'/split/f'{sid}.png', source/'images'/split/f'{sid}.png'
            if image_mode == 'symlink':
                destination.symlink_to(os.path.relpath(original, destination.parent))
            else:
                shutil.copy2(original, destination)
            (output/'labels'/split/f'{sid}.txt').write_text(label)
        data = {'path':str(output),'train':'images/train','val':'images/val','test':'images/test','names':{0:'towline'}}
        (output/'dataset.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
        # The actual written text is parsed and rasterized a second time.
        for record in records:
            sid, split = record['sample_id'], record['split']
            txt = (output/'labels'/split/f'{sid}.txt').read_text()
            result = rasterize(parse_labels(txt), (record['height'],record['width']))
            require(int(result.sum()) == record['tp'], f'Written label changed: {sid}')
            require((output/'images'/split/f'{sid}.png').is_file(), 'Broken RGB link')
        label_hashes = {str(p.relative_to(output)):sha256(p) for p in (output/'labels').rglob('*.txt')}
        report = {'status':'complete','converter_version':'towline-yolo-seg-1.0.0','source':str(source),
                  'counts':expected,'class_names':{0:'towline'},'image_mode':image_mode,'minimum_required_iou':minimum_iou,
                  'minimum_reconstruction_iou':min(r['iou'] for r in records),'mean_reconstruction_iou':float(np.mean([r['iou'] for r in records])),
                  'lost_pixels':sum(r['fn'] for r in records),'added_pixels':sum(r['fp'] for r in records),
                  'polygon_count':sum(r['polygons'] for r in records),'repaired_contours':sum(r['repaired_contours'] for r in records),
                  'source_sha256':fingerprints,'label_sha256':label_hashes,'environment':versions()}
        write_json(output/'preparation_summary.json', report)
        write_csv(output/'preparation_samples.csv',records)
        contact_sheet(examples, output/'conversion_contact.png')
        print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','label_sha256','environment')},indent=2))
    except BaseException:
        write_json(output/'preparation_summary.json', {'status':'incomplete'})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'towline_dataset')
    p.add_argument('--output',type=Path,default=ROOT/'towline_yolo_seg')
    p.add_argument('--expected-counts',type=int,nargs=3,default=[350,100,50],metavar=('TRAIN','VAL','TEST'))
    p.add_argument('--image-mode',choices=['symlink','copy'],default='symlink')
    p.add_argument('--minimum-iou',type=float,default=.98)
    args=p.parse_args()
    require(0 < args.minimum_iou <= 1, 'Minimum IoU must be in (0,1]')
    prepare(args.source,args.output,dict(zip(SPLITS,args.expected_counts)),args.image_mode,args.minimum_iou)


if __name__ == '__main__':
    main()
