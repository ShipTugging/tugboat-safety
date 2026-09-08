#!/usr/bin/env python3
"""Recheck every derived label against original PNG and the installed YOLO rasterizer."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from utils.common import ROOT, OUTPUTS, write_json, SPLITS, require, configure_runtime, read_mask, binary_counts, pixel_metrics, contact_sheet
from utils.dataset import prepared_dataset
from utils.polygons import parse_labels, rasterize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'towline_yolo_seg')
    parser.add_argument('--overlay',type=Path)
    parser.add_argument('--report',type=Path,default=OUTPUTS/'preparation'/'validation.json')
    args=parser.parse_args()
    configure_runtime()
    from ultralytics.data.utils import polygon2mask, img2label_paths
    from ultralytics.utils.ops import resample_segments
    root,report,_=prepared_dataset(args.dataset,verify_source=True)
    ious=[]; resampled_ious=[]; examples=[]; bins=set()
    for split in SPLITS:
        for image in sorted((root/'images'/split).glob('*.png')):
            label=root/'labels'/split/f'{image.stem}.txt'
            require(img2label_paths([str(image)])[0]==str(label),'Ultralytics resolved label path incorrectly')
            polygons=parse_labels(label.read_text())
            source=Path(report['source']);mask=read_mask(source/'masks'/split/f'{image.stem}_mask.png')
            h,w=mask.shape
            reconstructed=rasterize(polygons,mask.shape)
            actual=np.zeros(mask.shape,dtype=bool)
            resampled=np.zeros(mask.shape,dtype=bool)
            count=max(1000,max(map(len,polygons))+1)
            for polygon in polygons:
                points=polygon*np.array([w,h],dtype=np.float32)
                actual|=polygon2mask(mask.shape,[points.reshape(-1)],1).astype(bool)
                resampled_points=resample_segments([polygon.copy()],n=count)[0]*np.array([w,h],dtype=np.float32)
                resampled|=polygon2mask(mask.shape,[resampled_points.reshape(-1)],1).astype(bool)
            require(np.array_equal(actual,reconstructed),'Installed YOLO rasterization differs')
            counts=binary_counts(actual,mask);iou=pixel_metrics(counts)['iou']
            require(iou>=report['minimum_required_iou'] and counts['fp']==0,'Label changed or lost fidelity')
            ious.append(iou);resampled_ious.append(pixel_metrics(binary_counts(resampled,mask))['iou'])
            require(resampled_ious[-1]>=report['minimum_required_iou'],'YOLO segment resampling destroys mask fidelity')
            m=json.loads((source/'metadata'/split/f'{image.stem}.json').read_text())
            if args.overlay and split=='val' and m['sagBin'] not in bins:
                bins.add(m['sagBin'])
                with Image.open(image) as rgb:
                    examples.append((f'{image.stem} | bin {m["sagBin"]} | RGB / GT / actual YOLO raster | IoU {iou:.5f}',rgb.convert('RGB'),mask,actual))
    if args.overlay:
        args.overlay.parent.mkdir(parents=True,exist_ok=True);contact_sheet(examples,args.overlay)
    result={'validated':len(ious),'counts':report['counts'],'minimum_iou':min(ious),'mean_iou':float(np.mean(ious)),
                      'minimum_after_ultralytics_resampling_iou':min(resampled_ious),'mean_after_ultralytics_resampling_iou':float(np.mean(resampled_ious)),
                      'source_hashes_unchanged':True}
    write_json(args.report,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
