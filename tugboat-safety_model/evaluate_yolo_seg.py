#!/usr/bin/env python3
"""Report YOLO mask AP and original-PNG pixel metrics on a held-out split."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from utils.common import ROOT, require, configure_runtime, select_device, load_model, new_output, read_mask, save_prediction, binary_counts, pixel_metrics, connectivity, write_json, write_csv, contact_sheet, sha256, versions
from utils.dataset import prepared_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',required=True,type=Path)
    p.add_argument('--dataset',type=Path,default=ROOT/'towline_yolo_seg')
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--name',default='evaluation')
    p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--device',default='auto')
    p.add_argument('--conf',type=float,default=.25,help='Fixed threshold for PNG pixel metrics; AP uses 0.001')
    args=p.parse_args()
    require(0<=args.conf<=1,'Confidence must be in [0,1]')
    configure_runtime()
    root,report,_=prepared_dataset(args.dataset,splits=(args.split,))
    model,weights=load_model(args.weights);out=new_output(args.name);device=select_device(args.device)
    # AP needs a low confidence floor for its PR curve. Pixel metrics use an
    # independently recorded fixed threshold, never optimized on the test set.
    validation=model.val(data=str(root/'dataset.yaml'),split=args.split,imgsz=args.imgsz,batch=args.batch,
                         device=device,workers=0,conf=.001,mask_ratio=1,overlap_mask=False,
                         plots=True,project=str(out),name='yolo_validation',verbose=False)
    segmentation=dict(zip(['precision','recall','mAP50','mAP50_95'],map(float,validation.seg.mean_results())))
    source=Path(report['source']);rows=[];examples=[];seen_bins=set();totals=dict.fromkeys(['tp','fp','fn','tn'],0)
    for image in sorted((root/'images'/args.split).glob('*.png')):
        sid=image.stem
        original=source/'masks'/args.split/f'{sid}_mask.png'
        require(sha256(original)==report['source_sha256'][str(original.relative_to(source))],f'Original GT changed: {sid}')
        truth=read_mask(original)
        result=model.predict(str(image),imgsz=args.imgsz,device=device,conf=args.conf,retina_masks=True,classes=[0],save=False,verbose=False)[0]
        prediction=save_prediction(result,out/'predictions',sid)
        counts=binary_counts(prediction,truth)
        for key,value in counts.items():totals[key]+=value
        metadata=json.loads((source/'metadata'/args.split/f'{sid}.json').read_text())
        rows.append({'sample_id':sid,'split':args.split,'sag_bin':metadata['sagBin'],**counts,**pixel_metrics(counts),
                     **connectivity(prediction),'gt_components':connectivity(truth)['connected_components']})
        if metadata['sagBin'] not in seen_bins:
            seen_bins.add(metadata['sagBin'])
            with Image.open(image) as rgb:
                examples.append((f'{sid} | bin {metadata["sagBin"]} | RGB / original GT / prediction | IoU {rows[-1]["iou"]:.4f}',rgb.convert('RGB'),truth,prediction))
    summary={'status':'complete','weights':str(weights),'weights_sha256':sha256(weights),'split':args.split,'images':len(rows),
             'imgsz':args.imgsz,'device':device,'pixel_confidence':args.conf,'ap_confidence_floor':.001,
             'segmentation':segmentation,'pixel_micro':pixel_metrics(totals),
             'pixel_macro':{k:float(np.mean([r[k] for r in rows])) for k in pixel_metrics(totals)},
             'pixel_counts':totals,'mean_predicted_components':float(np.mean([r['connected_components'] for r in rows])),
             'mean_largest_component_fraction':float(np.mean([r['largest_component_fraction'] for r in rows])),
             'prepared_dataset_sha256':sha256(root/'preparation_summary.json'),'environment':versions(),
             'notes':'YOLO AP is instance/component-level polygon evaluation. Pixel metrics union all class-0 predictions and compare ORIGINAL full-resolution PNG masks. Test scores must not select training settings.'}
    write_json(out/'evaluation.json',summary);write_csv(out/'evaluation.csv',rows)
    write_csv(out/'metrics_summary.csv',[{'split':args.split,**segmentation,**{'micro_'+k:v for k,v in summary['pixel_micro'].items()},**{'macro_'+k:v for k,v in summary['pixel_macro'].items()}}])
    contact_sheet(examples,out/'evaluation_contact.png')
    print(json.dumps({k:summary[k] for k in ['split','images','segmentation','pixel_micro','pixel_macro']},indent=2))


if __name__=='__main__':main()
