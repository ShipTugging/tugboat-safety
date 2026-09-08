#!/usr/bin/env python3
"""Fine-tune a pretrained nano segmentation model; test images are never supplied."""
import argparse
import json
from pathlib import Path
import time
import random
import subprocess
import yaml
from utils.common import ROOT, DEFAULT_MODEL, configure_runtime, select_device, load_model, new_output, write_json, versions, sha256, require
from utils.dataset import prepared_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','baseline'],default='smoke')
    p.add_argument('--dataset',type=Path,default=ROOT/'towline_yolo_seg')
    p.add_argument('--model',default=DEFAULT_MODEL)
    p.add_argument('--imgsz',type=int)
    p.add_argument('--epochs',type=int)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--device',default='auto',help='auto selects MPS, then CUDA, then CPU; or specify cpu/mps/0')
    p.add_argument('--seed',type=int,default=1043)
    p.add_argument('--name',help='Fresh output directory under ai/outputs, defaults to mode')
    p.add_argument('--smoke-train',type=int,default=32)
    p.add_argument('--smoke-val',type=int,default=20)
    p.add_argument('--dry-run',action='store_true',help='Save exact configuration/split lists without downloading or training')
    args=p.parse_args()
    configure_runtime()
    root,report,data=prepared_dataset(args.dataset,splits=('train','val'))
    epochs=args.epochs if args.epochs is not None else (1 if args.mode=='smoke' else 30)
    imgsz=args.imgsz if args.imgsz is not None else (640 if args.mode=='smoke' else 960)
    require(epochs>=1 and imgsz>=128 and imgsz%32==0 and args.batch>=1,'Invalid epochs, batch or image size (multiple of 32 required)')
    out=new_output(args.name or args.mode)
    selected={}
    # A training-specific YAML intentionally has no test entry.
    training_data={'path':str(root),'names':{0:'towline'}}
    for split,limit in [('train',args.smoke_train),('val',args.smoke_val)]:
        images=sorted((root/'images'/split).glob('*.png'))
        if args.mode=='smoke':
            require(1<=limit<=len(images),f'Invalid smoke {split} subset size')
            random.Random(args.seed+(0 if split=='train' else 1)).shuffle(images)
            images=sorted(images[:limit])
        selected[split]=[image.stem for image in images]
        listing=out/f'{split}.txt'
        # Do not resolve image symlinks: labels live beside the derived image path.
        listing.write_text('\n'.join(str(image.absolute()) for image in images)+'\n')
        training_data[split]=str(listing)
    training_yaml=out/'training_data.yaml'
    training_yaml.write_text(yaml.safe_dump(training_data,sort_keys=False))
    device=select_device(args.device)
    kwargs=dict(data=str(training_yaml),epochs=epochs,imgsz=imgsz,batch=args.batch,device=device,
                seed=args.seed,deterministic=True,workers=0,cache=False,amp=False,
                optimizer='AdamW',lr0=.001,weight_decay=.0005,patience=10,
                mask_ratio=1,overlap_mask=False,mosaic=0.,mixup=0.,copy_paste=0.,
                degrees=1.,translate=.02,scale=.1,shear=0.,perspective=0.,flipud=0.,fliplr=0.,
                hsv_h=.01,hsv_s=.1,hsv_v=.1,close_mosaic=0,
                val=True,plots=True,project=str(out),name='train',exist_ok=False,save=True)
    run={'status':'planned' if args.dry_run else 'running','mode':args.mode,'arguments':vars(args)|{'dataset':str(args.dataset)},
         'training_arguments':kwargs,'selected_ids':selected,'test_used':False,
         'prepared_dataset_sha256':sha256(root/'preparation_summary.json'),'environment':versions()}
    write_json(out/'run_config.json',run)
    freeze=subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True)
    (out/'requirements-frozen.txt').write_text(freeze)
    if args.dry_run:
        print(json.dumps({'dry_run':True,'output':str(out),'train_count':len(selected['train']),'val_count':len(selected['val']),**kwargs},indent=2));return
    started=time.monotonic()
    try:
        model,checkpoint=load_model(args.model,pretrained=True)
        run['pretrained_checkpoint']=str(checkpoint);run['pretrained_sha256']=sha256(checkpoint)
        write_json(out/'run_config.json',run)
        model.train(**kwargs)
        best=Path(model.trainer.best)
        require(best.is_file(),'Training did not produce best.pt')
        run.update(status='complete',elapsed_seconds=time.monotonic()-started,best_weights=str(best),best_sha256=sha256(best))
    except BaseException as error:
        run.update(status='failed',elapsed_seconds=time.monotonic()-started,error=str(error));raise
    finally:
        write_json(out/'run_config.json',run)
    print(json.dumps({'output':str(out),'best_weights':str(best),'elapsed_seconds':run['elapsed_seconds']},indent=2))


if __name__=='__main__':main()
