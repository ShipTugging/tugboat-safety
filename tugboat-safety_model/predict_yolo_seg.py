#!/usr/bin/env python3
"""Save original-resolution towline masks and bright green overlays for RGB inputs."""
import argparse
from pathlib import Path
from utils.common import configure_runtime, select_device, load_model, new_output, save_prediction, connectivity, write_json, sha256, require


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',required=True,type=Path)
    p.add_argument('--source',required=True,type=Path,help='Image or directory of images (recursive)')
    p.add_argument('--name',default='prediction')
    p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--device',default='auto')
    p.add_argument('--conf',type=float,default=.25)
    p.add_argument('--limit',type=int,help='Optional deterministic first N input images')
    args=p.parse_args()
    require(args.source.exists(),'Input does not exist')
    files=([args.source] if args.source.is_file() else sorted(p for p in args.source.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.webp'}))
    if args.limit is not None:
        require(args.limit>0,'Limit must be positive');files=files[:args.limit]
    require(bool(files),'No input RGB images found')
    require(0<=args.conf<=1,'Confidence must be in [0,1]')
    configure_runtime();model,weights=load_model(args.weights);out=new_output(args.name)
    base=args.source.parent if args.source.is_file() else args.source
    records=[]
    for file in files:
        result=model.predict(str(file),imgsz=args.imgsz,device=select_device(args.device),conf=args.conf,
                             retina_masks=True,classes=[0],save=False,verbose=False)[0]
        relative=file.relative_to(base)
        # Preserve subfolders and original extension in stem to avoid JPG/PNG collisions.
        mask=save_prediction(result,out/relative.parent,relative.name)
        records.append({'source':str(file.resolve()),'mask':str(relative.parent/f'{relative.name}_mask.png'),
                        'overlay':str(relative.parent/f'{relative.name}_overlay.png'),'foreground_pixels':int(mask.sum()),**connectivity(mask)})
    write_json(out/'predictions.json',{'weights':str(weights),'weights_sha256':sha256(weights),'imgsz':args.imgsz,'confidence':args.conf,
                                      'device':select_device(args.device),'images':records})
    print(f'Saved {len(records)} predictions to {out}')


if __name__=='__main__':main()
