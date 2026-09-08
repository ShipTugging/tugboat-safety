from pathlib import Path
import json
import yaml
from .common import SPLITS, require, sha256


def prepared_dataset(path, splits=SPLITS, verify_source=False):
    root=Path(path).resolve()
    report=json.loads((root/'preparation_summary.json').read_text())
    require(report['status']=='complete','Run prepare_yolo_seg.py successfully first')
    data=yaml.safe_load((root/'dataset.yaml').read_text())
    require(data['names']=={0:'towline'},'Expected one class, 0=towline')
    require(Path(data['path']).resolve()==root,'Dataset moved; rerun preparation at its new location')
    ids=set()
    for split in splits:
        images=sorted((root/'images'/split).glob('*.png'))
        require(len(images)==report['counts'][split],f'{split}: image count changed')
        require({p.stem for p in images}=={p.stem for p in (root/'labels'/split).glob('*.txt')},'Missing or orphan label')
        for image in images:
            require(image.stem not in ids,'Cross-split sample ID duplication')
            ids.add(image.stem)
            require(image.is_file(),f'Broken image link: {image}')
            label=root/'labels'/split/f'{image.stem}.txt'
            require(sha256(label)==report['label_sha256'][str(label.relative_to(root))],f'Label changed: {label}')
    if verify_source:
        for relative,digest in report['source_sha256'].items():
            require(sha256(Path(report['source'])/relative)==digest,f'Source changed: {relative}')
    return root,report,data
