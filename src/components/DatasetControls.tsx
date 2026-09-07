import React from 'react';
import { Download, Camera, Square, Shuffle } from 'lucide-react';
import type { DatasetController } from '../hooks/useDatasetExporter';

export function DatasetControls({dataset:d}:{dataset:DatasetController;onRandomize:()=>void}) {
  return <section className="dataset-controls" aria-label="Towline dataset generator">
    <label className="dataset-toggle"><span><Camera size={15}/>Dataset Generation · 개발자</span><input type="checkbox" role="switch" checked={d.enabled} disabled={d.busy} onChange={e=>d.setEnabled(e.target.checked)}/></label>
    {d.enabled && <div className="dataset-body">
      <div className="dataset-settings">
        {([{label:'Dataset Size',value:d.count,set:d.setCount,min:1,max:5000},{label:'Random Seed',value:d.seed,set:d.setSeed,min:0,max:4294967295},{label:'Image Width',value:d.width,set:d.setWidth,min:64,max:4096},{label:'Image Height',value:d.height,set:d.setHeight,min:64,max:4096}]).map(f=><label key={f.label}>{f.label}<input aria-label={f.label} type="number" min={f.min} max={f.max} step={1} value={Number.isFinite(f.value)?f.value:''} disabled={d.busy} onChange={e=>f.set(e.target.valueAsNumber)}/></label>)}
      </div>
      <div className="dataset-actions">
        {d.busy?<button onClick={d.cancel}><Square size={13}/>생성 취소</button>:<button className="dataset-primary" onClick={d.start}>Generate Pilot Dataset</button>}
      </div>
      <div className="dataset-actions"><button onClick={d.single} disabled={d.busy}>Generate Single Sample</button><button onClick={d.randomPreview} disabled={d.busy}><Shuffle size={14}/>Preview Random Scenario</button></div>
      {(d.busy||d.status)&&<div className="dataset-progress"><progress max={100} value={d.progress}/><span role="status">{d.status}</span></div>}
      {d.error&&<p role="alert" className="dataset-error">{d.error}</p>}
      {d.archiveBlob&&<button className="dataset-download" onClick={d.download}><Download size={14}/>ZIP 다운로드 · {(d.archiveBlob.size/1048576).toFixed(1)} MB</button>}
      {d.preview&&<details className="dataset-preview" open><summary>RGB / Mask · Sag Ratio GT {d.preview.metadata.towline.sagRatioGT.toFixed(5)}</summary>
        <img style={{width:'100%'}} src={d.preview.rgb} alt="Towline RGB preview"/><img style={{width:'100%'}} src={d.preview.mask} alt="Towline binary mask preview"/>
        <details><summary>Scenario parameters / ground truth</summary><pre style={{maxHeight:260,overflow:'auto',fontSize:10}}>{JSON.stringify(d.preview.metadata,null,2)}</pre></details>
      </details>}
      <p className="dataset-note">PNG + Binary mask + JSON · 70/20/10 split · 50 centerline points<br/>11장 이상: Chrome 폴더 저장 또는 npm run dataset:generate<br/>Sag는 기하학적 특징이며 장력·위험 등급이 아닙니다.</p>
    </div>}
  </section>;
}
