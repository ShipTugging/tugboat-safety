import { useCallback, useEffect, useRef, useState } from 'react';
import { datasetConfig, generateDataset, type DatasetSink } from '../dataset/export';
import { validateConfig, type DatasetConfig } from '../dataset/config';
import type { TowlineCapture } from '../dataset/geometry';
import type { CaptureSample, SceneCaptureApi } from '../dataset/types';

interface Directory {
  getDirectoryHandle(name:string,options?:{create:boolean}):Promise<Directory>;
  getFileHandle(name:string,options?:{create:boolean}):Promise<{createWritable():Promise<{write(data:Blob|string):Promise<void>;close():Promise<void>}>}>;
  values():AsyncIterable<unknown>;
}
declare global {
  interface Window {
    showDirectoryPicker?:()=>Promise<Directory>;
    __towlineWrite?:DatasetSink;
    towlineDataset?:{ready:boolean;generate:(config?:Partial<DatasetConfig>)=>Promise<unknown>};
  }
}
async function directorySink():Promise<DatasetSink> {
  const parent=await window.showDirectoryPicker!();
  const root=await parent.getDirectoryHandle('towline_dataset',{create:true});
  for await(const _ of root.values())throw new Error('towline_dataset 폴더가 비어 있지 않습니다. 새 상위 폴더를 선택하세요.');
  return async(path,data,encoding)=>{
    const parts=path.split('/');let folder=root;
    for(const part of parts.slice(0,-1))folder=await folder.getDirectoryHandle(part,{create:true});
    const file=await folder.getFileHandle(parts[parts.length-1],{create:true}),writer=await file.createWritable();
    try {await writer.write(encoding==='utf8'?data:new Blob([Uint8Array.from(atob(data),c=>c.charCodeAt(0))]));}
    finally {await writer.close();}
  };
}
export function useDatasetExporter() {
  const [enabled,setEnabled]=useState(false),[count,setCount]=useState(datasetConfig.datasetSize),[seed,setSeed]=useState(datasetConfig.seed);
  const [width,setWidth]=useState(datasetConfig.imageWidth),[height,setHeight]=useState(datasetConfig.imageHeight);
  const [busy,setBusy]=useState(false),[progress,setProgress]=useState(0),[rejected,setRejected]=useState(0);
  const [status,setStatus]=useState(''),[error,setError]=useState(''),[sample,setSample]=useState<CaptureSample|null>(null);
  const [preview,setPreview]=useState<TowlineCapture|null>(null),[archiveBlob,setArchiveBlob]=useState<Blob|null>(null);
  const captureApi=useRef<SceneCaptureApi|null>(null),abort=useRef<AbortController|null>(null),previewIndex=useRef(0);
  const [ready,setReady]=useState(false);
  const setCaptureApi=useCallback((api:SceneCaptureApi|null)=>{captureApi.current=api;setReady(!!api);},[]);
  useEffect(()=>()=>abort.current?.abort(),[]);
  const cancel=useCallback(()=>abort.current?.abort(),[]);
  const download=useCallback(async()=>{if(archiveBlob){const {saveAs}=await import('file-saver');saveAs(archiveBlob,'towline_dataset.zip');}},[archiveBlob]);
  const run=async(config:DatasetConfig,mode:'preview'|'export',external?:DatasetSink)=>{
    if(abort.current)throw new Error('Generation already in progress');
    validateConfig(config);
    if(!captureApi.current)throw new Error('3D scene is not ready');
    const controller=new AbortController();abort.current=controller;
    setEnabled(true);setBusy(true);setProgress(0);setRejected(0);setError('');setStatus('준비 중');setArchiveBlob(null);
    try {
      let write:DatasetSink=async()=>{},zip:import('jszip')|undefined;
      if(external)write=external;
      else if(mode==='export') {
        if(window.showDirectoryPicker && config.datasetSize>10)write=await directorySink();
        else {
          if(config.datasetSize>10)throw new Error('이 브라우저는 폴더 저장을 지원하지 않습니다. npm run dataset:generate 명령을 사용하세요. ZIP은 최대 10장입니다.');
          const {default:JSZip}=await import('jszip');zip=new JSZip();
          let bytes=0;
          write=async(path,data,encoding)=>{bytes+=data.length;if(bytes>128*1024*1024)throw new Error('ZIP memory limit exceeded; use the local generator');zip!.file(`towline_dataset/${path}`,data,{base64:encoding==='base64'});};
        }
      }
      const result=await generateDataset(config,async next=>{
        setSample(next);
        const frame=await captureApi.current!.capture(next.id,controller.signal);
        if(!frame.towlineCapture)throw new Error('Towline capture is unavailable');
        return frame.towlineCapture;
      },write,controller.signal,(generated,rejectedSamples,last)=>{
        setProgress(generated/config.datasetSize*100);setRejected(rejectedSamples);
        setStatus(`Generating: ${generated} / ${config.datasetSize} · Rejected: ${rejectedSamples}`);
        if(last)setPreview(last);
      });
      if(zip) {
        const blob=await zip.generateAsync({type:'blob',compression:'STORE'});controller.signal.throwIfAborted();setArchiveBlob(blob);
        const {saveAs}=await import('file-saver');saveAs(blob,'towline_dataset.zip');
      }
      setStatus(mode==='preview'?'미리보기 완료':`${result.generatedSamples}장 저장 완료 · 제외 ${result.rejectedSamples}장`);
      return result;
    } catch(caught) {
      setError(caught instanceof Error?caught.message:'Generation failed');setStatus(controller.signal.aborted?'생성 취소됨':'생성 실패');throw caught;
    } finally {abort.current=null;setSample(null);setBusy(false);}
  };
  const current=()=>({...structuredClone(datasetConfig),datasetSize:count,seed,imageWidth:width,imageHeight:height});
  const start=()=>{void run(current(),'export').catch(error=>setError(error instanceof Error?error.message:'Generation failed')); };
  const single=()=>{void run({...current(),datasetSize:1},'export').catch(error=>setError(error instanceof Error?error.message:'Generation failed')); };
  const randomPreview=()=>{void run({...current(),datasetSize:1,seed:(seed+previewIndex.current++)>>>0},'preview').catch(error=>setError(error instanceof Error?error.message:'Generation failed')); };
  // Local automation uses the same React scene and capture pipeline as the UI.
  useEffect(()=>{
    if(!import.meta.env.DEV)return;
    window.towlineDataset={ready,generate:async overrides=>{
      if(!window.__towlineWrite)throw new Error('Local file sink is not connected');
      return run({...structuredClone(datasetConfig),...overrides},'export',window.__towlineWrite);
    }};
    return ()=>{delete window.towlineDataset;};
  });
  return {enabled,setEnabled,count,setCount,seed,setSeed,width,setWidth,height,setHeight,busy,progress,rejected,status,error,sample,preview,archiveBlob,setCaptureApi,start,single,randomPreview,cancel,download};
}
export type DatasetController=ReturnType<typeof useDatasetExporter>;
