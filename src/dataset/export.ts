import readme from '../../README_DATASET.md?raw';
import { datasetConfig, GENERATOR_VERSION, validateConfig, type DatasetConfig } from './config';
import { rejectionReasons, type TowlineCapture, type TowlineMetadata } from './geometry';
import { makeScenario, planScenarios } from './scenarios';
import type { CaptureSample } from './types';

export type DatasetSink=(path:string,data:string,encoding:'base64'|'utf8')=>Promise<void>;
export const csvColumns=['sample_id','split','scenario_seed','image_path','mask_path','metadata_path','sag_parameter','sag_ratio_gt','length_ratio_gt','curve_length','chord_length','max_sag_distance','merchant_distance','merchant_lateral_offset','camera_fov','camera_pitch','camera_roll','camera_yaw','mean_curvature_gt','max_curvature_gt','sag_bin','mask_pixels','visible_fraction'];
export function csvRow(m:TowlineMetadata) {
  const t=m.towline,c=m.camera,v=m.merchantVessel;
  return [m.sampleId,m.split,m.scenarioSeed,m.paths.image,m.paths.mask,m.paths.metadata,t.sagParameter,t.sagRatioGT,t.lengthRatioGT,t.curveLength,t.chordLength,t.maxSagDistance,v.distance,v.lateralOffset,c.fov,c.rotation[0],c.rotation[2],c.rotation[1],t.meanCurvatureGT,t.maxCurvatureGT,m.sagBin,m.validation.maskPixels,m.validation.visibleFraction].join(',');
}
export async function generateDataset(config:DatasetConfig,capture:(sample:CaptureSample)=>Promise<TowlineCapture>,write:DatasetSink,signal:AbortSignal,progress:(generated:number,rejected:number,last?:TowlineCapture)=>void) {
  validateConfig(config);
  const plan=planScenarios(config),rows:string[]=[csvColumns.join(',')];
  const rejectionLog:Array<{sampleIndex:number;attempt:number;scenarioSeed:number;reasons:string[]}>=[];
  const counts={train:0,val:0,test:0},bins=[0,0,0,0,0];
  let generatedSamples=0,status='incomplete';
  const startedAt=new Date().toISOString();
  const manifest=()=>({generatorVersion:GENERATOR_VERSION,startedAt,finishedAt:new Date().toISOString(),status,configuration:config,generatedSamples,rejectedSamples:rejectionLog.length,splitCounts:counts,sagBinCounts:bins});
  const text=(path:string,value:unknown)=>write(path,JSON.stringify(value,null,2)+'\n','utf8');
  await write('README_DATASET.md',readme,'utf8');
  await text('generation_manifest.json',manifest());
  try {
    for(let index=0;index<plan.length;index++) {
      let accepted=false;
      for(let attempt=0;attempt<config.validation.maxAttemptsPerSample;attempt++) {
        signal.throwIfAborted();
        const sample=makeScenario(config,index,attempt,plan[index]);
        let frame:TowlineCapture|undefined,reasons:string[]=[];
        try {frame=await capture(sample);reasons=rejectionReasons(frame.metadata,config);}
        catch(error) {
          if(error instanceof Error && /Invalid centerline|Degenerate endpoint|Non-finite Sag|Centerline cannot/.test(error.message)) reasons=[error.message];
          else throw error;
        }
        signal.throwIfAborted();
        if(reasons.length) {
          rejectionLog.push({sampleIndex:index,attempt,scenarioSeed:sample.towline!.scenarioSeed,reasons});
          progress(generatedSamples,rejectionLog.length);continue;
        }
        const f=frame!,m=f.metadata;
        await write(m.paths.image,f.rgb.split(',')[1],'base64');
        await write(m.paths.mask,f.mask.split(',')[1],'base64');
        await text(m.paths.metadata,m);
        rows.push(csvRow(m));generatedSamples++;counts[m.split]++;bins[m.sagBin]++;
        progress(generatedSamples,rejectionLog.length,f);accepted=true;break;
      }
      if(!accepted)throw new Error(`Sample ${index+1}: exceeded ${config.validation.maxAttemptsPerSample} attempts; inspect rejection_log.json`);
    }
    status='complete';
  } finally {
    await write('dataset_summary.csv',rows.join('\n')+'\n','utf8');
    await text('rejection_log.json',rejectionLog);
    await text('generation_manifest.json',manifest());
  }
  return manifest();
}
export {datasetConfig};
