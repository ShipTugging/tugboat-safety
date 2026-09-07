import { PerspectiveCamera, Vector3, type Curve } from 'three';
import type { CaptureSample } from './types';
import { GENERATOR_VERSION, type DatasetConfig } from './config';

export function measureCenterline(points:Vector3[]) {
  if(points.length<3 || points.some(p=>!p.toArray().every(Number.isFinite))) throw new Error('Invalid centerline coordinates');
  const a=points[0],b=points[points.length-1],chord=b.clone().sub(a),chordLength=chord.length();
  if(chordLength<1e-6) throw new Error('Degenerate endpoint chord');
  const unit=chord.clone().divideScalar(chordLength);
  let curveLength=0,maxSagDistance=0;
  const curvature:number[]=[];
  points.forEach((p,i)=>{
    const offset=p.clone().sub(a);
    maxSagDistance=Math.max(maxSagDistance,offset.clone().addScaledVector(unit,-offset.dot(unit)).length());
    if(i)curveLength+=p.distanceTo(points[i-1]);
    if(i>0&&i<points.length-1) {
      const left=p.clone().sub(points[i-1]),right=points[i+1].clone().sub(p);
      const denominator=left.length()*right.length()*points[i+1].distanceTo(points[i-1]);
      curvature.push(denominator>1e-12?2*left.cross(right).length()/denominator:0);
    }
  });
  const sagRatioGT=maxSagDistance/chordLength;
  if(!Number.isFinite(sagRatioGT)) throw new Error('Non-finite Sag Ratio');
  return {chordLength,curveLength,maxSagDistance,sagRatioGT,lengthRatioGT:curveLength/chordLength,
    meanCurvatureGT:curvature.reduce((a,b)=>a+b,0)/curvature.length,maxCurvatureGT:Math.max(...curvature)};
}
export function makeMetadata(sample:CaptureSample,camera:PerspectiveCamera,curve:Curve<Vector3>,mask:Uint8Array) {
  const s=sample.towline!,points=curve.getSpacedPoints(s.centerlinePoints-1),metrics=measureCenterline(points);
  const centerlineImage:number[][]=[],centerlineInFrame:boolean[]=[],centerlineVisible:boolean[]=[];
  for(const point of points) {
    const p=point.clone().project(camera),depth=-point.clone().applyMatrix4(camera.matrixWorldInverse).z;
    if(!p.toArray().every(Number.isFinite)||depth<=camera.near||depth>=camera.far) throw new Error('Centerline cannot be projected inside camera clipping planes');
    const u=(p.x+1)*sample.width/2,v=(1-p.y)*sample.height/2;
    const inside=u>=0&&u<sample.width&&v>=0&&v<sample.height;
    centerlineImage.push([u,v]);centerlineInFrame.push(inside);
    let visible=false;
    if(inside)for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++) {
      const x=Math.floor(u)+dx,y=Math.floor(v)+dy;
      if(x>=0&&x<sample.width&&y>=0&&y<sample.height&&mask[y*sample.width+x]===255)visible=true;
    }
    centerlineVisible.push(visible);
  }
  const first=centerlineImage[0],last=centerlineImage[centerlineImage.length-1];
  const id=`sample_${String(sample.index+1).padStart(6,'0')}`;
  return {
    generatorVersion:GENERATOR_VERSION,sampleId:id,split:s.split,scenarioSeed:s.scenarioSeed,scenarioGroup:s.scenarioGroup,
    simulationTime:sample.time,sagBin:s.sagBin,
    paths:{image:`images/${s.split}/${id}.png`,mask:`masks/${s.split}/${id}_mask.png`,metadata:`metadata/${s.split}/${id}.json`},
    image:{width:sample.width,height:sample.height,format:'PNG',coordinateConvention:'pixel edges, origin top-left; u right, v down; pixel centers at (x+0.5,y+0.5)'},
    camera:{mode:'TUG_AFT_DECK',mount:s.cameraMount,jitter:s.cameraJitter,fov:camera.fov,position:camera.position.toArray(),rotation:camera.rotation.toArray().slice(0,3),rotationOrder:camera.rotation.order,quaternion:camera.quaternion.toArray(),near:camera.near,far:camera.far,aspect:camera.aspect,projectionMatrix:camera.projectionMatrix.toArray()},
    tug:{position:sample.telemetry.tugPosition,rotation:sample.telemetry.tugRotation,heading:sample.telemetry.tugRotation[1]},
    merchantVessel:{position:sample.telemetry.shipPosition,heading:s.heading,distance:s.merchantDistance,lateralOffset:s.lateralOffset},
    towline:{sagParameter:s.sagParameter,radius:s.radius,brightness:s.brightness,...metrics,centerlineWorld:points.map(p=>p.toArray()),centerlineImage,centerlineInFrame,centerlineVisible},
    environment:{timeOfDay:sample.params.timeOfDay,lightIntensity:sample.params.sunIntensity,sunPosition:s.sunPosition,exposure:s.exposure,fog:sample.params.fogDensity,waveStrength:sample.params.waveStrength,shipSpeed:sample.params.shipSpeed,propellerRpm:sample.params.propellerRpm},
    validation:{maskPixels:mask.reduce((n,v)=>n+(v===255?1:0),0),inFrameFraction:centerlineInFrame.filter(Boolean).length/points.length,visibleFraction:centerlineVisible.filter(Boolean).length/points.length,projectedChordPixels:Math.hypot(first[0]-last[0],first[1]-last[1])},
  };
}
export type TowlineMetadata=ReturnType<typeof makeMetadata>;
export interface TowlineCapture {rgb:string;mask:string;metadata:TowlineMetadata}
export function rejectionReasons(m:TowlineMetadata,c:DatasetConfig) {
  const v=m.validation,r=c.validation,reasons:string[]=[];
  if(v.maskPixels<r.minMaskPixels)reasons.push('empty or tiny mask');
  if(v.inFrameFraction<r.minInFrameFraction)reasons.push('insufficient centerline in frame');
  if(v.visibleFraction<r.minVisibleFraction)reasons.push('excessive occlusion');
  if(v.projectedChordPixels<r.minProjectedChordPixels)reasons.push('insufficient projected span');
  return reasons;
}
