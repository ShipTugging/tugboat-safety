import { Euler, PerspectiveCamera, Vector3 } from 'three';
import { createInitialTelemetry } from '../simulation/physics';
import { getTowlineAnchors } from '../simulation/towline';
import { seededRandom } from './environment';
import { type DatasetConfig, type Split } from './config';
import type { CaptureSample } from './types';

export interface TowlineScenario {
  scenarioSeed: number; scenarioGroup: string; split: Split; sagBin: number;
  sagParameter: number; radius: number; brightness: number;
  merchantDistance: number; lateralOffset: number; heading: number;
  cameraMount: number[]; cameraJitter: number[];
  sunPosition: [number,number,number]; exposure: number; centerlinePoints: number;
}
// Bijective uint32 mapping: distinct attempt slots cannot collide within a run.
export function scenarioSeed(seed:number, slot:number) { return (seed + Math.imul(slot+1, 2654435761)) >>> 0; }
export function planScenarios(c:DatasetConfig) {
  const counts=[Math.floor(c.datasetSize*c.split.train),Math.floor(c.datasetSize*c.split.val),Math.floor(c.datasetSize*c.split.test)];
  const fractions=[c.split.train,c.split.val,c.split.test];
  const order=[0,1,2].sort((a,b)=>(c.datasetSize*fractions[b]-counts[b])-(c.datasetSize*fractions[a]-counts[a]));
  for(let i=0,left=c.datasetSize-counts.reduce((a,b)=>a+b,0);i<left;i++) counts[order[i]]++;
  const plan=counts.flatMap((count,i)=>Array.from({length:count},(_,j)=>({split:(['train','val','test'] as Split[])[i],sagBin:(j+c.seed+i)%5})));
  const random=seededRandom(c.seed ^ 0x5a17);
  for(let i=plan.length-1;i>0;i--) {const j=Math.floor(random()*(i+1));[plan[i],plan[j]]=[plan[j],plan[i]];}
  return plan;
}
export function makeScenario(c:DatasetConfig,index:number,attempt:number,plan:ReturnType<typeof planScenarios>[number]):CaptureSample {
  const seed=scenarioSeed(c.seed,index*c.validation.maxAttemptsPerSample+attempt), random=seededRandom(seed);
  const pick=(r:number[])=>r[0]+random()*(r[1]-r[0]);
  const binWidth=(c.towline.sagRange[1]-c.towline.sagRange[0])/5;
  const s:TowlineScenario={
    scenarioSeed:seed,scenarioGroup:`scenario_${seed}`,split:plan.split,sagBin:plan.sagBin,
    sagParameter:c.towline.sagRange[0]+(plan.sagBin+random())*binWidth,
    radius:pick(c.towline.radiusRange),brightness:pick(c.towline.brightnessRange),
    merchantDistance:pick(c.merchantVessel.distanceRange),lateralOffset:pick(c.merchantVessel.lateralOffsetRange),heading:pick(c.merchantVessel.headingRange),
    cameraMount:[...c.camera.mount],cameraJitter:[pick(c.camera.pitchRange),pick(c.camera.yawRange),pick(c.camera.rollRange)],
    sunPosition:[pick(c.environment.sunXRange),pick(c.environment.sunYRange),pick(c.environment.sunZRange)],
    exposure:pick(c.environment.brightnessRange),centerlinePoints:c.centerlinePoints,
  };
  const time=pick(c.environment.timeRange), telemetry=createInitialTelemetry(time*1000);
  telemetry.tugPosition=[0,.4,0]; telemetry.tugRotation=[0,0,0];
  // Translate merchant so the rotated chock lands at the sampled offset from the tug.
  const chock=new Vector3(3.5,2.7,-34.2).applyEuler(new Euler(0,s.heading,0));
  telemetry.shipPosition=[s.lateralOffset-chock.x,0,s.merchantDistance-chock.z];
  const anchors=getTowlineAnchors(telemetry,s.heading);
  telemetry.lineStartPoint=anchors.start.toArray();telemetry.lineEndPoint=anchors.end.toArray();
  return {id:`${index}:${attempt}:${seed}`,index,time,width:c.imageWidth,height:c.imageHeight,towline:s,telemetry,
    params:{tugSteeringAngle:0,towLineLength:anchors.start.distanceTo(anchors.end),shipSpeed:0,propellerRpm:0,cameraMode:'TUG_AFT_DECK',timeOfDay:'day',quickReleaseActive:false,soundEnabled:false,
      fogDensity:pick(c.environment.fogRange),sunIntensity:pick(c.environment.lightIntensityRange),waveStrength:pick(c.environment.waveStrengthRange),cameraFov:pick(c.camera.fovRange)}};
}
export function applyTowlineCamera(camera:PerspectiveCamera,sample:CaptureSample) {
  const s=sample.towline!, anchors=getTowlineAnchors(sample.telemetry,s.heading);
  camera.position.copy(new Vector3(...s.cameraMount).applyEuler(new Euler(...sample.telemetry.tugRotation)).add(new Vector3(...sample.telemetry.tugPosition)));
  const target=anchors.start.clone().lerp(anchors.end,.5);target.y-=s.sagParameter*.5;
  camera.up.set(0,1,0);camera.lookAt(target);
  camera.rotateX(s.cameraJitter[0]);camera.rotateY(s.cameraJitter[1]);camera.rotateZ(s.cameraJitter[2]);
  camera.fov=sample.params.cameraFov!;camera.aspect=sample.width/sample.height;camera.near=.15;camera.far=1800;
  camera.updateProjectionMatrix();camera.updateMatrixWorld(true);
}
