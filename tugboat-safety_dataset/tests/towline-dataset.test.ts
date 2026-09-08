import test from 'node:test';
import assert from 'node:assert/strict';
import { Vector3 } from 'three';
import { datasetConfig,validateConfig } from '../src/dataset/config';
import { makeScenario,planScenarios,scenarioSeed,applyTowlineCamera } from '../src/dataset/scenarios';
import { measureCenterline,makeMetadata } from '../src/dataset/geometry';
import { createTowlineCurve,getTowlineAnchors } from '../src/simulation/towline';
import { PerspectiveCamera } from 'three';

test('geometric sag uses perpendicular chord distance, including sloping endpoints',()=>{
  const points=[new Vector3(0,0,0),new Vector3(1,0,0),new Vector3(2,2,0)];
  const m=measureCenterline(points);
  assert.ok(Math.abs(m.maxSagDistance-Math.SQRT1_2)<1e-12);
  assert.ok(Math.abs(m.sagRatioGT-.25)<1e-12);
  assert.ok(Math.abs(m.curveLength-(1+Math.sqrt(5)))<1e-12);
  const straight=measureCenterline([new Vector3(),new Vector3(1,1,1),new Vector3(2,2,2)]);
  assert.ok(straight.sagRatioGT<1e-12);assert.equal(straight.lengthRatioGT,1);assert.equal(straight.maxCurvatureGT,0);
  assert.throws(()=>measureCenterline([new Vector3(),new Vector3(1,1,1),new Vector3()]));
  assert.throws(()=>measureCenterline([new Vector3(),new Vector3(NaN,1,1),new Vector3(2,2,2)]));
});
test('500 independent scenarios balance each split and all five bins',()=>{
  const plan=planScenarios(datasetConfig);
  for(const [split,count] of [['train',350],['val',100],['test',50]] as const) {
    const group=plan.filter(s=>s.split===split);assert.equal(group.length,count);
    for(let bin=0;bin<5;bin++)assert.equal(group.filter(s=>s.sagBin===bin).length,count/5);
  }
  const seeds=new Set(plan.flatMap((_,i)=>Array.from({length:20},(_,a)=>scenarioSeed(datasetConfig.seed,i*20+a))));
  assert.equal(seeds.size,10000);
  assert.deepEqual(plan,planScenarios(datasetConfig));
});
test('scenario reproduction and projection preserve exact rendered curve and rotated anchor',()=>{
  const plan=planScenarios(datasetConfig);
  for(let i=0;i<50;i++) {
    const sample=makeScenario(datasetConfig,i,0,plan[i]);assert.deepEqual(sample,makeScenario(datasetConfig,i,0,plan[i]));
    const s=sample.towline!,anchors=getTowlineAnchors(sample.telemetry,s.heading);
    assert.ok(Math.abs(anchors.start.x-s.lateralOffset)<1e-10);assert.ok(Math.abs(anchors.start.z-s.merchantDistance)<1e-10);
    const curve=createTowlineCurve(anchors.start,anchors.end,0,999,'CRITICAL',s.sagParameter);
    const camera=new PerspectiveCamera();applyTowlineCamera(camera,sample);
    const metadata=makeMetadata(sample,camera,curve,new Uint8Array(sample.width*sample.height));
    assert.equal(metadata.towline.centerlineWorld.length,50);
    assert.ok(metadata.towline.lengthRatioGT>1);
    assert.ok(metadata.towline.sagRatioGT>0);
    assert.ok(metadata.validation.inFrameFraction>=.7);
    assert.ok(metadata.validation.projectedChordPixels>60);
  }
});
test('invalid configuration is rejected before rendering; small split rounding totals exactly',()=>{
  for(const value of [NaN,0,5001,1.5])assert.throws(()=>validateConfig({...datasetConfig,datasetSize:value}));
  assert.throws(()=>validateConfig({...datasetConfig,split:{train:.7,val:.3,test:.1}}));
  assert.throws(()=>validateConfig({...datasetConfig,towline:{...datasetConfig.towline,radiusRange:[-1,1]}}));
  for(let n=1;n<25;n++)assert.equal(planScenarios({...datasetConfig,datasetSize:n}).length,n);
});
